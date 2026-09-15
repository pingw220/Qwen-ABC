"""Causal-LM training for ABC CPT and lyrics->ABC SFT (single process, one GPU).

Design choices, all recorded in the run's resolved config:

* full-parameter finetuning (0.8B fits comfortably on one 48 GB GPU), fp32
  master weights with bf16 autocast, gradient checkpointing;
* no packing: Qwen3.5 mixes Gated-DeltaNet linear attention with full
  attention, and packed sequences would leak recurrent state across
  document boundaries unless the kernels receive per-document cu_seqlens.
  Examples are length-bucketed and right-padded instead (right padding cannot
  influence earlier positions under causal masking);
* loss = sum of token NLL / number of supervised tokens in the whole
  optimizer update (not a mean of per-micro-batch means), so long and short
  songs weigh by tokens;
* SFT supervises completion tokens + EOS only; CPT supervises every token.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import yaml


@dataclass
class TrainConfig:
    mode: str  # "sft" | "cpt"
    data_dir: str
    output_dir: str
    model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    train_file: Optional[str] = None  # default: {mode}_train.jsonl in data_dir
    eval_file: Optional[str] = None  # default: {mode}_validation.jsonl
    train_limit: int = 0  # 0 = all; else first N examples in a seeded shuffle
    eval_limit: int = 256
    max_seq_len: int = 8192
    tokens_per_micro_batch: int = 16384  # padded tokens per forward pass
    tokens_per_update: int = 131072  # approx padded tokens per optimizer step
    epochs: float = 2.0
    max_steps: int = 0  # overrides epochs when > 0
    learning_rate: float = 5e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every_epoch: bool = True
    save_final: bool = True
    seed: int = 1234
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    log_every: int = 10
    extra: Dict = field(default_factory=dict)


def load_config(path: str, overrides: List[str]) -> TrainConfig:
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    for item in overrides:
        k, v = item.split("=", 1)
        raw[k] = yaml.safe_load(v)
    return TrainConfig(**raw)


# ------------------------------------------------------------------ data
def read_examples(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def encode(tokenizer, ex: dict, mode: str, max_len: int) -> Optional[dict]:
    eos = tokenizer.eos_token_id
    if mode == "sft":
        p = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"] + [eos]
        ids = p + c
        labels = [-100] * len(p) + c
    else:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
        labels = list(ids)
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": labels, "id": ex["id"]}


def make_batches(examples: List[dict], tokens_per_batch: int, rng: random.Random) -> List[List[int]]:
    """Length-bucketed batches whose padded size stays under the token budget."""
    order = sorted(range(len(examples)), key=lambda i: (len(examples[i]["input_ids"]), rng.random()))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(examples[i]["input_ids"])
        if cur and max(cur_max, L) * (len(cur) + 1) > tokens_per_batch:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(examples: List[dict], idx: List[int], pad_id: int) -> Dict[str, torch.Tensor]:
    L = max(len(examples[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), L), -100, dtype=torch.long)
    mask = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x = examples[i]
        n = len(x["input_ids"])
        ids[r, :n] = torch.tensor(x["input_ids"])
        labels[r, :n] = torch.tensor(x["labels"])
        mask[r, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _ce_chunk(hidden: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = torch.nn.functional.linear(hidden, weight).float()
    return torch.nn.functional.cross_entropy(logits, target, reduction="sum")


def token_nll_sum(model, batch: Dict[str, torch.Tensor], chunk: int = 2048) -> (torch.Tensor, int):
    """Summed NLL over supervised positions without materializing full logits.

    A 248k-row vocabulary makes (tokens x vocab) logits the dominant memory
    cost, so only supervised positions are projected, in checkpointed chunks.
    """
    hidden = model.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).last_hidden_state
    target = batch["labels"][:, 1:]
    sel = target != -100
    h = hidden[:, :-1][sel]
    t = target[sel]
    weight = model.get_output_embeddings().weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        if torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(
                _ce_chunk, h[i: i + chunk], t[i: i + chunk], weight, use_reentrant=False
            )
        else:
            total = total + _ce_chunk(h[i: i + chunk], t[i: i + chunk], weight)
    return total, int(t.numel())


# ------------------------------------------------------------------ model
def load_model(name: str, attn_impl: str, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn_impl)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def evaluate(model, examples: List[dict], pad_id: int, tokens_per_batch: int, device) -> Dict[str, float]:
    model.eval()
    rng = random.Random(0)
    total_nll, total_tok = 0.0, 0
    for idx in make_batches(examples, tokens_per_batch, rng):
        batch = {k: v.to(device) for k, v in collate(examples, idx, pad_id).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll, n = token_nll_sum(model, batch)
        total_nll += float(nll)
        total_tok += n
    model.train()
    loss = total_nll / max(total_tok, 1)
    return {"eval_loss": loss, "eval_ppl": math.exp(min(loss, 50)), "eval_tokens": total_tok}


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(int(total * cfg.warmup_ratio), 1)
    if step < warm:
        return cfg.learning_rate * (step + 1) / warm
    progress = (step - warm) / max(total - warm, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def train(cfg: TrainConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final_model").exists():
        raise SystemExit(f"{out}/final_model exists; refusing to overwrite a finished run")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")

    model, tok = load_model(cfg.model_name_or_path, cfg.attn_implementation)
    model.to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    pad_id = tok.pad_token_id

    data = Path(cfg.data_dir)
    train_rows = read_examples(Path(cfg.train_file) if cfg.train_file else data / f"{cfg.mode}_train.jsonl")
    eval_rows = read_examples(Path(cfg.eval_file) if cfg.eval_file else data / f"{cfg.mode}_validation.jsonl")
    rng = random.Random(cfg.seed)
    if cfg.train_limit:
        rng.shuffle(train_rows)
        train_rows = sorted(train_rows[: cfg.train_limit], key=lambda r: r["id"])
    if cfg.eval_limit:
        eval_rows = eval_rows[:: max(len(eval_rows) // cfg.eval_limit, 1)][: cfg.eval_limit]
    train_ex = [e for e in (encode(tok, r, cfg.mode, cfg.max_seq_len) for r in train_rows) if e]
    eval_ex = [e for e in (encode(tok, r, cfg.mode, cfg.max_seq_len) for r in eval_rows) if e]
    sup_tokens = sum(sum(1 for x in e["labels"] if x != -100) for e in train_ex)
    all_tokens = sum(len(e["input_ids"]) for e in train_ex)

    micro_per_epoch = len(make_batches(train_ex, cfg.tokens_per_micro_batch, random.Random(0)))
    accum = max(round(cfg.tokens_per_update / cfg.tokens_per_micro_batch), 1)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total_steps = cfg.max_steps or math.ceil(steps_per_epoch * cfg.epochs)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "norm" in n or "A_log" in n or "dt_bias" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, cfg.adam_beta2), eps=1e-8, fused=True,
    )

    resolved = asdict(cfg) | {
        "train_examples": len(train_ex), "eval_examples": len(eval_ex),
        "train_dropped_over_max_len": len(train_rows) - len(train_ex),
        "train_tokens": all_tokens, "train_supervised_tokens": sup_tokens,
        "micro_batches_per_epoch": micro_per_epoch, "grad_accum": accum,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "n_params": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    import transformers

    resolved["transformers"] = transformers.__version__
    (out / "resolved_config.json").write_text(json.dumps(resolved, indent=1), encoding="utf-8")
    print(json.dumps(resolved, indent=1), flush=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def emit(row: dict) -> None:
        row["time"] = round(time.time() - t0, 1)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    t0 = time.time()
    emit({"step": 0, **evaluate(model, eval_ex, pad_id, cfg.tokens_per_micro_batch, device)})
    model.train()
    step, epoch = 0, 0
    tokens_seen = 0
    done = False
    while not done:
        batches = make_batches(train_ex, cfg.tokens_per_micro_batch, random.Random(cfg.seed + epoch))
        for start in range(0, len(batches), accum):
            group = batches[start: start + accum]
            n_sup = sum(sum(1 for x in train_ex[i]["labels"][1:] if x != -100) for b in group for i in b)
            lr = lr_at(step, total_steps, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for idx in group:
                batch = {k: v.to(device, non_blocking=True) for k, v in collate(train_ex, idx, pad_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nll, _ = token_nll_sum(model, batch)
                (nll / n_sup).backward()
                loss_sum += float(nll)
                tokens_seen += int(batch["attention_mask"].sum())
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                emit({"step": step, "epoch": round(epoch + (start + len(group)) / len(batches), 3),
                      "loss": loss_sum / n_sup, "lr": lr, "grad_norm": float(gnorm), "tokens_seen": tokens_seen,
                      "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            if cfg.eval_every and step % cfg.eval_every == 0:
                emit({"step": step, **evaluate(model, eval_ex, pad_id, cfg.tokens_per_micro_batch, device)})
            if step >= total_steps:
                done = True
                break
        epoch += 1
        if cfg.save_every_epoch and not done:
            save(model, tok, out / f"epoch-{epoch}", resolved)
    emit({"step": step, "final": True, **evaluate(model, eval_ex, pad_id, cfg.tokens_per_micro_batch, device)})
    if cfg.save_final:
        save(model, tok, out / "final_model", resolved)
    emit({"step": step, "done": True, "elapsed_hours": round((time.time() - t0) / 3600, 3)})


def save(model, tok, path: Path, resolved: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # bf16 on disk (half the size); training keeps fp32 master weights
    state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)
    tok.save_pretrained(path)
    (path / "training_config.json").write_text(json.dumps(resolved, indent=1), encoding="utf-8")
