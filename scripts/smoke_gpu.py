#!/usr/bin/env python
"""GPU smoke test: kernels, memory at full length, train step speed, padding invariance, generation speed.

  scripts/smoke_gpu.py --data-dir DATA [--model Qwen/Qwen3.5-0.8B-Base]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.generate import generate_one  # noqa: E402
from qwen_abc.train import collate, encode, load_model, read_examples, token_nll_sum  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", type=Path, required=True)
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
ap.add_argument("--budgets", type=int, nargs="+", default=[6144, 8192, 12288, 16384])
args = ap.parse_args()

report = {"gpu": torch.cuda.get_device_name(0)}
try:
    import fla  # noqa: F401

    report["fla"] = getattr(fla, "__version__", "present")
except Exception as exc:  # noqa: BLE001
    report["fla"] = f"unavailable: {exc}"
try:
    import causal_conv1d  # noqa: F401

    report["causal_conv1d"] = True
except Exception:  # noqa: BLE001
    report["causal_conv1d"] = False

model, tok = load_model(args.model, "sdpa")
model.to("cuda")
model.train()  # checkpointing only engages in training mode
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
la = next(m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet"))
report["linear_attn_module"] = type(la).__name__
report["linear_attn_kernels"] = {k: getattr(getattr(la, k), "__module__", str(getattr(la, k)))
                                 for k in vars(la) if callable(getattr(la, k, None)) and not isinstance(getattr(la, k), torch.nn.Module)}
rows = read_examples(args.data_dir / "sft_train.jsonl")
ex = [e for e in (encode(tok, r, "sft", 8192) for r in rows) if e]
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

import random  # noqa: E402

from qwen_abc.train import make_batches  # noqa: E402

for budget in args.budgets:
    batches = make_batches(ex, budget, random.Random(0))
    # worst case for memory: the batch with the most padded tokens, then the longest sequence
    idx = max(batches, key=lambda b: (len(b) * max(len(ex[i]["input_ids"]) for i in b), len(b)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = {k: v.cuda() for k, v in collate(ex, idx, tok.pad_token_id).items()}
    ntok = int(batch["attention_mask"].sum())
    times = []
    try:
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, n = token_nll_sum(model, batch)
            (nll / n).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        report[f"train_budget_{budget}"] = {
            "tokens": ntok, "shape": list(batch["input_ids"].shape), "loss": float(nll.detach() / n),
            "sec_per_step": round(min(times[1:]), 3), "tok_per_sec": round(ntok / min(times[1:])),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except torch.OutOfMemoryError:
        report[f"train_budget_{budget}"] = {"tokens": ntok, "shape": list(batch["input_ids"].shape), "oom": True}
        opt.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)

# right padding must not change a sequence's loss
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    order = sorted(range(len(ex)), key=lambda i: len(ex[i]["input_ids"]))
    a = {k: v.cuda() for k, v in collate(ex, [order[0]], tok.pad_token_id).items()}
    b = {k: v.cuda() for k, v in collate(ex, [order[0], order[-1]], tok.pad_token_id).items()}
    na, _ = token_nll_sum(model, a)
    b0 = {k: v[:1] for k, v in b.items()}  # row 0 padded to the longest length
    nb, _ = token_nll_sum(model, b0)
report["right_padding_nll"] = [float(na), float(nb)]

model.config.use_cache = True
gen = generate_one(model, tok, rows[0]["prompt"], seed=0, max_new_tokens=512, temperature=0.8, top_p=0.95)
report["generate_512"] = {"tokens": gen["new_tokens"], "seconds": gen["seconds"],
                          "tok_per_sec": round(gen["new_tokens"] / max(gen["seconds"], 1e-6), 1)}
report["sample_head"] = gen["text"][:400]
report["sample_parse_ok"] = parse_abc(gen["text"]).ok
print(json.dumps(report, ensure_ascii=False, indent=1))
print("SMOKE_OK")
