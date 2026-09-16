#!/usr/bin/env python
"""GPU infrastructure checks for the long-structure experiments.

  scripts/check_infra.py --checkpoint CKPT --data-dir DATA --output OUT.json

1. batched (left-padded) greedy generation == batch-1 greedy generation?
2. sampled generation throughput at batch 1 vs batch B;
3. training-step peak memory / time for one sequence of 8k / 16k / 32k tokens
   (train mode, gradient checkpointing, chunked CE) = the practical context limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.generate import generate_batch, generate_one, load_for_generation  # noqa: E402
from qwen_abc.train import load_model, token_nll_sum  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-greedy", type=int, default=6)
    ap.add_argument("--greedy-tokens", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lengths", default="8192,16384,32768")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data_dir / "sft_test.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["song_id"])
    out = {"checkpoint": args.checkpoint, "gpu": torch.cuda.get_device_name(0)}

    model, tok = load_for_generation(args.checkpoint)
    prompts = [r["prompt"] for r in rows[:: max(len(rows) // args.n_greedy, 1)][: args.n_greedy]]
    single = [generate_one(model, tok, p, 0, args.greedy_tokens, 0.0, 1.0) for p in prompts]
    batched = generate_batch(model, tok, prompts, 0, args.greedy_tokens, 0.0, 1.0)
    cmp = []
    for s, b in zip(single, batched):
        ts, tb = tok(s["text"], add_special_tokens=False)["input_ids"], tok(b["text"], add_special_tokens=False)["input_ids"]
        agree = next((i for i, (x, y) in enumerate(zip(ts, tb)) if x != y), min(len(ts), len(tb)))
        cmp.append({"single_tokens": len(ts), "batched_tokens": len(tb), "identical": s["text"] == b["text"],
                    "agreeing_prefix_tokens": agree})
    out["greedy_batched_vs_single"] = cmp

    # throughput on full songs (sampled, T=0.8)
    tp = rows[: args.batch]
    t0 = time.time()
    s1 = [generate_one(model, tok, r["prompt"], 1000, 7680, 0.8, 0.95) for r in tp[:2]]
    one = sum(x["new_tokens"] for x in s1) / (time.time() - t0)
    t0 = time.time()
    sb = generate_batch(model, tok, [r["prompt"] for r in tp], 1000, 7680, 0.8, 0.95)
    el = time.time() - t0
    out["throughput"] = {"batch1_tokens_per_sec": round(one, 1),
                         f"batch{len(tp)}_tokens_per_sec": round(sum(x["new_tokens"] for x in sb) / el, 1),
                         f"batch{len(tp)}_seconds": round(el, 1),
                         "batch_new_tokens": [x["new_tokens"] for x in sb], "batch_hit_eos": [x["hit_eos"] for x in sb],
                         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(out, indent=1), flush=True)
    del model
    torch.cuda.empty_cache()

    # practical training context limit
    model, tok = load_model(args.checkpoint, "sdpa")
    model.cuda()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    mem = {}
    for L in [int(x) for x in args.lengths.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 50000, (1, L), device="cuda")
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, n = token_nll_sum(model, batch)
            (nll / n).backward()
            torch.cuda.synchronize()
            mem[L] = {"peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "fwd_bwd_seconds": round(time.time() - t0, 2), "finite_loss": bool(torch.isfinite(nll).item())}
        except torch.cuda.OutOfMemoryError as exc:
            mem[L] = {"oom": str(exc)[:200]}
        model.zero_grad(set_to_none=True)
    out["train_step_one_sequence"] = mem
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("CHECK_INFRA_DONE")


if __name__ == "__main__":
    main()
