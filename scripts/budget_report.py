#!/usr/bin/env python
"""Training budget accounting table (reports/LONG_STRUCTURE_EXPERIMENTS.md §Budget).

  scripts/budget_report.py --run E0=experiments/direct_sft_20260915_023500 --run E1=experiments/e1_v2_sft_<ts> ... [--json OUT]

Reads resolved_config.json, train_log.jsonl and job_info.txt of each run. Wall-clock is summed over
job segments (a segment ends where the next job logs "resumed"), GPU-hours = segment hours x world size.
Round-1 logs have no supervised-token counter, so supervised tokens seen are derived from the
per-epoch supervised count x epochs completed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def run_budget(path: Path) -> dict:
    cfg = json.loads((path / "resolved_config.json").read_text())
    rows = [json.loads(l) for l in (path / "train_log.jsonl").read_text().splitlines() if l.strip()]
    segments, cur, world = [], [], cfg.get("world_size", 1)
    for r in rows:
        if r.get("resumed") and cur:
            segments.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segments.append(cur)
    hours = gpu_hours = 0.0
    for seg in segments:
        w = next((r.get("world_size") for r in seg if r.get("world_size")), world)
        h = max(r.get("time", 0) for r in seg) / 3600
        hours += h
        gpu_hours += h * w
    steps = max((r["step"] for r in rows if "step" in r), default=0)
    tok = max((r.get("tokens_seen", 0) for r in rows), default=0)
    sup = max((r.get("sup_tokens_seen", 0) for r in rows), default=0)
    epochs_done = max((r.get("epoch", 0) for r in rows), default=0)
    if not sup and cfg.get("steps_per_epoch"):
        sup = int(cfg["train_supervised_tokens"] * steps / cfg["steps_per_epoch"])
    mem = max((r.get("max_mem_gb", 0) for r in rows), default=0)
    tps = [r["tokens_per_sec"] for r in rows if r.get("tokens_per_sec")]
    gpu = cfg.get("cuda_device")
    info = path / "job_info.txt"
    jobs = re.findall(r"job=(\d+)", info.read_text()) if info.exists() else []
    finished = any(r.get("done") for r in rows)
    probe = any(r.get("stopped_early_for_probe") for r in rows) and not finished
    return {
        "model": cfg["model_name_or_path"], "params": cfg["n_params"], "trainable_params": cfg.get("n_trainable_params", cfg["n_params"]),
        "gpu": gpu, "gpus": world, "max_seq_len": cfg["max_seq_len"], "tokens_per_micro_batch": cfg["tokens_per_micro_batch"],
        "grad_accum_per_gpu": cfg["grad_accum"], "micro_batches_per_update": cfg.get("micro_batches_per_update", cfg["grad_accum"]),
        "tokens_per_update_budget": cfg["tokens_per_update"], "optimizer_updates": steps, "planned_updates": cfg["total_steps"],
        "epochs": round(steps / cfg["steps_per_epoch"], 3) if cfg.get("steps_per_epoch") else epochs_done,
        "raw_tokens_seen": tok or int(cfg["train_tokens"] * steps / cfg["steps_per_epoch"]),
        "supervised_tokens_seen": sup, "peak_vram_gb": mem,
        "tokens_per_sec_median": sorted(tps)[len(tps) // 2] if tps else None,
        "wall_clock_hours": round(hours, 3), "gpu_hours": round(gpu_hours, 3), "slurm_jobs": jobs,
        "learning_rate": cfg["learning_rate"], "train_file": cfg.get("train_file") or cfg["data_dir"], "status": "finished" if finished else ("probe" if probe else "incomplete"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}
    for item in args.run:
        name, p = item.split("=", 1)
        out[name] = run_budget(Path(p))
    cols = ["model", "gpus", "gpu", "max_seq_len", "tokens_per_micro_batch", "micro_batches_per_update", "tokens_per_update_budget",
            "learning_rate", "optimizer_updates", "epochs", "raw_tokens_seen", "supervised_tokens_seen", "peak_vram_gb",
            "tokens_per_sec_median", "wall_clock_hours", "gpu_hours", "status"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for n, b in out.items():
        cells = []
        for c in cols:
            v = b[c]
            if isinstance(v, int) and v > 100000:
                v = f"{v / 1e6:.2f}M"
            elif c == "model":
                v = v.split("/")[-1]
            elif c == "gpu":
                v = (v or "").replace("NVIDIA ", "")
            cells.append(str(v))
        print(f"| {n} | " + " | ".join(cells) + " |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
