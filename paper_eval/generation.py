#!/usr/bin/env python
"""Generate and score every paper-final task for one model (resumable, shardable).

  python -m paper_eval.generation --model qwen_e3b --suite bon [--dry-run]
  python -m paper_eval.generation --model qwen_e3b --suite interventions --conditions key_p5,lyrics_all \
      --shard 0 --num-shards 4
  python -m paper_eval.generation --model mupt --suite ood --songs 5          # smoke test on 5 songs

Output: experiments/paper_final/gen/<model>/<condition>/<song>_<seed>.json, the same row
format as scripts/generate_eval.py (prompt, generation, parse/strict flags, metrics,
canonical song) plus the task's spec and intervention metadata. A file that exists is
never regenerated (``--only-missing`` is the only mode); a corrupt file is reported, moved
aside to ``*.corrupt`` and regenerated.

Batches are fixed from the full task list (sorted by prompt length) before existing
files are skipped, and batch b goes to shard b % num_shards, so concurrent shards never
write the same file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import MAX_NEW, MAX_TOTAL, MODELS, OUT_ROOT, ROOT, TEMPERATURE, TOP_P, jdump, load_rows, make_prompt
from .tasks import build_tasks


def task_path(t: dict) -> Path:
    return OUT_ROOT / "gen" / t["model"] / t["condition"] / f"{t['song_id']}_{t['seed_name']}.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def select_rows(songs_arg):
    rows = load_rows("test")
    if songs_arg is None or songs_arg == "all":
        return rows
    if Path(songs_arg).exists():
        ids = {l.strip() for l in open(songs_arg, encoding="utf-8") if l.strip()}
        return [r for r in rows if r["song_id"] in ids]
    return rows[: int(songs_arg)]


def is_complete(p: Path, task: dict) -> bool:
    """An output counts only if it parses and was generated from exactly this task's spec and seed;
    anything else is renamed aside (never deleted) and regenerated."""
    if not p.exists():
        return False
    try:
        row = json.loads(p.read_text(encoding="utf-8"))
        ok = "generation" in row and "parse_ok" in row
    except Exception:  # noqa: BLE001
        p.rename(p.with_suffix(".corrupt"))
        print(f"CORRUPT moved aside: {p}", flush=True)
        return False
    if ok and (row.get("spec") != task["spec"] or row.get("seed") != task["seed"]):
        k = 0
        while p.with_suffix(f".stale{k}").exists():
            k += 1
        p.rename(p.with_suffix(f".stale{k}"))
        print(f"STALE (task definition changed) moved aside: {p}", flush=True)
        return False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--suite", required=True, choices=["bon", "interventions", "replay", "ood"])
    ap.add_argument("--conditions", default=None, help="comma list (interventions/ood); default all")
    ap.add_argument("--songs", default="all", help="'all', N (first N by id), or a file of song ids")
    ap.add_argument("--seed", default=None, help="restrict to one seed name (S1..S4)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="build and verify tasks, write the manifest, generate nothing")
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; resuming is always on")
    ap.add_argument("--only-missing", action="store_true", help="accepted for clarity; always on")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    rows = select_rows(args.songs)
    conds = args.conditions.split(",") if args.conditions else None
    tasks, skipped = build_tasks(args.model, args.suite, rows, conds)
    if args.seed:
        tasks = [t for t in tasks if t["seed_name"] == args.seed]
    for t in tasks:
        t["prompt"] = make_prompt(cfg["fmt"], t["spec"])
    man_dir = OUT_ROOT / "gen" / args.model
    man_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.suite}{'_' + args.conditions.replace(',', '+') if args.conditions else ''}"
    if args.songs != "all":
        tag += f"_songs{Path(str(args.songs)).stem}"
    with open(man_dir / f"tasks_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps({k: v for k, v in t.items() if k not in ("spec", "prompt")}, ensure_ascii=False) + "\n")
    jdump(skipped, man_dir / f"skipped_{tag}.json")
    # fixed batches: replay is batched in reverse length order so its batch composition
    # (and padding) differs from the S1 'orig' run it replays
    order = sorted(tasks, key=lambda t: (len(t["prompt"]), t["condition"], t["song_id"], t["seed_name"]),
                   reverse=args.suite == "replay")
    batches = [order[k: k + args.batch_size] for k in range(0, len(order), args.batch_size)]
    mine = [b for i, b in enumerate(batches) if i % args.num_shards == args.shard]
    pending = [[t for t in b if not is_complete(task_path(t), t)] for b in mine]
    pending = [b for b in pending if b]
    n_pending = sum(len(b) for b in pending)
    print(f"model={args.model} suite={args.suite} tasks={len(tasks)} skipped={len(skipped)} "
          f"shard={args.shard}/{args.num_shards} batches={len(mine)} pending={n_pending}", flush=True)
    if args.dry_run or not pending:
        print("DRY_RUN_DONE" if args.dry_run else "NOTHING_TO_DO")
        return

    from qwen_abc.generate import load_for_generation
    from generate_eval import score_row  # scripts/ is on sys.path via common
    from .sampler import generate_batch_crn

    model, tok = load_for_generation(cfg["ckpt"])
    commit = _git_commit()
    done = 0
    t_start = time.time()
    for bi, batch in enumerate(pending):
        outs = generate_batch_crn(model, tok, [t["prompt"] for t in batch], [t["seed"] for t in batch],
                                  MAX_NEW, TEMPERATURE, TOP_P, MAX_TOTAL)
        for t, g in zip(batch, outs):
            row = {"song_id": t["song_id"], "kind": t["condition"], "model": t["model"], "condition": t["condition"],
                   "seed_name": t["seed_name"], "seed": t["seed"], "prompt": t["prompt"], "generation": g["text"],
                   "new_tokens": g["new_tokens"], "hit_eos": g["hit_eos"], "truncated": g["truncated"],
                   "seconds": g["seconds"], "batch_size": g["batch_size"], "batch_seconds": g["batch_seconds"],
                   "prompt_tokens": g["prompt_tokens"], "token_budget": g["token_budget"],
                   "temperature": TEMPERATURE, "top_p": TOP_P, "sampler": "gumbel_crn", "max_total": MAX_TOTAL,
                   "checkpoint": cfg["ckpt"], "fmt": cfg["fmt"], "commit": commit, "meta": t["meta"], "spec": t["spec"]}
            score_row(row, t["spec"], cfg["fmt"])
            jdump(row, task_path(t), indent=None)
        done += len(batch)
        print(f"[{bi + 1}/{len(pending)}] n={len(batch)} {outs[0]['batch_seconds']}s "
              f"eos={sum(g['hit_eos'] for g in outs)} parse={sum(1 for t in batch if json.loads(task_path(t).read_text())['parse_ok'])} "
              f"done={done}/{n_pending} elapsed={time.time() - t_start:.0f}s", flush=True)
    print("GENERATION_DONE")


if __name__ == "__main__":
    main()
