#!/usr/bin/env python
"""Late-section continuation: free-running generation of the last third of a song from a correct prefix.

  scripts/eval_continuation.py --checkpoint CKPT --data-dir data/generated/abc_v2_<ts> --output-dir OUT --format v1|v2

Prompt = the model's own whole-song prompt + the reference ABC (v1 or v2 syntax) of sections [0, k),
k = ceil(2N/3) (qwen_abc.longrange.continuation_start). The continuation is appended to the prefix
and the whole text is scored with the standard metrics; the late-song metrics
(late_section_present/exact, late_lyric_recall, early_eos, chorus motif preservation against the
reference choruses in the prefix) isolate what the model does after the prefix.
Every model (E0/E1/E2/E3) gets the same songs and cut points.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_eval import score_row  # noqa: E402
from qwen_abc.abc import song_to_abc  # noqa: E402
from qwen_abc.abc_v2 import spec_to_prompt_v2  # noqa: E402
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.generate import generate_batch, load_for_generation  # noqa: E402
from qwen_abc.longrange import continuation_prompt, continuation_start, split_abc_sections  # noqa: E402
from qwen_abc.metrics import aggregate  # noqa: E402
from qwen_abc.prompt import spec_to_prompt  # noqa: E402


def build(r, fmt):
    song = Song.from_json(r["song"])
    k = continuation_start(r["spec"])
    if fmt == "v2":
        c = continuation_prompt(song, r["spec"], k)
        return c["prompt"], c["prefix"], k
    header, blocks = split_abc_sections(song_to_abc(song))
    prefix = header + "".join(blocks[:k])
    return spec_to_prompt(r["spec"]) + prefix, prefix, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--format", choices=["v1", "v2"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-total", type=int, default=10240)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data_dir / f"songs_{args.split}.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["song_id"])
    gen_dir = args.output_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if len(r["spec"]["sections"]) >= 2 and not (gen_dir / f"{r['song_id']}_s0.json").exists()]
    if pending and not args.aggregate_only:
        model, tok = load_for_generation(args.checkpoint)
        pending.sort(key=lambda r: len(r["abc"]))
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i: i + args.batch_size]
            built = [build(r, args.format) for r in chunk]
            outs = generate_batch(model, tok, [b[0] for b in built], 1000 + i, args.max_total, args.temperature, args.top_p, args.max_total)
            for r, (prompt, prefix, k), g in zip(chunk, built, outs):
                row = {"song_id": r["song_id"], "kind": "continuation", "start_section": k, "n_sections": len(r["spec"]["sections"]),
                       "prefix_chars": len(prefix), "generation": prefix + g["text"], "continuation": g["text"],
                       "new_tokens": g["new_tokens"], "hit_eos": g["hit_eos"], "seconds": g["seconds"],
                       "temperature": args.temperature, "top_p": args.top_p, "checkpoint": args.checkpoint}
                score_row(row, r["spec"], args.format)
                (gen_dir / f"{r['song_id']}_s0.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            print(f"batch {i // args.batch_size + 1}: {len(chunk)} songs {outs[0]['batch_seconds']}s", flush=True)
    done = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(gen_dir.glob("*_s0.json"))]
    summary = {"checkpoint": args.checkpoint, "format": args.format, "n": len(done),
               "parse_success": sum(r["parse_ok"] for r in done) / max(len(done), 1),
               "hit_eos": sum(bool(r["hit_eos"]) for r in done) / max(len(done), 1),
               "generated": aggregate([r["metrics"] for r in done if "metrics" in r])}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("CONTINUATION_DONE", len(done))


if __name__ == "__main__":
    main()
