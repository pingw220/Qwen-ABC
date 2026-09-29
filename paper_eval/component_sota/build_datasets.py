#!/usr/bin/env python
"""Build the melody-only and harmonizer SFT datasets from the canonical ABC-v2 songs.

  python -m paper_eval.component_sota.build_datasets --task mel|chord|chord_lyr

Same songs and splits as E3b (abc_v2_20260915_120927: 10,243 / 273 / 225). The melody-only
training mixture mirrors E3b's exactly: every whole-song example plus the infill example of
the same deterministic 50% of songs, same target sections -- only the chords are removed.
Every example is checked: the chord-free completion re-parses to the same melody, and
harmonizer chord lines re-parse to the song's own chords. Outputs go to
experiments/component_sota/data/<task>/ (git-ignored; refuses to overwrite).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qwen_abc.abc import parse_abc
from qwen_abc.canonical import Song, comparable, normalize_chords
from qwen_abc.longrange import GAP_LINE, choose_infill_target, infill_example

from ..common import DATA_DIR, ROOT
from .formats import chord_lines, chord_prompt, mel_completion, mel_prompt, parse_chord_lines, strip_chords

OUT = ROOT / "experiments/component_sota/data"


def selected(song_id: str, fraction: float = 0.5) -> bool:  # identical to scripts/build_longrange_tasks.py
    return int.from_bytes(hashlib.sha256(f"infill-select:{song_id}".encode()).digest()[:8], "big") / 2 ** 64 < fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mel", "chord", "chord_lyr"])
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_abc_v2_dataset import ntok
    out = OUT / args.task
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for split in ("train", "validation", "test"):
        path = out / f"{split}.jsonl"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        n = {"whole_song": 0, "infill": 0, "checked": 0}
        tok_sum = sup_sum = 0
        with open(DATA_DIR / f"songs_{split}.jsonl", encoding="utf-8") as fin, open(path, "w", encoding="utf-8") as fo:
            for line in fin:
                r = json.loads(line)
                song = Song.from_json(r["song"])
                rows = []
                if args.task == "mel":
                    comp = mel_completion(song)
                    p = parse_abc(comp, r["song_id"])
                    assert p.ok and comparable(p.song)["notes"] == comparable(song)["notes"], r["song_id"]
                    rows.append({"id": f"mel:{r['song_id']}", "task": "whole_song", "prompt": mel_prompt(r["spec"]), "completion": comp})
                    t = choose_infill_target(r["song_id"], r["spec"])
                    if t is not None and (split == "test" or selected(r["song_id"])):
                        ex = infill_example(strip_chords(song), r["spec"], t)
                        assert ex["prompt"].count(GAP_LINE) == 1 and '"' not in ex["completion"]
                        rows.append({"id": f"melinfill:{r['song_id']}", "task": "infill", "prompt": ex["prompt"],
                                     "completion": ex["completion"], "target_section": t})
                else:
                    lyr = args.task == "chord_lyr"
                    comp = chord_lines(song)
                    back, _ = parse_chord_lines(comp, song)
                    assert [(c.onset, c.symbol) for c in back] == [(c.onset, c.symbol) for c in normalize_chords(song.chords, song.total_ticks)], r["song_id"]
                    rows.append({"id": f"{args.task}:{r['song_id']}", "task": args.task, "prompt": chord_prompt(song, r["spec"], lyr),
                                 "completion": comp})
                for row in rows:
                    if split == "test" and row["task"] == "infill":
                        continue  # test infill prompts are built by the evaluation, not here
                    row["song_id"] = r["song_id"]
                    row["prompt_tokens"] = ntok(row["prompt"])
                    row["completion_tokens"] = ntok(row["completion"])
                    tok_sum += row["prompt_tokens"] + row["completion_tokens"]
                    sup_sum += row["completion_tokens"]
                    n[row["task"] if row["task"] in n else "whole_song"] += 1
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n["checked"] += 1
        rep[split] = {**n, "tokens": tok_sum, "supervised_tokens": sup_sum}
        print(split, rep[split], flush=True)
    (out / "build_report.json").write_text(json.dumps(rep, indent=1))
    print("BUILD_OK", out)


if __name__ == "__main__":
    main()
