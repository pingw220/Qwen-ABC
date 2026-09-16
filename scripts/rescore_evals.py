#!/usr/bin/env python
"""Re-score cached generations in place with the current metric code (round-2 eval dirs only).

  scripts/rescore_evals.py DIR [DIR ...]      # each DIR contains generations/*.json
Refuses to touch round-1 experiment directories.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_eval import score_row  # noqa: E402

DATA = {}


def specs(data_dir: Path, split: str):
    key = (str(data_dir), split)
    if key not in DATA:
        DATA[key] = {json.loads(l)["song_id"]: json.loads(l)["spec"]
                     for l in open(data_dir / f"songs_{split}.jsonl", encoding="utf-8")}
    return DATA[key]


for d in sys.argv[1:]:
    d = Path(d)
    args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
    data_dir = Path(args.get("data_dir", "data/generated/abc_v2_20260915_120927"))
    split = args.get("split", "test")
    fmt = args.get("format", "v1")
    sp = specs(data_dir, split)
    n = 0
    for p in sorted((d / "generations").glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row["song_id"] not in sp:
            continue
        score_row(row, sp[row["song_id"]], fmt)
        p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"{d}: rescored {n}")
