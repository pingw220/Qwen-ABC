#!/usr/bin/env python
"""Round-2 ablation table (reports/LONG_STRUCTURE_EXPERIMENTS.md §Ablation table).

  scripts/ablation_table.py --row "E0|0.8B|ABC-v1|8K|131K|Direct SFT|experiments/direct_sft_.../|EVALDIR" ...

Each --row is name|model|abc|context|tokens_per_update|task|run_dir(or '-')|eval_dir. Structure/lyric/
musical numbers come from the eval dir's generations; test loss from run_dir/loss_test.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_r2 import load_rows, value  # noqa: E402

KEYS = [("section_plan_exact", "structure exact"), ("lyric_recall", "lyric recall"), ("early_eos", "early EOS"),
        ("pitch_range", "pitch range"), ("distinct_bar_frac", "distinct bars"), ("strict_valid", "strict valid")]

ap = argparse.ArgumentParser()
ap.add_argument("--row", action="append", required=True)
ap.add_argument("--subset", type=Path, default=None)
args = ap.parse_args()
subset = {l.strip() for l in open(args.subset)} if args.subset else None
print("| ID | Model | ABC | Context | Tokens/update | Training task | " + " | ".join(l for _, l in KEYS) + " | test loss | content loss |")
print("|---|---|---|---|---|---|" + "---|" * (len(KEYS) + 2))
for spec in args.row:
    name, model, abc, ctx, tpu, task, run, ev = spec.split("|")
    rows = load_rows(Path(ev))
    ids = [s for s in rows if subset is None or s in subset]
    cells = []
    for k, _ in KEYS:
        v = [value(rows[s], k) for s in ids]
        v = [x for x in v if x is not None]
        cells.append(f"{sum(v) / len(v):.3f}" if v else "–")
    loss = content = "–"
    p = Path(run) / "loss_test.json" if run != "-" else None
    if p and p.exists():
        d = json.loads(p.read_text())
        loss = f"{d['loss']:.4f}"
        bc = d["by_category"]
        cat = [c for c in bc if c not in ("structure", "header", "space", "eos")]
        content = f"{sum(bc[c]['loss'] * bc[c]['tokens'] for c in cat) / sum(bc[c]['tokens'] for c in cat):.4f}"
    print(f"| {name} | {model} | {abc} | {ctx} | {tpu} | {task} | " + " | ".join(cells) + f" | {loss} | {content} |")
