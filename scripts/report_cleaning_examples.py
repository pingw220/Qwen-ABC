#!/usr/bin/env python
"""Markdown fragments for reports/ABC_V2_DATA_CLEANING.md from an ABC-v2 build.

  scripts/report_cleaning_examples.py --v2-dir data/generated/abc_v2_<ts> --v1-dir data/generated/abc_v1_<ts> > fragment.md

Examples show only the lyric text adjacent to each moved boundary (at most 6
characters on either side), which is enough to judge the boundary.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.prompt import song_to_spec  # noqa: E402


def edge(spec, i, side, n=6):
    lines = spec["sections"][i]["lines"]
    text = "/".join(lines)
    return (text[-n:] if side == "end" else text[:n]) or "∅"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=8)
    args = ap.parse_args()
    log = [json.loads(l) for l in open(args.v2_dir / "cleaning_log.jsonl", encoding="utf-8")]
    by_split = {"train": [], "validation": [], "test": []}
    for r in log:
        by_split[r["split"]].append(r)
    chosen = []
    for split, n in (("train", args.n_train), ("test", args.n_test)):
        rows = sorted(by_split[split], key=lambda r: hashlib.sha256(r["song_id"].encode()).hexdigest())[:n]
        chosen += [(split, r) for r in rows]
    wanted = {r["song_id"] for _, r in chosen}
    v1 = {}
    for split in ("train", "test"):
        for line in open(args.v1_dir / f"songs_{split}.jsonl", encoding="utf-8"):
            sid = line[len('{"song_id": "'):].split('"', 1)[0]
            if sid in wanted:
                v1[sid] = json.loads(line)
    print("| # | split | song | boundary (section index) | shift (bars) | crossing kinds at old barline | before: …end ‖ start… | after: …end ‖ start… |")
    print("|---|---|---|---|---|---|---|---|")
    k = 0
    for split, r in chosen:
        song = Song.from_json(v1[r["song_id"]]["song"])
        spec_b = song_to_spec(song)
        from qwen_abc.canonical import Section
        song.sections = [Section(*s) for s in r["sections_after"]]
        spec_a = song_to_spec(song)
        for d in r["decisions"]:
            if d["action"] != "moved":
                continue
            i = d["section"]
            kinds = ", ".join(f"{c['kind']} {c['syllables_before']}|{c['syllables_after']}" for c in d["crossings"] if c["kind"] != "pickup")
            lb = f"{spec_b['sections'][i-1]['label']}:{spec_b['sections'][i-1]['bars']} …{edge(spec_b, i-1, 'end')} ‖ {spec_b['sections'][i]['label']}:{spec_b['sections'][i]['bars']} {edge(spec_b, i, 'start')}…"
            la = f"{spec_a['sections'][i-1]['label']}:{spec_a['sections'][i-1]['bars']} …{edge(spec_a, i-1, 'end')} ‖ {spec_a['sections'][i]['label']}:{spec_a['sections'][i]['bars']} {edge(spec_a, i, 'start')}…"
            k += 1
            print(f"| {k} | {split} | `{r['song_id'][:10]}` | {i} | {d['new_bar'] - d['old_bar']:+d} | {kinds} | {lb} | {la} |")
            break
    # shift distribution and pathologies
    shifts = Counter()
    for r in log:
        for d in r["decisions"]:
            if d["action"] == "moved":
                shifts[d["new_bar"] - d["old_bar"]] += 1
    print("\nshifts:", dict(sorted(shifts.items())))
    path = []
    for split in ("train", "validation", "test"):
        for line in open(args.v2_dir / f"songs_{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["pathologies"]:
                path.append((split, r["song_id"], r["pathologies"], r["flags"]["tempo_bpm"]))
    c = Counter(p for _, _, ps, _ in path for p in ps)
    print("\npathology songs:", len(path), dict(c))
    print("\n| split | song | flags | tempo |\n|---|---|---|---|")
    for split, sid, ps, t in path:
        if split != "train":
            print(f"| {split} | `{sid}` | {', '.join(ps)} | {t} |")


if __name__ == "__main__":
    main()
