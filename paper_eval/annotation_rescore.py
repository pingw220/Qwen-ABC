#!/usr/bin/env python
"""Re-score systems against human-verified labels once annotations exist.

  python -m paper_eval.annotation_rescore --annotations reports/paper_final/human_annotation_package/annotations \
      --system e3b_sel4=experiments/bon_r3b_20260918_100234/test_sum/generations [--system ...] [--out result.json]

For every annotated song and every system sample it computes, against the pseudo labels and
against the human-corrected labels:

* ``structure_exact``      the generated (label, bars) plan equals the whole plan
* ``section_label_acc``    per section (by index), label equal
* ``section_bars_acc``     per section, bar count equal
* ``lyric_recall``         sung prompt syllables; the human variant removes, per song, the syllables an
                           annotator marked as wrong text from the denominator (the model was still prompted
                           with them, so this is the recall of the *valid* lyric)

and decomposes each model section that deviates from the pseudo plan into

* ``label_error`` -- the model agrees with the human label (the pseudo label was wrong);
* ``model_error`` -- the model disagrees with both;

plus ``followed_wrong_label`` -- the model reproduced a pseudo label the human corrected.
Samples are averaged within a song; CIs are song-level paired bootstraps (common.paired_bootstrap).
Several annotators on one song: the most confident record is used, and label/bars agreement
between annotators is reported separately.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import jdump, load_rows, mean, paired_bootstrap


def load_annotations(d: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in sorted(Path(d).glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in (data if isinstance(data, list) else [data]):
            if r.get("song_id") and r.get("sections"):
                out[r["song_id"]].append(r)
    return dict(out)


def consensus(records: List[dict]) -> dict:
    return sorted(records, key=lambda r: (-(r.get("confidence") or 0), r.get("annotator_id", "")))[0]


def human_plan(rec: dict) -> List[Tuple[str, int]]:
    if rec.get("corrected_sections"):
        return [(s["label"], int(s["bars"])) for s in rec["corrected_sections"]]
    plan = []
    for s in sorted(rec["sections"], key=lambda s: s["index"]):
        label = s.get("correct_label") if s.get("label_correct") == "no" and s.get("correct_label") else s["pseudo_label"]
        bars = s.get("correct_bars") if s.get("bars_correct") == "no" and s.get("correct_bars") is not None else s["pseudo_bars"]
        plan.append((label, int(bars)))
    return plan


def _gen_plan(row: dict) -> Optional[List[Tuple[str, int]]]:
    if not row.get("song"):
        return None
    return [(s["label"], s["num_bars"]) for s in row["song"]["sections"]]


def sample_metrics(row: dict, pseudo: List[Tuple[str, int]], human: List[Tuple[str, int]], want: int, wrong: int) -> dict:
    gen = _gen_plan(row)
    m = {}
    for tag, ref in (("pseudo", pseudo), ("human", human)):
        if gen is None:
            m.update({f"structure_exact_{tag}": 0.0, f"section_label_acc_{tag}": 0.0, f"section_bars_acc_{tag}": 0.0})
            continue
        m[f"structure_exact_{tag}"] = float(gen == ref)
        m[f"section_label_acc_{tag}"] = sum(1 for i, r in enumerate(ref) if i < len(gen) and gen[i][0] == r[0]) / max(len(ref), 1)
        m[f"section_bars_acc_{tag}"] = sum(1 for i, r in enumerate(ref) if i < len(gen) and gen[i][1] == r[1]) / max(len(ref), 1)
    lr = ((row.get("metrics") or {}).get("lyric_recall")) if gen is not None else 0.0
    lr = lr or 0.0
    m["lyric_recall_pseudo"] = lr
    m["lyric_recall_human"] = min(1.0, lr * want / max(want - wrong, 1)) if want else lr
    dev = lab = mod = fol = 0
    if gen is not None and len(pseudo) == len(human):
        for i, (p, h) in enumerate(zip(pseudo, human)):
            g = gen[i] if i < len(gen) else None
            if g != p:
                dev += 1
                if g == h:
                    lab += 1
                else:
                    mod += 1
            elif p != h:
                fol += 1
    m["deviating_sections"] = dev
    m["label_error_sections"] = lab
    m["model_error_sections"] = mod
    m["followed_wrong_label_sections"] = fol
    return m


def load_system(d: Path, songs) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for sid in songs:
        for p in sorted(Path(d).glob(f"{sid}_*.json")):
            out[sid].append(json.loads(p.read_text(encoding="utf-8")))
    return dict(out)


def rescore(annotations: Dict[str, List[dict]], systems: Dict[str, Dict[str, List[dict]]],
            pseudo_specs: Dict[str, dict], n_boot: int = 10000) -> dict:
    from qwen_abc.prompt import spec_syllables
    songs = sorted(s for s in annotations if s in pseudo_specs)
    result = {"n_annotated_songs": len(songs), "systems": {}, "annotator_agreement": agreement(annotations)}
    label_changes = bars_changes = n_sec = 0
    for sid in songs:
        rec = consensus(annotations[sid])
        spec = pseudo_specs[sid]
        pseudo = [(s["label"], s["bars"]) for s in spec["sections"]]
        human = human_plan(rec)
        n_sec += len(pseudo)
        if len(pseudo) == len(human):
            label_changes += sum(1 for a, b in zip(pseudo, human) if a[0] != b[0])
            bars_changes += sum(1 for a, b in zip(pseudo, human) if a[1] != b[1])
    result["label_noise"] = {"sections": n_sec, "label_corrected": label_changes, "bars_corrected": bars_changes,
                             "songs_plan_corrected": sum(1 for s in songs if human_plan(consensus(annotations[s])) !=
                                                         [(x["label"], x["bars"]) for x in pseudo_specs[s]["sections"]])}
    for name, rows_by_song in systems.items():
        per_song: Dict[str, Dict[str, float]] = defaultdict(dict)
        keys = ("deviating_sections", "label_error_sections", "model_error_sections", "followed_wrong_label_sections")
        tot = {k: 0 for k in keys}
        n_samples = 0
        for sid in songs:
            rows = rows_by_song.get(sid, [])
            if not rows:
                continue
            rec = consensus(annotations[sid])
            spec = pseudo_specs[sid]
            pseudo = [(s["label"], s["bars"]) for s in spec["sections"]]
            want = len(spec_syllables(spec))
            wrong = sum(int(s.get("wrong_lyric_syllables") or 0) for s in rec["sections"])
            ms = [sample_metrics(r, pseudo, human_plan(rec), want, wrong) for r in rows]
            for k in ms[0]:
                per_song[k][sid] = mean([m[k] for m in ms])
            for m in ms:
                for k in keys:
                    tot[k] += m[k]
            n_samples += len(ms)
        res = {"n_songs": len(per_song.get("structure_exact_pseudo", {}))}
        for base in ("structure_exact", "section_label_acc", "section_bars_acc", "lyric_recall"):
            res[base] = paired_bootstrap(per_song[f"{base}_human"], per_song[f"{base}_pseudo"], n_boot=n_boot)
        tot["samples"] = n_samples
        tot["label_error_share_of_deviations"] = tot["label_error_sections"] / tot["deviating_sections"] if tot["deviating_sections"] else None
        res["deviation_decomposition"] = tot
        result["systems"][name] = res
    return result


def agreement(annotations: Dict[str, List[dict]]) -> dict:
    """Section-level agreement on the corrected label and bar count between annotator pairs."""
    pairs = lab = bars = 0
    for recs in annotations.values():
        if len(recs) < 2:
            continue
        plans = [human_plan(r) for r in recs]
        for a in range(len(plans)):
            for b in range(a + 1, len(plans)):
                for x, y in zip(plans[a], plans[b]):
                    pairs += 1
                    lab += x[0] == y[0]
                    bars += x[1] == y[1]
    return {"section_pairs": pairs, "label_agreement": lab / pairs if pairs else None,
            "bars_agreement": bars / pairs if pairs else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--system", action="append", default=[], help="name=generations_dir (repeatable)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ann = load_annotations(args.annotations)
    if not ann:
        raise SystemExit(f"no annotation records under {args.annotations}: human results are still PENDING")
    specs = {r["song_id"]: r["spec"] for r in load_rows("test", ids=list(ann))}
    systems = {}
    for s in args.system:
        name, _, d = s.partition("=")
        systems[name] = load_system(Path(d), sorted(ann))
    res = rescore(ann, systems, specs)
    print(f"annotated songs: {res['n_annotated_songs']}  label noise: {res['label_noise']}")
    print(f"annotator agreement: {res['annotator_agreement']}")
    for name, r in res["systems"].items():
        print(f"\n== {name} ({r['n_songs']} songs; human - pseudo, 95% paired bootstrap CI)")
        for k in ("structure_exact", "section_label_acc", "section_bars_acc", "lyric_recall"):
            x = r[k]
            if x["diff"] is None:
                continue
            print(f"  {k:20s} pseudo {x['b']:.3f}  human {x['a']:.3f}  diff {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]")
        print(f"  deviations: {r['deviation_decomposition']}")
    if args.out:
        jdump(res, args.out)


if __name__ == "__main__":
    main()
