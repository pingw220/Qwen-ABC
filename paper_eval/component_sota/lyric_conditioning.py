#!/usr/bin/env python
"""Lyric intervention vs reseed for every melody source (Table 2), with the paper-final estimators.

  python -m paper_eval.component_sota.lyric_conditioning [--sources e3b,mel,csl_off,csl_rt]

For each song: 4 samples with the original lyrics (S1-S4) and 4 with the whole lyric replaced by
another held-out song's (same syllable count per section; paper_eval.tasks 'lyrics_all').
* effect = cross (orig_i vs swap_j, i != j) - within (orig_i vs orig_j, i != j), per distance;
* seed-paired ratio d(orig_S1, swap_S1) / d(orig_S1, orig_S2);
* new-lyric recall, old-lyric leakage, chance level;
* Mandarin tone crossover (paper_eval.tone_melody).
Melodies are compared with chords stripped, so the full model is judged on its melody only.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

from qwen_abc.canonical import Song

from ..common import bootstrap_mean_ci, fmt_ci, load_rows, mean, paired_bootstrap, write_table
from ..intervention_analysis import all_distances, features
from ..interventions import recall_of, syllables_of
from ..tone_melody import agreement, rate, slot_pitches, tones_of
from .melody_eval import CS_REPORT, DISPLAY
from .sources import SEEDS, iter_melodies

DISTS = [("d_melody", "melody"), ("d_contour", "contour"), ("d_rhythm", "rhythm"), ("f_pc_js", "pitch-class JS"),
         ("f_interval_js", "interval JS"), ("f_density", "abs. Δ notes/bar")]


def load(src, cond):
    out = defaultdict(dict)
    for s in SEEDS:
        for r in iter_melodies(src, s, cond):
            out[r["song_id"]][s] = r
    return out


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="e3b,mel,csl_offc,csl_rtc,csl_off,csl_rt")
    args = ap.parse_args()
    rows = {r["song_id"]: r for r in load_rows("test")}
    recs, table = [], []
    for src in args.sources.split(","):
        O, B = load(src, "orig"), load(src, "lyrics_all")
        if not B:
            print("skip", src)
            continue
        eff = defaultdict(dict)
        paired, newrec, oldrec, chance, tone_a, tone_b = {}, {}, {}, {}, {}, {}
        for sid in sorted(set(O) & set(B)):
            o = {k: Song.from_json(v["song"]) for k, v in O[sid].items() if v["ok"]}
            b = {k: Song.from_json(v["song"]) for k, v in B[sid].items() if v["ok"]}
            if len(o) < 2 or not b:
                continue
            fo = {k: features(v) for k, v in o.items()}
            fb = {k: features(v) for k, v in b.items()}
            cross = [all_distances(o[i], b[j], fo[i], fb[j]) for i in o for j in b if i != j]
            within = [all_distances(o[i], o[j], fo[i], fo[j]) for i, j in itertools.combinations(sorted(o), 2)]
            if not cross or not within:
                continue
            for d, _ in DISTS:
                eff[d][sid] = (mean([x[d] for x in cross]), mean([x[d] for x in within]))
            if "S1" in o and "S2" in o and "S1" in b:
                dab = all_distances(o["S1"], b["S1"], fo["S1"], fb["S1"])["d_melody"]
                dac = all_distances(o["S1"], o["S2"], fo["S1"], fo["S2"])["d_melody"]
                paired[sid] = dab / (dac + 1e-6)
            spec_b = next(iter(B[sid].values()))["meta"].get("spec")
            old_sy = [x for sec in rows[sid]["spec"]["sections"] for x in syllables_of(sec)]
            new_sy = [x for sec in spec_b["sections"] for x in syllables_of(sec)] if spec_b else None
            if new_sy:
                newrec[sid] = mean([recall_of(x, new_sy) for x in b.values()])
                oldrec[sid] = mean([recall_of(x, old_sy) for x in b.values()])
                chance[sid] = mean([recall_of(x, new_sy) for x in o.values()])
                t_old, t_new = tones_of(old_sy), tones_of(new_sy)
                ao = rate([agreement(slot_pitches(x, old_sy), t_old) for x in o.values()])
                an = rate([agreement(slot_pitches(x, old_sy), t_new) for x in o.values()])
                bn = rate([agreement(slot_pitches(x, new_sy), t_new) for x in b.values()])
                bo = rate([agreement(slot_pitches(x, new_sy), t_old) for x in b.values()])
                if None not in (ao, an):
                    tone_a[sid] = ao - an
                if None not in (bn, bo):
                    tone_b[sid] = bn - bo
            recs.append({"source": src, "song_id": sid, **{f"cross_{d}": eff[d][sid][0] for d, _ in DISTS},
                         **{f"within_{d}": eff[d][sid][1] for d, _ in DISTS}, "paired_ratio": paired.get(sid),
                         "new_recall": newrec.get(sid), "old_leak": oldrec.get(sid), "chance": chance.get(sid),
                         "tone_adv_orig": tone_a.get(sid), "tone_adv_swap": tone_b.get(sid)})
        row = {"system": DISPLAY.get(src, src), "N songs": len(eff["d_melody"])}
        for key, lab, vals in (("new", "lyric recall after swap", newrec), ("old", "old-lyric leakage", oldrec),
                               ("ch", "chance (new lyric in orig samples)", chance)):
            m, lo, hi, n = bootstrap_mean_ci(list(vals.values()))
            row[lab] = fmt_ci(m, lo, hi)
        for d, lab in DISTS:
            diffs = [c - w for c, w in eff[d].values()]
            m, lo, hi, n = bootstrap_mean_ci(diffs)
            row[f"{lab}: swap − reseed"] = fmt_ci(m, lo, hi, 3, True)
            cm, wm = mean([c for c, _ in eff[d].values()]), mean([w for _, w in eff[d].values()])
            row[f"{lab}: ratio"] = f"{cm / wm:.3f}" if wm else "–"
        m, lo, hi, n = bootstrap_mean_ci([v for v in paired.values() if v is not None])
        row["seed-paired d(A,B)/d(A,C)"] = fmt_ci(m, lo, hi)
        for lab, vals in (("tone: own − other (orig)", tone_a), ("tone: own − other (swap)", tone_b)):
            m, lo, hi, n = bootstrap_mean_ci(list(vals.values()))
            row[lab] = fmt_ci(m, lo, hi, 3, True)
        table.append(row)
        print(src, row, flush=True)
    write_table(table, "lyric_conditioning", "Lyric intervention vs reseed (Table 2). swap − reseed > 0: replacing the lyrics moves the melody beyond sampling noise; ratio = mean cross / mean within distance",
                table_dir=CS_REPORT / "tables")
    pd.DataFrame(recs).to_parquet(CS_REPORT / "data" / "lyric_conditioning.parquet", index=False)
    print("LYRIC_CONDITIONING_DONE")


if __name__ == "__main__":
    main()
