#!/usr/bin/env python
"""Paired comparison of backing renderers on identical lead sheets: tuned SA3 MIDI-SAG (step 30k) vs
MuseControlLite (MIDI-SAG adapter). Reads data/backing.parquet and data/backing_sa3.parquet (both from
backing_eval, chord-source group), pairs by (song, chord source).

  python -m paper_eval.component_sota.backing_eval evaluate --render-dir render_sa3
  python -m paper_eval.component_sota.sa3_compare [--json out.json]
"""

from __future__ import annotations

import argparse
import json

from ..common import bootstrap_mean_ci, fmt_ci, paired_bootstrap, write_table
from .melody_eval import CS_REPORT

METRICS = [("chord_f1", "Chord F1 ↑"), ("chord_root_acc", "chord root acc. ↑"), ("rhythm_f1", "Rhythm F1 ↑"),
           ("bpm_acc_4pct", "BPM within 4% ↑"), ("key_acc", "Key accuracy ↑"), ("songeval_Coherence", "SongEval Coherence ↑"),
           ("songeval_Musicality", "SongEval Musicality ↑"), ("songeval_Naturalness", "SongEval Naturalness ↑"),
           ("aes_PQ", "Audiobox PQ ↑"), ("aes_CE", "Audiobox CE ↑")]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    mcl = pd.read_parquet(CS_REPORT / "data" / "backing.parquet")
    sa3 = pd.read_parquet(CS_REPORT / "data" / "backing_sa3.parquet")
    mcl, sa3 = mcl[mcl.group == "chords"], sa3[sa3.group == "chords"]
    key = lambda d: d.assign(k=d.song_id + "__" + d.condition).set_index("k")
    mcl, sa3 = key(mcl), key(sa3)
    rows, page = [], []
    for cond in ("all", "qwen", "am2", "ref"):
        M = mcl if cond == "all" else mcl[mcl.condition == cond]
        S = sa3 if cond == "all" else sa3[sa3.condition == cond]
        for c, lab in METRICS:
            if c not in M or c not in S:
                continue
            dm, ds = M[c].dropna().to_dict(), S[c].dropna().to_dict()
            d = paired_bootstrap(ds, dm)
            if not d["n"]:
                continue
            common = sorted(set(dm) & set(ds))
            m1 = bootstrap_mean_ci([dm[k] for k in common])
            m2 = bootstrap_mean_ci([ds[k] for k in common])
            r = {"chords": cond, "metric": lab, "MuseControlLite": fmt_ci(*m1[:3]), "SA3 step 30k": fmt_ci(*m2[:3]),
                 "SA3 − MCL [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N renders": d["n"]}
            rows.append(r)
            if cond == "all":
                page.append({"metric": lab, "mcl": f"{m1[0]:.3f}", "sa3": f"{m2[0]:.3f}",
                             "diff": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True)})
    write_table(rows, "backing_sa3_vs_mcl", "Backing renderer comparison on identical lead sheets (36 songs x 3 chord sources, 95.1 s excerpts): "
                "SA3 MIDI-SAG step 30k vs MuseControlLite; paired by render, bootstrap over renders", table_dir=CS_REPORT / "tables")
    if a.json:
        n = page and rows[0]["N renders"]
        json.dump({"rows": page, "note": f"Paired over {n} renders (36 songs x Qwen / AccoMontage2 / reference chords), same vocal and "
                   "lead sheet for both renderers, same detectors as the component round (BTC chords, BeatNet, chroma key, SongEval, "
                   "Audiobox). Metrics measure control adherence and predicted quality, not preference."}, open(a.json, "w"), indent=1)
    print("SA3_COMPARE_DONE")


if __name__ == "__main__":
    main()
