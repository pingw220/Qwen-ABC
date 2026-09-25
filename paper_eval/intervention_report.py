#!/usr/bin/env python
"""Tables for INTERVENTIONAL_CONTROLLABILITY.md from paper_eval.intervention_analysis outputs.

  python -m paper_eval.intervention_report --tag qwen_e3b --tag mupt --tag midi_llm
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, write_table

# condition -> [(column, label, baseline column on orig samples or None)]
ADHERENCE = {
    "bars": [("target_exact", "target section has the requested label+bars", "target_exact"),
             ("direction_ok", "bar change in the requested direction", None),
             ("nontarget_exact_frac", "non-target sections exact", "nontarget_exact_frac"),
             ("plan_exact", "whole plan exact", "plan_exact")],
    "label": [("target_label_ok", "target section has the requested label", "target_label_ok"),
              ("target_bars_ok", "target duration preserved", "target_bars_ok"),
              ("nontarget_exact_frac", "non-target sections exact", "nontarget_exact_frac"),
              ("plan_exact", "whole plan exact", "plan_exact")],
    "key": [("declared_key_ok", "declared key = requested", None),
            ("melody_in_new_key", "melody time in the new key's scale", None),
            ("melody_in_old_key", "melody time in the old key's scale", None),
            ("chord_roots_in_new_key", "chord roots in the new key", None),
            ("pc_shift_matches", "pitch-class profile shifted by the requested interval", None),
            ("pitch_mean_shift", "mean pitch shift (semitones)", None),
            ("plan_exact", "whole plan exact", None)],
    "tempo": [("tempo_ok", "tempo header = requested", None), ("duration_ratio", "duration ratio (new/orig)", None),
              ("target_duration_ratio", "requested duration ratio", None), ("bar_count_ok", "bar count = requested", None),
              ("plan_exact", "whole plan exact", None)],
    "lyrics": [("new_recall", "new lyrics sung (recall)", None), ("chance_new_recall_in_orig", "chance: new lyrics 'sung' by unmodified samples", None),
               ("old_leak_recall", "old lyrics still sung", None),
               ("new_recall_target_section", "new section lyrics sung in the target section", None),
               ("chance_new_target_in_orig", "chance: same, unmodified samples", None),
               ("old_leak_target_section", "old section lyrics still sung there", None),
               ("plan_exact", "whole plan exact", "plan_exact")],
}
DISTS = [("d_melody", "melody (onset,pitch,dur)"), ("d_contour", "contour (intervals)"), ("d_rhythm", "rhythm"),
         ("d_chord", "chord changes"), ("f_pc_js", "pitch-class JS"), ("f_interval_js", "interval JS"),
         ("f_density", "|Δ notes/bar|"), ("f_pitch_mean", "|Δ mean pitch|")]


def song_mean(df, col):
    g = df.dropna(subset=[col]).groupby("song_id")[col].mean()
    return g.to_dict()


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()
    S = pd.concat([pd.read_parquet(DATA_OUT / f"interventions_{t}.parquet") for t in args.tag], ignore_index=True)
    E = pd.concat([pd.read_parquet(DATA_OUT / f"interventions_effects_{t}.parquet") for t in args.tag], ignore_index=True)
    S.to_parquet(DATA_OUT / "interventions.parquet", index=False)
    E.to_parquet(DATA_OUT / "interventions_effects.parquet", index=False)
    rows, fig = [], defaultdict(dict)
    for (model, cond), df in S.groupby(["model", "condition"]):
        fam = df.family.iloc[0]
        iv = df[df.role == "intervened"]
        ob = df[df.role == "orig_baseline"]
        for col, label, base in ADHERENCE.get(fam, []):
            if col not in iv or iv[col].notna().sum() == 0:
                continue
            m, lo, hi, n = bootstrap_mean_ci(list(song_mean(iv, col).values()))
            r = {"model": model, "condition": cond, "measure": label, "intervened": fmt_ci(m, lo, hi), "N songs": n,
                 "samples": int(iv[col].notna().sum())}
            if base and base in ob and ob[base].notna().sum():
                mb, lob, hib, nb = bootstrap_mean_ci(list(song_mean(ob, base).values()))
                r["same measure, unmodified samples"] = fmt_ci(mb, lob, hib)
            rows.append(r)
            fig[model][f"{cond}:{col}"] = [m, lo, hi, n]
    write_table(rows, "interventions_adherence", "Target-specific adherence to one-control interventions (song-level mean over seeds; 95% bootstrap)")
    # output change: cross - within
    erows, srows = [], []
    for (model, cond, kind), df in E.groupby(["model", "condition", "kind"]):
        if kind != "whole" and cond == "replay":
            continue
        r = {"model": model, "condition": cond, "N songs": df.song_id.nunique()}
        target = srows if kind != "whole" else erows
        if kind != "whole":
            r["sections"] = kind
        for d, label in DISTS:
            c, w = f"cross_{d}", f"within_{d}"
            if c not in df or w not in df:
                continue
            sub = df.dropna(subset=[c, w])
            if not len(sub):
                continue
            diff = (sub[c] - sub[w]).tolist()
            m, lo, hi, n = bootstrap_mean_ci(diff)
            ratio = float(sub[c].mean() / sub[w].mean()) if sub[w].mean() else float("nan")
            r[f"{label}: cross − within"] = fmt_ci(m, lo, hi, 3, True)
            r[f"{label}: ratio"] = f"{ratio:.2f}"
            fig[model][f"effect:{cond}:{kind}:{d}"] = [m, lo, hi, n, float(sub[c].mean()), float(sub[w].mean())]
        if kind == "whole" and "paired_ratio_melody" in df:
            pr = df.dropna(subset=["paired_ratio_melody"])
            if len(pr):
                m, lo, hi, n = bootstrap_mean_ci(pr["paired_ratio_melody"].tolist())
                r["seed-paired d(A,B)/d(A,C), melody"] = fmt_ci(m, lo, hi, 2)
        if kind == "whole" and "cross_transposed_d_melody" in df:
            t = df.dropna(subset=["cross_transposed_d_melody"])
            if len(t):
                diff = (t["cross_transposed_d_melody"] - t["within_d_melody"]).tolist()
                m, lo, hi, n = bootstrap_mean_ci(diff)
                r["melody vs orig TRANSPOSED: cross − within"] = fmt_ci(m, lo, hi, 3, True)
        target.append(r)
    write_table(erows, "interventions_effects", "Whole-song output change: mean distance between unmodified and intervened samples (different seeds) minus between two unmodified samples (reseed floor); > 0 = the control moved the output beyond sampling noise")
    write_table(srows, "interventions_effects_sections", "Section-level output change (cross − within) in the target section vs the sections before and after it")
    jdump(fig, DATA_OUT / "interventions_summary.json")
    print("INTERVENTION_REPORT_DONE")


if __name__ == "__main__":
    main()
