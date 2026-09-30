#!/usr/bin/env python
"""Lyrics->melody evaluation of every melody source on the 225 test songs.

  python -m paper_eval.component_sota.melody_eval [--sources ref,e3b,mel,csl_off,csl_rt] [--workers 16]

Writes reports/component_sota/data/melody_samples_<condition>.parquet (one row per sample, failures kept)
and tables/lyrics_to_melody.* (Table 1) with song-level bootstrap CIs and paired differences.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from multiprocessing import Pool

from qwen_abc.canonical import Song
from qwen_abc.metrics import song_metrics
from qwen_abc.prompt import split_syllables

from ..common import REPORT_DIR, bootstrap_mean_ci, fmt_ci, load_rows, mean, paired_bootstrap, write_table
from ..interventions import note_tokens, recall_of, syllables_of
from ..musicality import song_stats
from ..seqsim import lcs_sim
from .melody_metrics import dd_score, dd_tb_score, md_score, pd_pc_score, pd_score, phrase_lengths
from .sources import SEEDS, iter_melodies

CS_REPORT = REPORT_DIR.parent / "component_sota"
CJK = re.compile(r"[㐀-鿿豈-﫿]")
DISPLAY = {"ref": "Pseudo-reference", "e3b": "Qwen Full (E3b) → melody", "mel": "Qwen Melody-Only",
           "csl_off": "CSL-L2M official, whole song", "csl_offc": "CSL-L2M official, section-chunked",
           "csl_rt": "CSL-L2M retrained, whole song", "csl_rtc": "CSL-L2M retrained, section-chunked"}
STRUCTURED = {"ref", "e3b", "mel"}     # sources that receive and can follow the section plan


def score(args):
    row, ref_json, spec = args
    rec = {"song_id": row["song_id"], "source": row["source"], "seed": row["seed"], "gen_success": float(bool(row["ok"])),
           "failure": row.get("failure")}
    if not row["ok"]:
        for k in ("lyric_recall", "lyric_recall_cjk") + (("section_plan_exact",) if row["source"] in STRUCTURED else ()):
            rec[k] = 0.0   # a failed song counts as zero; structure only for sources that receive the plan
        return rec
    s, ref = Song.from_json(row["song"]), Song.from_json(ref_json)
    m = song_metrics(s, spec)
    for k in ("lyric_recall", "lyric_precision", "section_lyric_recall", "cram_syllable_frac", "multi_syllable_note_frac",
              "melisma_note_frac", "wordless_note_frac", "notes_per_bar", "pitch_range", "mean_abs_interval",
              "distinct_bar_frac", "pitch_4gram_repeat_frac", "syncopation_frac", "section_plan_exact",
              "section_count_match", "section_label_seq_match", "section_bars_exact_frac", "late_section_exact",
              "abs_total_bar_error"):
        if k in m and m[k] is not None and not (isinstance(m[k], float) and math.isnan(m[k])):
            rec[k] = float(m[k])
    if row["source"] not in STRUCTURED:   # no plan input: structure metrics are not applicable
        for k in ("section_plan_exact", "section_count_match", "section_label_seq_match", "section_bars_exact_frac",
                  "late_section_exact", "abs_total_bar_error", "section_lyric_recall"):
            rec.pop(k, None)
    cjk = [x for sec in spec["sections"] for x in syllables_of(sec) if CJK.fullmatch(x)]
    rec["lyric_recall_cjk"] = recall_of(s, cjk)
    rec["PD"] = pd_score(s, ref)
    rec["PD_pc"] = pd_pc_score(s, ref)
    rec["DD"] = dd_score(s, ref)
    rec["DD_tb"] = dd_tb_score(s, ref)
    rec["pitch_mean_minus_ref"] = sum(n.pitch for n in s.notes) / len(s.notes) - sum(n.pitch for n in ref.notes) / len(ref.notes)
    rec["MD"] = md_score(s, ref)
    rec["sim_pitch"] = lcs_sim(note_tokens(s, "pitch"), note_tokens(ref, "pitch"))
    rec["sim_interval"] = lcs_sim(note_tokens(s, "contour"), note_tokens(ref, "contour"))
    rec["sim_rhythm"] = lcs_sim(note_tokens(s, "rhythm"), note_tokens(ref, "rhythm"))
    sc = song_stats(s)["scalars"]
    for k in ("rhythm_entropy", "repeat_pitch_frac", "step_frac", "contour_up_frac"):
        if sc.get(k) is not None:
            rec[k] = sc[k]
    ps = [n.pitch for n in s.notes]
    rec["pitch_std"] = float((sum((p - sum(ps) / len(ps)) ** 2 for p in ps) / len(ps)) ** 0.5)
    pl = phrase_lengths(s)
    rec["phrase_len_notes"] = sum(pl) / len(pl) if pl else None
    rec["notes_per_syllable"] = m.get("notes_per_syllable")
    rec["dur_mean_beats"] = m.get("mean_duration_beats")
    return rec


TABLE_COLS = [("gen_success", "generation success", 3), ("PD", "PD ↑", 3), ("PD_pc", "PD (pitch class) ↑", 3), ("DD", "DD ↑", 3), ("DD_tb", "DD, timebase-invariant ↑", 3), ("MD", "MD ↓", 2),
              ("pitch_range", "pitch range", 1), ("notes_per_bar", "notes/bar", 2), ("lyric_recall", "lyric recall", 3),
              ("lyric_recall_cjk", "lyric recall (Han only)", 3), ("cram_syllable_frac", "crammed syll.", 3),
              ("melisma_note_frac", "melisma notes", 3), ("section_plan_exact", "exact structure", 3),
              ("sim_interval", "interval sim. to ref ↑", 3), ("rhythm_entropy", "rhythm entropy", 2),
              ("distinct_bar_frac", "distinct bars", 3)]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="ref,e3b,mel,csl_offc,csl_off,csl_rtc,csl_rt")
    ap.add_argument("--condition", default="orig")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rows = {r["song_id"]: r for r in load_rows("test")}
    jobs = []
    for src in args.sources.split(","):
        seeds = ["ref"] if src == "ref" else SEEDS
        n = 0
        for sd in seeds:
            for r in iter_melodies(src, sd if sd != "ref" else "S1", args.condition):
                jobs.append((r, rows[r["song_id"]]["song"], rows[r["song_id"]]["spec"]))
                n += 1
        print(src, n, flush=True)
    with Pool(args.workers) as p:
        recs = p.map(score, jobs, chunksize=4)
    df = pd.DataFrame(recs)
    (CS_REPORT / "data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(CS_REPORT / "data" / f"melody_samples_{args.condition}.parquet", index=False)
    # --- table 1
    per = {src: {c: df[df.source == src].groupby("song_id")[c].mean().dropna().to_dict() for c, _, _ in TABLE_COLS if c in df}
           for src in df.source.unique()}
    table = []
    for src in [s for s in args.sources.split(",") if s in per]:
        row = {"system": DISPLAY.get(src, src), "N songs": len(set(df[df.source == src].song_id)),
               "samples": int((df.source == src).sum())}
        for c, label, d in TABLE_COLS:
            v = per[src].get(c, {})
            m, lo, hi, n = bootstrap_mean_ci(list(v.values()))
            row[label] = fmt_ci(m, lo, hi, d) if m is not None else "n/a"
        table.append(row)
    write_table(table, "lyrics_to_melody" + ("" if args.condition == "orig" else f"_{args.condition}"), "Lyrics→melody on the 225 test songs (mean over 4 samples per song; failures = 0 in rates; "
                "PD/DD/MD vs the pseudo-reference as in CSL-L2M). Structure n/a for systems without plan input.",
                table_dir=CS_REPORT / "tables")
    pairs = [("mel", "csl_offc"), ("mel", "csl_rtc"), ("mel", "e3b"), ("csl_rtc", "csl_offc"), ("e3b", "csl_offc"), ("e3b", "csl_rtc"),
             ("mel", "csl_off"), ("mel", "csl_rt"), ("csl_offc", "csl_off"), ("csl_rtc", "csl_rt")]
    drows = []
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        for c, label, _ in TABLE_COLS:
            if c in per[a] and c in per[b]:
                d = paired_bootstrap(per[a][c], per[b][c])
                if d["n"]:
                    drows.append({"comparison": f"{DISPLAY[a]} − {DISPLAY[b]}", "metric": label,
                                  "difference [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N songs": d["n"],
                                  "resolved": "yes" if d["lo"] > 0 or d["hi"] < 0 else "no"})
    write_table(drows, "lyrics_to_melody_paired" + ("" if args.condition == "orig" else f"_{args.condition}"), "Paired song-level differences (bootstrap 10,000)", table_dir=CS_REPORT / "tables")
    print("MELODY_EVAL_DONE")


if __name__ == "__main__":
    main()
