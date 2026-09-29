#!/usr/bin/env python
"""Joint vs cascaded lead-sheet generation (Table 5), assembled from the melody and chord results.

  python -m paper_eval.component_sota.joint_cascade     # after melody_eval and chord_eval

A *system* = (melody source, chord source). Melody-stage metrics come from
data/melody_samples_orig.parquet (seed S1), chord-stage metrics from data/chords.parquet; a system
succeeds on a song only if both stages produced output. Cost = measured seconds per song of each
stage (Qwen/MuPT: amortized batch-16 GPU seconds; CSL-L2M: per-song GPU seconds; AccoMontage2:
per-song CPU seconds).
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..common import PAPER_FINAL_GEN, bootstrap_mean_ci, fmt_ci, mean, paired_bootstrap, write_table
from .melody_eval import CS_REPORT, DISPLAY
from .sources import CS_ROOT

SYSTEMS = [("Qwen Joint (E3b)", "e3b", "joint"), ("Qwen Melody-Only → Qwen Chord", "mel", "qwen"),
           ("Qwen Melody-Only → AccoMontage2", "mel", "am2"), ("CSL-L2M (official) → Qwen Chord", "csl_offc", "qwen"),
           ("CSL-L2M (official) → AccoMontage2", "csl_offc", "am2"), ("CSL-L2M (retrained) → Qwen Chord", "csl_rtc", "qwen"),
           ("CSL-L2M (retrained) → AccoMontage2", "csl_rtc", "am2"), ("E3b melody → Qwen Chord (re-harmonized)", "e3b", "qwen"),
           ("E3b melody → AccoMontage2", "e3b", "am2")]
MEL_COLS = [("PD", "PD ↑"), ("DD", "DD ↑"), ("MD", "MD ↓"), ("lyric_recall", "lyric recall"), ("cram_syllable_frac", "crammed syll."),
            ("section_plan_exact", "exact structure")]
CH_COLS = [("chord_tone", "chord-tone ratio"), ("strong_chord_tone", "strong-beat chord-tone"), ("chroma_compat", "chroma compat."),
           ("root_in_key", "roots in key"), ("cadence_plausible", "cadence I/V"), ("chords_per_bar", "chords/bar")]


def stage_seconds():
    """Median measured seconds per song for each stage."""
    import glob
    import statistics as st
    out = {}
    def med(paths, key):
        v = []
        for p in paths:
            try:
                r = json.loads(open(p).read())
            except Exception:  # noqa: BLE001
                continue
            x = r.get(key) if key != "meta.seconds" else (r.get("meta") or {}).get("seconds")
            if isinstance(x, (int, float)):
                v.append(x)
        return st.median(v) if v else None
    out["e3b"] = med(glob.glob(str(PAPER_FINAL_GEN / "qwen_e3b/orig/*_S1.json")), "seconds")
    out["mel"] = med(glob.glob(str(CS_ROOT / "gen/qwen_mel/orig/*_S1.json")), "seconds")
    for k in ("csl_off", "csl_offc", "csl_rt", "csl_rtc"):
        out[k] = med(glob.glob(str(CS_ROOT / f"csl/{k}/orig/*_S1.json")), "meta.seconds")
    out["qwen"] = med(glob.glob(str(CS_ROOT / "chords/chord/*/*_S1.json")), "seconds")
    out["am2"] = med(glob.glob(str(CS_ROOT / "chords/am2/*/*_S1.json")), "seconds")
    out["joint"] = 0.0
    return out


def main():
    import pandas as pd
    mel = pd.read_parquet(CS_REPORT / "data" / "melody_samples_orig.parquet")
    mel = mel[mel.seed.isin(["S1", "ref"])]
    ch = pd.read_parquet(CS_REPORT / "data" / "chords.parquet")
    ch = ch[ch.chord_seed == "S1"]
    secs = stage_seconds()
    table, per = [], {}
    for name, ms, cs in SYSTEMS:
        m = mel[mel.source == ms].set_index("song_id")
        c = ch[(ch.melody == ms) & (ch.chords == cs)].set_index("song_id")
        if not len(m) or not len(c):
            print("skip", name)
            continue
        ids = sorted(set(m.index) | set(c.index))
        ok = {s: float(m.gen_success.get(s, 0) == 1 and c.ok.get(s, 0) == 1) for s in ids}
        row = {"system": name, "stages": 1 if cs == "joint" else 2, "N songs": len(ids)}
        mm, lo, hi, n = bootstrap_mean_ci(list(ok.values()))
        row["success (both stages)"] = fmt_ci(mm, lo, hi)
        per[name] = {"success": ok}
        for col, lab in MEL_COLS:
            if col in m and m[col].notna().any():
                v = m[col].dropna().to_dict()
                mm, lo, hi, n = bootstrap_mean_ci(list(v.values()))
                row[lab] = fmt_ci(mm, lo, hi, 2 if col == "MD" else 3)
                per[name][col] = v
            else:
                row[lab] = "n/a"
        for col, lab in CH_COLS:
            if col in c and c[col].notna().any():
                v = c[col].dropna().to_dict()
                mm, lo, hi, n = bootstrap_mean_ci(list(v.values()))
                row[lab] = fmt_ci(mm, lo, hi, 2 if col == "chords_per_bar" else 3)
                per[name][col] = v
        a, b = secs.get(ms), secs.get(cs)
        row["seconds/song (melody + chords)"] = f"{a:.1f} + {b:.1f}" if (a is not None and b is not None) else "–"
        row["local section edit"] = {"e3b": "yes (infill, measured)", "mel": "trained with infill, not evaluated"}.get(ms, "no")
        table.append(row)
    write_table(table, "joint_vs_cascade", "Joint vs cascaded lead-sheet systems on the 225 test songs (seed S1 for every stage). Structure n/a where the melody model receives no plan",
                table_dir=CS_REPORT / "tables")
    drows = []
    base = "Qwen Joint (E3b)"
    for other in [s for s, _, _ in SYSTEMS if s != base and s in per]:
        for col, lab in [("success", "success")] + MEL_COLS + CH_COLS:
            if col in per.get(base, {}) and col in per[other]:
                d = paired_bootstrap(per[base][col], per[other][col])
                if d["n"]:
                    drows.append({"comparison": f"Joint − [{other}]", "metric": lab, "difference [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
    write_table(drows, "joint_vs_cascade_paired", "Paired differences: Qwen Joint minus each cascade (same songs)", table_dir=CS_REPORT / "tables")
    json.dump(secs, open(CS_REPORT / "data" / "stage_seconds.json", "w"), indent=1)
    print("JOINT_CASCADE_DONE")


if __name__ == "__main__":
    main()
