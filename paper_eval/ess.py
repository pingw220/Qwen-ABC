#!/usr/bin/env python
"""Explicit Structural State (ESS) ablation and long-range (position-resolved) analysis.

  python -m paper_eval.ess

Training/representation ablation only -- inference-time selection is reported separately
(FAIR_DECODING.md). Two decoding settings:
* T0.8, one sample per song: the round-2 evaluations (legacy; E1c and E3 exist only here);
* T1.0 (canonical), four paired-seed samples per song: E0, E1, E1-long, E3b (new this round).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .collect import iter_system, sample_record
from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, load_rows, mean, paired_bootstrap, write_table
from .sections import section_records

ABLATION_T08 = [("E0", "legacy:E0@T0.8", "ABC-v1, raw boundaries"), ("E1c", "legacy:E1c@T0.8", "+ boundary cleaning"),
                ("E1", "legacy:E1@T0.8", "+ ESS (header + [r:k])"), ("E1-long", "legacy:E1long@T0.8", "ESS, 758 updates (matched control)"),
                ("E3b", "legacy:E3b@T0.8", "ESS + late-section reconstruction, 758 updates"),
                ("E3 (524K batch)", "legacy:E3@T0.8", "ESS + reconstruction at 4x batch"),
                ("E0 seed 2", "legacy:E0s2@T0.8", "training-seed replicate"), ("E1 seed 2", "legacy:E1s2@T0.8", "training-seed replicate")]
ABLATION_T10 = [("E0", "new:qwen_e0/orig", "ABC-v1, raw boundaries"), ("E1", "new:qwen_e1/orig", "+ cleaning + ESS"),
                ("E1-long", "new:qwen_e1long/orig", "ESS, 758 updates"), ("E3b", "new:qwen_e3b/orig", "ESS + late-section reconstruction")]
METRICS = [("section_plan_exact", "exact structure"), ("section_label_seq_match", "label sequence exact"),
           ("section_bars_exact_frac", "sections exact (per section)"), ("abs_total_bar_error", "|total bars − request|"),
           ("early_eos", "early termination"), ("late_fail", "late-section failure (last third)"),
           ("lyric_recall", "lyric recall"), ("strict_valid", "strict validity"),
           ("counter_self_consistent_frac", "countdown = bars left (written)"), ("counter_plan_frac", "countdown = plan"),
           ("length_err", "|total bars ratio − 1|")]
BUCKET_METRICS = [("fail", "section not exact"), ("label_fail", "wrong label"), ("bar_err", "|bar error|"),
                  ("lyric_omission", "lyric omission"), ("cram", "cramming"), ("bad_bar_frac", "bars with wrong duration"),
                  ("premature_eos", "premature EOS"), ("counter_inconsistent", "countdown ≠ bars left")]


def per_song(spec_sys: str, specs: Dict[str, dict], want_sections: bool = True):
    """Per-song means of sample metrics, and per-song per-bucket means of section outcomes."""
    by_song = defaultdict(list)
    for song, sample, row in iter_system(spec_sys):
        by_song[song].append(row)
    song_m = {m: {} for m, _ in METRICS}
    bucket = {b: {k: {} for k, _ in BUCKET_METRICS} for b in range(5)}
    for song, rows in by_song.items():
        recs = [sample_record(r) for r in rows]
        for r in recs:
            if "late_section_exact" in r:
                r["late_fail"] = 1 - r["late_section_exact"]
            if "total_bars_ratio" in r:
                r["length_err"] = abs(r["total_bars_ratio"] - 1)
        for m, _ in METRICS:
            v = mean([r.get(m) for r in recs])
            if v is not None:
                song_m[m][song] = v
        if not want_sections:
            continue
        acc = {b: defaultdict(list) for b in range(5)}
        for r in rows:
            fmt_v1 = "[r:" not in (r.get("generation") or "")
            for s in section_records(r, specs[song], with_counters=not fmt_v1):
                b = s["bucket"]
                acc[b]["fail"].append(1 - s["exact"])
                acc[b]["label_fail"].append(1 - s["label_ok"])
                if "bar_err" in s:
                    acc[b]["bar_err"].append(s["bar_err"])
                if "lyric_recall" in s:
                    acc[b]["lyric_omission"].append(1 - s["lyric_recall"])
                if s.get("cram") is not None:
                    acc[b]["cram"].append(s["cram"])
                if s.get("bad_bar_frac") is not None:
                    acc[b]["bad_bar_frac"].append(s["bad_bar_frac"])
                acc[b]["premature_eos"].append(s.get("premature_eos", 0.0))
                if "counter_self_ok" in s:
                    acc[b]["counter_inconsistent"].append(1 - s["counter_self_ok"])
        for b in range(5):
            for k, vals in acc[b].items():
                if vals:
                    bucket[b][k][song] = sum(vals) / len(vals)
    return song_m, bucket, len(by_song)


def main() -> None:
    specs = {r["song_id"]: r["spec"] for r in load_rows("test")}
    out = {}
    for tag, systems in (("T0.8", ABLATION_T08), ("T1.0", ABLATION_T10)):
        res = {}
        for name, spec_sys, desc in systems:
            sm, bk, n = per_song(spec_sys, specs, want_sections=not name.endswith("seed 2") and "524K" not in name)
            res[name] = {"song": sm, "bucket": bk, "n": n, "desc": desc}
            print(tag, name, n)
        out[tag] = res
        rows = []
        for name, _, desc in systems:
            r = res[name]
            row = {"model": name, "change": desc, "N": r["n"]}
            for m, label in METRICS:
                a, lo, hi, n = bootstrap_mean_ci(list(r["song"][m].values()))
                row[label] = fmt_ci(a, lo, hi, 3 if "bars −" not in label else 2)
            rows.append(row)
        write_table(rows, f"ess_ablation_{tag.replace('.', '')}",
                    f"ESS / long-structure ablation at {tag} ({'1 sample' if tag == 'T0.8' else 'mean of 4 samples'} per song, 225 test songs); training/representation only, no selection")
        steps = [("E1c", "E0"), ("E1", "E1c"), ("E1-long", "E1"), ("E3b", "E1-long"), ("E3b", "E1"), ("E1", "E0")]
        drows = []
        for a, b in steps:
            if a not in res or b not in res:
                continue
            for m, label in METRICS[:8]:
                d = paired_bootstrap(res[a]["song"][m], res[b]["song"][m])
                drows.append({"step": f"{a} − {b}", "metric": label, "delta [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True),
                              "N": d["n"], "significant": "yes" if d["lo"] is not None and (d["lo"] > 0 or d["hi"] < 0) else "no"})
        write_table(drows, f"ess_ablation_deltas_{tag.replace('.', '')}", f"Paired ESS-ablation steps at {tag}")
        # long-range buckets
        brows = []
        for name, _, _ in systems:
            if name.endswith("seed 2") or "524K" in name:
                continue
            for k, label in BUCKET_METRICS:
                row = {"model": name, "metric": label}
                for b in range(5):
                    m_, lo, hi, n = bootstrap_mean_ci(list(res[name]["bucket"][b][k].values()))
                    row[f"{20 * b}-{20 * (b + 1)}%"] = fmt_ci(m_, lo, hi, 3) if m_ is not None else "–"
                brows.append(row)
        write_table(brows, f"long_range_{tag.replace('.', '')}", f"Failure rates by normalized song position ({tag}); song-level bootstrap")
    # compact json for figures
    fig = {}
    for tag, res in out.items():
        fig[tag] = {}
        for name, r in res.items():
            fig[tag][name] = {"metrics": {m: bootstrap_mean_ci(list(r["song"][m].values())) for m, _ in METRICS},
                              "buckets": {k: [bootstrap_mean_ci(list(r["bucket"][b][k].values())) for b in range(5)]
                                          for k, _ in BUCKET_METRICS}}
    # late-vs-early paired contrast per model (does failure grow with position?)
    trend = []
    for tag, res in out.items():
        for name, r in res.items():
            if name.endswith("seed 2") or "524K" in name:
                continue
            for k, label in BUCKET_METRICS[:4]:
                d = paired_bootstrap(r["bucket"][4][k], r["bucket"][0][k])
                trend.append({"decoding": tag, "model": name, "metric": label, "last 20% − first 20%": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
    write_table(trend, "long_range_trend", "Change in failure rate from the first to the last fifth of the song (paired over songs)")
    jdump(fig, DATA_OUT / "ess_long_range.json")
    print("ESS_DONE")


if __name__ == "__main__":
    main()
