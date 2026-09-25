#!/usr/bin/env python
"""Pseudo-label noise: headline metrics on all 225 test songs vs the 34-song human-checked clean
subset, and failure probability vs pseudo-label quality indicators.

  python -m paper_eval.annotation_noise

Quality indicators are the 12 rule checks that selected the clean subset
(scripts/select_clean_subset.py RULES), counted per song, plus the corpus' own confidence
fields (quality tier, key-ensemble agreement). No human re-annotation exists beyond the 34-song
selection; human_annotation_package/ prepares the extension.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .collect import iter_system, sample_record
from .common import DATA_OUT, bootstrap_mean_ci, clean_ids, fmt_ci, jdump, load_rows, mean, spearman, write_table

SYSTEMS = {"Qwen-ABC E3b (single)": "new:qwen_e3b/orig", "Qwen-ABC E3b (selector@4, legacy)": "legacy:E3b@T1.0x4#sel",
           "MuPT (single)": "new:mupt/orig", "MIDI-LLM (single)": "midi:orig",
           "Qwen E1 (single)": "new:qwen_e1/orig", "Qwen E0 (single)": "new:qwen_e0/orig"}
METRICS = [("section_plan_exact", "exact structure", True), ("strict_valid", "strict validity", True),
           ("lyric_recall", "lyric recall", True), ("cram_syllable_frac", "cramming", False),
           ("lyric_alignment_valid", "no lyric-alignment error", True)]


def per_song(spec: str):
    sel = spec.endswith("#sel")
    spec = spec.replace("#sel", "")
    by = defaultdict(list)
    for sid, s, row in iter_system(spec):
        by[sid].append((s, row))
    out = defaultdict(dict)
    for sid, rows in by.items():
        if sel:
            from .decoding import selector_pick
            recs = [sample_record(selector_pick(rows))]
        else:
            recs = [sample_record(r) for _, r in rows]
        for m, _, _ in METRICS:
            v = mean([r.get(m) for r in recs])
            if v is not None:
                out[m][sid] = v
    return out


def diff_ci(a, b, n_boot=10000, seed=0):
    """Unpaired difference of means (a - b), each group resampled independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def main() -> None:
    import sys
    sys.path.insert(0, "scripts")
    from select_clean_subset import RULES
    rows = {r["song_id"]: r for r in load_rows("test")}
    clean = set(clean_ids())
    n_fail = {sid: sum(1 for fn in RULES.values() if not fn(r, r["flags"], r["cleaning"])) for sid, r in rows.items()}
    tier = {sid: (r["song"].get("meta") or {}).get("quality_tier") for sid, r in rows.items()}
    agree = {sid: (r["song"].get("meta") or {}).get("key_agreement") for sid, r in rows.items()}
    table, attrib, dose = [], [], []
    for name, spec in SYSTEMS.items():
        ps = per_song(spec)
        if not ps.get("section_plan_exact"):
            continue
        for m, label, higher in METRICS:
            v = ps[m]
            allv = [v[s] for s in rows if s in v]
            cl = [v[s] for s in clean if s in v]
            rest = [v[s] for s in rows if s in v and s not in clean]
            if not cl or not rest:
                continue
            ma = bootstrap_mean_ci(allv)
            mc = bootstrap_mean_ci(cl)
            d = diff_ci(cl, rest)
            table.append({"system": name, "metric": label, "all 225": fmt_ci(*ma[:3]), "N all": ma[3],
                          "clean 34": fmt_ci(*mc[:3]), "N clean": mc[3],
                          "clean − other 191 [95% CI]": fmt_ci(*d, 3, True)})
            if m in ("section_plan_exact", "strict_valid", "lyric_recall"):
                err_all = 1 - np.mean(allv)
                err_cl = 1 - np.mean(cl)
                # bootstrap the share of the full-set error rate that disappears on clean songs
                rng = np.random.default_rng(1)
                shares = []
                A, C = np.asarray(allv), np.asarray(cl)
                for _ in range(10000):
                    ea = 1 - rng.choice(A, len(A)).mean()
                    ec = 1 - rng.choice(C, len(C)).mean()
                    if ea > 0:
                        shares.append((ea - ec) / ea)
                lo, hi = np.quantile(shares, [0.025, 0.975]) if shares else (float("nan"), float("nan"))
                attrib.append({"system": name, "metric": label, "error rate, all": f"{err_all:.3f}",
                               "error rate, clean": f"{err_cl:.3f}",
                               "share of error absent on clean songs [95% CI]": fmt_ci((err_all - err_cl) / err_all if err_all else float("nan"), lo, hi, 2)})
        # failure vs number of failed quality rules (dose-response), exact-structure failure
        v = ps["section_plan_exact"]
        buckets = {"0": [], "1": [], "2": [], "3+": []}
        for s, x in v.items():
            buckets["3+" if n_fail[s] >= 3 else str(n_fail[s])].append(1 - x)
        row = {"system": name}
        for b, xs in buckets.items():
            m_, lo, hi, n = bootstrap_mean_ci(xs)
            row[f"{b} rules failed"] = f"{fmt_ci(m_, lo, hi)} (n={n})"
        # rank correlation with bootstrap CI
        ids = sorted(v)
        x = np.array([n_fail[s] for s in ids], float)
        y = np.array([1 - v[s] for s in ids], float)
        rho = spearman(x, y)
        rng = np.random.default_rng(2)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(ids), len(ids))
            r_ = spearman(x[i], y[i])
            if not np.isnan(r_):
                bs.append(r_)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        row["Spearman(#rules failed, structure failure)"] = fmt_ci(rho, lo, hi, 2, True)
        for grp, fld in (("key_agreement", agree), ("quality_tier", tier)):
            for val in sorted({g for g in fld.values() if g}):
                xs = [1 - v[s] for s in v if fld[s] == val]
                m_, lo, hi, n = bootstrap_mean_ci(xs)
                row[f"{grp}={val}"] = f"{fmt_ci(m_, lo, hi)} (n={n})"
        dose.append(row)
    write_table(table, "annotation_noise", "Headline metrics on all 225 pseudo-labelled test songs vs the 34-song human-checked clean subset")
    write_table(attrib, "annotation_noise_attribution", "Share of the full-set error rate that is absent on clean songs (upper-bound estimate of label-noise-attributable error; clean songs are also simpler)")
    write_table(dose, "annotation_noise_dose", "Exact-structure failure rate by number of pseudo-label quality rules the song fails, and by corpus confidence fields")
    from collections import Counter
    jdump({"n_rules_failed": n_fail, "hist": dict(Counter(n_fail.values())), "tier": dict(Counter(tier.values())),
           "key_agreement": dict(Counter(agree.values()))}, DATA_OUT / "annotation_noise_songs.json")
    print("ANNOTATION_NOISE_DONE")


if __name__ == "__main__":
    main()
