#!/usr/bin/env python
"""Control adherence in-distribution vs compositional vs extrapolative OOD plans.

  python -m paper_eval.ood_analysis

Per model, per OOD condition: the condition's own adherence metric, whole-plan exactness,
validity, lyric recall and cramming, each paired against the same songs' ``orig`` S1 sample,
plus adherence as a function of plan surprisal (bits under the training distribution).
"""

from __future__ import annotations

from collections import defaultdict

from qwen_abc.canonical import Song

from .collect import iter_new, sample_record
from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, load_rows, mean, paired_bootstrap, write_table
from .ood import OOD_CONDITIONS, REGIME, load_train_stats, plan_surprisal

MODELS = ["qwen_e3b", "qwen_e1", "qwen_e0", "mupt"]


def adherence(cond: str, row: dict) -> dict:
    rec = sample_record(row)
    meta = row.get("meta") or {}
    spec = row.get("spec")
    out = {"plan_exact": rec.get("section_plan_exact", 0.0), "strict_valid": rec.get("strict_valid", 0.0),
           "lyric_recall": rec.get("lyric_recall", 0.0), "cram": rec.get("cram_syllable_frac"),
           "label_seq": rec.get("section_label_seq_match", 0.0), "count_ok": rec.get("section_count_match", 0.0),
           "tempo_ok": rec.get("tempo_match", 0.0), "early_eos": rec.get("early_eos"),
           "counter_plan": rec.get("counter_plan_frac")}
    s = Song.from_json(row["song"]) if row.get("song") else None
    tgt = meta.get("target_section")
    if cond == "orig" and spec:
        from .interventions import pick_target_section
        tgt = pick_target_section(spec, row["song_id"])
    if cond in ("ood_long20", "ood_long28", "orig") and tgt is not None:
        want = spec["sections"][tgt]["bars"] if spec else None
        out["target_ok"] = float(s is not None and tgt < len(s.sections) and s.sections[tgt].num_bars == want
                                 and s.sections[tgt].label == spec["sections"][tgt]["label"])
    # the condition's own adherence metric
    own = {"ood_chorus_first": "plan_exact", "ood_end_on_verse": "plan_exact", "ood_long20": "target_ok",
           "ood_long28": "target_ok", "ood_sections_p3": "plan_exact", "ood_sections_p6": "plan_exact",
           "ood_tempo": "tempo_ok", "orig": "plan_exact"}[cond]
    out["own"] = out.get(own)
    return out


def main() -> None:
    rows_test = {r["song_id"]: r for r in load_rows("test")}
    stats = load_train_stats()
    table, deltas, scatter = [], [], []
    for model in MODELS:
        base = {sid: adherence("orig", r) for sid, s, r in iter_new(model, "orig") if s == "S1"}
        # orig target_ok needs the target index: take it from any intervention/ood row's meta
        conds = ["orig"] + list(OOD_CONDITIONS)
        per = {}
        for cond in conds:
            if cond == "orig":
                vals = base
                sur = {sid: plan_surprisal(stats, rows_test[sid]["spec"]) for sid in vals}
            else:
                vals, sur = {}, {}
                for sid, s, r in iter_new(model, cond):
                    vals[sid] = adherence(cond, r)
                    sur[sid] = (r.get("meta") or {}).get("surprisal")
            if not vals:
                continue
            per[cond] = vals
            row = {"model": model, "condition": cond, "regime": REGIME[cond], "N": len(vals)}
            for k, label in (("own", "condition adherence"), ("plan_exact", "exact plan"), ("strict_valid", "strict valid"),
                             ("lyric_recall", "lyric recall"), ("cram", "cramming")):
                m, lo, hi, n = bootstrap_mean_ci([v.get(k) for v in vals.values()])
                row[label] = fmt_ci(m, lo, hi)
            row["mean max label|bars surprisal (bits)"] = f"{mean([x['max_label_bars_bits'] for x in sur.values() if x]) or 0:.2f}"
            row["mean section-count surprisal (bits)"] = f"{mean([x['n_sections_bits'] for x in sur.values() if x]) or 0:.2f}"
            table.append(row)
            for sid, v in vals.items():
                if sur.get(sid):
                    scatter.append({"model": model, "condition": cond, "regime": REGIME[cond], "song_id": sid,
                                    **{f"s_{k}": x for k, x in sur[sid].items()}, **{k: x for k, x in v.items() if x is not None}})
            if cond != "orig":
                for k in ("plan_exact", "strict_valid", "lyric_recall"):
                    d = paired_bootstrap({s: v[k] for s, v in vals.items()}, {s: v[k] for s, v in base.items()})
                    deltas.append({"model": model, "condition": cond, "regime": REGIME[cond], "metric": k,
                                   "OOD − ID [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
        # regime pooled (song-level mean over the regime's conditions)
        for regime in ("in-distribution", "compositional", "extrapolative"):
            cs = [c for c in per if REGIME[c] == regime]
            pooled = defaultdict(list)
            for c in cs:
                for sid, v in per[c].items():
                    pooled[sid].append(v["own"])
            m, lo, hi, n = bootstrap_mean_ci([mean(x) for x in pooled.values()])
            table.append({"model": model, "condition": f"[{regime}, pooled]", "regime": regime, "N": n,
                          "condition adherence": fmt_ci(m, lo, hi)})
    write_table(table, "ood_generalization", "Control adherence by regime (seed S1; song-level 95% bootstrap CI). "
                "Condition adherence: exact plan (reordering, extra sections), target-section bars (long sections), tempo header (tempo)")
    write_table(deltas, "ood_deltas", "Paired OOD − in-distribution differences (same songs, seed S1)")
    import pandas as pd
    pd.DataFrame(scatter).to_parquet(DATA_OUT / "ood_samples.parquet", index=False)
    print("OOD_DONE")


if __name__ == "__main__":
    main()
