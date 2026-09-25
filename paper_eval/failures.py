#!/usr/bin/env python
"""Lyric-failure analysis and automatic multi-label failure taxonomy.

  python -m paper_eval.failures

Tag thresholds come from the held-out references wherever a metric has a corpus band
(reference 95th/5th percentile), so a "failure" means leaving the range real songs occupy.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from qwen_abc.canonical import Song
from qwen_abc.metrics import song_metrics
from qwen_abc.prompt import split_syllables

from .collect import iter_system, sample_record
from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, load_rows, mean, spearman, write_table
from .sections import section_records

TAX_SYSTEMS = {"Qwen-ABC (E3b)": "new:qwen_e3b/orig", "MuPT": "new:mupt/orig", "MIDI-LLM": "midi:orig",
               "Qwen E0 (no ESS)": "new:qwen_e0/orig"}
LYRIC_SYSTEMS = {"E3b single": "new:qwen_e3b/orig", "E3b legacy x4": "legacy:E3b@T1.0x4",
                 "R3-A syllable budget x4": "legacy:R3A@T1.0x4", "R3-C repaired targets x4": "legacy:R3C@T1.0x4",
                 "decode-time '~' ban x4": "legacy:nocram@T1.0x4", "R5 free assignment x4": "legacy:R5@T1.0x4"}
TAGS = ["parse_failure", "invalid_syntax", "wrong_section_count", "wrong_section_label", "wrong_bar_count",
        "premature_eos", "excess_continuation", "lyric_omission", "lyric_duplication", "lyric_order_failure",
        "syllable_cramming", "timing_inconsistency", "degenerate_melody", "extreme_pitch_range", "repetitive_loop",
        "missing_chords", "front_end_refusal", "suspected_annotation_error"]
LYRIC_ERRS = {"lyric_overflow", "orphan_melisma", "lyric_bar_overflow"}


def ref_bands(refs):
    vals = defaultdict(list)
    for r in refs.values():
        m = song_metrics(Song.from_json(r["song"]), r["spec"])
        for k in ("cram_syllable_frac", "pitch_range", "distinct_bar_frac", "max_identical_consecutive_bars",
                  "chord_time_coverage", "lyric_precision", "wordless_note_frac", "int_repeat_frac"):
            if k in m:
                vals[k].append(m[k])
    q = lambda xs, p: float(np.quantile(xs, p))
    return {k: {"p01": q(v, 0.01), "p05": q(v, 0.05), "p50": q(v, 0.5), "p95": q(v, 0.95), "p99": q(v, 0.99)} for k, v in vals.items()}


def tags_for(row: dict, bands: dict, n_rule_fail: int) -> set:
    t = set()
    if row.get("failure") == "refused_input_g2p":
        return {"front_end_refusal"}
    if not row.get("parse_ok"):
        return {"parse_failure"}
    m = row.get("metrics") or {}
    errs = set((row.get("errors") or {}).keys())
    if errs - LYRIC_ERRS:   # any structural parser error (MIDI-LLM: its own validator's errors)
        t.add("invalid_syntax")
    if not m.get("section_count_match", 1):
        t.add("wrong_section_count")
    elif not m.get("section_label_seq_match", 1):
        t.add("wrong_section_label")
    if m.get("section_count_match", 1) and not m.get("section_bars_seq_match", 1):
        t.add("wrong_bar_count")
    if m.get("early_eos"):
        t.add("premature_eos")
    if m.get("more_sections_than_requested"):
        t.add("excess_continuation")
    if m.get("lyric_recall", 1) < 0.9:
        t.add("lyric_omission")
    if m.get("lyric_precision", 1) < 0.95:   # >5% of sung syllables are not in the requested order
        t.add("lyric_duplication")
    if m.get("lyric_recall", 1) - m.get("section_lyric_recall", 1) > 0.1:
        t.add("lyric_order_failure")
    if m.get("cram_syllable_frac", 0) > bands["cram_syllable_frac"]["p95"]:
        t.add("syllable_cramming")
    if m.get("bar_duration_ok_frac", 1) < 1 or m.get("counter_self_consistent_frac", 1) < 0.99:
        t.add("timing_inconsistency")
    if m.get("distinct_bar_frac", 1) < bands["distinct_bar_frac"]["p01"] or m.get("pitch_range", 99) < bands["pitch_range"]["p01"] \
            or m.get("int_repeat_frac", 0) > bands["int_repeat_frac"]["p99"]:
        t.add("degenerate_melody")
    if m.get("pitch_range", 0) > bands["pitch_range"]["p99"]:
        t.add("extreme_pitch_range")
    if m.get("max_identical_consecutive_bars", 1) > max(bands["max_identical_consecutive_bars"]["p99"], 3):
        t.add("repetitive_loop")
    if m.get("chord_time_coverage", 1) < min(bands["chord_time_coverage"]["p05"], 0.5):
        t.add("missing_chords")
    control_fail = t & {"wrong_section_count", "wrong_section_label", "wrong_bar_count", "lyric_omission", "syllable_cramming"}
    if control_fail and n_rule_fail >= 3:
        t.add("suspected_annotation_error")
    return t


def main() -> None:
    import sys
    sys.path.insert(0, "scripts")
    from select_clean_subset import RULES
    refs = {r["song_id"]: r for r in load_rows("test")}
    n_fail = {sid: sum(1 for fn in RULES.values() if not fn(r, r["flags"], r["cleaning"])) for sid, r in refs.items()}
    bands = ref_bands(refs)
    jdump(bands, DATA_OUT / "reference_bands.json")
    # ---------------------------------------------------------------- taxonomy
    tax_rows, examples, per_sample = [], {}, []
    for name, spec in TAX_SYSTEMS.items():
        counts, n = Counter(), 0
        by_song = defaultdict(list)
        for sid, s, row in iter_system(spec):
            tg = tags_for(row, bands, n_fail.get(sid, 0))
            n += 1
            counts.update(tg)
            by_song[sid].append(tg)
            per_sample.append({"system": name, "song_id": sid, "sample": s, **{k: float(k in tg) for k in TAGS}})
            for k in tg:
                examples.setdefault(name, {}).setdefault(k, [])
                if len(examples[name][k]) < 3:
                    m = row.get("metrics") or {}
                    examples[name][k].append({"song_id": sid, "sample": s, "strict_ok": row.get("strict_ok"),
                                              **{x: round(m[x], 3) if isinstance(m.get(x), float) else m.get(x)
                                                 for x in ("section_plan_exact", "lyric_recall", "cram_syllable_frac",
                                                           "pitch_range", "distinct_bar_frac", "eos_section_index")}})
        if not n:
            continue
        row = {"system": name, "N samples": n, "songs": len(by_song),
               "no tag": f"{sum(1 for v in by_song.values() for t in v if not t) / n:.3f}"}
        for k in TAGS:
            song_rates = [mean([float(k in t) for t in v]) for v in by_song.values()]
            m_, lo, hi, _ = bootstrap_mean_ci(song_rates)
            row[k] = fmt_ci(m_, lo, hi)
        tax_rows.append(row)
    write_table(tax_rows, "failure_taxonomy", "Share of samples carrying each failure tag (multi-label; single samples, song-level 95% bootstrap)")
    jdump(examples, DATA_OUT / "failure_examples.json")
    import pandas as pd
    pd.DataFrame(per_sample).to_parquet(DATA_OUT / "failure_tags.parquet", index=False)
    # co-occurrence for the main system
    ps = pd.DataFrame(per_sample)
    e3 = ps[ps.system == "Qwen-ABC (E3b)"][TAGS]
    if len(e3):
        co = (e3.T @ e3).astype(int)
        co.to_csv(DATA_OUT / "failure_cooccurrence_e3b.csv")
    # ---------------------------------------------------------------- lyric failures
    lyr = []
    ref_m = {sid: song_metrics(Song.from_json(r["song"]), r["spec"]) for sid, r in refs.items()}
    for name, spec in [("Pseudo-GT reference", None)] + list(LYRIC_SYSTEMS.items()):
        vals = defaultdict(lambda: defaultdict(list))
        if spec is None:
            for sid, m in ref_m.items():
                for k in ("cram_syllable_frac", "lyric_recall", "multi_syllable_note_frac", "wordless_note_frac", "melisma_note_frac", "lyric_precision", "strict_valid"):
                    if k in m:
                        vals[k][sid].append(m[k])
                vals["strict_valid"][sid] = [1.0]
        else:
            for sid, s, row in iter_system(spec):
                rec = sample_record(row)
                for k in ("cram_syllable_frac", "lyric_recall", "multi_syllable_note_frac", "wordless_note_frac", "melisma_note_frac", "lyric_precision", "strict_valid"):
                    if rec.get(k) is not None:
                        vals[k][sid].append(rec[k])
        row = {"system": name}
        for k, label in (("cram_syllable_frac", "crammed syllables"), ("lyric_recall", "lyric recall"),
                         ("multi_syllable_note_frac", "notes with >1 syllable"), ("wordless_note_frac", "wordless notes"),
                         ("melisma_note_frac", "melisma notes"), ("lyric_precision", "lyric precision"), ("strict_valid", "strict validity")):
            m_, lo, hi, n = bootstrap_mean_ci([mean(v) for v in vals[k].values()])
            row[label] = fmt_ci(m_, lo, hi)
        row["N songs"] = len(vals["lyric_recall"])
        lyr.append(row)
    write_table(lyr, "lyric_failures", "Lyric realization for the current model, the corpus, and every earlier cramming attack (single samples, mean over 4 where available)")
    # syllable-per-bar pressure and position, E3b single samples vs reference
    press = defaultdict(lambda: defaultdict(list))
    corr = {"song_sylls": [], "cram": [], "recall": [], "ref_cram": []}
    edges = [0, 2, 4, 6, 8, 1e9]
    lab = ["<2", "2-4", "4-6", "6-8", ">=8"]
    by_song = defaultdict(list)
    for sid, s, row in iter_system("new:qwen_e3b/orig"):
        by_song[sid].append(row)
    for sid, rows in by_song.items():
        spec = refs[sid]["spec"]
        for row in rows:
            for sec in section_records(row, spec):
                if not sec["req_syllables"]:
                    continue
                dens = sec["req_syllables"] / max(sec["req_bars"], 1)
                b = lab[max(i for i, e in enumerate(edges[:-1]) if dens >= e)]
                if sec.get("cram") is not None:
                    press[b]["cram"].append((sid, sec["cram"]))
                if "lyric_recall" in sec:
                    press[b]["recall"].append((sid, sec["lyric_recall"]))
        recs = [sample_record(r) for r in rows]
        corr["song_sylls"].append(sum(len(split_syllables(line)) for sec in refs[sid]["spec"]["sections"] for line in sec["lines"]))
        corr["cram"].append(mean([r.get("cram_syllable_frac") for r in recs]) or 0)
        corr["recall"].append(mean([r.get("lyric_recall") for r in recs]) or 0)
        corr["ref_cram"].append(ref_m[sid].get("cram_syllable_frac", 0))
    # reference cramming per density bucket
    ref_press = defaultdict(list)
    for sid, r in refs.items():
        s = Song.from_json(r["song"])
        fake = {"song": r["song"], "generation": r["abc"], "hit_eos": True}
        for sec in section_records(fake, r["spec"]):
            if sec["req_syllables"] and sec.get("cram") is not None:
                dens = sec["req_syllables"] / max(sec["req_bars"], 1)
                ref_press[lab[max(i for i, e in enumerate(edges[:-1]) if dens >= e)]].append((sid, sec["cram"]))
    prow = []
    for b in lab:
        def song_level(pairs):
            d = defaultdict(list)
            for sid, v in pairs:
                d[sid].append(v)
            return [mean(v) for v in d.values()]
        mc = bootstrap_mean_ci(song_level(press[b]["cram"]))
        mr = bootstrap_mean_ci(song_level(press[b]["recall"]))
        rc = bootstrap_mean_ci(song_level(ref_press[b]))
        prow.append({"syllables per bar (requested)": b, "sections (E3b samples)": len(press[b]["cram"]),
                     "E3b cramming": fmt_ci(*mc[:3]), "reference cramming": fmt_ci(*rc[:3]),
                     "E3b section lyric recall": fmt_ci(*mr[:3])})
    write_table(prow, "lyric_pressure", "Cramming and lyric recall by requested syllable density (song-level means within each bucket)")
    jdump({"spearman_song_length_vs_cram": spearman(corr["song_sylls"], corr["cram"]),
           "spearman_song_length_vs_recall": spearman(corr["song_sylls"], corr["recall"]),
           "spearman_reference_cram_vs_generated_cram": spearman(corr["ref_cram"], corr["cram"]),
           "n_songs": len(corr["cram"])}, DATA_OUT / "lyric_correlations.json")
    print("FAILURES_DONE")


if __name__ == "__main__":
    main()
