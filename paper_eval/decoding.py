#!/usr/bin/env python
"""Single sample vs selector@4 vs oracle@4 (and oracle@8), and the main three-family comparison.

  python -m paper_eval.decoding      # FAIR_DECODING.md inputs: tables/decoding_comparison.*, model_comparison.*

* single      -- per song, the mean over its samples (expected single-sample score)
* selector@4  -- the `sum` rule of scripts/select_best_of_n.py over 4 samples (inference-only signals)
* oracle@k    -- per metric, the best of k samples (upper bound; direction-aware)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List

from .collect import iter_system, sample_record
from .common import DATA_OUT, TABLE_DIR, bootstrap_mean_ci, fmt_ci, jdump, mean, paired_bootstrap, write_table

# metric -> (higher is better?, zero-on-failure rate?)
METRICS = {
    "gen_success": (True, True), "parse_ok": (True, True), "strict_valid": (True, True),
    "section_plan_exact": (True, True), "section_count_match": (True, True), "section_label_seq_match": (True, True),
    "section_bars_exact_frac": (True, True), "abs_total_bar_error": (False, False), "early_eos": (False, False),
    "more_sections_than_requested": (False, False), "counter_self_consistent_frac": (True, False),
    "counter_plan_frac": (True, False), "lyric_recall": (True, True), "lyric_precision": (True, False),
    "lyric_exact": (True, True), "section_lyric_recall": (True, True), "cram_syllable_frac": (False, False),
    "wordless_note_frac": (False, False), "melisma_note_frac": (None, False), "pitch_range": (None, False),
    "notes_per_bar": (None, False), "chords_per_bar": (None, False), "chord_tone_frac": (None, False),
    "melody_in_key_frac": (True, False), "distinct_bar_frac": (None, False), "syncopation_frac": (None, False),
}
HEADLINE = ["gen_success", "strict_valid", "section_plan_exact", "section_bars_exact_frac", "abs_total_bar_error",
            "early_eos", "lyric_recall", "lyric_exact", "cram_syllable_frac", "pitch_range"]

SYSTEMS = {
    # name: (system spec, samples used, family, description)
    "E3b (legacy x4)": ("legacy:E3b@T1.0x4", 4, "Qwen-ABC"),
    "E3b (paired-seed x4)": ("new:qwen_e3b/orig", 4, "Qwen-ABC"),
    "MuPT (legacy x4)": ("legacy:MuPT@T1.0x4", 4, "MuPT"),
    "MuPT (paired-seed x4)": ("new:mupt/orig", 4, "MuPT"),
    "MIDI-LLM (mode A x4)": ("midi:orig", 4, "MIDI-LLM"),
    "E1-long (paired-seed x4)": ("new:qwen_e1long/orig", 4, "Qwen-ABC"),
    "E1 (paired-seed x4)": ("new:qwen_e1/orig", 4, "Qwen-ABC"),
    "E0 (paired-seed x4)": ("new:qwen_e0/orig", 4, "Qwen-ABC"),
}
ORACLE8 = {"E3b": ("E3b (legacy x4)", "E3b (paired-seed x4)"), "MuPT": ("MuPT (legacy x4)", "MuPT (paired-seed x4)")}


def load_samples(spec: str) -> Dict[str, List[dict]]:
    by_song = defaultdict(list)
    for song, sample, row in iter_system(spec):
        by_song[song].append((sample, row))
    return by_song


def selector_pick(rows: List[tuple]) -> dict:
    from select_best_of_n import rank_key  # scripts/ on sys.path via common
    idx = []
    for sample, row in rows:
        seed = int("".join(ch for ch in sample if ch.isdigit()) or 0)
        idx.append((rank_key(row, "sum", seed), sample, row))
    return max(idx, key=lambda t: t[0])[2]


def per_song_scores(by_song: Dict[str, List[tuple]], k: int = 4, songs=None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{mode: {metric: {song: value}}} for single / first / selector / oracle."""
    out = {m: defaultdict(dict) for m in ("single", "first", "selector", "oracle")}
    for song, rows in by_song.items():
        if songs is not None and song not in songs:
            continue
        rows = sorted(rows, key=lambda t: t[0])[:k]
        recs = [sample_record(r) for _, r in rows]
        sel = sample_record(selector_pick(rows))
        for met, (higher, _) in METRICS.items():
            vals = [r.get(met) for r in recs]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if vals:
                out["single"][met][song] = sum(vals) / len(vals)
                if higher is not None:
                    out["oracle"][met][song] = max(vals) if higher else min(vals)
            if recs[0].get(met) is not None:
                out["first"][met][song] = recs[0][met]
            if sel.get(met) is not None:
                out["selector"][met][song] = sel[met]
    return out


def main() -> None:
    cache = {}
    results = {}
    for name, (spec, k, fam) in SYSTEMS.items():
        by_song = load_samples(spec)
        n_samples = {s: len(v) for s, v in by_song.items()}
        cache[name] = by_song
        if not by_song:
            print(f"skip {name}: no samples yet")
            continue
        full = [s for s, n in n_samples.items() if n >= k]
        scores = per_song_scores(by_song, k, set(full))
        secs = [r.get("seconds") or 0 for rows in by_song.values() for _, r in rows if isinstance(r.get("seconds"), (int, float))]
        results[name] = {"scores": scores, "n_songs": len(full), "n_songs_any": len(by_song), "family": fam,
                         "sec_per_sample": mean(secs), "spec": spec}
        print(f"{name}: songs with >= {k} samples {len(full)} / {len(by_song)}")
    # oracle@8
    for tag, (a, b) in ORACLE8.items():
        if a in cache and b in cache and cache[a] and cache[b]:
            merged = {s: [("L" + x[0], x[1]) for x in cache[a][s]] + [("N" + x[0], x[1]) for x in cache[b].get(s, [])]
                      for s in cache[a] if s in cache[b]}
            sc = per_song_scores(merged, 8, {s for s, v in merged.items() if len(v) >= 8})
            results[f"{tag} (x8 pooled)"] = {"scores": {"oracle8": sc["oracle"], "single": sc["single"]},
                                             "n_songs": len(merged), "family": tag}
    # ---- decoding table
    rows_out, raw = [], {}
    for name, r in results.items():
        sc = r["scores"]
        for mode in ("single", "selector", "oracle", "oracle8"):
            if mode not in sc:
                continue
            row = {"system": name, "mode": {"single": "single (mean of samples)", "selector": "selector@4",
                                            "oracle": "oracle@4", "oracle8": "oracle@8"}[mode], "N songs": r["n_songs"]}
            for met in HEADLINE:
                vals = sc[mode].get(met, {})
                m, lo, hi, n = bootstrap_mean_ci(list(vals.values()))
                row[met] = fmt_ci(m, lo, hi, 3 if met != "pitch_range" and met != "abs_total_bar_error" else 2)
                raw.setdefault(name, {}).setdefault(mode, {})[met] = {"mean": m, "lo": lo, "hi": hi, "n": n}
            rows_out.append(row)
    write_table(rows_out, "decoding_comparison", "Single sample (mean over 4), selector@4 and oracle@4/8 on the 225 test songs; 95% song-level bootstrap CI")
    # ---- paired deltas: selector - single, oracle - selector
    deltas = []
    for name, r in results.items():
        sc = r["scores"]
        if "selector" not in sc:
            continue
        for met in ("strict_valid", "section_plan_exact", "lyric_recall", "cram_syllable_frac", "pitch_range"):
            for a, b in (("selector", "single"), ("oracle", "selector")):
                d = paired_bootstrap(sc[a].get(met, {}), sc[b].get(met, {}))
                deltas.append({"system": name, "metric": met, "comparison": f"{a} − {b}",
                               "delta [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, signed=True), "N": d["n"]})
    write_table(deltas, "decoding_deltas", "Paired song-level differences between decoding modes")
    # ---- cost
    cost = []
    for name, r in results.items():
        if r.get("sec_per_sample"):
            cost.append({"system": name, "samples/song": 4 if "x4" in name else 1,
                         "GPU-s per sample": f"{r['sec_per_sample']:.1f}",
                         "GPU-h per 225 songs, single": f"{r['sec_per_sample'] * 225 / 3600:.2f}",
                         "GPU-h per 225 songs, @4": f"{4 * r['sec_per_sample'] * 225 / 3600:.2f}"})
    write_table(cost, "decoding_cost", "Inference cost (seconds are amortized batch-16 wall time for Qwen/MuPT; per-song process time for MIDI-LLM)")
    jdump(raw, DATA_OUT / "decoding_raw.json")
    jdump({n: {mode: {met: vals for met, vals in d.items()} for mode, d in r["scores"].items()}
           for n, r in results.items()}, DATA_OUT / "decoding_per_song.json", indent=None)
    print("DECODING_DONE")


if __name__ == "__main__":
    main()
