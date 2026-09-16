#!/usr/bin/env python
"""Independent validation of an ABC-v2 build against its ABC-v1 parent (re-derived from written files).

  scripts/validate_abc_v2_dataset.py --v2-dir data/generated/abc_v2_<ts> --v1-dir data/generated/abc_v1_<ts> --report OUT.json

Hard failures (non-zero exit): split differs from v1, song-id overlap, any
example that does not parse strictly / round-trip / keep v1's notes, chords,
bars, labels and lyrics, wrong counters, prompt that does not rebuild, MIDI
round-trip failure, identical lyrics across splits.
Also re-verifies leakage without the build's document-frequency filter, and
reports token lengths and section-length distributions before/after cleaning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_abc_dataset import midi_roundtrip_ok  # noqa: E402
from qwen_abc.abc import parse_abc, song_to_abc  # noqa: E402
from qwen_abc.abc_v2 import counter_report, spec_to_prompt_v2, strip_v2  # noqa: E402
from qwen_abc.canonical import Song, comparable  # noqa: E402
from qwen_abc.prompt import spec_syllables, spec_to_prompt, song_to_spec  # noqa: E402
from qwen_abc.splits import SPLITS, ShingleIndex, lyric_shingles, lyrics_exact_hash, melody_shingles  # noqa: E402


def pct(values, q):
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)] if values else None


def check_one(args):
    v2_line, sft_line, v1c_line, v1_line = args
    r, sft, v1c, v1 = json.loads(v2_line), json.loads(sft_line), json.loads(v1c_line), json.loads(v1_line)
    sid = r["song_id"]
    assert sft["song_id"] == sid == v1c["song_id"] == v1["song_id"], "file order differs"
    song = Song.from_json(r["song"])
    old = Song.from_json(v1["song"])
    c = {}
    p = parse_abc(sft["completion"], sid)
    c["v2_parse_ok"] = p.ok
    c["v2_strict"] = p.strict_ok
    c["v2_roundtrip"] = p.ok and comparable(p.song) == comparable(song)
    c["v1clean_roundtrip"] = parse_abc(v1c["completion"]).ok and comparable(parse_abc(v1c["completion"]).song) == comparable(song)
    c["strip_equals_v1clean"] = strip_v2(sft["completion"]) == v1c["completion"]
    c["v1clean_writer_reproduces"] = song_to_abc(song) == v1c["completion"]
    a, b = comparable(old), comparable(song)
    c["notes_and_lyrics_equal_v1"] = a["notes"] == b["notes"]
    c["chords_equal_v1"] = a["chords"] == b["chords"]
    c["bars_equal_v1"] = a["bar_beats"] == b["bar_beats"]
    c["meta_equal_v1"] = (a["meter_num"], a["tempo_bpm"], a["key"]) == (b["meter_num"], b["tempo_bpm"], b["key"])
    c["section_labels_equal_v1"] = [s[0] for s in a["sections"]] == [s[0] for s in b["sections"]]
    c["sections_tile_bars"] = sum(s.num_bars for s in song.sections) == len(song.bar_beats) and \
        all(x.start_bar + x.num_bars == y.start_bar for x, y in zip(song.sections, song.sections[1:]))
    c["spec_rebuilds"] = song_to_spec(song) == r["spec"]
    c["prompt_v2_rebuilds"] = spec_to_prompt_v2(r["spec"]) == sft["prompt"]
    c["prompt_v1clean_rebuilds"] = spec_to_prompt(r["spec"]) == v1c["prompt"]
    c["prompt_lyrics_equal_abc_lyrics"] = spec_syllables(r["spec"]) == [x for n in song.notes if n.lyric for x in n.lyric]
    cr = counter_report(sft["completion"], r["spec"])
    c["counters_all_correct"] = all(cr[k] == 1.0 for k in ("counter_present_frac", "counter_self_consistent_frac",
                                                          "counter_plan_frac", "header_plan_frac", "section_header_frac",
                                                          "sections_ending_on_counter_1"))
    c["counter_bars_equal_bars"] = cr["counter_bars"] == len(song.bar_beats)
    c["midi_roundtrip"] = midi_roundtrip_ok(p.song) if p.ok else False
    info = {
        "song_id": sid,
        "lens_before": [s.num_bars for s in old.sections], "lens_after": [s.num_bars for s in song.sections],
        "tokens_v2": sft["prompt_tokens"] + sft["completion_tokens"] + 1,
        "tokens_v1clean": v1c["prompt_tokens"] + v1c["completion_tokens"] + 1,
        "tokens_v1": v1["prompt_tokens"] + v1["abc_tokens"] + 1,
        "prompt_v2": sft["prompt_tokens"], "abc_v2": sft["completion_tokens"] + 1,
        "lyric_sh": sorted(lyric_shingles(song)), "melody_sh": sorted(melody_shingles(song)), "lyrics_hash": lyrics_exact_hash(song),
        "n_chords": len(song.chords), "n_notes": len(song.notes), "n_sections": len(song.sections), "n_bars": len(song.bar_beats),
    }
    return c, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    d2, d1 = args.v2_dir, args.v1_dir
    rep, hard = {"v2_dir": str(d2), "v1_dir": str(d1)}, []
    m2 = json.load(open(d2 / "split_manifest.json"))
    m1 = json.load(open(d1 / "split_manifest.json"))
    rep["split_identical_to_v1"] = m1 == m2
    if m1 != m2:
        hard.append("split differs from v1")
    rep["split_sizes"] = {k: len(v) for k, v in m2.items()}
    ids = {k: set(v) for k, v in m2.items()}
    rep["song_id_overlap"] = {f"{a}/{b}": len(ids[a] & ids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(rep["song_id_overlap"].values()):
        hard.append("song id overlap")

    checks = {s: Counter() for s in SPLITS}
    infos = {s: [] for s in SPLITS}
    failures = []
    with Pool(args.workers) as pool:
        for split in SPLITS:
            files = [open(d2 / f"songs_{split}.jsonl", encoding="utf-8"), open(d2 / f"sft_{split}.jsonl", encoding="utf-8"),
                     open(d2 / f"sft_v1clean_{split}.jsonl", encoding="utf-8"), open(d1 / f"songs_{split}.jsonl", encoding="utf-8")]
            for c, info in pool.imap(check_one, zip(*files), chunksize=16):
                checks[split]["examples"] += 1
                for k, ok in c.items():
                    checks[split][k] += int(ok)
                    if not ok and len(failures) < 50:
                        failures.append((split, info["song_id"], k))
                infos[split].append(info)
    rep["checks"] = {s: dict(v) for s, v in checks.items()}
    for s, v in checks.items():
        bad = {k: v["examples"] - n for k, n in v.items() if k != "examples" and n != v["examples"]}
        if bad:
            hard.append(f"{s}: {bad}")
    rep["check_failure_examples"] = failures

    # ------------------------------------------------ leakage (re-derived, no df filter and with the build's filter)
    li, mi, exact = ShingleIndex(), ShingleIndex(), {}
    for x in infos["train"]:
        li.add(x["song_id"], x["lyric_sh"])
        mi.add(x["song_id"], x["melody_sh"])
        exact.setdefault(x["lyrics_hash"], x["song_id"])
    tli, tmi = ShingleIndex(), ShingleIndex()
    for x in infos["test"]:
        tli.add(x["song_id"], x["lyric_sh"])
        tmi.add(x["song_id"], x["melody_sh"])
    leak = {}
    for split in ("validation", "test"):
        row = {}
        for label, max_df in (("build_filter_df100", 100), ("no_df_filter", 10 ** 9)):
            lyr = [li.best_match(set(x["lyric_sh"]), max_df)[1] for x in infos[split]]
            mel = [mi.best_match(set(x["melody_sh"]), max_df)[1] for x in infos[split]]
            row[label] = {"lyrics_max": max(lyr), "lyrics_p95": pct(lyr, 0.95), "lyrics_ge_0.3": sum(v >= 0.3 for v in lyr),
                          "melody_max": max(mel), "melody_p95": pct(mel, 0.95), "melody_ge_0.3": sum(v >= 0.3 for v in mel)}
        row["exact_lyrics_duplicate_of_train"] = sum(1 for x in infos[split] if x["lyrics_hash"] in exact)
        leak[split] = row
    val_vs_test = [tli.best_match(set(x["lyric_sh"]))[1] for x in infos["validation"]]
    leak["validation_vs_test_lyrics_max"] = max(val_vs_test)
    rep["leakage_vs_train"] = leak
    if leak["test"]["exact_lyrics_duplicate_of_train"] or leak["validation"]["exact_lyrics_duplicate_of_train"]:
        hard.append("exact lyric duplicates across splits")

    # ------------------------------------------------ tokens
    tok = {}
    for split in SPLITS:
        row = {}
        for key in ("tokens_v1", "tokens_v1clean", "tokens_v2", "prompt_v2", "abc_v2"):
            v = [x[key] for x in infos[split]]
            row[key] = {"median": pct(v, 0.5), "p90": pct(v, 0.9), "p95": pct(v, 0.95), "p99": pct(v, 0.99), "max": max(v),
                        "sum": sum(v)}
        v = [x["tokens_v2"] for x in infos[split]]
        row["v2_exceeding"] = {str(L): sum(1 for t in v if t > L) for L in (4096, 8192, 16384, 32768)}
        row["v2_over_v1_token_ratio"] = sum(x["tokens_v2"] for x in infos[split]) / sum(x["tokens_v1"] for x in infos[split])
        tok[split] = row
    rep["tokens"] = tok

    # ------------------------------------------------ section lengths before/after cleaning
    sec = {}
    for split in SPLITS:
        before = [n for x in infos[split] for n in x["lens_before"]]
        after = [n for x in infos[split] for n in x["lens_after"]]

        def dist(v):
            h = Counter(min(n, 33) for n in v)
            return {"n": len(v), "mean": sum(v) / len(v), "median": pct(v, 0.5), "p05": pct(v, 0.05), "p95": pct(v, 0.95),
                    "le_1_bar": sum(1 for n in v if n <= 1), "le_2_bars": sum(1 for n in v if n <= 2),
                    "gt_32_bars": sum(1 for n in v if n > 32), "multiple_of_4": sum(1 for n in v if n % 4 == 0) / len(v),
                    "hist_capped_33": {str(k): h[k] for k in sorted(h)}}

        sec[split] = {"before": dist(before), "after": dist(after),
                      "songs_with_changed_lengths": sum(1 for x in infos[split] if x["lens_before"] != x["lens_after"])}
    rep["section_lengths"] = sec
    rep["hard_failures"] = hard
    args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in rep if k not in ("tokens", "section_lengths")}, indent=1, ensure_ascii=False)[:6000])
    if hard:
        print("V2_VALIDATION_FAILED", hard)
        sys.exit(1)
    print("V2_VALIDATION_OK")


if __name__ == "__main__":
    main()
