#!/usr/bin/env python
"""Round-2 comparison tables with paired bootstrap CIs.

  scripts/compare_r2.py --run E0=experiments/r2eval_E0_<ts>/v1test_T0.8 --run E1=... [--baseline E0]
      [--subset reports/clean_test_song_ids.txt] [--reference-data data/generated/abc_v2_<ts> --split test]
      [--groups structure,lyrics,music,repetition,coherence,counters] [--json OUT.json] > table.md

Each --run points at an evaluation directory with generations/*_s0.json rows (any
format written by scripts/generate_eval.py). Songs are paired by song_id; only
songs present in every run (and in --subset) are used. Rows whose ABC did not
parse count as failures for rate metrics and are excluded from content metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = {
    "validity": [
        ("strict_valid", "strict-valid ABC"), ("parse_ok", "parse success"), ("hit_eos", "stopped with EOS"),
        ("lyric_alignment_valid", "no lyric-alignment error"), ("bar_duration_ok_frac", "bars with correct duration"),
    ],
    "structure": [
        ("section_plan_exact", "exact full structure (labels + bars per section)"),
        ("section_label_seq_match", "exact section label sequence"),
        ("section_bars_seq_match", "exact bars-per-section sequence"),
        ("section_bars_exact_frac", "sections with exact label+bars (per section)"),
        ("bar_count_match", "total bar count = request"), ("abs_total_bar_error", "|total bars - requested|"),
        ("section_completion_ratio", "section completion ratio"), ("early_eos", "early EOS (EOS before last section)"),
        ("fewer_sections_than_requested", "fewer sections than requested"),
        ("late_section_present", "late-section completion (last third, label present)"),
        ("late_section_exact", "late sections exact (label+bars)"),
        ("bridge_completion", "bridge completion"), ("outro_completion", "outro completion"),
    ],
    "lyrics": [
        ("lyric_recall", "lyric recall"), ("lyric_precision", "lyric precision"), ("lyric_exact", "lyrics exactly as requested"),
        ("section_lyric_recall", "section-local lyric recall"), ("late_lyric_recall", "late-song lyric recall (last third)"),
        ("late_section_lyric_recall", "lyric recall in late sections"),
        ("multi_syllable_note_frac", "notes with several syllables (cramming)"), ("cram_syllable_frac", "syllables sung crammed"),
        ("melisma_note_frac", "melisma note fraction"), ("notes_per_syllable", "notes per syllable"),
    ],
    "music": [
        ("pitch_range", "pitch range (semitones)"), ("mean_abs_interval", "mean |interval|"),
        ("int_repeat_frac", "repeated-pitch intervals"), ("int_step_frac", "steps (1-2)"), ("int_skip_frac", "skips (3-4)"),
        ("int_leap_frac", "leaps (5-7)"), ("large_leap_frac", "leaps > 7"), ("notes_per_bar", "notes per bar"),
        ("mean_duration_beats", "mean duration (beats)"), ("dur_16th_frac", "16th-note durations"),
        ("offbeat_16th_frac", "16th off-beat onsets"), ("offbeat_8th_frac", "8th off-beat onsets"),
        ("syncopation_frac", "syncopation proxy"), ("chords_per_bar", "chords per bar"),
        ("chord_tone_frac", "chord-tone agreement"),
        ("melody_in_key_frac", "melody in declared key (duration)"),
        ("chord_root_in_key_frac", "chord roots in key"),
        ("chord_all_tones_in_key_frac", "chords entirely in key"),
    ],
    "repetition": [
        ("distinct_bar_frac", "distinct non-empty bars"), ("repeated_bar_frac", "bars repeating an earlier bar"),
        ("max_identical_consecutive_bars", "longest identical-bar run"),
        ("consecutive_repeat_bar_frac", "bar = previous bar"), ("pitch_4gram_repeat_frac", "pitch 4-gram repetition"),
        ("interval_4gram_repeat_frac", "interval 4-gram repetition"),
        ("consecutive_motif_repeat_frac", "bar motif = previous bar motif (transposition-inv.)"),
    ],
    "coherence": [
        ("verse_verse_motif_sim", "verse-verse motif similarity"), ("chorus_chorus_motif_sim", "chorus-chorus motif similarity"),
        ("chorus_motif_preservation", "chorus motif preservation (4-gram)"), ("cross_label_motif_sim", "across-label section similarity"),
        ("within_section_adjacent_bar_motif_sim", "within-section adjacent-bar similarity"),
        ("late_minus_early_pitch_range", "late − early pitch range"),
        ("late_minus_early_rhythm_diversity", "late − early rhythm diversity"),
    ],
    "counters": [
        ("counter_present_frac", "bars with a counter"), ("counter_self_consistent_frac", "counter = bars actually left"),
        ("counter_plan_frac", "counter = plan"), ("header_plan_frac", "section header = plan"),
        ("sections_ending_on_counter_1", "sections ending on [r:1]"),
    ],
}
RATE_KEYS = {"strict_valid", "parse_ok", "lyric_recall", "lyric_precision", "lyric_exact", "section_lyric_recall",
             "bar_count_match", "section_plan_exact", "section_label_seq_match", "section_bars_seq_match",
             "section_bars_exact_frac", "lyric_alignment_valid", "section_completion_ratio", "late_section_present",
             "late_section_exact", "late_lyric_recall", "late_section_lyric_recall"}


def load_rows(eval_dir: Path):
    rows = {}
    d = eval_dir / "generations"
    for p in sorted(d.glob("*_s0.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows[r["song_id"]] = r
    return rows


def value(row, key):
    if key == "strict_valid":
        return float(row["strict_ok"])
    if key == "parse_ok":
        return float(row["parse_ok"])
    if key == "hit_eos":
        return float(bool(row.get("hit_eos")))
    m = row.get("metrics")
    if m is None:
        return 0.0 if key in RATE_KEYS else None
    v = m.get(key)
    return None if v is None or v != v else float(v)


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def reference_metrics(data_dir: Path, split: str, ids, fmt_v2: bool):
    from qwen_abc.abc import parse_abc
    from qwen_abc.abc_v2 import counter_report
    from qwen_abc.canonical import Song
    from qwen_abc.metrics import song_metrics

    out = {}
    with open(data_dir / f"songs_{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            sid = line[len('{"song_id": "'):].split('"', 1)[0]
            if sid not in ids:
                continue
            r = json.loads(line)
            res = parse_abc(r["abc"], sid)
            m = song_metrics(Song.from_json(r["song"]), r["spec"], res)
            if fmt_v2:
                m.update(counter_report(r["abc"], r["spec"]))
            out[sid] = {"strict_ok": res.strict_ok, "parse_ok": True, "hit_eos": True, "metrics": m}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=eval_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--subset", type=Path, default=None)
    ap.add_argument("--reference-data", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default="validity,structure,lyrics,music,repetition,coherence")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.run)
    names = list(runs)
    base = args.baseline or names[0]
    rows = {n: load_rows(Path(p)) for n, p in runs.items()}
    common = set.intersection(*(set(r) for r in rows.values()))
    if args.subset:
        wanted = {l.strip() for l in open(args.subset, encoding="utf-8") if l.strip()}
        common &= wanted
    common = sorted(common)
    ref = reference_metrics(args.reference_data, args.split, set(common), "counters" in args.groups) if args.reference_data else {}
    if args.title:
        print(f"### {args.title}\n")
    print(f"Songs in every run{' and in the subset' if args.subset else ''}: **{len(common)}**. "
          f"Paired differences vs **{base}**; `*` = 95% paired bootstrap CI excludes 0.\n")
    dump = {"songs": common, "runs": runs, "metrics": {}}
    for group in args.groups.split(","):
        print(f"**{group}**\n")
        head = "| metric | ref | " + " | ".join(names) + "".join(f" | {n} − {base}" for n in names if n != base) + " |"
        print(head)
        print("|" + "---|" * (2 + len(names) + len(names) - 1))
        for key, label in GROUPS[group]:
            per = {n: [value(rows[n][s], key) for s in common] for n in names}
            if all(all(v is None for v in vals) for vals in per.values()):
                continue
            rv = [value(ref[s], key) for s in common if s in ref]
            rv = [x for x in rv if x is not None]
            cells = [f"{sum(rv) / len(rv):.3f}" if rv else "–"]
            dump["metrics"][key] = {}
            for n in names:
                v = [x for x in per[n] if x is not None]
                cells.append(f"{sum(v) / len(v):.3f}" if v else "–")
                dump["metrics"][key][n] = {"mean": sum(v) / len(v) if v else None, "n": len(v)}
            for n in names:
                if n == base:
                    continue
                pairs = [(a, b) for a, b in zip(per[n], per[base]) if a is not None and b is not None]
                if len(pairs) < 5:
                    cells.append("–")
                    continue
                d = [a - b for a, b in pairs]
                lo, hi = boot(d)
                star = " *" if lo > 0 or hi < 0 else ""
                cells.append(f"{sum(d) / len(d):+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
                dump["metrics"][key][f"{n}-{base}"] = {"mean": sum(d) / len(d), "ci": [lo, hi], "n": len(d)}
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()
    if args.json:
        args.json.write_text(json.dumps(dump, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
