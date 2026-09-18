#!/usr/bin/env python
"""Pick one of n sampled generations per song, using only what the prompt already gives.

  scripts/select_best_of_n.py --generations <eval>/generations --out experiments/bon_<ts>/lex --rule lex
  scripts/generate_eval.py --aggregate-only --seeds 1 --read-generations experiments/bon_<ts>/lex/generations ...

Sampling at T=1.0 gives the best structure and the most corpus-like melodies but
loses strict validity against T=0.8 (0.729 vs 0.844), and occasionally writes a
whole song out of key. Drawing n samples and keeping the best one is a decoding
fix for that, and it needs no training.

**The selector may only use signals available at inference time.** Everything it
reads is computed from the prompt and the generation: the requested section plan,
the requested lyrics, the requested key, and the generation's own syntax. It
never looks at the reference lead sheet, so a rule that wins here is one that
would work on a new song with no ground truth. ``SAFE_SIGNALS`` is the whitelist;
anything outside it is rejected rather than silently used.

Rules are compared on the validation subset and only the winner is applied to
the test set, so the choice of rule is not fitted to the test songs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

# Metrics computed from prompt + generation alone. Anything reference-derived
# (e.g. melody_distance_to_reference) is deliberately absent.
SAFE_SIGNALS = {
    "strict_valid", "parse_ok", "early_eos",
    "section_plan_exact", "section_bars_exact_frac", "section_count_match",
    "section_label_seq_match", "counter_plan_frac", "counter_self_consistent_frac",
    "lyric_recall", "lyric_precision", "lyric_alignment_valid",
    "melody_in_key_frac", "chord_root_in_key_frac", "chord_all_tones_in_key_frac",
    "chord_tone_frac", "cram_syllable_frac", "multi_syllable_note_frac",
    "key_match", "meter_match", "tempo_match", "bar_duration_ok_frac",
}


def signal(row: dict, name: str, default: float = 0.0) -> float:
    if name not in SAFE_SIGNALS:
        raise SystemExit(f"{name} is not an inference-time signal; add it to SAFE_SIGNALS only if it is")
    if name == "strict_valid":
        return float(row.get("strict_ok", False))
    if name == "parse_ok":
        return float(row.get("parse_ok", False))
    v = (row.get("metrics") or {}).get(name, default)
    return float(v) if isinstance(v, (int, float)) else default


def in_key(row: dict) -> float:
    return 0.5 * (signal(row, "melody_in_key_frac") + signal(row, "chord_root_in_key_frac"))


def rank_key(row: dict, rule: str, seed: int):
    """Sort key; the largest wins. Ties fall back to the lowest seed."""
    if rule == "seed0":
        return (-seed,)
    if rule == "valid":
        # the first strict-valid sample, else the first sample
        return (signal(row, "strict_valid"), -seed)
    if rule == "lex":
        # validity, then the requested plan, then lyrics and harmony, then cramming
        return (signal(row, "strict_valid"),
                signal(row, "section_plan_exact"),
                round(signal(row, "lyric_recall"), 3),
                round(in_key(row), 3),
                -round(signal(row, "cram_syllable_frac"), 3),
                -seed)
    if rule == "sum":
        return (2.0 * signal(row, "strict_valid")
                + 1.5 * signal(row, "section_plan_exact")
                + 1.5 * signal(row, "lyric_recall")
                + 1.0 * in_key(row)
                + 0.5 * signal(row, "chord_tone_frac")
                - 1.0 * signal(row, "cram_syllable_frac")
                - 1.0 * signal(row, "early_eos"),
                -seed)
    raise SystemExit(f"unknown rule {rule}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, required=True, help="dir of <song>_s<k>.json")
    ap.add_argument("--out", type=Path, required=True, help="output dir; chosen rows land in <out>/generations")
    ap.add_argument("--rule", default="lex", choices=["seed0", "valid", "lex", "sum"])
    args = ap.parse_args()

    by_song: dict[str, list[tuple[int, Path]]] = {}
    for p in sorted(args.generations.glob("*_s*.json")):
        song, _, seed = p.stem.rpartition("_s")
        if seed.isdigit():
            by_song.setdefault(song, []).append((int(seed), p))
    if not by_song:
        raise SystemExit(f"no <song>_s<k>.json under {args.generations}")

    out_gen = args.out / "generations"
    out_gen.mkdir(parents=True, exist_ok=True)
    chosen_seeds, records, n_with_valid = Counter(), [], 0
    for song, entries in sorted(by_song.items()):
        rows = [(seed, json.loads(p.read_text(encoding="utf-8")), p) for seed, p in sorted(entries)]
        best_seed, best_row, best_path = max(rows, key=lambda t: rank_key(t[1], args.rule, t[0]))
        shutil.copyfile(best_path, out_gen / f"{song}_s0.json")
        chosen_seeds[best_seed] += 1
        n_with_valid += any(r.get("strict_ok") for _s, r, _p in rows)
        records.append({
            "song_id": song, "n_samples": len(rows), "chosen_seed": best_seed,
            "chosen": {k: signal(best_row, k) for k in
                       ("strict_valid", "section_plan_exact", "lyric_recall", "cram_syllable_frac")},
            "in_key": round(in_key(best_row), 4),
            "strict_valid_per_seed": [bool(r.get("strict_ok")) for _s, r, _p in rows],
        })

    n = len(records)
    summary = {
        "rule": args.rule,
        "source_generations": str(args.generations),
        "songs": n,
        "samples_per_song": records[0]["n_samples"] if records else 0,
        "chosen_seed_histogram": dict(sorted(chosen_seeds.items())),
        "songs_with_at_least_one_strict_valid": n_with_valid / max(n, 1),
        "chosen_strict_valid": sum(r["chosen"]["strict_valid"] for r in records) / max(n, 1),
        "chosen_structure_exact": sum(r["chosen"]["section_plan_exact"] for r in records) / max(n, 1),
        "chosen_lyric_recall": sum(r["chosen"]["lyric_recall"] for r in records) / max(n, 1),
        "chosen_cram": sum(r["chosen"]["cram_syllable_frac"] for r in records) / max(n, 1),
        "safe_signals": sorted(SAFE_SIGNALS),
        "per_song": records,
    }
    (args.out / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rule={args.rule} songs={n} samples/song={summary['samples_per_song']}")
    print(f"  any sample strict-valid: {summary['songs_with_at_least_one_strict_valid']:.3f}"
          f"   chosen strict-valid: {summary['chosen_strict_valid']:.3f}")
    print(f"  chosen structure {summary['chosen_structure_exact']:.3f}"
          f"  lyric recall {summary['chosen_lyric_recall']:.3f}  cram {summary['chosen_cram']:.3f}")
    print(f"  seeds chosen: {summary['chosen_seed_histogram']}")
    print(f"SELECT_DONE -> {out_gen}")


if __name__ == "__main__":
    main()
