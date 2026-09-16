#!/usr/bin/env python
"""Rule-based candidates for the clean held-out test subset (reports/CLEAN_TEST_SUBSET.md).

  scripts/select_clean_subset.py --v2-dir data/generated/abc_v2_<ts> --output-dir reports/clean_subset

Selection uses only the reference data (pseudo-label consistency checks), never
model output. Writes candidates.json (every song with its failed rules) and
inspect.md (prompt + ABC excerpt of each candidate for manual reading). The
final list after manual inspection is clean_test_song_ids.txt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RULES = {
    "no_pathology_flag": lambda r, f, c: not r["pathologies"],
    "tempo_60_170_bpm": lambda r, f, c: 60 <= f["tempo_bpm"] <= 170,
    "stable_key": lambda r, f, c: (f["melody_in_key_frac"] or 0) >= (0.85 if f["key_agreement"] == "high" else 0.92)
    and (f["chord_root_in_key_frac"] or 0) >= 0.75,
    "regular_bars_le_10pct": lambda r, f, c: f["irregular_bar_frac"] <= 0.10,
    "no_section_lyric_cut": lambda r, f, c: c["cuts_after"] == 0,
    "note_density_2_to_6_per_bar": lambda r, f, c: 2.0 <= f["notes_per_bar"] <= 6.0,
    "aligner_cramming_le_5pct": lambda r, f, c: f["multi_syllable_note_frac"] <= 0.05,
    "wordless_le_10pct": lambda r, f, c: f["wordless_note_frac"] <= 0.10,
    "chord_tone_ge_55pct": lambda r, f, c: (f["chord_tone_frac"] or 0) >= 0.55,
    "no_1bar_section": lambda r, f, c: f["tiny_sections"] == 0,
    "at_least_4_sections": lambda r, f, c: f["sections"] >= 4,
    "pitch_range_le_28": lambda r, f, c: f["pitch_range"] <= 28,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.v2_dir / "songs_test.jsonl", encoding="utf-8")]
    fails, cands, per_song = Counter(), [], {}
    for r in rows:
        failed = [name for name, fn in RULES.items() if not fn(r, r["flags"], r["cleaning"])]
        per_song[r["song_id"]] = failed
        for x in failed:
            fails[x] += 1
        if not failed:
            cands.append(r)
    (args.output_dir / "candidates.json").write_text(json.dumps(
        {"rules": list(RULES), "n_test": len(rows), "n_candidates": len(cands), "rule_failures": dict(fails),
         "candidates": [r["song_id"] for r in cands], "failed_rules_per_song": per_song}, indent=1), encoding="utf-8")
    out = ["# Clean-subset candidates (manual inspection sheet; contains lyrics, not committed)\n"]
    for r in cands:
        f = r["flags"]
        out.append(f"## {r['song_id']}\n")
        out.append(f"tempo {f['tempo_bpm']} key {r['spec']['key']} ({f['key_agreement']}), in-key {f['melody_in_key_frac']}, "
                   f"chord-tone {f['chord_tone_frac']}, notes/bar {f['notes_per_bar']}, range {f['pitch_range']}, "
                   f"sections {[(s['label'], s['bars']) for s in r['spec']['sections']]}\n")
        out.append("```\n" + "\n".join(r["abc_v1clean"].split("\n")[:40]) + "\n```\n")
    (args.output_dir / "inspect.md").write_text("\n".join(out), encoding="utf-8")
    print(len(cands), dict(fails))


if __name__ == "__main__":
    main()
