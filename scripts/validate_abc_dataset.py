#!/usr/bin/env python
"""Strict checks on a built dataset. Re-derives everything from the written files.

Exit status is non-zero if a hard invariant fails (split overlap, round-trip
failure, CPT/SFT song-set mismatch), so a Slurm pipeline stops before training.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.canonical import Song, comparable  # noqa: E402
from qwen_abc.prompt import spec_syllables, spec_to_prompt  # noqa: E402
from qwen_abc.splits import SPLITS, ShingleIndex, lyric_shingles, melody_shingles  # noqa: E402
from qwen_abc.theory import parse_chord_symbol  # noqa: E402

CJK = re.compile(r"^[㐀-鿿豈-﫿]+$")
LATIN = re.compile(r"[a-z]")


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(int(q * len(values)), len(values) - 1)]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--context-limits", default="2048,4096,6144,8192")
    args = ap.parse_args()
    d = args.data_dir
    build = json.load(open(d / "build_report.json", encoding="utf-8"))
    hard_fail = []
    rep = {"data_dir": str(d)}

    songs, sft, cpt = {}, {}, {}
    for split in SPLITS:
        songs[split] = read_jsonl(d / f"songs_{split}.jsonl")
        sft[split] = read_jsonl(d / f"sft_{split}.jsonl")
        cpt[split] = read_jsonl(d / f"cpt_{split}.jsonl")
    rep["counts"] = {
        split: {"songs": len(songs[split]), "sft_examples": len(sft[split]), "cpt_examples": len(cpt[split])}
        for split in SPLITS
    }

    # ------------------------------------------------ split invariants
    ids = {split: {r["song_id"] for r in songs[split]} for split in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        inter = ids[a] & ids[b]
        if inter:
            hard_fail.append(f"song overlap {a}/{b}: {len(inter)}")
    for split in SPLITS:
        if {r["song_id"] for r in sft[split]} != ids[split] or {r["song_id"] for r in cpt[split]} != ids[split]:
            hard_fail.append(f"{split}: CPT/SFT/song sets differ")
    exact = {}
    for kind, getter in (("abc_text", lambda r: r["completion"]), ("prompt", lambda r: r["prompt"]),
                         ("lyrics_text", lambda r: "\n".join(l for s in r["_spec"]["sections"] for l in s["lines"]))):
        seen = {}
        for split in SPLITS:
            for r, s in zip(sft[split], songs[split]):
                r["_spec"] = s["spec"]
                h = hashlib.sha256(getter(r).encode("utf-8")).hexdigest()
                seen.setdefault(h, set()).add(split)
        exact[kind] = sum(1 for v in seen.values() if len(v) > 1)
    rep["exact_duplicates_across_splits"] = exact
    if exact["abc_text"]:
        hard_fail.append(f"identical ABC across splits: {exact['abc_text']}")

    # near-duplicate residue after the build's filter
    canon = {split: [Song.from_json(r["song"]) for r in songs[split]] for split in SPLITS}
    li, mi = ShingleIndex(), ShingleIndex()
    for s in canon["train"]:
        li.add(s.song_id, lyric_shingles(s))
        mi.add(s.song_id, melody_shingles(s))
    resid = {}
    for split in ("validation", "test"):
        lyr = [li.best_match(lyric_shingles(s))[1] for s in canon[split]]
        mel = [mi.best_match(melody_shingles(s))[1] for s in canon[split]]
        resid[split] = {"lyrics_containment_max": max(lyr, default=0), "lyrics_p95": pct(lyr, 0.95),
                        "melody_containment_max": max(mel, default=0), "melody_p95": pct(mel, 0.95)}
    rep["held_out_vs_train_near_duplicate_residue"] = resid

    # ------------------------------------------------ parse + round trip
    parse_ok = strict_ok = equal = prompt_ok = lyric_match = 0
    malformed = []
    total = 0
    for split in SPLITS:
        for r, s, song in zip(sft[split], songs[split], canon[split]):
            total += 1
            res = parse_abc(r["completion"], r["song_id"])
            parse_ok += res.ok
            strict_ok += res.strict_ok
            if res.ok and comparable(res.song) == comparable(song):
                equal += 1
            else:
                malformed.append(r["song_id"])
            if res.errors:
                malformed.append(r["song_id"])
            prompt_ok += spec_to_prompt(s["spec"]) == r["prompt"]
            abc_sylls = [x for n in song.notes if n.lyric for x in n.lyric]
            lyric_match += spec_syllables(s["spec"]) == abc_sylls
    rep["abc"] = {
        "examples": total, "parse_success": parse_ok, "strict_valid": strict_ok, "roundtrip_equal": equal,
        "prompt_rebuild_equal": prompt_ok, "prompt_lyrics_equal_abc_lyrics": lyric_match,
        "malformed_song_ids": sorted(set(malformed))[:50],
    }
    if equal != total:
        hard_fail.append(f"round trip failed on {total - equal} examples")
    if lyric_match != total:
        hard_fail.append(f"prompt lyrics differ from ABC lyrics on {total - lyric_match} examples")
    conv = build["conversion"]
    rep["midi_roundtrip"] = {"equal": conv["midi_roundtrip_equal"], "attempted": conv["attempted"]}

    # ------------------------------------------------ tokens
    limits = [int(x) for x in args.context_limits.split(",")]
    tok = {}
    excluded_long = [json.loads(l) for l in open(d / "exclusions.jsonl", encoding="utf-8")]
    excluded_long = [e for e in excluded_long if e["reason"] == "exceeds_max_seq_len"]
    for split in SPLITS:
        sft_len = [r["prompt_tokens"] + r["completion_tokens"] + 1 for r in sft[split]]
        tok[split] = {
            "sft_total": {q: pct(sft_len, q) for q in (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)},
            "prompt_median": statistics.median(r["prompt_tokens"] for r in sft[split]) if sft[split] else None,
            "completion_median": statistics.median(r["completion_tokens"] for r in sft[split]) if sft[split] else None,
            "sum_sft_tokens": sum(sft_len),
            "sum_cpt_tokens": sum(r["tokens"] + 1 for r in cpt[split]),
            "exceeding": {str(L): round(sum(1 for x in sft_len if x > L) / max(len(sft_len), 1), 4) for L in limits},
            "excluded_by_build_gate": sum(1 for e in excluded_long if e["split"] == split),
        }
    rep["tokens"] = tok

    # ------------------------------------------------ content coverage
    cov = Counter()
    sections = Counter()
    songs_with_latin = 0
    for split in SPLITS:
        for r, song in zip(songs[split], canon[split]):
            st = r["stats"]
            cov["source_syllables"] += st.get("syllables", 0)
            cov["syllables_dropped"] += st.get("unmatched_syllable_dropped", 0)
            cov["syllables_joined"] += st.get("unmatched_syllable_joined", 0)
            cov["source_notes_dropped"] += st.get("note_same_onset_dropped", 0) + st.get("note_after_end_dropped", 0)
            cov["notes_truncated"] += st.get("note_overlap_truncated", 0)
            cov["bars"] += len(song.bar_beats)
            cov["irregular_bars"] += sum(1 for b in song.bar_beats if b != song.meter_num)
            notes = song.notes
            cov["notes"] += len(notes)
            cov["attack_notes"] += sum(1 for n in notes if n.lyric)
            cov["melisma_notes"] += sum(1 for n in notes if n.melisma)
            cov["wordless_notes"] += sum(1 for n in notes if not n.lyric and not n.melisma)
            cov["multi_syllable_notes"] += sum(1 for n in notes if n.lyric and len(n.lyric) > 1)
            sylls = [x for n in notes if n.lyric for x in n.lyric]
            cov["abc_syllables"] += len(sylls)
            cov["cjk_syllables"] += sum(1 for x in sylls if CJK.match(x))
            cov["latin_syllables"] += sum(1 for x in sylls if LATIN.search(x))
            songs_with_latin += any(LATIN.search(x) for x in sylls)
            # syllables carried by 2+ notes
            run = 0
            for n in notes:
                if n.lyric:
                    if run >= 2:
                        cov["melismatic_syllables"] += 1
                    run = 1
                elif n.melisma:
                    run += 1
                else:
                    if run >= 2:
                        cov["melismatic_syllables"] += 1
                    run = 0
            if run >= 2:
                cov["melismatic_syllables"] += 1
            total_ticks = song.total_ticks
            cov["ticks"] += total_ticks
            chords = song.chords
            cov["chord_ticks"] += sum(c.duration for c in chords
                                      if (parse_chord_symbol(c.symbol) or {}).get("root_pc") is not None)
            cov["chords"] += len(chords)
            cov["invalid_chords"] += sum(1 for c in chords if parse_chord_symbol(c.symbol) is None)
            onsets = [c.onset for c in chords]
            under = 0
            for n in notes:
                i = bisect.bisect_right(onsets, n.onset) - 1
                if i >= 0 and n.onset < chords[i].offset:
                    info = parse_chord_symbol(chords[i].symbol)
                    under += info is not None and info["root_pc"] is not None
            cov["notes_under_chord"] += under
            for sec in song.sections:
                sections[sec.label] += sec.num_bars
    n_songs = sum(len(canon[s]) for s in SPLITS)
    rep["coverage"] = {
        "lyric_syllables_kept_frac": round(1 - cov["syllables_dropped"] / max(cov["source_syllables"], 1), 5),
        "lyric_syllables_joined_frac": round(cov["syllables_joined"] / max(cov["source_syllables"], 1), 5),
        "notes_with_syllable_frac": round(cov["attack_notes"] / cov["notes"], 4),
        "melisma_note_frac": round(cov["melisma_notes"] / cov["notes"], 4),
        "wordless_note_frac": round(cov["wordless_notes"] / cov["notes"], 4),
        "melismatic_syllable_frac": round(cov["melismatic_syllables"] / max(cov["attack_notes"], 1), 4),
        "multi_syllable_note_frac": round(cov["multi_syllable_notes"] / cov["notes"], 5),
        "notes_per_syllable": round(cov["notes"] / max(cov["abc_syllables"], 1), 4),
        "chord_time_coverage": round(cov["chord_ticks"] / cov["ticks"], 4),
        "notes_under_chord_frac": round(cov["notes_under_chord"] / cov["notes"], 4),
        "invalid_chords": cov["invalid_chords"],
        "irregular_bar_frac": round(cov["irregular_bars"] / cov["bars"], 4),
        "section_bar_share": {k: round(v / cov["bars"], 4) for k, v in sections.most_common()},
        "cjk_syllable_frac": round(cov["cjk_syllables"] / cov["abc_syllables"], 4),
        "latin_syllable_frac": round(cov["latin_syllables"] / cov["abc_syllables"], 4),
        "songs_with_latin_syllables_frac": round(songs_with_latin / n_songs, 4),
        "raw": dict(cov),
    }
    rep["build_exclusions"] = build["exclusion_reasons"]
    rep["hard_failures"] = hard_fail
    args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False)[:6000])
    if hard_fail:
        print("VALIDATION_FAILED", hard_fail)
        sys.exit(1)
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
