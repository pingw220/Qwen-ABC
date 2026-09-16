#!/usr/bin/env python
"""Build the ABC-v2 dataset (cleaned section boundaries + explicit structural state).

  scripts/build_abc_v2_dataset.py --v1-dir data/generated/abc_v1_<ts> --output-dir data/generated/abc_v2_<ts>

Derived from the ABC-v1 build's canonical songs, never from the corpus again,
so the split, melody, chords, lyrics, bar lengths and tempo/key are identical by
construction; only section boundaries (cleaning) and the text format change.

Per split it writes
  songs_{split}.jsonl        cleaned canonical song, spec, ABC-v2, ABC-v1 of the cleaned song, cleaning decisions, flags
  sft_{split}.jsonl          ABC-v2 prompt -> ABC-v2 completion              (experiment E1)
  sft_v1clean_{split}.jsonl  ABC-v1 prompt -> ABC-v1 completion, cleaned      (cleaning-only ablation)
plus cleaning_log.jsonl (every changed song), split_manifest.json (must equal v1's) and build_report.json.
Deterministic: sorted song ids, no randomness; sha256 of every output is recorded.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc, song_to_abc  # noqa: E402
from qwen_abc.abc_v2 import ABC_V2_VERSION, counter_report, song_to_abc_v2, spec_to_prompt_v2, strip_v2  # noqa: E402
from qwen_abc.canonical import Song, comparable  # noqa: E402
from qwen_abc.cleaning import CLEANING_VERSION, CleanConfig, clean_sections, cut_count, is_pathological, pathology_flags  # noqa: E402
from qwen_abc.prompt import spec_syllables, spec_to_prompt, song_to_spec  # noqa: E402
from qwen_abc.splits import SPLITS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_abc_dataset import midi_roundtrip_ok, sha256_file  # noqa: E402

_TOK = None


def tokenizer():
    """The Qwen3.5 fast tokenizer via `tokenizers` (identical ids to AutoTokenizer, add_special_tokens=False)."""
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer

        root = Path(__file__).resolve().parents[1]
        path = sorted(glob.glob(str(root / ".hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/tokenizer.json")))[0]
        _TOK = Tokenizer.from_file(path)
    return _TOK


def ntok(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def convert(line: str) -> dict:
    row = json.loads(line)
    sid = row["song_id"]
    song = Song.from_json(row["song"])
    assert song_to_abc(song) == row["abc"], f"{sid}: v1 writer no longer reproduces the v1 build"
    cuts_before, pickups_before = cut_count(song)
    flags = pathology_flags(song)
    secs, decisions = clean_sections(song, CleanConfig())
    clean = copy.deepcopy(song)
    clean.sections = secs
    cuts_after, pickups_after = cut_count(clean)
    spec = song_to_spec(clean)
    abc_v1c = song_to_abc(clean)
    abc_v2 = song_to_abc_v2(clean)
    checks = {}
    p2 = parse_abc(abc_v2, sid)
    checks["v2_parse_strict"] = p2.strict_ok
    checks["v2_roundtrip_equal"] = p2.ok and comparable(p2.song) == comparable(clean)
    p1 = parse_abc(abc_v1c, sid)
    checks["v1clean_roundtrip_equal"] = p1.ok and comparable(p1.song) == comparable(clean)
    checks["strip_v2_equals_v1clean"] = strip_v2(abc_v2) == abc_v1c
    a, b = comparable(song), comparable(clean)
    checks["music_unchanged_by_cleaning"] = all(a[k] == b[k] for k in ("meter_num", "tempo_bpm", "key", "bar_beats", "notes", "chords"))
    checks["bars_conserved"] = sum(s.num_bars for s in clean.sections) == len(clean.bar_beats) and \
        all(clean.sections[i].start_bar + clean.sections[i].num_bars == clean.sections[i + 1].start_bar
            for i in range(len(clean.sections) - 1)) and clean.sections[0].start_bar == 0
    checks["labels_unchanged"] = [s.label for s in song.sections] == [s.label for s in clean.sections]
    checks["prompt_lyrics_equal_abc_lyrics"] = spec_syllables(spec) == [x for n in clean.notes if n.lyric for x in n.lyric]
    cr = counter_report(abc_v2, spec)
    checks["counters_correct"] = all(cr[k] == 1.0 for k in cr if k.endswith("_frac") or k.startswith("sections_ending"))
    checks["midi_roundtrip_equal"] = midi_roundtrip_ok(clean)
    prompt_v2 = spec_to_prompt_v2(spec)
    prompt_v1c = spec_to_prompt(spec)
    moved = [d for d in decisions if d.action == "moved"]
    return {
        "song_id": sid, "split": row["split"],
        "song": clean.to_json(), "abc": abc_v2, "abc_v1clean": abc_v1c, "spec": spec,
        "prompt": prompt_v2, "prompt_v1clean": prompt_v1c,
        "stats": row["stats"],
        "cleaning": {
            "version": CLEANING_VERSION,
            "changed": bool(moved),
            "cuts_before": cuts_before, "cuts_after": cuts_after,
            "pickups_before": pickups_before, "pickups_after": pickups_after,
            "sections_before": [(s.label, s.start_bar, s.num_bars) for s in song.sections],
            "sections_after": [(s.label, s.start_bar, s.num_bars) for s in clean.sections],
            "decisions": [{"section": d.index, "old_bar": d.old_bar, "new_bar": d.new_bar, "action": d.action,
                           "crossings": d.crossings} for d in decisions if d.action != "kept_clean"],
        },
        "flags": flags, "pathologies": is_pathological(flags),
        "checks": checks,
        "prompt_tokens": ntok(prompt_v2), "abc_tokens": ntok(abc_v2),
        "prompt_v1clean_tokens": ntok(prompt_v1c), "abc_v1clean_tokens": ntok(abc_v1c),
        "v1_prompt_tokens": row["prompt_tokens"], "v1_abc_tokens": row["abc_tokens"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N songs per split")
    args = ap.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v1_manifest = json.load(open(args.v1_dir / "split_manifest.json", encoding="utf-8"))
    report = {"schema_version": ABC_V2_VERSION, "cleaning_version": CLEANING_VERSION, "clean_config": vars(CleanConfig()),
              "v1_dir": str(args.v1_dir), "v1_build_sha256": json.load(open(args.v1_dir / "build_report.json"))["files_sha256"],
              "args": {k: str(v) for k, v in vars(args).items()}}
    files, failures, manifest = {}, Counter(), {}
    log_rows = []
    agg = {}
    with Pool(args.workers) as pool:
        for split in SPLITS:
            with open(args.v1_dir / f"songs_{split}.jsonl", encoding="utf-8") as fh:
                lines = [l for l in fh if l.strip()]
            if args.limit:
                lines = lines[: args.limit]
            ids = []
            c = Counter()
            with open(out / f"songs_{split}.jsonl", "w", encoding="utf-8") as f_song, \
                    open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f_sft, \
                    open(out / f"sft_v1clean_{split}.jsonl", "w", encoding="utf-8") as f_v1c:
                for r in pool.imap(convert, lines, chunksize=16):
                    ids.append(r["song_id"])
                    for k, ok in r["checks"].items():
                        c[f"check_{k}"] += int(ok)
                        if not ok:
                            failures[k] += 1
                    cl = r["cleaning"]
                    c["songs"] += 1
                    c["songs_changed"] += cl["changed"]
                    c["cuts_before"] += cl["cuts_before"]
                    c["cuts_after"] += cl["cuts_after"]
                    c["songs_with_cut_before"] += cl["cuts_before"] > 0
                    c["songs_with_cut_after"] += cl["cuts_after"] > 0
                    c["pickups_before"] += cl["pickups_before"]
                    c["pickups_after"] += cl["pickups_after"]
                    for d in cl["decisions"]:
                        c[f"action_{d['action']}"] += 1
                        if d["action"] == "moved":
                            c[f"shift_{d['new_bar'] - d['old_bar']:+d}"] += 1
                    for p in r["pathologies"]:
                        c[f"pathology_{p}"] += 1
                    c["songs_with_pathology"] += bool(r["pathologies"])
                    for tag, key in (("v1", "v1"), ("v1clean", "v1clean"), ("v2", "v2")):
                        pass
                    c["tokens_v1_sft"] += r["v1_prompt_tokens"] + r["v1_abc_tokens"] + 1
                    c["tokens_v1clean_sft"] += r["prompt_v1clean_tokens"] + r["abc_v1clean_tokens"] + 1
                    c["tokens_v2_sft"] += r["prompt_tokens"] + r["abc_tokens"] + 1
                    c["tokens_v1_abc"] += r["v1_abc_tokens"]
                    c["tokens_v2_abc"] += r["abc_tokens"]
                    if cl["changed"]:
                        log_rows.append({"song_id": r["song_id"], "split": split, **{k: cl[k] for k in (
                            "cuts_before", "cuts_after", "sections_before", "sections_after", "decisions")}})
                    f_song.write(json.dumps({k: r[k] for k in (
                        "song_id", "split", "song", "abc", "abc_v1clean", "spec", "stats", "cleaning", "flags", "pathologies",
                        "prompt_tokens", "abc_tokens", "prompt_v1clean_tokens", "abc_v1clean_tokens",
                        "v1_prompt_tokens", "v1_abc_tokens")}, ensure_ascii=False) + "\n")
                    f_sft.write(json.dumps({"id": f"sft:{r['song_id']}", "song_id": r["song_id"], "prompt": r["prompt"],
                                            "completion": r["abc"], "prompt_tokens": r["prompt_tokens"],
                                            "completion_tokens": r["abc_tokens"]}, ensure_ascii=False) + "\n")
                    f_v1c.write(json.dumps({"id": f"sft:{r['song_id']}", "song_id": r["song_id"], "prompt": r["prompt_v1clean"],
                                            "completion": r["abc_v1clean"], "prompt_tokens": r["prompt_v1clean_tokens"],
                                            "completion_tokens": r["abc_v1clean_tokens"]}, ensure_ascii=False) + "\n")
            manifest[split] = ids
            agg[split] = dict(c)
            for name in (f"songs_{split}.jsonl", f"sft_{split}.jsonl", f"sft_v1clean_{split}.jsonl"):
                files[name] = sha256_file(out / name)
            print(f"{split}: {c['songs']} songs, changed {c['songs_changed']}, cuts {c['cuts_before']} -> {c['cuts_after']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    with open(out / "cleaning_log.jsonl", "w", encoding="utf-8") as fh:
        for row in log_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    files["cleaning_log.jsonl"] = sha256_file(out / "cleaning_log.jsonl")
    files["split_manifest.json"] = sha256_file(out / "split_manifest.json")
    report["split_equals_v1"] = args.limit > 0 or manifest == v1_manifest
    report["per_split"] = agg
    report["check_failures"] = dict(failures)
    report["files_sha256"] = files
    report["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"split_equals_v1": report["split_equals_v1"], "check_failures": report["check_failures"]}, indent=1))
    if failures or not report["split_equals_v1"]:
        print("BUILD_V2_FAILED")
        sys.exit(1)
    print("BUILD_V2_OK")


if __name__ == "__main__":
    main()
