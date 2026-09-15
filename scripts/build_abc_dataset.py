#!/usr/bin/env python
"""Build the Qwen-ABC dataset from SheetSage-Pro lead sheets.

Order of operations (the split is fixed BEFORE any example is constructed):

1. read the corpus manifest and the inherited MIDI-LLM v4 song split;
2. convert every song -> canonical -> ABC, and prove the round trip
   (ABC parse == canonical, MIDI melody == canonical) per song;
3. remove held-out songs that near-duplicate train (lyrics or melody);
4. apply the token-length gate (same gate for every split);
5. only then write CPT and SFT examples, per split, from the kept songs.

Deterministic: songs are processed and written in sorted song_id order and
no randomness is used. The report records sha256 of every output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc import SCHEMA_VERSION  # noqa: E402
from qwen_abc.abc import parse_abc, song_to_abc  # noqa: E402
from qwen_abc.canonical import TICKS_PER_BEAT, Song, comparable  # noqa: E402
from qwen_abc.prompt import spec_to_prompt, song_to_spec  # noqa: E402
from qwen_abc.source import song_from_leadsheet  # noqa: E402
from qwen_abc.splits import (  # noqa: E402
    SPLITS,
    containment_profile,
    load_inherited_split,
    lyric_shingles,
    lyrics_exact_hash,
    melody_shingles,
    resolve_leakage,
)

MUSIC_ACC = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc")
DEFAULT_VIEWS = MUSIC_ACC / "sheetsage-pro/dataset/processed/sheetsage_zh_training_views_v2_lvcr"
DEFAULT_SPLIT = MUSIC_ACC / "MIDI-LLM-phoneme-lyric-v1/manifests/phoneme_leadsheet_section_v4"


def midi_roundtrip_ok(song: Song) -> bool:
    import mido
    from qwen_abc.midi import MIDI_PER_TICK, song_to_midi

    tmp = f"/dev/shm/qwen_abc_{os.getpid()}.mid"
    song_to_midi(song, tmp)
    try:
        mid = mido.MidiFile(tmp, charset="utf-8")
    finally:
        os.unlink(tmp)
    for track in mid.tracks:
        if not any(m.type == "track_name" and m.name == "melody" for m in track):
            continue
        t, active, got = 0, {}, []
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = t
            elif msg.type in ("note_off", "note_on") and msg.note in active:
                s = active.pop(msg.note)
                got.append((s // MIDI_PER_TICK, (t - s) // MIDI_PER_TICK, msg.note))
        want = [(n.onset, n.duration, n.pitch) for n in song.notes]
        return sorted(got) == sorted(want)
    return False


def convert_one(args):
    row, views_root = args
    sid = row["song_id"]
    path = views_root / "phonemes" / sid / "leadsheet.json"
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        song, stats = song_from_leadsheet(data, sid)
        song.meta.update({"quality_tier": row.get("quality_tier")})
        abc = song_to_abc(song)
        parsed = parse_abc(abc, sid)
        rt_equal = parsed.ok and comparable(parsed.song) == comparable(song)
        midi_ok = midi_roundtrip_ok(song)
        spec = song_to_spec(song)
        return {
            "song_id": sid,
            "ok": True,
            "song": song.to_json(),
            "abc": abc,
            "spec": spec,
            "prompt": spec_to_prompt(spec),
            "stats": stats,
            "abc_parse_errors": dict(parsed.errors),
            "abc_roundtrip_equal": rt_equal,
            "midi_roundtrip_equal": midi_ok,
            "lyric_sh": sorted(lyric_shingles(song)),
            "melody_sh": sorted(melody_shingles(song)),
            "lyrics_hash": lyrics_exact_hash(song),
            "num_notes": len(song.notes),
            "num_attacks": sum(1 for n in song.notes if n.lyric),
            "total_beats": song.total_ticks // TICKS_PER_BEAT,
        }
    except Exception as exc:  # recorded, never silently dropped
        return {"song_id": sid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--max-seq-len", type=int, default=8192, help="gate on SFT prompt+completion tokens")
    ap.add_argument("--lyric-dup-threshold", type=float, default=0.30)
    ap.add_argument("--melody-dup-threshold", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N songs of the manifest (sorted)")
    args = ap.parse_args()

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows = [json.loads(l) for l in open(args.views_root / "manifest.jsonl", encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["song_id"])
    inherited = load_inherited_split(args.split_dir)
    report = {
        "schema_version": SCHEMA_VERSION,
        "views_root": str(args.views_root),
        "split_dir": str(args.split_dir),
        "args": {k: str(v) for k, v in vars(args).items()},
        "corpus_songs": len(rows),
        "inherited_split_songs": dict(Counter(inherited.values())),
    }
    exclusions = []
    candidates = []
    for r in rows:
        if r["song_id"] not in inherited:
            exclusions.append({"song_id": r["song_id"], "split": None, "reason": "not_in_inherited_split"})
        else:
            candidates.append(r)
    if args.limit:
        candidates = candidates[: args.limit]

    results = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(convert_one, [(r, args.views_root) for r in candidates], chunksize=8)):
            results[res["song_id"]] = res
            if (i + 1) % 500 == 0:
                print(f"converted {i + 1}/{len(candidates)} in {time.time() - started:.0f}s", flush=True)

    agg_stats: Counter = Counter()
    parse_err: Counter = Counter()
    for sid in sorted(results):
        res = results[sid]
        split = inherited[sid]
        if not res["ok"]:
            exclusions.append({"song_id": sid, "split": split, "reason": "conversion_failed", "detail": res["error"]})
            continue
        agg_stats.update(res["stats"])
        parse_err.update(res["abc_parse_errors"])
        if not res["abc_roundtrip_equal"]:
            exclusions.append({"song_id": sid, "split": split, "reason": "abc_roundtrip_mismatch"})
        elif not res["midi_roundtrip_equal"]:
            exclusions.append({"song_id": sid, "split": split, "reason": "midi_roundtrip_mismatch"})
        elif res["num_attacks"] == 0:
            exclusions.append({"song_id": sid, "split": split, "reason": "no_lyrics"})
    excluded = {e["song_id"] for e in exclusions}
    report["conversion"] = {
        "attempted": len(candidates),
        "failed": sum(1 for r in results.values() if not r["ok"]),
        "abc_roundtrip_equal": sum(1 for r in results.values() if r.get("abc_roundtrip_equal")),
        "midi_roundtrip_equal": sum(1 for r in results.values() if r.get("midi_roundtrip_equal")),
        "aggregate_stats": dict(agg_stats),
        "writer_output_parse_errors": dict(parse_err),
    }

    # ---------------------------------------------------------- leakage
    pool_rows = {
        sid: {"split": inherited[sid], "lyric_sh": set(r["lyric_sh"]), "melody_sh": set(r["melody_sh"]),
              "lyrics_hash": r["lyrics_hash"]}
        for sid, r in results.items() if sid not in excluded
    }
    profiles = {}
    for split in ("validation", "test"):
        prof = containment_profile(pool_rows, split, "train")
        profiles[split] = {
            kind: {f"ge_{t}": sum(1 for x in vals if x >= t) for t in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)}
            | {"n": len(vals)}
            for kind, vals in prof.items()
        }
    report["held_out_vs_train_containment"] = profiles
    final_split, leak_ex = resolve_leakage(pool_rows, args.lyric_dup_threshold, args.melody_dup_threshold)
    exclusions.extend(leak_ex)

    # ---------------------------------------------------------- tokens
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    kept = sorted(final_split)
    lengths = {}
    batch = 256
    for i in range(0, len(kept), batch):
        ids = kept[i: i + batch]
        p = tok([results[s]["prompt"] for s in ids], add_special_tokens=False)["input_ids"]
        c = tok([results[s]["abc"] for s in ids], add_special_tokens=False)["input_ids"]
        for s, pi, ci in zip(ids, p, c):
            lengths[s] = (len(pi), len(ci))
    for s in kept:
        pl, cl = lengths[s]
        if pl + cl + 1 > args.max_seq_len:
            exclusions.append({"song_id": s, "split": final_split[s], "reason": "exceeds_max_seq_len",
                               "tokens": pl + cl + 1})
            del final_split[s]

    # ---------------------------------------------------------- write
    counts = Counter()
    files = {}
    for split in SPLITS:
        ids = sorted(s for s, sp in final_split.items() if sp == split)
        counts[split] = len(ids)
        with open(out / f"songs_{split}.jsonl", "w", encoding="utf-8") as f_song, \
                open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f_sft, \
                open(out / f"cpt_{split}.jsonl", "w", encoding="utf-8") as f_cpt:
            for s in ids:
                r = results[s]
                pl, cl = lengths[s]
                f_song.write(json.dumps({
                    "song_id": s, "split": split, "song": r["song"], "abc": r["abc"], "spec": r["spec"],
                    "stats": r["stats"], "prompt_tokens": pl, "abc_tokens": cl,
                }, ensure_ascii=False) + "\n")
                f_sft.write(json.dumps({
                    "id": f"sft:{s}", "song_id": s, "prompt": r["prompt"], "completion": r["abc"],
                    "prompt_tokens": pl, "completion_tokens": cl,
                }, ensure_ascii=False) + "\n")
                f_cpt.write(json.dumps({
                    "id": f"cpt:{s}", "song_id": s, "text": r["abc"], "tokens": cl,
                }, ensure_ascii=False) + "\n")
        for name in (f"songs_{split}.jsonl", f"sft_{split}.jsonl", f"cpt_{split}.jsonl"):
            files[name] = sha256_file(out / name)

    with open(out / "exclusions.jsonl", "w", encoding="utf-8") as fh:
        for e in sorted(exclusions, key=lambda e: (e["song_id"], e["reason"])):
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    split_manifest = {split: sorted(s for s, sp in final_split.items() if sp == split) for split in SPLITS}
    with open(out / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=1)
    files["split_manifest.json"] = sha256_file(out / "split_manifest.json")
    files["exclusions.jsonl"] = sha256_file(out / "exclusions.jsonl")

    reasons = Counter((e["reason"], e["split"]) for e in exclusions)
    report["exclusion_reasons"] = {f"{reason}|{split}": n for (reason, split), n in sorted(reasons.items(), key=str)}
    report["final_split_songs"] = dict(counts)
    report["token_lengths"] = {
        split: sorted(lengths[s][0] + lengths[s][1] + 1 for s in split_manifest[split]) for split in SPLITS
    }
    report["files_sha256"] = files
    report["elapsed_sec"] = round(time.time() - started, 1)
    with open(out / "build_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("final_split_songs", "exclusion_reasons", "conversion")},
                     indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
