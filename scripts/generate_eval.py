#!/usr/bin/env python
"""Generate ABC for held-out songs and score it.

  scripts/generate_eval.py --checkpoint CKPT --data-dir DATA --output-dir OUT \
      [--split test] [--num-songs 100] [--seeds 1] [--probes]

Writes per song: prompt, raw generation, parsed metrics; aggregate metrics
next to the same metrics computed on the reference ABC of the same songs.
--probes adds conditional-sensitivity runs:
  * lyrics swap: song i's structure with song j's lyric lines (re-flowed into
    the same sections) -> does the melody change?
  * tempo / key change with the same lyrics -> does the output follow?
Generation resumes: songs whose generation file already exists are skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.generate import generate_one, load_for_generation  # noqa: E402
from qwen_abc.metrics import aggregate, distribution_distances, lcs_len, song_metrics  # noqa: E402
from qwen_abc.prompt import spec_to_prompt, split_syllables  # noqa: E402
from qwen_abc.theory import canonical_key, parse_key_name  # noqa: E402


def pick_songs(data_dir: Path, split: str, n: int):
    rows = [json.loads(l) for l in open(data_dir / f"songs_{split}.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["song_id"])
    step = max(len(rows) // n, 1) if n else 1
    return rows[::step][:n] if n else rows


def reflow_lyrics(spec: dict, donor: dict) -> dict:
    """Keep spec's structure; fill its lyric sections with donor syllables in proportion."""
    out = copy.deepcopy(spec)
    donor_sylls = [s for sec in donor["sections"] for line in sec["lines"] for s in split_syllables(line)]
    want = [sum(len(split_syllables(l)) for l in sec["lines"]) for sec in spec["sections"]]
    total = sum(want)
    pos = 0
    for sec, k in zip(out["sections"], want):
        if not k:
            continue
        take = round(k * len(donor_sylls) / max(total, 1))
        chunk = donor_sylls[pos: pos + take]
        pos += take
        n_lines = max(len(sec["lines"]), 1)
        per = max(-(-len(chunk) // n_lines), 1)
        from qwen_abc.prompt import join_syllables

        sec["lines"] = [join_syllables(chunk[i: i + per]) for i in range(0, len(chunk), per)]
    return out


def melody_distance(a: Song, b: Song) -> float:
    """1 - normalized LCS over (onset-in-bar, pitch) note tokens; 0 = identical melody."""
    def toks(s):
        starts = s.bar_starts()
        import bisect
        out = []
        for n in s.notes:
            i = bisect.bisect_right(starts, n.onset) - 1
            out.append(f"{n.onset - starts[max(i, 0)]}:{n.pitch}:{n.duration}")
        return out
    ta, tb = toks(a), toks(b)
    return 1 - lcs_len(ta, tb) / max(len(ta), len(tb), 1)


def run_one(model, tok, prompt, spec, seed, args, out_path: Path, meta: dict):
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    gen = generate_one(model, tok, prompt, seed, args.max_new_tokens, args.temperature, args.top_p)
    res = parse_abc(gen["text"], meta.get("song_id", "gen"))
    row = {**meta, "seed": seed, "prompt": prompt, "generation": gen["text"], "new_tokens": gen["new_tokens"],
           "hit_eos": gen["hit_eos"], "seconds": gen["seconds"], "parse_ok": res.ok, "strict_ok": res.strict_ok,
           "errors": dict(res.errors), "warnings": dict(res.warnings)}
    if res.ok:
        row["metrics"] = song_metrics(res.song, spec, res)
        row["song"] = res.song.to_json()
        try:
            import tempfile
            from qwen_abc.midi import song_to_midi
            with tempfile.NamedTemporaryFile(suffix=".mid") as fh:
                song_to_midi(res.song, fh.name)
            row["midi_ok"] = len(res.song.notes) > 0
        except Exception as exc:  # noqa: BLE001
            row["midi_ok"] = False
            row["midi_error"] = str(exc)
    else:
        row["midi_ok"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-songs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=7680)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--num-probe-songs", type=int, default=20)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    songs = pick_songs(args.data_dir, args.split, args.num_songs)
    model, tok = load_for_generation(args.checkpoint)

    gen_rows, ref_rows, gen_songs, ref_songs = [], [], [], []
    for i, r in enumerate(songs):
        ref = Song.from_json(r["song"])
        ref_parse = parse_abc(r["abc"], r["song_id"])
        ref_rows.append(song_metrics(ref, r["spec"], ref_parse))
        ref_songs.append(ref)
        for seed in range(args.seeds):
            prompt = spec_to_prompt(r["spec"])
            row = run_one(model, tok, prompt, r["spec"], 1000 + seed, args,
                          out / "generations" / f"{r['song_id']}_s{seed}.json",
                          {"song_id": r["song_id"], "kind": "main"})
            gen_rows.append(row)
            if row.get("song"):
                gen_songs.append(Song.from_json(row["song"]))
            print(f"[{i + 1}/{len(songs)}] {r['song_id']} seed={seed} parse={row['parse_ok']} strict={row['strict_ok']} "
                  f"eos={row['hit_eos']} toks={row['new_tokens']} {row['seconds']}s "
                  f"lyric_recall={row.get('metrics', {}).get('lyric_recall')}", flush=True)

    n = len(gen_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "num_generations": n,
        "parse_success": sum(r["parse_ok"] for r in gen_rows) / n,
        "strict_valid": sum(r["strict_ok"] for r in gen_rows) / n,
        "midi_success": sum(bool(r.get("midi_ok")) for r in gen_rows) / n,
        "hit_eos": sum(r["hit_eos"] for r in gen_rows) / n,
        "error_counts": {},
        "generated": aggregate([r["metrics"] for r in gen_rows if "metrics" in r]),
        "reference": aggregate(ref_rows),
        "distribution_js_vs_reference": distribution_distances(gen_songs, ref_songs),
    }
    for r in gen_rows:
        for k, v in r["errors"].items():
            summary["error_counts"][k] = summary["error_counts"].get(k, 0) + v

    # ------------------------------------------------ conditional sensitivity
    if args.probes:
        probe = songs[: args.num_probe_songs]
        base = {r["song_id"]: row for r, row in zip(songs, gen_rows[:: args.seeds])}
        swaps, tempo_rows, key_rows, reseed = [], [], [], []
        for i, r in enumerate(probe):
            donor = probe[(i + 1) % len(probe)]
            b = base[r["song_id"]]
            if not b.get("song"):
                continue
            b_song = Song.from_json(b["song"])
            # same seed, different lyrics
            spec2 = reflow_lyrics(r["spec"], donor["spec"])
            row = run_one(model, tok, spec_to_prompt(spec2), spec2, 1000, args,
                          out / "probes" / f"{r['song_id']}_lyricswap.json",
                          {"song_id": r["song_id"], "kind": "lyric_swap", "donor": donor["song_id"]})
            if row.get("song"):
                swaps.append({"melody_distance": melody_distance(b_song, Song.from_json(row["song"])),
                              "lyric_recall_new": row["metrics"]["lyric_recall"]})
            # same prompt, different seed: the noise floor for melody distance
            row = run_one(model, tok, spec_to_prompt(r["spec"]), r["spec"], 2000, args,
                          out / "probes" / f"{r['song_id']}_reseed.json", {"song_id": r["song_id"], "kind": "reseed"})
            if row.get("song"):
                reseed.append({"melody_distance": melody_distance(b_song, Song.from_json(row["song"]))})
            # tempo change
            spec3 = copy.deepcopy(r["spec"])
            spec3["tempo_bpm"] = int(round(r["spec"]["tempo_bpm"] * 1.25))
            row = run_one(model, tok, spec_to_prompt(spec3), spec3, 1000, args,
                          out / "probes" / f"{r['song_id']}_tempo.json", {"song_id": r["song_id"], "kind": "tempo"})
            if row.get("song"):
                s3 = Song.from_json(row["song"])
                tempo_rows.append({"tempo_followed": s3.tempo_bpm == spec3["tempo_bpm"],
                                   "notes_per_bar_ratio": row["metrics"]["notes_per_bar"] / max(b["metrics"]["notes_per_bar"], 1e-9)})
            # key change: +5 semitones
            parsed = parse_key_name(r["spec"].get("key"))
            if parsed:
                spec4 = copy.deepcopy(r["spec"])
                pc, mode = parsed
                names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
                spec4["key"] = canonical_key(f"{names[(pc + 5) % 12]} {mode}")
                row = run_one(model, tok, spec_to_prompt(spec4), spec4, 1000, args,
                              out / "probes" / f"{r['song_id']}_key.json", {"song_id": r["song_id"], "kind": "key"})
                if row.get("song"):
                    s4 = Song.from_json(row["song"])
                    tonic = parse_key_name(spec4["key"])[0]
                    in_scale = _in_scale_frac(s4, tonic, mode)
                    old_in_new = _in_scale_frac(b_song, tonic, mode)
                    key_rows.append({"key_followed": s4.key == spec4["key"], "in_new_scale": in_scale,
                                     "baseline_in_new_scale": old_in_new})
        summary["probes"] = {
            "lyric_swap": aggregate([{k: float(v) for k, v in x.items()} for x in swaps]),
            "reseed_noise_floor": aggregate(reseed),
            "tempo_change": aggregate([{k: float(v) for k, v in x.items()} for x in tempo_rows]),
            "key_change": aggregate([{k: float(v) for k, v in x.items()} for x in key_rows]),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("generated", "reference")}, indent=1))
    print("EVAL_DONE")


def _in_scale_frac(song: Song, tonic: int, mode: str) -> float:
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10, 11)
    if not song.notes:
        return 0.0
    return sum(1 for n in song.notes if (n.pitch - tonic) % 12 in steps) / len(song.notes)


if __name__ == "__main__":
    main()
