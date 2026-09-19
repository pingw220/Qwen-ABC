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

Round-2 options: --format v2 (ABC-v2 prompts; counters scored), --batch-size N
(left-padded batched sampling for the main set; see qwen_abc/generate.py),
--read-generations DIR + --rescore (re-score cached generations of an earlier
run with the current metric code, writing only into --output-dir), and
--song-id-list FILE (one song id per line).
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
from qwen_abc.abc_v2 import counter_report, spec_to_prompt_v2  # noqa: E402
from qwen_abc.abc_v3 import spec_to_prompt_v3  # noqa: E402
from qwen_abc.generate import generate_batch, generate_one, load_for_generation  # noqa: E402
from qwen_abc.metrics import aggregate, distribution_distances, lcs_len, song_metrics  # noqa: E402
from qwen_abc.prompt import spec_to_prompt, split_syllables  # noqa: E402
from qwen_abc.theory import canonical_key, parse_key_name  # noqa: E402


PROMPT_FN = [spec_to_prompt]  # set from --format in main(); probes use the same prompt format


def pick_songs(data_dir: Path, split: str, n: int, song_ids=None):
    """Every k-th song (sorted by id), or an explicit id list. Streams the (large) song file."""
    path = data_dir / f"songs_{split}.jsonl"
    if song_ids:
        wanted = set(song_ids)
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line[len('{"song_id": "'):].split('"', 1)[0] in wanted:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r["song_id"])
        return rows[:n] if n else rows
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
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


def score_row(row: dict, spec: dict, fmt: str) -> dict:
    """(Re)compute parse, metrics, counters and MIDI export for a generation row in place."""
    res = parse_abc(row["generation"], row.get("song_id", "gen"))
    row.update(parse_ok=res.ok, strict_ok=res.strict_ok, errors=dict(res.errors), warnings=dict(res.warnings))
    row.pop("metrics", None)
    row.pop("song", None)
    if res.ok:
        m = song_metrics(res.song, spec, res)
        m["hit_eos"] = float(bool(row.get("hit_eos")))
        m["early_eos"] = float(bool(row.get("hit_eos")) and m.get("fewer_sections_than_requested", 0.0) == 1.0)
        m["new_tokens"] = row.get("new_tokens")
        if fmt in ("v2", "v3"):  # v3 completions are ABC-v2 text, counters and all
            m.update(counter_report(row["generation"], spec))
        row["metrics"] = m
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
    return row


def run_one(model, tok, prompt, spec, seed, args, out_path: Path, meta: dict, temperature=None):
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    temp = args.temperature if temperature is None else temperature
    gen = generate_one(model, tok, prompt, seed, args.max_new_tokens, temp, args.top_p, args.max_total,
                       no_cram=getattr(args, "no_cram", False))
    row = {**meta, "seed": seed, "prompt": prompt, "generation": gen["text"], "new_tokens": gen["new_tokens"],
           "hit_eos": gen["hit_eos"], "seconds": gen["seconds"], "temperature": temp, "top_p": args.top_p}
    score_row(row, spec, args.format)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def generate_main_batched(model, tok, songs, args, gen_dir: Path, make_prompt) -> None:
    """Batched sampling for every main-set song whose generation file is missing (sorted by length)."""
    pending = []
    for r in songs:
        for seed in range(args.seeds):
            path = gen_dir / f"{r['song_id']}_s{seed}.json"
            if not path.exists():
                pending.append((len(r["abc"]), r, seed, path))
    pending.sort(key=lambda x: (x[0], x[1]["song_id"], x[2]))
    for k in range(0, len(pending), args.batch_size):
        chunk = pending[k: k + args.batch_size]
        prompts = [make_prompt(r["spec"]) for _, r, _, _ in chunk]
        outs = generate_batch(model, tok, prompts, 1000 + k, args.max_new_tokens, args.temperature, args.top_p,
                              args.max_total, no_cram=getattr(args, "no_cram", False))
        for (_, r, seed, path), prompt, gen in zip(chunk, prompts, outs):
            row = {"song_id": r["song_id"], "kind": "main", "seed": seed, "batch_seed": 1000 + k, "prompt": prompt,
                   "generation": gen["text"], "new_tokens": gen["new_tokens"], "hit_eos": gen["hit_eos"],
                   "seconds": gen["seconds"], "batch_size": gen["batch_size"], "batch_seconds": gen["batch_seconds"],
                   "temperature": args.temperature, "top_p": args.top_p}
            score_row(row, r["spec"], args.format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        print(f"[batch {k // args.batch_size + 1}/{-(-len(pending) // args.batch_size)}] {len(chunk)} songs "
              f"{outs[0]['batch_seconds']}s tokens={sum(g['new_tokens'] for g in outs)} "
              f"eos={sum(g['hit_eos'] for g in outs)}", flush=True)


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
    ap.add_argument("--song-ids-from", type=Path, default=None, help="jsonl with song_id fields (e.g. an SFT file)")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--num-probe-songs", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help=">1: generate only this shard, write no summary")
    ap.add_argument("--probes-only", action="store_true", help="generate probe outputs only (for a separate job)")
    ap.add_argument("--greedy-probes-only", action="store_true", help="generate the greedy probe set only")
    ap.add_argument("--aggregate-only", action="store_true", help="no model: summarize cached generations")
    ap.add_argument("--format", choices=["v1", "v2", "v3"], default="v1",
                    help="prompt format (v2: ABC-v2 prompt + counter metrics; v3: + syllable budget and continuation marks)")
    ap.add_argument("--batch-size", type=int, default=1, help=">1: batched sampling for the main set")
    ap.add_argument("--max-total", type=int, default=8192, help="prompt + generation token cap")
    ap.add_argument("--read-generations", type=Path, default=None, help="cached generations dir (default OUT/generations)")
    ap.add_argument("--rescore", action="store_true", help="re-score cached rows with the current metric code")
    ap.add_argument("--no-cram", action="store_true",
                    help="constrained decoding: forbid '~' inside a w: line, so no note can carry several syllables")
    ap.add_argument("--song-id-list", type=Path, default=None, help="text file, one song id per line")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    ids = [json.loads(l)["song_id"] for l in open(args.song_ids_from, encoding="utf-8")] if args.song_ids_from else None
    if args.song_id_list:
        ids = [l.strip() for l in open(args.song_id_list, encoding="utf-8") if l.strip()]
    make_prompt = {"v2": spec_to_prompt_v2, "v3": spec_to_prompt_v3}.get(args.format, spec_to_prompt)
    PROMPT_FN[0] = make_prompt
    gen_dir = args.read_generations or (out / "generations")
    songs = pick_songs(args.data_dir, args.split, args.num_songs, ids)
    sharded = args.num_shards > 1
    model = tok = None
    if not args.aggregate_only:
        model, tok = load_for_generation(args.checkpoint)

    def cached_or_generate(prompt, spec, seed, path, meta, temperature=None):
        if args.aggregate_only:
            if not path.exists():
                raise SystemExit(f"missing generation {path}; run the generation shards first")
            row = json.loads(path.read_text(encoding="utf-8"))
            return score_row(row, spec, args.format) if args.rescore else row
        return run_one(model, tok, prompt, spec, seed, args, path, meta, temperature)

    gen_rows, ref_rows, gen_songs, ref_songs = [], [], [], []
    main_songs = [] if (args.probes_only or args.greedy_probes_only) else (songs[args.shard:: args.num_shards] if sharded else songs)
    if args.batch_size > 1 and main_songs and not args.aggregate_only:
        generate_main_batched(model, tok, main_songs, args, gen_dir, make_prompt)
    for i, r in enumerate(main_songs):
        ref = Song.from_json(r["song"])
        ref_parse = parse_abc(r["abc"], r["song_id"])
        ref_rows.append(song_metrics(ref, r["spec"], ref_parse))
        ref_songs.append(ref)
        for seed in range(args.seeds):
            prompt = make_prompt(r["spec"])
            row = cached_or_generate(prompt, r["spec"], 1000 + seed,
                                     gen_dir / f"{r['song_id']}_s{seed}.json",
                                     {"song_id": r["song_id"], "kind": "main"})
            gen_rows.append(row)
            if row.get("song"):
                gen_songs.append(Song.from_json(row["song"]))
                # copying the reference is not the goal on held-out songs, but it is the
                # sanity signal on memorized ones
                row["metrics"]["melody_distance_to_reference"] = melody_distance(gen_songs[-1], ref)
            print(f"[{i + 1}/{len(main_songs)}] {r['song_id']} seed={seed} parse={row['parse_ok']} strict={row['strict_ok']} "
                  f"eos={row['hit_eos']} toks={row['new_tokens']} {row['seconds']}s "
                  f"lyric_recall={row.get('metrics', {}).get('lyric_recall')}", flush=True)

    if sharded and not args.probes_only:
        print("SHARD_DONE")
        return
    if args.greedy_probes_only:
        run_greedy_probes(songs[: args.num_probe_songs], out, cached_or_generate)
        print("GREEDY_PROBES_DONE")
        return
    if args.probes_only:
        probe_songs = songs[: args.num_probe_songs]
        run_probes(probe_songs, args, out, cached_or_generate)
        print("PROBES_DONE")
        return
    n = len(gen_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "num_generations": n,
        "parse_success": sum(r["parse_ok"] for r in gen_rows) / n,
        "strict_valid": sum(r["strict_ok"] for r in gen_rows) / n,
        "midi_success": sum(bool(r.get("midi_ok")) for r in gen_rows) / n,
        "hit_eos": sum(r["hit_eos"] for r in gen_rows) / n,
        "format": args.format, "temperature": args.temperature, "top_p": args.top_p, "batch_size": args.batch_size,
        "generations_dir": str(gen_dir), "rescored": args.rescore,
        "error_counts": {},
        "generated": aggregate([r["metrics"] for r in gen_rows if "metrics" in r]),
        "reference": aggregate(ref_rows),
        "distribution_js_vs_reference": distribution_distances(gen_songs, ref_songs),
    }
    for r in gen_rows:
        for k, v in r["errors"].items():
            summary["error_counts"][k] = summary["error_counts"].get(k, 0) + v

    if args.probes or args.aggregate_only and (out / "probes").exists():
        summary["probes"] = run_probes(songs[: args.num_probe_songs], args, out, cached_or_generate)
    if args.aggregate_only and (out / "probes_greedy").exists():
        summary["probes_greedy"] = run_greedy_probes(songs[: args.num_probe_songs], out, cached_or_generate)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("generated", "reference")}, indent=1))
    print("EVAL_DONE")


def run_probes(probe, args, out, gen):
    """Conditional sensitivity on the first N evaluation songs (same seed as the main run)."""
    # probes keep their own baseline generations (same prompt and seed as the main run) so a
    # probe job never races a generation shard writing the same file
    base = {}
    for r in probe:
        base[r["song_id"]] = gen(PROMPT_FN[0](r["spec"]), r["spec"], 1000,
                                 out / "probes" / f"{r['song_id']}_base.json",
                                 {"song_id": r["song_id"], "kind": "probe_base"})
    swaps, tempo_rows, key_rows, reseed = [], [], [], []
    for i, r in enumerate(probe):
        donor = probe[(i + 1) % len(probe)]
        b = base[r["song_id"]]
        if not b.get("song"):
            continue
        b_song = Song.from_json(b["song"])
        # same seed, different lyrics
        spec2 = reflow_lyrics(r["spec"], donor["spec"])
        row = gen(PROMPT_FN[0](spec2), spec2, 1000, out / "probes" / f"{r['song_id']}_lyricswap.json",
                  {"song_id": r["song_id"], "kind": "lyric_swap", "donor": donor["song_id"]})
        if row.get("song"):
            swaps.append({"melody_distance": melody_distance(b_song, Song.from_json(row["song"])),
                          "lyric_recall_new": row["metrics"]["lyric_recall"]})
        # same prompt, different seed: the noise floor for melody distance
        row = gen(PROMPT_FN[0](r["spec"]), r["spec"], 2000, out / "probes" / f"{r['song_id']}_reseed.json",
                  {"song_id": r["song_id"], "kind": "reseed"})
        if row.get("song"):
            reseed.append({"melody_distance": melody_distance(b_song, Song.from_json(row["song"]))})
        # tempo change
        spec3 = copy.deepcopy(r["spec"])
        spec3["tempo_bpm"] = int(round(r["spec"]["tempo_bpm"] * 1.25))
        row = gen(PROMPT_FN[0](spec3), spec3, 1000, out / "probes" / f"{r['song_id']}_tempo.json",
                  {"song_id": r["song_id"], "kind": "tempo"})
        if row.get("song"):
            s3 = Song.from_json(row["song"])
            tempo_rows.append({"tempo_followed": float(s3.tempo_bpm == spec3["tempo_bpm"]),
                               "notes_per_bar_ratio": row["metrics"]["notes_per_bar"] / max(b["metrics"]["notes_per_bar"], 1e-9),
                               "melody_distance": melody_distance(b_song, s3)})
        # key change: +5 semitones
        parsed = parse_key_name(r["spec"].get("key"))
        if parsed:
            spec4 = copy.deepcopy(r["spec"])
            pc, mode = parsed
            names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
            spec4["key"] = canonical_key(f"{names[(pc + 5) % 12]} {mode}")
            row = gen(PROMPT_FN[0](spec4), spec4, 1000, out / "probes" / f"{r['song_id']}_key.json",
                      {"song_id": r["song_id"], "kind": "key"})
            if row.get("song"):
                s4 = Song.from_json(row["song"])
                tonic = parse_key_name(spec4["key"])[0]
                key_rows.append({"key_followed": float(s4.key == spec4["key"]),
                                 "in_new_scale": _in_scale_frac(s4, tonic, mode),
                                 "baseline_in_new_scale": _in_scale_frac(b_song, tonic, mode),
                                 "pitch_mean_shift": row["metrics"].get("pitch_mean", 0) - b["metrics"].get("pitch_mean", 0)})
    return {
        "n_songs": len(probe),
        "lyric_swap": aggregate(swaps),
        "reseed_noise_floor": aggregate(reseed),
        "tempo_change": aggregate(tempo_rows),
        "key_change": aggregate(key_rows),
    }


def run_greedy_probes(probe, out, gen):
    """Deterministic sensitivity: greedy decoding, so any change is caused by the condition.

    * lyric swap: same structure/tempo/key, another song's lyrics re-flowed into the sections
    * key change by a tritone (+6): shares only 2 of 7 scale degrees, unlike +5
    """
    d = out / "probes_greedy"
    swaps, keys = [], []
    names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    for i, r in enumerate(probe):
        donor = probe[(i + 1) % len(probe)]
        base = gen(PROMPT_FN[0](r["spec"]), r["spec"], 0, d / f"{r['song_id']}_base.json",
                   {"song_id": r["song_id"], "kind": "greedy_base"}, 0.0)
        spec2 = reflow_lyrics(r["spec"], donor["spec"])
        swap = gen(PROMPT_FN[0](spec2), spec2, 0, d / f"{r['song_id']}_lyricswap.json",
                   {"song_id": r["song_id"], "kind": "greedy_lyric_swap", "donor": donor["song_id"]}, 0.0)
        parsed = parse_key_name(r["spec"].get("key"))
        key = None
        if parsed:
            pc, mode = parsed
            spec4 = copy.deepcopy(r["spec"])
            spec4["key"] = canonical_key(f"{names[(pc + 6) % 12]} {mode}")
            key = gen(PROMPT_FN[0](spec4), spec4, 0, d / f"{r['song_id']}_key6.json",
                      {"song_id": r["song_id"], "kind": "greedy_key6"}, 0.0)
        if not base.get("song"):
            continue
        b = Song.from_json(base["song"])
        if swap.get("song"):
            s2 = Song.from_json(swap["song"])
            swaps.append({"melody_distance": melody_distance(b, s2),
                          "pitch_contour_distance": contour_distance(b, s2),
                          "lyric_recall_new": swap["metrics"]["lyric_recall"],
                          "lyric_recall_of_old_lyrics": _recall_of(s2, r["spec"])})
        if key is not None and key.get("song"):
            s4 = Song.from_json(key["song"])
            tonic = parse_key_name(spec4["key"])[0]
            keys.append({"key_followed": float(s4.key == spec4["key"]),
                         "in_new_scale": _in_scale_frac(s4, tonic, mode),
                         "baseline_in_new_scale": _in_scale_frac(b, tonic, mode),
                         "in_old_scale": _in_scale_frac(s4, pc, mode),
                         "baseline_in_old_scale": _in_scale_frac(b, pc, mode)})
    return {"n_songs": len(probe), "lyric_swap": aggregate(swaps), "key_tritone": aggregate(keys)}


def contour_distance(a: Song, b: Song) -> float:
    """1 - normalized LCS over the pitch-interval sequence (timing ignored)."""
    ia = [str(y.pitch - x.pitch) for x, y in zip(a.notes, a.notes[1:])]
    ib = [str(y.pitch - x.pitch) for x, y in zip(b.notes, b.notes[1:])]
    return 1 - lcs_len(ia, ib) / max(len(ia), len(ib), 1)


def _recall_of(song: Song, spec: dict) -> float:
    from qwen_abc.prompt import spec_syllables
    want = spec_syllables(spec)
    got = [x for n in song.notes if n.lyric for x in n.lyric]
    return lcs_len(want, got) / max(len(want), 1)


def _in_scale_frac(song: Song, tonic: int, mode: str) -> float:
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10, 11)
    if not song.notes:
        return 0.0
    return sum(1 for n in song.notes if (n.pitch - tonic) % 12 in steps) / len(song.notes)


if __name__ == "__main__":
    main()
