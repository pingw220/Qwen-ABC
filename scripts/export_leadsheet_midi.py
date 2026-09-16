#!/usr/bin/env python
"""ABC -> lead sheet MIDI for the MIDI-SAG / MuseControlLite renderer.

  scripts/export_leadsheet_midi.py --sample-dir experiments/listen_now_<ts> \
      --outdir experiments/midi_sag_<ts>/leadsheets

Every ``*.abc`` under the sample directory becomes one flat
``<relative__path>.mid`` plus a ``.json`` audit, because MIDI-SAG's runner globs
``<input-dir>/*.mid`` non-recursively and names its outputs after the file stem.
The audit records the structure plan (tags, starts, window spans) the renderer
will condition on, and any reason a lead sheet was skipped.

The files are exactly what ``MIDI-SAG/tools/midi_llm_adapter/
run_midi_llm_to_midi_sag.py`` reads from MIDI-LLM: conductor markers, melody
with lyric events, chords, click. See ``qwen_abc/leadsheet_midi.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mido  # noqa: E402

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.fastsinger import write_fastsinger_inputs  # noqa: E402
from qwen_abc.leadsheet_midi import (  # noqa: E402
    SAO_WINDOW_SECONDS,
    renderable_problems,
    song_to_leadsheet_midi,
    summarize,
)


def check_fastsinger_pairing(midi_path: Path, txt_path: Path) -> str:
    """FastSinger pairs symbol k with note k, so the two files must agree exactly."""
    notes = sum(
        1 for tr in mido.MidiFile(str(midi_path)).tracks for m in tr
        if m.type == "note_on" and m.velocity > 0
    )
    symbols = sum(len(line.replace("，", "")) for line in txt_path.read_text(encoding="utf-8").splitlines())
    if notes != symbols:
        return f"FastSinger pairing is off: {notes} notes vs {symbols} symbols"
    return ""


def flat_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts if p not in ("generated",)]
    if not parts:
        parts = [rel.name]
    return "__".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("abc", nargs="*", type=Path, help="individual .abc files")
    ap.add_argument("--sample-dir", type=Path, default=None, help="convert every .abc under this directory")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--only", action="append", default=None,
                    help="keep only paths containing this substring (repeatable), e.g. --only E3b_best_T1.0")
    ap.add_argument("--max-segment-seconds", type=float, default=SAO_WINDOW_SECONDS - 1.0)
    ap.add_argument("--no-fastsinger", action="store_true",
                    help="skip the FastSinger melody/lyric pair (vocal synthesis inputs)")
    args = ap.parse_args()

    files = list(args.abc)
    root = args.sample_dir or Path(".")
    if args.sample_dir:
        files += sorted(p for p in args.sample_dir.rglob("*.abc") if not p.name.endswith(".clean.abc"))
    if args.only:
        files = [f for f in files if any(o in str(f) for o in args.only)]
    if not files:
        raise SystemExit(__doc__)

    args.outdir.mkdir(parents=True, exist_ok=True)
    audits, skipped = [], []
    for f in files:
        if f.stat().st_size == 0:
            skipped.append({"source": str(f), "problems": ["empty file"]})
            continue
        res = parse_abc(f.read_text(encoding="utf-8"), f.stem)
        if not res.ok:
            skipped.append({"source": str(f), "problems": ["ABC does not parse"]})
            continue
        name = flat_name(f, root) if args.sample_dir else f.stem
        song = res.song
        song.song_id = name
        out = args.outdir / f"{name}.mid"
        audit = song_to_leadsheet_midi(song, str(out), max_segment_seconds=args.max_segment_seconds)
        audit["source"] = str(f)
        problems = renderable_problems(song, audit)
        if problems:
            out.unlink(missing_ok=True)
            audit["problems"] = problems
            skipped.append(audit)
            print(f"skip {name}: {'; '.join(problems)}")
            continue
        if not args.no_fastsinger:
            fs_midi, fs_txt = args.outdir / f"{name}.fs.mid", args.outdir / f"{name}.fs.txt"
            audit["fastsinger"] = write_fastsinger_inputs(song, str(fs_midi), str(fs_txt))
            mismatch = check_fastsinger_pairing(fs_midi, fs_txt)
            if mismatch:
                out.unlink(missing_ok=True)
                audit["problems"] = [mismatch]
                skipped.append(audit)
                print(f"skip {name}: {mismatch}")
                continue
        (args.outdir / f"{name}.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        audits.append(audit)
        sung = audit.get("fastsinger", {}).get("notes_sung", 0)
        print(f"{name}: {audit['duration_s']:6.1f}s  {audit['n_bars']:3d} bars  "
              f"{audit['n_segments']:2d} segments (max {audit['max_segment_span_s']:5.1f}s)  "
              f"{audit['n_lyric_notes']:3d} lyrics  {sung:3d} sung notes -> {out}")

    index = {"summary": summarize(audits), "leadsheets": audits, "skipped": skipped}
    (args.outdir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LEADSHEET_DONE written={len(audits)} skipped={len(skipped)} -> {args.outdir}")


if __name__ == "__main__":
    main()
