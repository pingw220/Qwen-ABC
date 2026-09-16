#!/usr/bin/env python
"""ABC (v1 or v2) -> MusicXML, for opening lead sheets in MuseScore/Finale/Dorico.

  scripts/export_musicxml.py IN.abc [IN.abc ...] [--outdir DIR]
  scripts/export_musicxml.py --sample-dir experiments/listen_now_<ts>   # every .abc under it

The canonical parser is used rather than music21's ABC reader, so ABC-v2 counters
(`[r:k]`) and section comments are handled exactly as the project defines them.
Each score carries the melody with its lyrics, chord symbols, per-bar meters
(irregular bars are real), the key, the tempo and a rehearsal mark per section.
Notes that cross a barline are split and tied, as an engraver would write them.
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.canonical import TICKS_PER_BEAT, Song  # noqa: E402
from qwen_abc.theory import parse_chord_symbol  # noqa: E402

QL = Fraction(1, TICKS_PER_BEAT)  # one tick in quarter lengths


def song_to_score(song: Song, title: str = ""):
    from music21 import bar, chord, clef, expressions, harmony, key, metadata, meter, note, stream, tempo

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title or song.song_id, composer="Qwen-ABC")
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    if song.key:
        tonic, mode = song.key.split()
        part.insert(0, key.Key(tonic.replace("b", "-"), mode))
    part.insert(0, tempo.MetronomeMark(number=song.tempo_bpm))

    starts = song.bar_starts()
    section_at = {s.start_bar: s.label for s in song.sections}
    chords = {c.onset: c.symbol for c in song.chords}
    # note segments per bar, splitting anything that crosses a barline
    segs: dict[int, list] = {i: [] for i in range(len(song.bar_beats))}
    for n in song.notes:
        t, remaining, first = n.onset, n.duration, True
        while remaining > 0:
            b = max([i for i, s in enumerate(starts) if s <= t], default=0)
            bar_end = starts[b] + song.bar_beats[b] * TICKS_PER_BEAT
            take = min(remaining, bar_end - t)
            segs[b].append((t, take, n, first, remaining > take))
            t += take
            remaining -= take
            first = False

    current_meter = None
    for b, beats in enumerate(song.bar_beats):
        m = stream.Measure(number=b + 1)
        if beats != current_meter:
            m.insert(0, meter.TimeSignature(f"{beats}/4"))
            current_meter = beats
        if b in section_at:
            mark = expressions.RehearsalMark(section_at[b])
            m.insert(0, mark)
        cursor = starts[b]
        bar_end = starts[b] + beats * TICKS_PER_BEAT
        for onset, dur, src, is_first, ties_on in segs[b]:
            if onset > cursor:
                r = note.Rest(quarterLength=float((onset - cursor) * QL))
                m.insert(float((cursor - starts[b]) * QL), r)
            el = note.Note(src.pitch, quarterLength=float(dur * QL))
            if is_first and src.lyric:
                el.lyric = "".join(src.lyric)
            if ties_on or not is_first:
                from music21.tie import Tie

                el.tie = Tie("continue" if (ties_on and not is_first) else ("start" if ties_on else "stop"))
            m.insert(float((onset - starts[b]) * QL), el)
            cursor = onset + dur
        if cursor < bar_end:
            m.insert(float((cursor - starts[b]) * QL), note.Rest(quarterLength=float((bar_end - cursor) * QL)))
        for onset, symbol in chords.items():
            if starts[b] <= onset < bar_end:
                info = parse_chord_symbol(symbol)
                try:
                    cs = harmony.NoChord() if (not info or info["root_pc"] is None) else harmony.ChordSymbol(
                        symbol.replace("b", "-").replace("N.C.", "N.C."))
                except Exception:  # noqa: BLE001 — a symbol music21 cannot spell stays a text mark
                    cs = expressions.TextExpression(symbol)
                m.insert(float((onset - starts[b]) * QL), cs)
        if b == len(song.bar_beats) - 1:
            m.rightBarline = bar.Barline("final")
        part.append(m)
    score.insert(0, part)
    return score


def convert(path: Path, outdir: Path) -> Path:
    res = parse_abc(path.read_text(encoding="utf-8"), path.stem)
    if not res.ok:
        raise SystemExit(f"{path}: could not parse")
    score = song_to_score(res.song, title=path.parent.name if path.stem == "reference" else f"{path.parent.name} ({path.parent.name})")
    out = outdir / (path.stem + ".musicxml")
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("abc", nargs="*", type=Path)
    ap.add_argument("--sample-dir", type=Path, default=None, help="convert every .abc under this directory, in place")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    files = list(args.abc)
    if args.sample_dir:
        files += sorted(args.sample_dir.rglob("*.abc"))
    if not files:
        raise SystemExit(__doc__)
    n = 0
    for f in files:
        out = convert(f, args.outdir or f.parent)
        n += 1
        print(f"{f} -> {out}")
    print(f"MUSICXML_DONE {n}")


if __name__ == "__main__":
    main()
