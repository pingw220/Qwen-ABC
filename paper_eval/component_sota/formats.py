"""Task formats of the component round, all derived from the canonical ABC-v2 lead sheet.

* **Melody-only (``mel``)** -- identical conditioning to E3b (plan, lyrics, key, meter, tempo,
  ``section i/N``); the completion is the ABC-v2 lead sheet with every chord symbol removed.
  The infill variant is E3b's masked-section task on the chord-free song.
* **Harmonizer (``chord``)** -- the melody is given read-only in the prompt (ABC-v2 without chord
  symbols; lyrics lines stripped unless ``with_lyrics``) and the completion is only a chord line
  per bar: ``[r:k] b:SYM b:SYM`` (beat offset in the bar : chord symbol; ``-`` = no change). The
  model cannot alter the melody because it never writes it.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Tuple

from qwen_abc.abc_v2 import PROMPT_HEADER_V2, spec_to_prompt_v2, song_to_abc_v2
from qwen_abc.canonical import TICKS_PER_BEAT, Chord, Song, normalize_chords
from qwen_abc.prompt import COMPLETION_MARKER, _beats_text

MEL_HEADER = "Task: write a melody in ABC notation (melody, aligned lyrics, bar countdown)."
CHORD_HEADER = "Task: harmonize the melody below. For every bar write its chord changes as beat:chord."
CHORD_MARKER = "Chords:\n"


def strip_chords(song: Song) -> Song:
    s = copy.deepcopy(song)
    s.chords = []
    return s


def mel_prompt(spec: Dict) -> str:
    p = spec_to_prompt_v2(spec)
    assert p.startswith(PROMPT_HEADER_V2)
    return MEL_HEADER + p[len(PROMPT_HEADER_V2):]


def mel_completion(song: Song) -> str:
    return song_to_abc_v2(strip_chords(song))


# ------------------------------------------------------------------ harmonizer
def _melody_block(song: Song, with_lyrics: bool) -> str:
    abc = song_to_abc_v2(strip_chords(song))
    if with_lyrics:
        return abc
    return "\n".join(l for l in abc.split("\n") if not l.startswith("w:")) + ("" if abc.endswith("\n") else "\n")


def chord_prompt(song: Song, spec: Dict, with_lyrics: bool = False) -> str:
    nominal = int(str(spec["meter"]).split("/")[0])
    out = [CHORD_HEADER, f"Meter: {spec['meter']}", f"Tempo: {spec['tempo_bpm']} BPM"]
    if spec.get("key"):
        out.append(f"Key: {spec['key']}")
    out.append("Structure:")
    n = len(spec["sections"])
    for i, sec in enumerate(spec["sections"]):
        out.append(f"P:{sec['label']} | {sec['bars']} bars{_beats_text(sec['beats'], nominal)} | section {i + 1}/{n}")
    return "\n".join(out) + "\n\nMelody:\n" + _melody_block(song, with_lyrics) + "\n" + CHORD_MARKER


def chord_lines(song: Song) -> str:
    """Per section a ``P:label`` line, then one line per bar with that bar's chord changes."""
    starts = song.bar_starts()
    chords = normalize_chords(song.chords, song.total_ticks)
    out = []
    for sec in song.sections:
        out.append(f"P:{sec.label}")
        for b in range(sec.start_bar, min(sec.start_bar + sec.num_bars, len(starts))):
            lo = starts[b]
            hi = starts[b + 1] if b + 1 < len(starts) else song.total_ticks
            ev = [f"{(c.onset - lo) / TICKS_PER_BEAT:g}:{c.symbol}" for c in chords if lo <= c.onset < hi]  # off-beat chords keep .5/.25
            out.append(f"[r:{sec.start_bar + sec.num_bars - b}] " + (" ".join(ev) if ev else "-"))
    return "\n".join(out) + "\n"


_EV = re.compile(r"(\d+(?:\.\d+)?):(\S+)")


def parse_chord_lines(text: str, melody: Song) -> Tuple[List[Chord], Dict[str, float]]:
    """Chord lines -> chords on ``melody``'s bar grid. Bars are read in order; extra lines are
    dropped and missing bars get no new chord (both counted)."""
    starts = melody.bar_starts()
    bars: List[List[Tuple[int, str]]] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("P:") or s.startswith("%"):
            continue
        s = re.sub(r"^\[r:\s*\d+\s*\]\s*", "", s)
        bars.append([(float(b), sym) for b, sym in _EV.findall(s)])
    chords = []
    for b, ev in enumerate(bars[: len(starts)]):
        beats = melody.bar_beats[b]
        for beat, sym in ev:
            if 0 <= beat < beats:
                chords.append(Chord(starts[b] + int(round(beat * TICKS_PER_BEAT)), 0, sym))
    stats = {"chord_bars_written": len(bars), "bars": len(starts),
             "chord_bar_count_match": float(len(bars) == len(starts))}
    return normalize_chords(chords, melody.total_ticks), stats


def with_chords(melody: Song, chords: List[Chord]) -> Song:
    s = copy.deepcopy(melody)
    s.chords = normalize_chords(chords, s.total_ticks)
    return s
