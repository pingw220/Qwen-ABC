"""Canonical, representation-neutral lead sheet.

All time is integer ``ticks`` where one tick is a quarter of a beat (a
sixteenth note when the beat is a quarter note). The corpus quantizes melody
onsets to exactly this grid and chords to whole beats, so nothing finer is
needed.

A song is a sequence of bars (each with its own beat count, so irregular
downbeats survive), sections that start on bar boundaries, a monophonic
melody, and chord segments. Lyrics live on notes:

* ``lyric`` is a tuple of syllable strings sung starting on that note
  (normally one; more than one when an unmatched syllable was joined on);
* ``melisma`` marks a note that continues the previous syllable;
* a note with neither is wordless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

TICKS_PER_BEAT = 4
SECTION_LABELS = ("intro", "verse", "prechorus", "chorus", "bridge", "instrumental", "outro", "other")


@dataclass
class Note:
    onset: int
    duration: int
    pitch: int
    lyric: Optional[Tuple[str, ...]] = None
    melisma: bool = False
    line: Optional[int] = None  # lyric line index; prompt-side only, not in ABC

    @property
    def offset(self) -> int:
        return self.onset + self.duration


@dataclass
class Chord:
    onset: int
    duration: int
    symbol: str  # ABC chord symbol, e.g. "Bm7", "F#/A#", "N.C."

    @property
    def offset(self) -> int:
        return self.onset + self.duration


@dataclass
class Section:
    label: str
    start_bar: int
    num_bars: int


@dataclass
class Song:
    song_id: str
    meter_num: int  # nominal beats per bar (beat = quarter note)
    tempo_bpm: int
    key: Optional[str]  # "B major", "G# minor"
    bar_beats: List[int]
    sections: List[Section]
    notes: List[Note]
    chords: List[Chord]
    language: str = "zh"
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- time
    def bar_starts(self) -> List[int]:
        starts, t = [], 0
        for beats in self.bar_beats:
            starts.append(t)
            t += beats * TICKS_PER_BEAT
        return starts

    @property
    def total_ticks(self) -> int:
        return sum(self.bar_beats) * TICKS_PER_BEAT

    @property
    def meter(self) -> str:
        return f"{self.meter_num}/4"

    # ---------------------------------------------------------------- io
    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        for n in d["notes"]:
            if n["lyric"] is not None:
                n["lyric"] = list(n["lyric"])
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Song":
        return cls(
            song_id=d["song_id"],
            meter_num=d["meter_num"],
            tempo_bpm=d["tempo_bpm"],
            key=d.get("key"),
            bar_beats=list(d["bar_beats"]),
            sections=[Section(**s) for s in d["sections"]],
            notes=[
                Note(**{**n, "lyric": tuple(n["lyric"]) if n.get("lyric") is not None else None})
                for n in d["notes"]
            ],
            chords=[Chord(**c) for c in d["chords"]],
            language=d.get("language", "zh"),
            meta=d.get("meta", {}),
        )


# ------------------------------------------------------------- normalizing
def normalize_chords(chords: List[Chord], total_ticks: int) -> List[Chord]:
    """Sort, clip to the song, drop empty segments, merge equal neighbours.

    Chord *duration* is not independent information in a lead sheet: a chord
    lasts until the next one starts. Gaps are filled by extending the
    previous chord, which is how ABC reads chord symbols anyway.
    """
    rows = sorted((c for c in chords if c.onset < total_ticks), key=lambda c: c.onset)
    out: List[Chord] = []
    for c in rows:
        onset = max(0, c.onset)
        if out and out[-1].onset == onset:
            out[-1] = Chord(onset, 0, c.symbol)  # later symbol at same onset wins
            continue
        if out and out[-1].symbol == c.symbol:
            continue
        out.append(Chord(onset, 0, c.symbol))
    for i, c in enumerate(out):
        end = out[i + 1].onset if i + 1 < len(out) else total_ticks
        c.duration = end - c.onset
    return [c for c in out if c.duration > 0]


def comparable(song: Song, with_lines: bool = False) -> Dict[str, Any]:
    """A plain structure for equality checks in round-trip tests."""
    return {
        "meter_num": song.meter_num,
        "tempo_bpm": song.tempo_bpm,
        "key": song.key,
        "bar_beats": list(song.bar_beats),
        "sections": [(s.label, s.start_bar, s.num_bars) for s in song.sections],
        "notes": [
            (n.onset, n.duration, n.pitch, tuple(n.lyric) if n.lyric else None, n.melisma)
            + ((n.line,) if with_lines else ())
            for n in song.notes
        ],
        "chords": [(c.onset, c.duration, c.symbol) for c in normalize_chords(song.chords, song.total_ticks)],
    }
