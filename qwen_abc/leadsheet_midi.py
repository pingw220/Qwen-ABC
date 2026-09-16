"""ABC lead sheet -> the MIDI a MIDI-SAG / MuseControlLite render reads.

MIDI-SAG's adapter (``MIDI-SAG/tools/midi_llm_adapter/``) turns a four-track
lead sheet MIDI into the five things
``MuseControlLite_inference_continuation.py`` conditions on: a melody with
lyrics (sung by SoulX-Singer, then tracked by RMVPE), a chord track, a click
track (the rhythm condition) and section markers (the structure embedding *and*
the 47.55 s generation windows). That adapter was written for MIDI-LLM output;
this module writes the same file from a canonical :class:`Song`, so an ABC lead
sheet renders through the identical pipeline.

Two things are different from a MIDI-LLM lead sheet, both in our favour:

* the click is the score's own bar/beat grid rather than a separate detected
  track, so click, notes, chords and tempo map are one timeline by
  construction (MIDI-LLM's click disagrees with its tempo header by 1-19 BPM);
* section starts are exact bar boundaries, so no section-stitching pile-up can
  occur.

The one adaptation the renderer forces is :func:`plan_segments`: MuseControlLite
generates in 47.55 s windows anchored on section starts, so a section longer
than one window is split into equal parts at bar boundaries. The split parts
keep the section's own label and tag, so the structure condition is unchanged;
only the window plan sees more anchors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .canonical import TICKS_PER_BEAT, Song
from .midi import song_to_midi

# `structure2id` in MuseControlLite_inference_continuation.py -- the structure
# embedding is nn.Embedding(8, 176) over exactly these tags.
MIDI_SAG_TAGS = ("intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus")

# Our section vocabulary (canonical.SECTION_LABELS) -> MIDI-SAG's. Only two are
# not the identity: a pre-chorus is sung like a verse, and MIDI-SAG spells an
# instrumental section "inst". "other" has no counterpart and falls back to the
# adapter's own default.
LABEL_TO_TAG: Dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "prechorus": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "instrumental": "inst",
    "outro": "outro",
    "other": "verse",
}

# Stable Audio Open 1.0 window: 2097152 samples at 44.1 kHz.
SAO_WINDOW_SECONDS = 2097152 / 44100

# General MIDI percussion: bass drum on the downbeat, side stick on other beats.
# Two pitches are what lets the adapter recover downbeats from the click alone.
DOWNBEAT_PITCH = 36
BEAT_PITCH = 37


@dataclass
class Segment:
    """One structure anchor: a section, or one part of a split long section."""

    index: int
    label: str
    tag: str
    start_bar: int
    num_bars: int
    start_tick: int
    start_s: float
    end_s: float
    part: Optional[str] = None  # "2/3" when this is part of a split section

    @property
    def span_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def marker_text(self) -> str:
        """The spelling parse_leadsheet.py reads: "<index>:<label>"."""
        return f"{self.index}:{self.label}"


def seconds_per_tick(song: Song) -> float:
    return 60.0 / max(song.tempo_bpm, 1) / TICKS_PER_BEAT


def plan_segments(
    song: Song, max_segment_seconds: float = SAO_WINDOW_SECONDS - 1.0
) -> List[Segment]:
    """Section anchors for the renderer, splitting anything wider than a window.

    A split is at bar boundaries into the fewest equal parts that each fit, so a
    16-bar chorus at 60 BPM becomes two 8-bar anchors with the same tag rather
    than a window that would truncate it.
    """
    spt = seconds_per_tick(song)
    starts = song.bar_starts()
    total_ticks = song.total_ticks
    if not song.sections:
        return [
            Segment(0, "verse", "verse", 0, len(song.bar_beats), 0, 0.0, total_ticks * spt)
        ]

    ordered = sorted(song.sections, key=lambda s: s.start_bar)
    segments: List[Segment] = []
    for i, sec in enumerate(ordered):
        if sec.start_bar >= len(starts):
            continue
        end_bar = ordered[i + 1].start_bar if i + 1 < len(ordered) else len(song.bar_beats)
        end_bar = min(max(end_bar, sec.start_bar + 1), len(song.bar_beats))
        n_bars = end_bar - sec.start_bar
        start_tick = starts[sec.start_bar]
        end_tick = starts[end_bar] if end_bar < len(starts) else total_ticks
        tag = LABEL_TO_TAG.get(sec.label, "verse")

        span_s = (end_tick - start_tick) * spt
        n_parts = max(1, math.ceil(span_s / max_segment_seconds)) if max_segment_seconds > 0 else 1
        n_parts = min(n_parts, n_bars)  # a part is at least one bar

        cut_bars = [sec.start_bar + round(k * n_bars / n_parts) for k in range(n_parts)] + [end_bar]
        for k in range(n_parts):
            b0, b1 = cut_bars[k], cut_bars[k + 1]
            t0 = starts[b0]
            t1 = starts[b1] if b1 < len(starts) else total_ticks
            segments.append(
                Segment(
                    index=len(segments),
                    label=sec.label,
                    tag=tag,
                    start_bar=b0,
                    num_bars=b1 - b0,
                    start_tick=t0,
                    start_s=t0 * spt,
                    end_s=t1 * spt,
                    part=f"{k + 1}/{n_parts}" if n_parts > 1 else None,
                )
            )
    return segments


def song_to_leadsheet_midi(
    song: Song,
    path: str,
    max_segment_seconds: float = SAO_WINDOW_SECONDS - 1.0,
) -> Dict:
    """Write the lead sheet MIDI and return an audit of what the renderer will see."""
    segments = plan_segments(song, max_segment_seconds=max_segment_seconds)
    song_to_midi(
        song,
        path,
        markers=[(s.start_tick, s.marker_text) for s in segments],
        click=(DOWNBEAT_PITCH, BEAT_PITCH),
    )
    spt = seconds_per_tick(song)
    duration_s = song.total_ticks * spt
    return {
        "path": path,
        "song_id": song.song_id,
        "tempo_bpm": song.tempo_bpm,
        "key": song.key,
        "duration_s": round(duration_s, 3),
        "n_bars": len(song.bar_beats),
        "n_beats": sum(song.bar_beats),
        "n_notes": len(song.notes),
        "n_lyric_notes": sum(1 for n in song.notes if n.lyric),
        "n_chords": len(song.chords),
        "window_seconds": round(SAO_WINDOW_SECONDS, 3),
        "n_sections": len(song.sections),
        "n_segments": len(segments),
        "n_split_sections": len({s.label + str(s.start_bar) for s in segments if s.part}),
        "max_segment_span_s": round(max((s.span_s for s in segments), default=0.0), 3),
        "structure_starts": [round(s.start_s, 6) for s in segments],
        "structure_tags": [s.tag for s in segments],
        "segments": [
            {
                "index": s.index,
                "label": s.label,
                "tag": s.tag,
                "part": s.part,
                "start_bar": s.start_bar,
                "num_bars": s.num_bars,
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "span_s": round(s.span_s, 3),
            }
            for s in segments
        ],
    }


def renderable_problems(song: Song, audit: Dict) -> List[str]:
    """Reasons the renderer would reject or mangle this lead sheet, or []."""
    problems: List[str] = []
    if not song.notes:
        problems.append("no melody notes: SoulX-Singer has nothing to sing")
    if not any(n.lyric for n in song.notes):
        problems.append("no lyrics: every note would be sung as 啦")
    if not song.chords:
        problems.append("no chord track: the chord condition would be empty")
    if audit["duration_s"] < 10.0:
        problems.append(f"duration {audit['duration_s']:.1f}s is too short to render")
    if audit["max_segment_span_s"] >= SAO_WINDOW_SECONDS:
        problems.append(
            f"a segment spans {audit['max_segment_span_s']:.1f}s, past the "
            f"{SAO_WINDOW_SECONDS:.2f}s window"
        )
    onsets = sorted(n.onset for n in song.notes)
    overlaps = sum(
        1
        for a, b in zip(sorted(song.notes, key=lambda n: n.onset), sorted(song.notes, key=lambda n: n.onset)[1:])
        if a.offset > b.onset
    )
    if overlaps:
        problems.append(f"{overlaps} overlapping melody note(s): the melody must be monophonic")
    if len(onsets) != len(set(onsets)):
        problems.append("melody notes share an onset: SoulX-Singer would drop one and its syllable")
    return problems


def summarize(audits: Sequence[Dict]) -> Dict:
    return {
        "n": len(audits),
        "total_duration_s": round(sum(a["duration_s"] for a in audits), 1),
        "total_segments": sum(a["n_segments"] for a in audits),
        "split_sections": sum(a["n_split_sections"] for a in audits),
    }
