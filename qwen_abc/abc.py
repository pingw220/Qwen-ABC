"""Canonical Song <-> lyrics-aware ABC text. The format is specified in reports/ABC_SCHEMA.md.

Writer guarantees (checked by tests and by the dataset validator):
* ``parse_abc(song_to_abc(song))`` reproduces ``comparable(song)`` exactly;
* every bar's written content equals its declared meter;
* accidentals are unambiguous under both ABC propagation conventions.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .canonical import TICKS_PER_BEAT, Chord, Note, Section, Song, normalize_chords
from .theory import (
    NATURAL_PC,
    abc_note_name,
    abc_to_key,
    key_signature,
    key_to_abc,
    parse_chord_symbol,
    spell_pitch,
)

UNIT_TICKS = 2  # L:1/8 with TICKS_PER_BEAT = 4 (sixteenth-note ticks)
WHOLE_NOTE_TICKS = 16
ACC_TEXT = {1: "^", -1: "_", 0: "=", 2: "^^", -2: "__"}
ACC_VALUE = {"^": 1, "_": -1, "=": 0, "^^": 2, "__": -2}


# =================================================================== writer
def _duration_text(ticks: int) -> str:
    if ticks % UNIT_TICKS == 0:
        n = ticks // UNIT_TICKS
        return "" if n == 1 else str(n)
    return "/" if ticks == 1 else f"{ticks}/"


def _lyric_token(note: Note) -> str:
    if note.lyric:
        return "~".join(note.lyric)
    return "_" if note.melisma else "*"


class _BarSpeller:
    def __init__(self, key: Optional[str]):
        self.key = key
        self.sig = key_signature(key)
        self.reset()

    def reset(self) -> None:
        self.by_octave: Dict[Tuple[str, int], int] = {}
        self.by_letter: Dict[str, int] = {}

    def token(self, pitch: int) -> str:
        letter, alter, octave = spell_pitch(pitch, self.key)
        eff_oct = self.by_octave.get((letter, octave), self.sig[letter])
        eff_letter = self.by_letter.get(letter, self.sig[letter])
        acc = ""
        if eff_oct != alter or eff_letter != alter:
            acc = ACC_TEXT[alter]
            self.by_octave[(letter, octave)] = alter
            self.by_letter[letter] = alter
        return acc + abc_note_name(letter, octave)


def song_to_abc(song: Song, bars_per_line: int = 4, bar_prefix=None, section_header=None) -> str:
    """ABC-v1 text. ``bar_prefix(sec_idx, n_sections, bar_in_sec, sec_bars, line_start)`` and
    ``section_header(sec_idx, n_sections, section)`` are ABC-v2 hooks (see abc_v2.py); with
    both None the output is exactly the v1 format."""
    lines = [
        "X:1",
        f"M:{song.meter_num}/4",
        "L:1/8",
        f"Q:1/4={song.tempo_bpm}",
        f"K:{key_to_abc(song.key)}",
    ]
    notes = sorted(song.notes, key=lambda n: n.onset)
    total = song.total_ticks
    chords = normalize_chords(song.chords, total)
    chord_onsets = [c.onset for c in chords]
    chord_at = {c.onset: c.symbol for c in chords}
    bar_starts = song.bar_starts()
    speller = _BarSpeller(song.key)
    current_meter = song.meter_num
    ni = 0

    n_sections = len(song.sections)
    for si, sec in enumerate(song.sections):
        lines.append(f"P:{sec.label}")
        if section_header is not None:
            lines.append(section_header(si, n_sections, sec))
        bars = list(range(sec.start_bar, sec.start_bar + sec.num_bars))
        for chunk_start in range(0, len(bars), bars_per_line):
            chunk = bars[chunk_start: chunk_start + bars_per_line]
            music: List[str] = []
            lyric_bars: List[List[str]] = []
            for ci, b in enumerate(chunk):
                speller.reset()
                start = bar_starts[b]
                end = start + song.bar_beats[b] * TICKS_PER_BEAT
                tokens: List[str] = []
                lyr: List[str] = []
                if bar_prefix is not None:
                    prefix = bar_prefix(si, n_sections, chunk_start + ci, sec.num_bars, ci == 0)
                    if prefix:
                        tokens.append(prefix)
                if song.bar_beats[b] != current_meter:
                    current_meter = song.bar_beats[b]
                    tokens.append(f"[M:{current_meter}/4]")
                t = start
                while t < end:
                    while ni < len(notes) and notes[ni].offset <= t:
                        ni += 1
                    ci = bisect.bisect_right(chord_onsets, t)
                    next_chord = chord_onsets[ci] if ci < len(chord_onsets) else total
                    note = notes[ni] if ni < len(notes) and notes[ni].onset <= t else None
                    if note is not None:
                        seg_end = min(note.offset, next_chord, end)
                        text = speller.token(note.pitch) + _duration_text(seg_end - t)
                        if seg_end < note.offset:
                            text += "-"
                        if t == note.onset:
                            lyr.append(_lyric_token(note))
                    else:
                        next_on = notes[ni].onset if ni < len(notes) else total
                        seg_end = min(next_on, next_chord, end)
                        text = "z" + _duration_text(seg_end - t)
                    if t in chord_at:
                        text = f'"{chord_at[t]}"' + text
                    tokens.append(text)
                    t = seg_end
                while lyr and lyr[-1] == "*":
                    lyr.pop()
                music.append(" ".join(tokens) + " |")
                lyric_bars.append(lyr)
            lines.append(" ".join(music))
            last = max((i for i, seg in enumerate(lyric_bars) if seg), default=-1)
            if last >= 0:
                words: List[str] = []
                for i in range(last + 1):
                    words.extend(lyric_bars[i])
                    if i < last:
                        words.append("|")
                lines.append("w: " + " ".join(words))
    return "\n".join(lines) + "\n"


# =================================================================== parser
@dataclass
class ParseResult:
    song: Optional[Song]
    errors: Counter = field(default_factory=Counter)  # structural problems
    warnings: Counter = field(default_factory=Counter)  # tolerated deviations
    bar_ticks: List[int] = field(default_factory=list)  # content actually written per bar
    declared_beats: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.song is not None

    @property
    def strict_ok(self) -> bool:
        return self.song is not None and not self.errors


_MUSIC_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
   |(?P<inline>\[[A-Za-z]:[^\]]*\])
   |(?P<chord>"[^"]*")
   |(?P<bar>\|\]|\|\||\[\||:\||\|:|::|\|)
   |(?P<note>(?P<acc>\^\^|\^|__|_|=)?(?P<letter>[A-Ga-g])(?P<oct>[',]*)(?P<dur>\d*/*\d*)(?P<tie>-)?)
   |(?P<rest>[zx](?P<rdur>\d*/*\d*))
   |(?P<other>.)
    """,
    re.VERBOSE,
)
_FIELD_LINE = re.compile(r"^([A-Za-z]):(.*)$")


def _parse_duration(text: str, unit_ticks: Fraction) -> Tuple[int, bool]:
    m = re.fullmatch(r"(\d*)(/*)(\d*)", text)
    num = int(m.group(1)) if m.group(1) else 1
    slashes = len(m.group(2))
    if m.group(3):
        den = int(m.group(3)) * (2 ** max(slashes - 1, 0))
    else:
        den = 2 ** slashes
    if den == 0:
        return 0, False
    value = Fraction(num, den) * unit_ticks
    on_grid = value.denominator == 1
    return max(int(round(value)), 0), on_grid


def _meter_beats(text: str) -> Optional[Fraction]:
    text = text.strip()
    if text in ("C", "C|"):
        return Fraction(4)
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if not m or int(m.group(2)) == 0:
        return None
    return Fraction(int(m.group(1)) * 4, int(m.group(2)))  # in quarter-note beats


def parse_abc(text: str, song_id: str = "parsed") -> ParseResult:
    res = ParseResult(song=None)
    err, warn = res.errors, res.warnings

    meter_beats: Fraction = Fraction(4)
    nominal: Optional[Fraction] = None
    unit_ticks = Fraction(UNIT_TICKS)
    tempo: Optional[int] = None
    key: Optional[str] = None
    in_body = False

    notes: List[Note] = []
    chords: List[Chord] = []
    sections: List[Tuple[str, int]] = []
    pending_section: Optional[str] = None
    bar_start_tick = 0
    t = 0
    bar_has_content = False
    by_octave: Dict[Tuple[str, int], int] = {}
    sig = key_signature(None)
    tie_from: Optional[int] = None
    line_items: List[Tuple[str, int]] = []  # ("note", idx) | ("bar", 0) for the last music line
    lyrics_applied = True

    def close_bar() -> None:
        nonlocal bar_start_tick, t, bar_has_content, by_octave
        content = t - bar_start_tick
        declared = meter_beats * TICKS_PER_BEAT
        res.bar_ticks.append(content)
        res.declared_beats.append(int(meter_beats) if meter_beats.denominator == 1 else -1)
        if content != declared:
            err["bar_duration_mismatch"] += 1
        if content % TICKS_PER_BEAT:
            pad = TICKS_PER_BEAT - content % TICKS_PER_BEAT
            t += pad
            warn["bar_padded_to_whole_beat"] += 1
        bar_start_tick = t
        bar_has_content = False
        by_octave = {}

    def start_bar_if_needed() -> None:
        nonlocal pending_section
        if not bar_has_content and pending_section is not None:
            sections.append((pending_section, len(res.bar_ticks)))
            pending_section = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        fm = _FIELD_LINE.match(line)
        if fm:
            name, value = fm.group(1), fm.group(2).strip()
            if name == "w":
                if not in_body:
                    err["lyrics_before_body"] += 1
                    continue
                if lyrics_applied:
                    err["lyrics_without_music_line"] += 1
                    continue
                _apply_lyrics(value, line_items, notes, err)
                lyrics_applied = True
                continue
            if name == "W":
                continue
            if name == "M":
                mb = _meter_beats(value)
                if mb is None:
                    err["bad_meter"] += 1
                else:
                    if bar_has_content:
                        err["meter_change_mid_bar"] += 1
                    meter_beats = mb
                    nominal = nominal if nominal is not None else mb
            elif name == "L":
                mm = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
                if mm and int(mm.group(2)):
                    unit_ticks = Fraction(WHOLE_NOTE_TICKS * int(mm.group(1)), int(mm.group(2)))
                else:
                    err["bad_unit_length"] += 1
            elif name == "Q":
                qm = re.search(r"(?:(\d+)/(\d+)\s*=\s*)?(\d+(?:\.\d+)?)\s*$", value)
                if qm:
                    bpm = float(qm.group(3))
                    if qm.group(1):  # convert to quarter-note bpm
                        bpm *= Fraction(int(qm.group(1)), int(qm.group(2))) * 4
                    tempo = int(round(bpm))
                else:
                    err["bad_tempo"] += 1
            elif name == "K":
                key = abc_to_key(value)
                if key is None and value.strip().lower() not in ("none", ""):
                    err["bad_key"] += 1
                sig = key_signature(key)
                in_body = True
            elif name == "P":
                if in_body:
                    label = value.strip().lower()
                    if bar_has_content:
                        err["section_mid_bar"] += 1
                    pending_section = label
            elif name == "X" or name == "T" or name in "CNOSZRBDFGHIm":
                pass
            else:
                warn[f"ignored_field_{name}"] += 1
            continue
        if not in_body:
            err["music_before_K"] += 1
            in_body = True

        # ------------------------------------------------ a music line
        line_items = []
        lyrics_applied = False
        for m in _MUSIC_TOKEN.finditer(line):
            kind = m.lastgroup
            if kind == "ws":
                continue
            if kind == "inline":
                inner = m.group("inline")[1:-1]
                name, value = inner[0], inner[2:]
                if name == "M":
                    mb = _meter_beats(value)
                    if mb is None:
                        err["bad_meter"] += 1
                    else:
                        if bar_has_content:
                            err["meter_change_mid_bar"] += 1
                        meter_beats = mb
                elif name == "K":
                    key = abc_to_key(value)
                    sig = key_signature(key)
                elif name == "P":
                    pending_section = value.strip().lower()
                elif name == "L":
                    mm = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value.strip())
                    if mm and int(mm.group(2)):
                        unit_ticks = Fraction(WHOLE_NOTE_TICKS * int(mm.group(1)), int(mm.group(2)))
                continue
            if kind == "chord":
                symbol = m.group("chord")[1:-1]
                if symbol[:1] in ("^", "_", "<", ">", "@"):
                    warn["annotation_ignored"] += 1
                    continue
                start_bar_if_needed()
                if parse_chord_symbol(symbol) is None:
                    err["invalid_chord_symbol"] += 1
                if chords and chords[-1].onset == t:
                    chords[-1] = Chord(t, 0, symbol)
                else:
                    chords.append(Chord(t, 0, symbol))
                continue
            if kind == "bar":
                if bar_has_content or t > bar_start_tick:
                    close_bar()
                    line_items.append(("bar", 0))
                else:
                    warn["empty_bar_ignored"] += 1
                continue
            if kind == "note":
                start_bar_if_needed()
                letter = m.group("letter")
                octave = 4 if letter.isupper() else 5
                octave += m.group("oct").count("'") - m.group("oct").count(",")
                L = letter.upper()
                if m.group("acc"):
                    alter = ACC_VALUE[m.group("acc")]
                    by_octave[(L, octave)] = alter
                else:
                    alter = by_octave.get((L, octave), sig[L])
                pitch = 12 * (octave + 1) + NATURAL_PC[L] + alter
                dur, on_grid = _parse_duration(m.group("dur"), unit_ticks)
                if not on_grid:
                    err["off_grid_duration"] += 1
                if dur <= 0:
                    err["zero_duration"] += 1
                    continue
                bar_has_content = True
                if tie_from is not None:
                    prev = notes[tie_from]
                    if prev.pitch == pitch and prev.offset == t:
                        prev.duration += dur
                        t += dur
                        tie_from = tie_from if m.group("tie") else None
                        continue
                    err["broken_tie"] += 1
                    tie_from = None
                notes.append(Note(t, dur, pitch))
                line_items.append(("note", len(notes) - 1))
                if m.group("tie"):
                    tie_from = len(notes) - 1
                t += dur
                continue
            if kind == "rest":
                start_bar_if_needed()
                dur, on_grid = _parse_duration(m.group("rdur"), unit_ticks)
                if not on_grid:
                    err["off_grid_duration"] += 1
                if tie_from is not None:
                    err["broken_tie"] += 1
                    tie_from = None
                bar_has_content = True
                t += dur
                continue
            err["unsupported_token"] += 1

    if bar_has_content or t > bar_start_tick:
        err["missing_final_barline"] += 1
        close_bar()
    if tie_from is not None:
        err["dangling_tie"] += 1

    if not res.bar_ticks:
        err["no_bars"] += 1
        return res

    bar_beats = [0] * len(res.bar_ticks)
    for i, content in enumerate(res.bar_ticks):
        bar_beats[i] = -(-content // TICKS_PER_BEAT)
    if any(b == 0 for b in bar_beats):
        err["empty_bar"] += sum(1 for b in bar_beats if b == 0)
        # keep indices stable: an empty bar still occupies zero beats
    if pending_section is not None:
        warn["trailing_section_label"] += 1
    if not sections or sections[0][1] != 0:
        sections.insert(0, ("other", 0))
        warn["no_leading_section_label"] += 1
    merged: List[Tuple[str, int]] = []
    for label, start in sections:
        if merged and merged[-1][1] == start:
            merged[-1] = (label, start)
            err["empty_section"] += 1
        else:
            merged.append((label, start))
    section_objs = [
        Section(label, start, (merged[i + 1][1] if i + 1 < len(merged) else len(bar_beats)) - start)
        for i, (label, start) in enumerate(merged)
    ]
    nominal_num = nominal if nominal is not None else Fraction(4)
    total_ticks = sum(bar_beats) * TICKS_PER_BEAT
    song = Song(
        song_id=song_id,
        meter_num=int(nominal_num) if nominal_num.denominator == 1 else 4,
        tempo_bpm=tempo if tempo is not None else 120,
        key=key,
        bar_beats=bar_beats,
        sections=section_objs,
        notes=notes,
        chords=normalize_chords(chords, total_ticks),
    )
    if tempo is None:
        err["missing_tempo"] += 1
    if key is None:
        warn["missing_key"] += 1
    res.song = song
    return res


def _lyric_units(text: str) -> List[Tuple[str, str]]:
    """Split a w: body into ("syl", text) | ("hold",) | ("skip",) | ("bar",)."""
    out: List[Tuple[str, str]] = []
    for chunk in text.split():
        buf = ""
        i = 0
        while i < len(chunk):
            ch = chunk[i]
            if ch == "\\" and i + 1 < len(chunk):
                buf += chunk[i + 1]
                i += 2
                continue
            if ch in "|*_-":
                if buf:
                    out.append(("syl", buf))
                    buf = ""
                if ch == "|":
                    out.append(("bar", ""))
                elif ch == "*":
                    out.append(("skip", ""))
                elif ch == "_":
                    out.append(("hold", ""))
                i += 1
                continue
            buf += ch
            i += 1
        if buf:
            out.append(("syl", buf))
    return out


def _apply_lyrics(text: str, items: List[Tuple[str, int]], notes: List[Note], err: Counter) -> None:
    p = 0
    for kind, value in _lyric_units(text):
        if kind == "bar":
            while p < len(items) and items[p][0] != "bar":
                p += 1
            if p >= len(items):
                err["lyric_bar_overflow"] += 1
            p += 1
            continue
        while p < len(items) and items[p][0] == "bar":
            p += 1
        if p >= len(items):
            err["lyric_overflow"] += 1
            continue
        idx = items[p][1]
        p += 1
        note = notes[idx]
        if kind == "syl":
            parts = tuple(x for x in value.split("~") if x)
            note.lyric = parts or None
        elif kind == "hold":
            if idx == 0 or not (notes[idx - 1].lyric or notes[idx - 1].melisma):
                err["orphan_melisma"] += 1
            else:
                note.melisma = True
