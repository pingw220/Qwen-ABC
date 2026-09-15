"""SheetSage-Pro ``leadsheet.json`` (phoneme_leadsheet_v1) -> canonical Song.

Timing: every time in the corpus is mapped onto the song's own beat track
(``beats``, from beat_this), so a note at 17.52 s becomes "beat 33 + 2/4".
Tempo drift therefore costs nothing metrically; only the absolute seconds are
lost (see reports/ABC_SCHEMA.md, "lossy conversions").
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .canonical import TICKS_PER_BEAT, Chord, Note, Section, Song, normalize_chords
from .theory import canonical_key, corpus_chord_to_abc, respell_chord_symbol

LABEL_ALIASES = {  # mirrors MIDI-LLM midi_llm/section_schema.py
    "intro": "intro", "introduction": "intro", "start": "intro", "verse": "verse", "vers": "verse",
    "prechorus": "prechorus", "pre-chorus": "prechorus", "pre_chorus": "prechorus", "prehook": "prechorus",
    "chorus": "chorus", "hook": "chorus", "refrain": "chorus", "bridge": "bridge", "middle8": "bridge",
    "inst": "instrumental", "instrumental": "instrumental", "solo": "instrumental", "break": "instrumental",
    "interlude": "instrumental", "outro": "outro", "end": "outro", "ending": "outro", "fadeout": "outro",
    "silence": "other", "other": "other", "no_label": "other",
}
_LYRIC_STRIP = re.compile(r"[\s\-_*~|\\%\"\[\]{}]+")


def normalize_label(raw: Any) -> str:
    return LABEL_ALIASES.get(str(raw or "").strip().lower(), "other")


def clean_syllable(text: Optional[str]) -> str:
    """Make a syllable safe for an ABC ``w:`` line (no separators inside)."""
    if not text:
        return ""
    return _LYRIC_STRIP.sub("", text.strip().lower())


class BeatGrid:
    """Seconds <-> fractional beat index on a (possibly drifting) beat track."""

    def __init__(self, times: List[float]):
        if len(times) < 2:
            raise ValueError("beat track needs at least two beats")
        self.times = times
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        self.period = gaps[len(gaps) // 2]

    def position(self, t: float) -> float:
        times = self.times
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return (t - times[0]) / self.period
        if i >= len(times) - 1:
            return len(times) - 1 + (t - times[-1]) / self.period
        return i + (t - times[i]) / (times[i + 1] - times[i])

    def tick(self, t: float) -> Tuple[int, float]:
        """Nearest tick and the rounding residual in ticks."""
        raw = self.position(t) * TICKS_PER_BEAT
        q = int(round(raw))
        return q, raw - q


def _alignment_by_note(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Upstream note->syllable links, plus MIDI-LLM's melisma recovery.

    Recovery (identical rule to ``_augment_alignment_with_melisma`` in
    MIDI-LLM ``midi_llm/section_extract.py``): an unlinked note whose onset
    lies inside a syllable's ``[start_sec, end_sec)`` window continues that
    syllable.
    """
    by_note: Dict[str, Dict[str, Any]] = {}
    for item in (data.get("note_lyric_alignment") or {}).get("items", []):
        if item.get("lyric_unit_id") and item.get("note_id"):
            by_note[item["note_id"]] = {
                "unit": item["lyric_unit_id"],
                "cont": bool(item.get("is_lyric_continuation")),
            }
    windows = sorted(
        (float(s["start_sec"]), float(s["end_sec"]), s["syllable_id"])
        for s in data.get("syllables", [])
        if s.get("start_sec") is not None and s.get("end_sec") is not None
    )
    starts = [w[0] for w in windows]
    inherited = 0
    for n in data.get("melody_notes", []):
        if n["note_id"] in by_note:
            continue
        onset = float(n["onset_sec"])
        i = bisect.bisect_right(starts, onset) - 1
        if i >= 0 and windows[i][0] <= onset < windows[i][1]:
            by_note[n["note_id"]] = {"unit": windows[i][2], "cont": True}
            inherited += 1
    return by_note, inherited


def song_from_leadsheet(data: Dict[str, Any], song_id: Optional[str] = None) -> Tuple[Song, Dict[str, Any]]:
    """Convert one corpus lead sheet. Returns (song, conversion statistics)."""
    stats: Counter = Counter()
    song_id = song_id or data["song_id"]
    glob = data.get("global") or {}
    meter = str(glob.get("meter") or "4/4")
    m = re.fullmatch(r"(\d+)/4", meter)
    if not m:
        raise ValueError(f"unsupported meter {meter!r}")
    meter_num = int(m.group(1))

    beats = data["beats"]
    grid = BeatGrid([float(b["time_sec"]) for b in beats])
    last_beat = len(beats) - 1
    total_ticks = last_beat * TICKS_PER_BEAT

    # ---------------------------------------------------------------- bars
    downbeats = [i for i, b in enumerate(beats) if b.get("is_downbeat")]
    boundaries = sorted({0, *[i for i in downbeats if 0 < i < last_beat], last_beat})

    # sections: snap each start to the nearest beat, force a barline there
    raw_sections = sorted(data.get("sections") or [], key=lambda s: float(s["start_sec"]))
    section_beats: List[Tuple[int, str]] = []
    for s in raw_sections:
        b = int(round(grid.position(float(s["start_sec"]))))
        b = min(max(b, 0), last_beat)
        label = normalize_label(s.get("label"))
        if section_beats and section_beats[-1][0] == b:
            section_beats[-1] = (b, label)
            stats["section_collapsed"] += 1
            continue
        if b >= last_beat:
            stats["section_after_end_dropped"] += 1
            continue
        section_beats.append((b, label))
    if not section_beats:
        section_beats = [(0, "other")]
    elif section_beats[0][0] != 0:  # the first section absorbs any lead-in
        section_beats[0] = (0, section_beats[0][1])
        stats["first_section_extended_to_start"] += 1
    for b, _ in section_beats:
        if b not in boundaries:
            stats["section_start_off_downbeat"] += 1
            bisect.insort(boundaries, b)
    stats["irregular_bars_raw"] = sum(1 for a, b in zip(boundaries, boundaries[1:]) if b - a != meter_num)
    boundaries = _regularize_bars(boundaries, {b for b, _ in section_beats}, meter_num, stats)
    bar_beats = [b - a for a, b in zip(boundaries, boundaries[1:])]
    bar_index_of_beat = {b: i for i, b in enumerate(boundaries)}
    sections: List[Section] = []
    for i, (b, label) in enumerate(section_beats):
        start_bar = bar_index_of_beat[b]
        end_bar = bar_index_of_beat[section_beats[i + 1][0]] if i + 1 < len(section_beats) else len(bar_beats)
        sections.append(Section(label, start_bar, end_bar - start_bar))
    stats["bars"] = len(bar_beats)
    stats["irregular_bars"] = sum(1 for x in bar_beats if x != meter_num)

    # ------------------------------------------------------------- lyrics
    syllables = {s["syllable_id"]: s for s in data.get("syllables", [])}
    line_index = {l["line_id"]: i for i, l in enumerate(data.get("lines", []))}
    by_note, inherited = _alignment_by_note(data)
    stats["melisma_inherited"] = inherited

    # ------------------------------------------------------------- melody
    rows = []
    for n in data.get("melody_notes", []):
        on_t = n.get("onset_grid_time_sec", n["onset_sec"])
        off_t = n.get("offset_grid_time_sec", n["offset_sec"])
        on, r1 = grid.tick(float(on_t))
        off, r2 = grid.tick(float(off_t))
        stats["onset_residual_gt_0.25tick"] += int(abs(r1) > 0.25)
        rows.append((on, off, int(n["pitch_midi"]), n["note_id"]))
    rows.sort(key=lambda r: (r[0], r[1]))

    notes: List[Note] = []
    note_units: List[Optional[Dict[str, Any]]] = []
    for on, off, pitch, note_id in rows:
        on = min(max(on, 0), total_ticks - 1)
        off = min(off, total_ticks)
        if notes and on == notes[-1].onset:
            stats["note_same_onset_dropped"] += 1
            continue
        if notes and on < notes[-1].offset:
            notes[-1].duration = on - notes[-1].onset
            stats["note_overlap_truncated"] += 1
        if off <= on:
            off = on + 1
            stats["note_zero_duration_extended"] += 1
            if off > total_ticks:
                stats["note_after_end_dropped"] += 1
                continue
        notes.append(Note(on, off - on, pitch))
        note_units.append(by_note.get(note_id))
    stats["notes"] = len(notes)

    # attach lyrics in note order
    attacked: Dict[str, int] = {}  # syllable_id -> note index of its attack
    last_unit: Optional[str] = None
    for i, (note, link) in enumerate(zip(notes, note_units)):
        if not link or link["unit"] not in syllables:
            last_unit = None  # a wordless note ends any melisma run
            continue
        unit = link["unit"]
        text = clean_syllable(syllables[unit].get("normalized_text") or syllables[unit].get("text"))
        if not text:
            stats["syllable_empty_after_cleaning"] += 1
            last_unit = None
            continue
        if unit == last_unit:
            note.melisma = True
            note.line = notes[attacked[unit]].line
            stats["melisma_notes"] += 1
            if not link["cont"]:
                stats["repeat_attack_as_melisma"] += 1
            continue
        if unit in attacked:
            stats["noncontiguous_repeat_as_wordless"] += 1
            continue
        if link["cont"]:
            stats["continuation_promoted_to_attack"] += 1
        note.lyric = (text,)
        note.line = line_index.get(syllables[unit].get("line_id"))
        attacked[unit] = i
        last_unit = unit
    _enforce_melisma_contiguity(notes, stats)

    # unmatched syllables: join onto a neighbour attack in the same line
    unit_to_note = dict(attacked)
    all_units = [s["syllable_id"] for s in data.get("syllables", [])]
    stats["syllables"] = len(all_units)
    pending_before: Dict[int, List[str]] = {}
    for pos, unit in enumerate(all_units):
        if unit in unit_to_note:
            continue
        text = clean_syllable(syllables[unit].get("normalized_text") or syllables[unit].get("text"))
        if not text:
            continue
        line = syllables[unit].get("line_id")
        prev = next((all_units[j] for j in range(pos - 1, -1, -1) if all_units[j] in unit_to_note), None)
        nxt = next((all_units[j] for j in range(pos + 1, len(all_units)) if all_units[j] in unit_to_note), None)
        if prev is not None and syllables[prev].get("line_id") == line:
            idx = unit_to_note[prev]
            notes[idx].lyric = notes[idx].lyric + (text,)
            stats["unmatched_syllable_joined"] += 1
        elif nxt is not None and syllables[nxt].get("line_id") == line:
            pending_before.setdefault(unit_to_note[nxt], []).append(text)
            stats["unmatched_syllable_joined"] += 1
        else:
            stats["unmatched_syllable_dropped"] += 1
    for idx, texts in pending_before.items():
        notes[idx].lyric = tuple(texts) + notes[idx].lyric

    # ------------------------------------------------------------- chords
    key = canonical_key(glob.get("key"))
    chords: List[Chord] = []
    for c in data.get("recognized_chords", []):
        symbol, known = corpus_chord_to_abc(c.get("symbol", "N"), c.get("root"), c.get("quality"), c.get("bass"))
        symbol = respell_chord_symbol(symbol, key)  # enharmonic spelling follows the key signature
        stats["chord_unrecognized_quality"] += int(not known)
        on, _ = grid.tick(float(c["start_sec"]))
        chords.append(Chord(min(max(on, 0), total_ticks), 0, symbol))
    chords = normalize_chords(chords, total_ticks)
    stats["chords"] = len(chords)

    tempo = glob.get("tempo_bpm")
    tempo_bpm = int(round(float(tempo))) if tempo else int(round(60.0 / grid.period))
    lines_text = [l.get("normalized_text") or l.get("text") or "" for l in data.get("lines", [])]
    song = Song(
        song_id=song_id,
        meter_num=meter_num,
        tempo_bpm=tempo_bpm,
        key=key,
        bar_beats=bar_beats,
        sections=sections,
        notes=notes,
        chords=chords,
        language=str(glob.get("language") or data.get("language") or "zh"),
        meta={"key_agreement": glob.get("key_agreement"), "source_key": glob.get("key"),
              "num_source_lines": len(lines_text)},
    )
    return song, dict(stats)


def _regularize_bars(boundaries: List[int], anchors: set, meter_num: int, stats: Counter) -> List[int]:
    """Remove spurious downbeats between trusted ones.

    beat_this often places an extra downbeat mid-bar (2 + 2 beats in 4/4) or
    misses one (an 8-beat bar). A maximal run of consecutive off-length bars
    whose total is a whole number of nominal bars is re-barred into nominal
    bars; a run of two or more bars with another total becomes nominal bars
    plus one shorter final bar. The run's first and last boundaries (ordinary downbeats, section
    starts or song edges) stay, so every beat keeps its position and only
    barline placement inside the run changes. Runs never cross a section start.
    """
    out = [boundaries[0]]
    i = 0
    n = len(boundaries) - 1  # number of bars
    while i < n:
        length = boundaries[i + 1] - boundaries[i]
        if length == meter_num:
            out.append(boundaries[i + 1])
            i += 1
            continue
        j = i
        while j < n and boundaries[j + 1] - boundaries[j] != meter_num and \
                (j == i or boundaries[j] not in anchors):
            j += 1
        start, end = boundaries[i], boundaries[j]
        total = end - start
        if total % meter_num == 0:  # every bar in the run is off-length, so this always changes it
            out.extend(range(start + meter_num, end + 1, meter_num))
            stats["irregular_runs_regularized"] += 1
        elif j - i >= 2 and total > meter_num:
            # e.g. nine 2-beat "bars" before a section start: pack whole bars from
            # the run's start and leave the remainder as the run's last bar
            out.extend(range(start + meter_num, end, meter_num))
            out.append(end)
            stats["irregular_runs_packed_with_remainder"] += 1
        else:
            out.extend(boundaries[i + 1: j + 1])
        i = j
    return out


def _enforce_melisma_contiguity(notes: List[Note], stats: Counter) -> None:
    """A melisma note must directly follow its syllable's notes."""
    active = False
    for n in notes:
        if n.lyric:
            active = True
        elif n.melisma:
            if not active:
                n.melisma = False
                stats["orphan_melisma_as_wordless"] += 1
        else:
            active = False
