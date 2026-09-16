"""Conservative section-boundary cleaning and pathology flags (ABC-v2 data).

The corpus section boundaries (All-In-One + SongPrep, snapped to downbeats) often
fall one syllable early or late, so a lyric line is split across two sections
(REPORT.md §18). This module moves a section boundary to another *existing*
barline only when that clearly repairs the split. Notes, chords, lyrics, bar
lengths and every onset are untouched: only ``Section.start_bar`` /
``num_bars`` change, so musical time is preserved exactly.

A lyric line *crosses* a boundary at tick T when it has sung notes (attacks or
melisma continuations) both before and after T. A crossing is tolerated as a
**pickup** (anacrusis) when all of the line's notes before T lie in the bar just
before T and they carry fewer syllables than the part after T (or at most
``pickup_max_syllables``). Engraving puts the rehearsal mark on the downbeat and
the pickup in the previous bar, so pickups are not errors. Every other crossing
is a **cut**.

Rule per internal boundary, left to right:

1. no cut at the current barline -> keep;
2. candidates are barlines within ``max_shift_bars`` that leave both adjacent
   sections at least ``min_section_bars`` bars (or their original length if
   already shorter) and do not cross a neighbouring boundary;
3. the direction is implied by the cut: a tail spill (the misplaced syllables
   are after the barline) moves the boundary later, a head cut earlier; cuts that
   disagree on direction are left alone. The nearest candidate in that direction
   is taken if every repaired line then lies wholly on one side (turning a cut
   into a "pickup" is not a repair), no other line is cut, no line that was
   intact at the old barline crosses the new one, and no intact sung line lies
   wholly between the two barlines (it would silently change section);
4. a cut is only repaired when its misplaced part is small (at most
   ``repair_max_syllables`` and ``repair_max_fraction`` of the line); a balanced
   split usually means two corpus lyric lines were merged, which gives no evidence
   where the boundary belongs;
5. otherwise keep the original boundary and log the reason
   ("conflicting_directions", "no_cut_free_candidate", "balanced_split_kept").
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .canonical import TICKS_PER_BEAT, Section, Song
from .theory import parse_chord_symbol, parse_key_name

CLEANING_VERSION = "section_boundary_snap_v2"


@dataclass
class CleanConfig:
    max_shift_bars: int = 2
    min_section_bars: int = 2
    pickup_max_syllables: int = 2
    repair_max_syllables: int = 3  # a cut is repaired only if its misplaced part is this small ...
    repair_max_fraction: float = 1 / 3  # ... and at most this share of the line's syllables


@dataclass
class BoundaryDecision:
    index: int  # section index whose start moves
    old_bar: int
    new_bar: int
    cuts_before: int
    cuts_after: int
    action: str  # kept_clean | moved | balanced_split_kept | conflicting_directions | no_cut_free_candidate
    crossings: List[dict] = field(default_factory=list)


def _line_events(song: Song) -> Dict[int, List[Tuple[int, int]]]:
    """line index -> [(onset tick, syllables started on this note)] for sung notes."""
    lines: Dict[int, List[Tuple[int, int]]] = {}
    last_line: Optional[int] = None
    for n in sorted(song.notes, key=lambda n: n.onset):
        if n.lyric:
            last_line = n.line
            if n.line is not None:
                lines.setdefault(n.line, []).append((n.onset, len(n.lyric)))
        elif n.melisma:
            line = n.line if n.line is not None else last_line
            if line is not None:
                lines.setdefault(line, []).append((n.onset, 0))
        else:
            last_line = None
    return lines


def crossings_at(song: Song, bar: int, lines: Dict[int, List[Tuple[int, int]]],
                 cfg: CleanConfig, starts: Optional[List[int]] = None) -> List[dict]:
    starts = starts or song.bar_starts()
    if bar <= 0 or bar >= len(starts):
        return []
    T = starts[bar]
    prev_bar_start = starts[bar - 1]
    out = []
    for line, ev in lines.items():
        before = [e for e in ev if e[0] < T]
        after = [e for e in ev if e[0] >= T]
        if not before or not after:
            continue
        sb = sum(e[1] for e in before)
        sa = sum(e[1] for e in after)
        in_prev_bar = min(e[0] for e in before) >= prev_bar_start
        # a pickup is the *start* of a line: few syllables, all in the previous bar
        pickup = in_prev_bar and sb < sa and (sb <= cfg.pickup_max_syllables or sb * 2 < sa)
        if pickup:
            kind = "pickup"
        elif sa == 0:
            kind = "melisma_spill"  # only continuation notes of the last syllable cross: no text moves
        elif sa < sb:
            kind = "tail"
        else:
            kind = "head_or_mid"
        out.append({"line": line, "syllables_before": sb, "syllables_after": sa, "kind": kind})
    return out


TOLERATED = ("pickup", "melisma_spill")


def _repairable(c: dict, cfg: CleanConfig) -> bool:
    minority = min(c["syllables_before"], c["syllables_after"])
    total = c["syllables_before"] + c["syllables_after"]
    return minority <= cfg.repair_max_syllables and minority <= cfg.repair_max_fraction * total


def count_cuts(crossings: List[dict]) -> int:
    return sum(1 for c in crossings if c["kind"] not in TOLERATED)


def clean_sections(song: Song, cfg: Optional[CleanConfig] = None) -> Tuple[List[Section], List[BoundaryDecision]]:
    """Return cleaned sections (new objects) and one decision per internal boundary."""
    cfg = cfg or CleanConfig()
    starts = song.bar_starts()
    lines = _line_events(song)
    secs = [Section(s.label, s.start_bar, s.num_bars) for s in song.sections]
    total_bars = len(song.bar_beats)
    decisions: List[BoundaryDecision] = []
    for i in range(1, len(secs)):
        b0 = secs[i].start_bar
        cr0 = crossings_at(song, b0, lines, cfg, starts)
        c0 = count_cuts(cr0)
        if c0 == 0:
            decisions.append(BoundaryDecision(i, b0, b0, 0, 0, "kept_clean", cr0))
            continue
        if any(not _repairable(c, cfg) for c in cr0 if c["kind"] not in TOLERATED):
            # a balanced split (e.g. two corpus lines merged into one) gives no evidence where the boundary belongs
            decisions.append(BoundaryDecision(i, b0, b0, c0, c0, "balanced_split_kept", cr0))
            continue
        cut = [c for c in cr0 if c["kind"] not in TOLERATED]
        # a tail spill (few syllables after the barline) moves the boundary later, a head cut earlier
        directions = {+1 if c["syllables_after"] < c["syllables_before"] else -1 for c in cut}
        if len(directions) != 1:
            decisions.append(BoundaryDecision(i, b0, b0, c0, c0, "conflicting_directions", cr0))
            continue
        direction = directions.pop()
        cut_lines = {c["line"] for c in cut}
        prev_start = secs[i - 1].start_bar
        next_start = secs[i + 1].start_bar if i + 1 < len(secs) else total_bars
        min_prev = max(min(cfg.min_section_bars, b0 - prev_start), 1)
        min_cur = max(min(cfg.min_section_bars, next_start - b0), 1)
        chosen = None
        for k in range(1, cfg.max_shift_bars + 1):
            b = b0 + direction * k
            if b - prev_start < min_prev or next_start - b < min_cur:
                break
            cr = crossings_at(song, b, lines, cfg, starts)
            # repaired lines must lie wholly on one side (a pickup or melisma spill is not a repair);
            # other lines may only cross in a tolerated way
            crossed0 = {c["line"] for c in cr0}
            lo, hi = sorted((starts[b0], starts[b]))
            # a sung line lying wholly between the old and new barline would change section: not a repair
            transferred = [line for line, ev in lines.items() if line not in crossed0
                           and sum(e[1] for e in ev) > 0 and all(lo <= e[0] < hi for e in ev)]
            if count_cuts(cr) == 0 and not any(c["line"] in cut_lines and c["kind"] != "melisma_spill" for c in cr) \
                    and {c["line"] for c in cr} <= crossed0 and not transferred:  # never split or move an intact line
                chosen = b
                break
        if chosen is None:
            decisions.append(BoundaryDecision(i, b0, b0, c0, c0, "no_cut_free_candidate", cr0))
            continue
        b = chosen
        secs[i - 1].num_bars = b - prev_start
        secs[i].start_bar = b
        secs[i].num_bars = next_start - b
        decisions.append(BoundaryDecision(i, b0, b, c0, 0, "moved", cr0))
    return secs, decisions


def cut_count(song: Song, cfg: Optional[CleanConfig] = None) -> Tuple[int, int]:
    """(non-pickup cuts, pickup crossings) summed over all internal section boundaries."""
    cfg = cfg or CleanConfig()
    starts = song.bar_starts()
    lines = _line_events(song)
    cuts = pickups = 0
    for s in song.sections[1:]:
        cr = crossings_at(song, s.start_bar, lines, cfg, starts)
        cuts += count_cuts(cr)
        pickups += sum(1 for c in cr if c["kind"] == "pickup")
    return cuts, pickups


# ------------------------------------------------------------------ flags
SCALE = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}


def pathology_flags(song: Song) -> Dict[str, object]:
    """Rule-based quality flags. They never change data; they feed exclusion lists and the clean subset."""
    flags: Dict[str, object] = {}
    notes = song.notes
    nbars = max(len(song.bar_beats), 1)
    beats = max(sum(song.bar_beats), 1)
    lens = [s.num_bars for s in song.sections]
    flags["tiny_sections"] = sum(1 for s in song.sections if s.num_bars <= 1)
    flags["long_sections"] = sum(1 for n in lens if n > 32)
    flags["tempo_bpm"] = song.tempo_bpm
    flags["double_time_suspect"] = song.tempo_bpm >= 180 and len(notes) / beats < 0.55
    flags["half_time_suspect"] = song.tempo_bpm < 60
    flags["irregular_bar_frac"] = round(sum(1 for b in song.bar_beats if b != song.meter_num) / nbars, 4)
    key = parse_key_name(song.key)
    if key and notes:
        tonic, mode = key
        inkey = sum(n.duration for n in notes if (n.pitch - tonic) % 12 in SCALE[mode])
        flags["melody_in_key_frac"] = round(inkey / max(sum(n.duration for n in notes), 1), 4)
        roots = [parse_chord_symbol(c.symbol) for c in song.chords]
        rooted = [(c, r) for c, r in zip(song.chords, roots) if r and r["root_pc"] is not None]
        tot = sum(c.duration for c, _ in rooted)
        flags["chord_root_in_key_frac"] = round(
            sum(c.duration for c, r in rooted if (r["root_pc"] - tonic) % 12 in SCALE[mode]) / tot, 4) if tot else None
    else:
        flags["melody_in_key_frac"] = None
        flags["chord_root_in_key_frac"] = None
    flags["key_agreement"] = song.meta.get("key_agreement")
    flags["notes_per_bar"] = round(len(notes) / nbars, 3)
    sung = sum(len(n.lyric) for n in notes if n.lyric)
    flags["multi_syllable_note_frac"] = round(sum(1 for n in notes if n.lyric and len(n.lyric) > 1) / max(len(notes), 1), 4)
    flags["wordless_note_frac"] = round(sum(1 for n in notes if not n.lyric and not n.melisma) / max(len(notes), 1), 4)
    flags["syllables"] = sung
    # chord-tone agreement of the melody (pseudo-label consistency between two extractors)
    onsets = [c.onset for c in song.chords]
    tone = dur = 0
    for n in notes:
        i = bisect.bisect_right(onsets, n.onset) - 1
        if i < 0 or n.onset >= song.chords[i].offset:
            continue
        info = parse_chord_symbol(song.chords[i].symbol)
        if info and info["root_pc"] is not None:
            dur += n.duration
            tone += n.duration * (n.pitch % 12 in info["pcs"])
    flags["chord_tone_frac"] = round(tone / dur, 4) if dur else None
    pitches = [n.pitch for n in notes]
    flags["pitch_range"] = (max(pitches) - min(pitches)) if pitches else 0
    flags["sections"] = len(song.sections)
    flags["bars"] = len(song.bar_beats)
    return flags


def is_pathological(flags: Dict[str, object]) -> List[str]:
    """Clear-cut pathologies only (used for reporting and the clean subset, not for dropping training songs)."""
    reasons = []
    if flags["double_time_suspect"]:
        reasons.append("double_time_suspect")
    if flags["long_sections"]:
        reasons.append("section_over_32_bars")
    if flags["irregular_bar_frac"] > 0.25:
        reasons.append("irregular_bars_over_25pct")
    if flags["melody_in_key_frac"] is not None and flags["melody_in_key_frac"] < 0.7:
        reasons.append("melody_in_key_under_70pct")
    if flags["multi_syllable_note_frac"] > 0.10:
        reasons.append("aligner_cramming_over_10pct")
    if flags["notes_per_bar"] < 1.0 or flags["notes_per_bar"] > 9.0:
        reasons.append("implausible_note_density")
    if flags["pitch_range"] > 36:
        reasons.append("pitch_range_over_3_octaves")
    return reasons
