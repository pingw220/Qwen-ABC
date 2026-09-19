"""Repair the one pathology the model copies from its targets: notes that carry several syllables.

The corpus puts two or more syllables on a single note in 5.7% of syllables,
because the note↔lyric matcher that produced the labels could not place them all.
The model learned it and does it at three times that rate, which is audible as a
burst of words on one pitch. Telling the model how many syllables a section has
(R3-A) did not stop it, and best-of-n reranking bottoms out at 0.086, so the
targets themselves are what has to change.

:func:`split_crammed_notes` subdivides such a note into one note per syllable,
same pitch, equal durations. That invents rhythm — but only within the note's own
span, it is what a transcriber writes when a singer sings two syllables on one
beat, and it is strictly less invention than leaving a note that cannot be sung
as written. A note too short to subdivide keeps its remaining syllables and is
counted.

Nothing else moves: pitches, onsets, the bar grid, chords, sections and the
lyric line each syllable belongs to are untouched, so the *prompt* built from a
repaired song is byte-identical to the original's.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from .canonical import Note, Song

MIN_SPLIT_TICKS = 1  # one tick is a sixteenth note; nothing shorter exists in the corpus


def split_crammed_notes(song: Song) -> Tuple[Song, Dict[str, int]]:
    """Return a copy whose notes carry at most one syllable each, plus counts.

    A note that spans a section boundary is left alone: splitting it would move
    its later syllables into the next section, which changes the *prompt* as well
    as the target and would cost the experiment its single variable.
    """
    starts = song.bar_starts()
    boundaries = {starts[s.start_bar] for s in song.sections if 0 < s.start_bar < len(starts)}
    stats = {"notes": len(song.notes), "crammed_notes": 0, "notes_added": 0, "syllables_separated": 0,
             "crosses_a_section_boundary": 0,
             # a note can be too short to take every syllable even when it can take some:
             # whatever is left rides on the last piece, and that piece is still crammed
             "notes_too_short_to_split": 0, "notes_still_crammed": 0, "syllables_left_joined": 0}
    out: List[Note] = []
    for n in song.notes:
        k = len(n.lyric) if n.lyric else 0
        if k < 2:
            out.append(n)
            continue
        stats["crammed_notes"] += 1
        if any(n.onset < b < n.offset for b in boundaries):
            stats["crosses_a_section_boundary"] += 1
            stats["notes_still_crammed"] += 1
            stats["syllables_left_joined"] += k - 1
            out.append(n)
            continue
        parts = min(k, max(1, n.duration // MIN_SPLIT_TICKS))
        if parts < 2:
            stats["notes_too_short_to_split"] += 1
            stats["notes_still_crammed"] += 1
            stats["syllables_left_joined"] += k - 1
            out.append(n)
            continue
        if parts < k:
            stats["syllables_left_joined"] += k - parts
            stats["notes_still_crammed"] += 1
        # equal split; the remainder goes to the last piece so the span is exact
        base, extra = divmod(n.duration, parts)
        onset = n.onset
        for i in range(parts):
            dur = base + (extra if i == parts - 1 else 0)
            # the last piece carries any syllables that had no room of their own
            syls = (n.lyric[i],) if i < parts - 1 else tuple(n.lyric[i:])
            out.append(replace(n, onset=onset, duration=dur, lyric=syls, melisma=False))
            onset += dur
        stats["notes_added"] += parts - 1
        stats["syllables_separated"] += parts - 1
    repaired = replace(song, notes=out)
    return repaired, stats


def check_repair(before: Song, after: Song, stats: Dict[str, int]) -> Dict[str, bool]:
    """Everything except the melody's subdivision must be identical.

    Every entry here can fail; a check that cannot is not a check.
    """
    def syllables(song: Song):
        return [(n.line, s) for n in song.notes if n.lyric for s in n.lyric]

    left_joined = sum(1 for n in after.notes if n.lyric and len(n.lyric) > 1)
    return {
        # each syllable keeps its identity, its order and the lyric line it belongs
        # to, which is what makes the prompt come out byte-identical
        "syllables_and_lines_preserved": syllables(before) == syllables(after),
        "total_duration_unchanged": (max((n.offset for n in before.notes), default=0)
                                     == max((n.offset for n in after.notes), default=0)),
        "original_onsets_still_present": {n.onset for n in before.notes} <= {n.onset for n in after.notes},
        "note_count_matches_stats": len(after.notes) == len(before.notes) + stats["notes_added"],
        "notes_sorted_and_non_overlapping": all(a.offset <= b.onset for a, b in zip(after.notes, after.notes[1:])),
        "only_notes_without_room_still_carry_several": left_joined == stats["notes_still_crammed"],
        "melisma_flags_untouched_elsewhere": ([n.melisma for n in before.notes if not n.lyric]
                                              == [n.melisma for n in after.notes if not n.lyric]),
        "bars_unchanged": before.bar_beats == after.bar_beats,
        "chords_unchanged": [(c.onset, c.duration, c.symbol) for c in before.chords]
        == [(c.onset, c.duration, c.symbol) for c in after.chords],
        "sections_unchanged": [(s.label, s.start_bar, s.num_bars) for s in before.sections]
        == [(s.label, s.start_bar, s.num_bars) for s in after.sections],
    }
