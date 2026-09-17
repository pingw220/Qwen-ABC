"""Canonical Song -> FastSinger inputs (melody MIDI + note-aligned lyric text).

FastSinger (``music_acc/fastsinger/inference.py``) is the singing synthesizer
whose voice we use for listening samples. ``midi_lyric_to_jlines`` pairs the two
input files **by position only**: symbol k sings note k of
``pretty_midi.PrettyMIDI(midi).instruments[0].notes``, with no timing link at
all. Too many symbols and it slices the tail off; too few and it pads ``#`` at
the very end, so a single missing symbol shifts every later word onto the wrong
note. Both files are therefore built here from one note list.

The text format, as ``midi_lyric_to_jlines`` reads it: ``，`` is deleted,
newlines separate phrases, and every remaining symbol takes one note -- a
Chinese character sings, ``#`` holds the previous syllable (a melisma).

What is left out, and why:

* wordless notes (an instrumental line, not the vocal) are not sung at all, so
  they are absent from both files -- if they stayed, FastSinger would sing the
  next word on them;
* a note carrying more than one syllable (the corpus joins an unmatched
  syllable onto its neighbour) is divided between them, since one note sings
  one symbol -- but only as far as each syllable still lasts ``MIN_SYLLABLE_S``;
  the rest are dropped, because a lead sheet that asks for 19 syllables inside
  0.19 s is asking for a burst of clicks, not words;
* non-Chinese syllables: FastSinger's Mandarin lexicon has no pronunciation for
  them, so they are sung as a held vowel instead of derailing the line.

Every one of these is counted in the returned stats, never done silently.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import mido

from .canonical import TICKS_PER_BEAT, Song
from .midi import PPQ

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

MELISMA = "#"
MIN_NOTE_S = 0.03  # shorter than this cannot carry a phoneme
# A syllable needs about this long to be sung as a word rather than a click.
# It matters because a lead sheet can pile many syllables onto one note (the
# ABC `w:` line holding more syllables than the bar has notes leaves the
# remainder on the last note): 19 syllables on a 0.19 s note were measured.
# Dividing that note by the note floor alone sings a machine-gun burst, so
# syllables that have no room at this length are dropped and counted instead.
MIN_SYLLABLE_S = 0.12
LINE_GAP_S = 0.6  # a rest this long is a breath: start a new phrase
HOLD_GAP_S = 1.0  # a melisma note further than this from its syllable is not a held vowel
# FastSinger's own comment: notes should sit inside C#3..D5.
COMFORTABLE_RANGE = (49, 74)


def _syllables(lyric: Optional[Tuple[str, ...]]) -> Tuple[List[str], int]:
    """(Chinese syllables on this note, count of syllables with no Chinese)."""
    if not lyric:
        return [], 0
    return [c for s in lyric for c in CJK.findall(s)], sum(1 for s in lyric if not CJK.search(s))


def sung_notes(song: Song, line_gap_s: float = LINE_GAP_S) -> Tuple[List[Dict], Dict]:
    """The vocal line as ``[{start, end, pitch, symbol, line}]`` plus stats.

    Onsets are never moved; ends are clamped to the next onset so the line is
    monophonic, which is what the synthesizer's timeline assumes.
    """
    spt = 60.0 / max(song.tempo_bpm, 1) / TICKS_PER_BEAT
    stats = {
        "notes_in_song": len(song.notes),
        "notes_split": 0,
        "wordless_dropped": 0,
        "extra_syllables_dropped": 0,
        "non_chinese_syllables": 0,
        "melisma_without_syllable_dropped": 0,
        "short_notes_dropped": 0,
        "out_of_comfortable_range": 0,
    }
    ordered = sorted(song.notes, key=lambda n: (n.onset, n.pitch))
    rows: List[Dict] = []
    singing = False  # a syllable has been attacked and can still be held
    last_end: Optional[float] = None
    line = 0

    for i, n in enumerate(ordered):
        chars, non_chinese = _syllables(n.lyric)
        stats["non_chinese_syllables"] += non_chinese
        start = n.onset * spt
        end = n.offset * spt
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1].onset * spt)

        # The corpus joins a syllable it could not place onto its neighbour, so
        # a note can carry several. One note sings one symbol, so the note is
        # divided between them rather than losing all but the first.
        if len(chars) > 1:
            stats["notes_split"] += 1
            room = max(1, int((end - start) // MIN_SYLLABLE_S))
            if room < len(chars):
                stats["extra_syllables_dropped"] += len(chars) - room
                chars = chars[:room]
            step = (end - start) / len(chars)
            for k, c in enumerate(chars):
                s0 = start + k * step
                if rows and last_end is not None and k == 0 and (s0 - last_end) > line_gap_s:
                    line += 1
                rows.append({"start": s0, "end": s0 + step, "pitch": n.pitch, "symbol": c, "line": line})
                last_end = s0 + step
            singing = True
            continue

        symbol = chars[0] if chars else None
        if symbol is None:
            # A held vowel only if something is being sung and the rest before
            # it is short enough to be one breath.
            held = singing and (n.melisma or non_chinese) and last_end is not None \
                and start - last_end <= HOLD_GAP_S
            if not held:
                stats["melisma_without_syllable_dropped" if n.melisma else "wordless_dropped"] += 1
                if not n.melisma:
                    singing = False
                continue
            symbol = MELISMA
        else:
            singing = True

        if end - start < MIN_NOTE_S:
            stats["short_notes_dropped"] += 1
            continue
        if not COMFORTABLE_RANGE[0] <= n.pitch <= COMFORTABLE_RANGE[1]:
            stats["out_of_comfortable_range"] += 1

        if rows and last_end is not None and (start - last_end) > line_gap_s:
            line += 1
        rows.append({"start": start, "end": end, "pitch": n.pitch, "symbol": symbol, "line": line})
        last_end = end

    # A phrase that opens on a held vowel has nothing to hold: drop that head,
    # then renumber so the lines stay consecutive.
    kept: List[Dict] = []
    for row in rows:
        starts_line = not kept or kept[-1]["line"] != row["line"]
        if starts_line and row["symbol"] == MELISMA:
            stats["melisma_without_syllable_dropped"] += 1
            continue
        kept.append(row)
    renumber = {old: new for new, old in enumerate(dict.fromkeys(r["line"] for r in kept))}
    for row in kept:
        row["line"] = renumber[row["line"]]

    stats["notes_sung"] = len(kept)
    stats["lines"] = len(renumber)
    stats["symbols"] = len(kept)
    return kept, stats


def write_fastsinger_inputs(
    song: Song, midi_path: str, txt_path: str, line_gap_s: float = LINE_GAP_S
) -> Dict:
    """Write the melody MIDI and lyric text FastSinger reads, and return stats."""
    rows, stats = sung_notes(song, line_gap_s=line_gap_s)

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ, charset="utf-8")
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="melody", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(song.tempo_bpm, 1)), time=0))
    spb = 60.0 / max(song.tempo_bpm, 1)
    events: List[Tuple[int, int, mido.Message]] = []
    for r in rows:
        on = int(round(r["start"] / spb * PPQ))
        off = max(on + 1, int(round(r["end"] / spb * PPQ)))
        events.append((on, 1, mido.Message("note_on", channel=0, note=r["pitch"], velocity=90)))
        events.append((off, 0, mido.Message("note_off", channel=0, note=r["pitch"], velocity=0)))
    last = 0
    for t, _order, msg in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(msg.copy(time=t - last))
        last = t
    mid.tracks.append(track)
    mid.save(midi_path)

    lines: List[List[str]] = []
    previous = None
    for r in rows:
        if r["line"] != previous:
            lines.append([])
            previous = r["line"]
        lines[-1].append(r["symbol"])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join("".join(line) for line in lines) + "\n")

    stats.update({"midi_path": midi_path, "lyric_path": txt_path,
                  "duration_s": round(rows[-1]["end"], 3) if rows else 0.0})
    return stats
