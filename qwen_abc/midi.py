"""Canonical Song -> MIDI (melody with lyric events, block chords, markers).

Time is metrical: one constant tempo from ``Q:``. The corpus' beat-level
tempo drift is not represented (see ABC_SCHEMA.md, lossy conversions).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mido

from .canonical import TICKS_PER_BEAT, Song
from .theory import key_to_abc, parse_chord_symbol

PPQ = 480
MIDI_PER_TICK = PPQ // TICKS_PER_BEAT
CLICK_TICKS = PPQ // 8  # audible but well inside one beat


def _abs_to_delta(events: List[Tuple[int, int, mido.Message]]) -> List[mido.Message]:
    events.sort(key=lambda e: (e[0], e[1]))
    out, last = [], 0
    for t, _, msg in events:
        out.append(msg.copy(time=t - last))
        last = t
    return out


def song_to_midi(
    song: Song,
    path: str,
    chord_velocity: int = 56,
    markers: Optional[Sequence[Tuple[int, str]]] = None,
    click: Optional[Tuple[int, int]] = None,
) -> None:
    """Write the song as a type-1 MIDI file.

    ``markers`` replaces the default one-marker-per-section conductor text with
    explicit ``(tick, text)`` pairs (ticks in canonical units, not PPQ), for
    consumers that want a different marker spelling. ``click`` is a
    ``(downbeat_pitch, beat_pitch)`` pair that adds a ``click_raw`` track on the
    percussion channel, one note per beat of the score's own bar grid; the
    downbeat pitch marks the first beat of every bar.
    """
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ, charset="utf-8")

    # conductor: tempo, meter changes, key, section markers
    cond: List[Tuple[int, int, mido.Message]] = []
    cond.append((0, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(song.tempo_bpm, 1)))))
    key_name = key_to_abc(song.key)
    try:
        cond.append((0, 0, mido.MetaMessage("key_signature", key=key_name)))
    except (ValueError, KeyError):
        pass
    current = None
    for start, beats in zip(song.bar_starts(), song.bar_beats):
        if beats != current and beats > 0:
            cond.append((start * MIDI_PER_TICK, 1, mido.MetaMessage("time_signature", numerator=beats, denominator=4)))
            current = beats
    starts = song.bar_starts()
    if markers is None:
        markers = [
            (starts[sec.start_bar], sec.label) for sec in song.sections if sec.start_bar < len(starts)
        ]
    for tick, text in markers:
        cond.append((tick * MIDI_PER_TICK, 2, mido.MetaMessage("marker", text=text)))
    track = mido.MidiTrack(_abs_to_delta(cond))
    track.insert(0, mido.MetaMessage("track_name", name="conductor", time=0))
    mid.tracks.append(track)

    # melody with lyrics on attacks only (a melisma never re-sings its text)
    mel: List[Tuple[int, int, mido.Message]] = []
    for n in song.notes:
        on, off = n.onset * MIDI_PER_TICK, n.offset * MIDI_PER_TICK
        if n.lyric:
            mel.append((on, 1, mido.MetaMessage("lyrics", text="".join(n.lyric))))
        mel.append((on, 2, mido.Message("note_on", channel=0, note=min(max(n.pitch, 0), 127), velocity=90)))
        mel.append((off, 0, mido.Message("note_off", channel=0, note=min(max(n.pitch, 0), 127), velocity=0)))
    track = mido.MidiTrack(_abs_to_delta(mel))
    track.insert(0, mido.MetaMessage("track_name", name="melody", time=0))
    track.insert(1, mido.Message("program_change", channel=0, program=73, time=0))  # flute: easy to hear
    mid.tracks.append(track)

    # chords: close voicing around C4, bass an octave below the root/bass note
    ch: List[Tuple[int, int, mido.Message]] = []
    for c in song.chords:
        info = parse_chord_symbol(c.symbol)
        if not info or info["root_pc"] is None:
            continue
        on, off = c.onset * MIDI_PER_TICK, c.offset * MIDI_PER_TICK
        pitches = sorted({60 + pc for pc in info["pcs"]})
        bass = 36 + (info["bass_pc"] if info["bass_pc"] is not None else info["root_pc"])
        for p in [bass] + pitches:
            ch.append((on, 1, mido.Message("note_on", channel=1, note=p, velocity=chord_velocity)))
            ch.append((off, 0, mido.Message("note_off", channel=1, note=p, velocity=0)))
    track = mido.MidiTrack(_abs_to_delta(ch))
    track.insert(0, mido.MetaMessage("track_name", name="chords", time=0))
    track.insert(1, mido.Message("program_change", channel=1, program=0, time=0))
    mid.tracks.append(track)

    # click: the metrical grid the score is written on, one note per beat
    if click is not None:
        downbeat_pitch, beat_pitch = click
        cl: List[Tuple[int, int, mido.Message]] = []
        for start, beats in zip(starts, song.bar_beats):
            for b in range(beats):
                on = (start + b * TICKS_PER_BEAT) * MIDI_PER_TICK
                pitch = downbeat_pitch if b == 0 else beat_pitch
                cl.append((on, 1, mido.Message("note_on", channel=9, note=pitch, velocity=100)))
                cl.append((on + CLICK_TICKS, 0, mido.Message("note_off", channel=9, note=pitch, velocity=0)))
        track = mido.MidiTrack(_abs_to_delta(cl))
        track.insert(0, mido.MetaMessage("track_name", name="click_raw", time=0))
        mid.tracks.append(track)

    mid.save(path)


def midi_melody(path: str) -> List[Tuple[float, float, int]]:
    """Read back (onset_beats, dur_beats, pitch) from track 'melody' (for round-trip checks)."""
    mid = mido.MidiFile(path, charset="utf-8")
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name != "melody":
            continue
        t, active, out = 0, {}, []
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = t
            elif msg.type in ("note_off", "note_on") and msg.note in active:
                start = active.pop(msg.note)
                out.append((start / mid.ticks_per_beat, (t - start) / mid.ticks_per_beat, msg.note))
        return sorted(out)
    return []
