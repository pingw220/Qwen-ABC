"""The lead sheet MIDI the MIDI-SAG / MuseControlLite renderer reads."""
import mido

from qwen_abc.canonical import Chord, Note, Section, Song
from qwen_abc.leadsheet_midi import (
    BEAT_PITCH,
    DOWNBEAT_PITCH,
    LABEL_TO_TAG,
    MIDI_SAG_TAGS,
    SAO_WINDOW_SECONDS,
    plan_segments,
    renderable_problems,
    song_to_leadsheet_midi,
)
from qwen_abc.midi import song_to_midi


def song(n_bars=8, tempo=100, sections=None, beats=4):
    notes = [Note(16 * b, 8, 60 + (b % 5), ("啊",)) for b in range(n_bars)]
    sections = sections or [Section("verse", 0, n_bars // 2), Section("chorus", n_bars // 2, n_bars - n_bars // 2)]
    return Song("t", beats, tempo, "C major", [beats] * n_bars, sections, notes,
                [Chord(0, 16 * n_bars, "C")])


def tracks(path):
    mid = mido.MidiFile(path)
    return {next((m.name for m in t if m.type == "track_name"), ""): t for t in mid.tracks}


def test_click_track_has_one_note_per_beat_and_a_downbeat_per_bar(tmp_path):
    s = song(n_bars=6)
    path = str(tmp_path / "a.mid")
    song_to_leadsheet_midi(s, path)
    click = tracks(path)["click_raw"]
    ons = [m for m in click if m.type == "note_on"]
    assert len(ons) == sum(s.bar_beats) == 24
    assert all(m.channel == 9 for m in ons)
    assert sum(1 for m in ons if m.note == DOWNBEAT_PITCH) == len(s.bar_beats)
    assert sum(1 for m in ons if m.note == BEAT_PITCH) == 24 - 6


def test_irregular_bars_keep_their_own_beat_count(tmp_path):
    s = song(n_bars=3)
    s.bar_beats = [4, 3, 4]
    path = str(tmp_path / "b.mid")
    song_to_leadsheet_midi(s, path)
    ons = [m for m in tracks(path)["click_raw"] if m.type == "note_on"]
    assert len(ons) == 11
    assert sum(1 for m in ons if m.note == DOWNBEAT_PITCH) == 3


def test_markers_are_indexed_and_tags_are_in_the_midi_sag_vocabulary(tmp_path):
    s = song(sections=[Section("intro", 0, 2), Section("prechorus", 2, 3), Section("instrumental", 5, 3)])
    path = str(tmp_path / "c.mid")
    audit = song_to_leadsheet_midi(s, path)
    texts = [m.text for m in tracks(path)["conductor"] if m.type == "marker"]
    assert texts == ["0:intro", "1:prechorus", "2:instrumental"]
    assert audit["structure_tags"] == ["intro", "verse", "inst"]
    assert set(audit["structure_tags"]) <= set(MIDI_SAG_TAGS)
    assert set(LABEL_TO_TAG.values()) <= set(MIDI_SAG_TAGS)


def test_a_section_wider_than_the_generation_window_is_split_at_bars(tmp_path):
    # 32 bars of 4/4 at 40 BPM = 192 s, four times the 47.55 s window.
    s = song(n_bars=32, tempo=40, sections=[Section("chorus", 0, 32)])
    segments = plan_segments(s)
    assert len(segments) == 5
    assert [seg.num_bars for seg in segments] == [6, 7, 6, 7, 6]
    assert all(seg.span_s < SAO_WINDOW_SECONDS for seg in segments)
    assert all(seg.tag == "chorus" and seg.part for seg in segments)
    assert [seg.index for seg in segments] == [0, 1, 2, 3, 4]
    audit = song_to_leadsheet_midi(s, str(tmp_path / "d.mid"))
    assert audit["n_sections"] == 1 and audit["n_segments"] == 5


def test_structure_starts_are_section_start_times_in_seconds(tmp_path):
    s = song(n_bars=8, tempo=120, sections=[Section("verse", 0, 4), Section("chorus", 4, 4)])
    audit = song_to_leadsheet_midi(s, str(tmp_path / "e.mid"))
    assert audit["structure_starts"] == [0.0, 8.0]  # 4 bars * 4 beats / 2 beats per second
    assert audit["duration_s"] == 16.0


def test_utf8_lyrics_survive_midos_latin1_meta_encoding(tmp_path):
    s = song(n_bars=2)
    s.notes = [Note(0, 8, 60, ("愛",)), Note(16, 8, 62, ("你",))]
    path = str(tmp_path / "f.mid")
    song_to_leadsheet_midi(s, path)
    raw = [m.text for m in tracks(path)["melody"] if m.type == "lyrics"]
    # mido reads meta text as latin-1; MIDI-SAG's parser recovers UTF-8 this way.
    assert [t.encode("latin-1").decode("utf-8") for t in raw] == ["愛", "你"]


def test_renderable_problems_flags_what_the_renderer_cannot_use(tmp_path):
    s = song(n_bars=8)
    audit = song_to_leadsheet_midi(s, str(tmp_path / "g.mid"))
    assert renderable_problems(s, audit) == []

    mute = song(n_bars=8)
    for n in mute.notes:
        n.lyric = None
    audit = song_to_leadsheet_midi(mute, str(tmp_path / "h.mid"))
    assert any("no lyrics" in p for p in renderable_problems(mute, audit))

    stacked = song(n_bars=8)
    stacked.notes.append(Note(0, 8, 64, ("疊",)))
    audit = song_to_leadsheet_midi(stacked, str(tmp_path / "i.mid"))
    assert any("onset" in p or "monophonic" in p for p in renderable_problems(stacked, audit))


def test_plain_song_to_midi_is_unchanged_without_the_new_arguments(tmp_path):
    s = song(n_bars=4)
    a, b = str(tmp_path / "p.mid"), str(tmp_path / "q.mid")
    song_to_midi(s, a)
    song_to_midi(s, b, markers=None, click=None)
    assert open(a, "rb").read() == open(b, "rb").read()
    assert "click_raw" not in tracks(a)
    assert [m.text for m in tracks(a)["conductor"] if m.type == "marker"] == ["verse", "chorus"]
