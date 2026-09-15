import pytest

from qwen_abc.abc import parse_abc, song_to_abc
from qwen_abc.canonical import Chord, Note, Section, Song
from qwen_abc.metrics import distribution_distances, lcs_len, song_metrics
from qwen_abc.prompt import song_to_spec


def demo_song():
    notes = [Note(0, 4, 67, ("我",), line=0), Note(4, 2, 69, ("们",), line=0), Note(6, 2, 71, None, melisma=True, line=0),
             Note(8, 8, 72, ("走",), line=1), Note(24, 4, 64), Note(28, 12, 62, ("吧",), line=1)]
    return Song("demo", 4, 100, "G major", [4, 4, 2], [Section("verse", 0, 2), Section("outro", 2, 1)], notes,
                [Chord(0, 16, "G"), Chord(16, 16, "Em7"), Chord(32, 8, "D/F#")])


def test_midi_roundtrip_melody_and_lyrics(tmp_path):
    mido = pytest.importorskip("mido")
    from qwen_abc.midi import MIDI_PER_TICK, midi_melody, song_to_midi

    song = demo_song()
    path = tmp_path / "demo.mid"
    song_to_midi(song, str(path))
    got = midi_melody(str(path))
    assert [(round(o * 4), round(d * 4), p) for o, d, p in got] == [(n.onset, n.duration, n.pitch) for n in song.notes]
    mid = mido.MidiFile(str(path), charset="utf-8")
    lyrics = [m.text for t in mid.tracks for m in t if m.type == "lyrics"]
    assert lyrics == ["我", "们", "走", "吧"]  # melisma note writes no text
    sigs = [(m.numerator, m.denominator) for t in mid.tracks for m in t if m.type == "time_signature"]
    assert sigs == [(4, 4), (2, 4)]
    assert MIDI_PER_TICK == 120


def test_metrics_on_reference_are_self_consistent():
    song = demo_song()
    spec = song_to_spec(song)
    res = parse_abc(song_to_abc(song))
    m = song_metrics(res.song, spec, res)
    assert m["lyric_recall"] == 1.0 and m["lyric_precision"] == 1.0 and m["lyric_exact"] == 1.0
    assert m["section_plan_exact"] == 1.0 and m["bar_beats_match_frac"] == 1.0 and m["strict_valid"] == 1.0
    assert m["notes_per_syllable"] == pytest.approx(6 / 4)
    assert m["melisma_note_frac"] == pytest.approx(1 / 6)
    assert m["wordless_note_frac"] == pytest.approx(1 / 6)
    # G chord (G B D) over G A B -> 4 + 0 + 2 of 8 ticks; Em7 (E G B D) over C for 8 ticks -> 0
    assert 0 < m["chord_tone_frac"] < 1
    assert all(v == 0 for v in distribution_distances([song], [song]).values())


def test_lyric_recall_detects_missing_and_extra_syllables():
    song = demo_song()
    spec = song_to_spec(song)
    song.notes[3].lyric = None  # drop 走
    song.notes[4].lyric = ("啊",)  # sing an extra syllable
    m = song_metrics(song, spec)
    assert m["lyric_recall"] == pytest.approx(3 / 4)
    assert m["lyric_precision"] == pytest.approx(3 / 4)
    assert lcs_len(list("abcde"), list("axcye")) == 3
