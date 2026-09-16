"""FastSinger inputs: symbol k must sing note k, so the two files must agree exactly."""
import mido

from qwen_abc.canonical import Chord, Note, Section, Song
from qwen_abc.fastsinger import MELISMA, sung_notes, write_fastsinger_inputs


def song(notes, tempo=60, n_bars=4):
    return Song("fs", 4, tempo, "C major", [4] * n_bars, [Section("verse", 0, n_bars)], notes,
                [Chord(0, 16 * n_bars, "C")])


def written(tmp_path, s, **kw):
    mid, txt = str(tmp_path / "a.mid"), str(tmp_path / "a.txt")
    stats = write_fastsinger_inputs(s, mid, txt, **kw)
    notes = [m for tr in mido.MidiFile(mid).tracks for m in tr if m.type == "note_on" and m.velocity > 0]
    text = open(txt, encoding="utf-8").read()
    return notes, text, stats


def test_note_count_and_symbol_count_always_match(tmp_path):
    s = song([Note(0, 4, 60, ("我",)), Note(4, 4, 62, None, melisma=True), Note(8, 8, 64, ("爱",)),
              Note(16, 8, 65, None), Note(24, 8, 67, ("你",))])
    notes, text, stats = written(tmp_path, s)
    symbols = "".join(text.split())
    assert len(notes) == len(symbols) == stats["symbols"]
    assert symbols == f"我{MELISMA}爱你"


def test_a_wordless_note_is_not_sung_and_does_not_shift_the_words(tmp_path):
    # The note at tick 16 has no lyric and is not a melisma: an instrumental
    # note, not part of the vocal. If it stayed, 你 would be sung on it.
    s = song([Note(0, 8, 60, ("我",)), Note(16, 8, 65, None), Note(24, 8, 67, ("你",))])
    notes, text, stats = written(tmp_path, s)
    assert "".join(text.split()) == "我你"
    assert [n.note for n in notes] == [60, 67]
    assert stats["wordless_dropped"] == 1


def test_a_note_with_several_syllables_is_divided_between_them(tmp_path):
    s = song([Note(0, 8, 60, ("喜", "怒", "哀")), Note(8, 8, 62, ("乐",))])
    notes, text, stats = written(tmp_path, s)
    assert "".join(text.split()) == "喜怒哀乐"
    assert len(notes) == 4 and stats["notes_split"] == 1
    assert stats["extra_syllables_dropped"] == 0
    starts, mid = [], mido.MidiFile(str(tmp_path / "a.mid"))
    t = 0
    for m in mid.tracks[0]:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            starts.append(t)
    # 2 beats (960 PPQ ticks) split three ways, then the next note on beat 2.
    assert starts == [0, 320, 640, 960]


def test_a_melisma_with_nothing_to_hold_is_dropped(tmp_path):
    s = song([Note(0, 4, 60, None, melisma=True), Note(8, 8, 62, ("啊",))])
    notes, text, stats = written(tmp_path, s)
    assert "".join(text.split()) == "啊"
    assert stats["melisma_without_syllable_dropped"] == 1


def test_non_chinese_syllables_are_held_rather_than_left_to_the_mandarin_g2p(tmp_path):
    s = song([Note(0, 8, 60, ("戴",)), Note(8, 8, 62, ("mask",))])
    notes, text, stats = written(tmp_path, s)
    assert "".join(text.split()) == f"戴{MELISMA}"
    assert stats["non_chinese_syllables"] == 1


def test_a_breath_starts_a_new_line(tmp_path):
    # 60 BPM: tick 0-8 is 0-2 s, the next note starts at 6 s.
    s = song([Note(0, 8, 60, ("一",)), Note(24, 8, 62, ("二",))], n_bars=8)
    _notes, text, stats = written(tmp_path, s)
    assert text.splitlines() == ["一", "二"]
    assert stats["lines"] == 2


def test_the_line_is_monophonic_and_ends_are_clamped(tmp_path):
    s = song([Note(0, 16, 60, ("長",)), Note(4, 4, 62, ("短",))])
    rows, _stats = sung_notes(s)
    assert rows[0]["end"] <= rows[1]["start"]
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(rows, rows[1:]))
