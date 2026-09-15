import random

import pytest

from qwen_abc.abc import parse_abc, song_to_abc
from qwen_abc.canonical import Chord, Note, Section, Song, comparable
from qwen_abc.prompt import song_to_spec, spec_syllables, spec_to_prompt
from qwen_abc.theory import canonical_key, key_signature, spell_pitch


def make_song(**kw):
    base = dict(
        song_id="t", meter_num=4, tempo_bpm=92, key="G major",
        bar_beats=[4, 4], sections=[Section("verse", 0, 2)],
        notes=[], chords=[Chord(0, 32, "G")],
    )
    base.update(kw)
    return Song(**base)


def roundtrip(song):
    text = song_to_abc(song)
    res = parse_abc(text)
    assert res.ok, text
    assert not res.errors, (dict(res.errors), text)
    assert comparable(res.song) == comparable(song), text
    return text, res


def test_one_syllable_per_note_and_rest():
    notes = [Note(0, 4, 71, ("我",)), Note(4, 4, 71, ("還",)), Note(12, 4, 69, ("記",))]
    text, _ = roundtrip(make_song(notes=notes))
    assert "w: 我 還 記" in text
    assert "z2" in text  # the gap between tick 8 and 12 is a rest


def test_melisma_uses_underscore_and_is_not_a_new_syllable():
    notes = [Note(0, 2, 71, ("得",)), Note(2, 2, 72, None, melisma=True), Note(4, 4, 74, ("那",))]
    text, res = roundtrip(make_song(notes=notes))
    assert "w: 得 _ 那" in text
    assert res.song.notes[1].melisma and res.song.notes[1].lyric is None


def test_wordless_notes_use_star_and_trailing_stars_are_omitted():
    notes = [Note(0, 4, 67), Note(4, 4, 71, ("啊",)), Note(8, 4, 72)]
    text, _ = roundtrip(make_song(notes=notes))
    assert "w: * 啊" in text
    assert "w: * 啊 *" not in text


def test_instrumental_line_has_no_lyric_line():
    notes = [Note(0, 8, 67), Note(8, 8, 69)]
    text, _ = roundtrip(make_song(notes=notes))
    assert "w:" not in text


def test_tie_across_barline_keeps_one_syllable():
    notes = [Note(12, 8, 74, ("风",)), Note(20, 4, 72, ("起",))]
    text, res = roundtrip(make_song(notes=notes))
    assert "-" in text.split("w:")[0]
    assert len(res.song.notes) == 2 and res.song.notes[0].duration == 8
    assert "w: * | 起" not in text  # the tied continuation does not consume a syllable
    assert "w: 风 | 起" in text


def test_chord_change_inside_a_note_splits_with_tie():
    notes = [Note(0, 8, 71, ("长",))]
    chords = [Chord(0, 4, "G"), Chord(4, 28, "Em7")]
    text, res = roundtrip(make_song(notes=notes, chords=chords))
    assert '"G"B2- "Em7"B2' in text
    assert len(res.song.notes) == 1


def test_irregular_bar_uses_inline_meter():
    song = make_song(bar_beats=[4, 2, 4], sections=[Section("verse", 0, 3)],
                     notes=[Note(16, 8, 67, ("短",)), Note(24, 4, 69, ("小",))],
                     chords=[Chord(0, 40, "C")])
    text, res = roundtrip(song)
    assert "[M:2/4]" in text and "[M:4/4]" in text
    assert res.song.bar_beats == [4, 2, 4]


def test_sections_and_three_four():
    song = make_song(meter_num=3, bar_beats=[3, 3, 3, 3],
                     sections=[Section("intro", 0, 1), Section("verse", 1, 2), Section("chorus", 3, 1)],
                     notes=[Note(12, 6, 64, ("一",)), Note(18, 6, 65, ("二",)), Note(36, 12, 67, ("三",))],
                     chords=[Chord(0, 12, "N.C."), Chord(12, 36, "C")])
    text, res = roundtrip(song)
    assert "M:3/4" in text and "P:intro" in text and "P:chorus" in text
    assert [s.label for s in res.song.sections] == ["intro", "verse", "chorus"]


def test_joined_syllables_and_english():
    notes = [Note(0, 4, 67, ("hey",)), Note(4, 4, 67, ("boy",)), Note(8, 4, 69, ("坦", "白"))]
    text, res = roundtrip(make_song(notes=notes))
    assert "坦~白" in text
    spec = song_to_spec(res.song)
    assert spec["sections"][0]["lines"] == ["hey boy 坦白"]
    assert spec_syllables(spec) == ["hey", "boy", "坦", "白"]


@pytest.mark.parametrize("key", ["C major", "B major", "Db major", "F# major", "G# minor", "Eb minor", "D minor"])
def test_chromatic_pitches_roundtrip_in_every_key(key):
    rng = random.Random(0)
    notes, t = [], 0
    while t < 64:
        d = rng.choice([1, 2, 3, 4, 6])
        notes.append(Note(t, d, rng.randint(52, 84), (f"s{t}",)))
        t += d + rng.choice([0, 0, 1, 2])
    song = make_song(key=key, bar_beats=[4] * 5, sections=[Section("verse", 0, 5)],
                     notes=[n for n in notes if n.offset <= 80], chords=[Chord(0, 80, "C")])
    roundtrip(song)


def test_accidental_is_explicit_under_both_conventions():
    # C major: ^F4 then F5 natural must be written as =f (pitch-class propagation)
    song = make_song(key="C major", notes=[Note(0, 4, 66, ("a",)), Note(4, 4, 77, ("b",))])
    text, _ = roundtrip(song)
    assert "^F" in text and "=f" in text


def test_key_normalization_and_spelling():
    assert canonical_key("C# major") == "Db major"
    assert canonical_key("Ab minor") == "G# minor"
    assert key_signature("F# major")["E"] == 1
    assert spell_pitch(65, "F# major")[:2] == ("E", 1)


def test_prompt_contains_structure_and_lyrics():
    notes = [Note(0, 4, 67, ("我",), line=0), Note(4, 4, 67, ("们",), line=0), Note(8, 4, 69, ("走",), line=1)]
    prompt = spec_to_prompt(song_to_spec(make_song(notes=notes)))
    assert "Tempo: 92 BPM" in prompt and "Key: G major" in prompt
    assert "P:verse | 2 bars\n我们\n走\n" in prompt
    assert prompt.endswith("ABC:\n")


def test_parser_reports_malformed_input():
    bad = "X:1\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nP:verse\n\"Q7\"C2 D2 E2 | F2 G2 A2 B2 |\nw: a b c d e f g h i\n"
    res = parse_abc(bad)
    assert res.ok and not res.strict_ok
    assert res.errors["bar_duration_mismatch"] == 1
    assert res.errors["invalid_chord_symbol"] == 1
    assert res.errors["lyric_overflow"] == 2


def test_parser_standard_lyric_conventions():
    text = "X:1\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nC2 D2 E2 F2 | G8 |\nw: hel-lo * wor~ld | la\n"
    res = parse_abc(text)
    assert res.strict_ok, dict(res.errors)
    lyr = [n.lyric for n in res.song.notes]
    assert lyr == [("hel",), ("lo",), None, ("wor", "ld"), ("la",)]
