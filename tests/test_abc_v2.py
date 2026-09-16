"""ABC-v2: section headers, per-bar countdown, stripping, parsing and counter evaluation."""
from qwen_abc.abc import parse_abc, song_to_abc
from qwen_abc.abc_v2 import counter_report, song_to_abc_v2, spec_to_prompt_v2, strip_v2
from qwen_abc.canonical import Chord, Note, Section, Song, comparable
from qwen_abc.midi import midi_melody, song_to_midi
from qwen_abc.prompt import song_to_spec, spec_syllables


def two_section_song():
    notes = [Note(0, 4, 67, ("我",)), Note(4, 4, 69, None, melisma=True), Note(8, 8, 71, ("爱",)),
             Note(16, 16, 72, ("你",)), Note(36, 4, 74, ("啊",)), Note(44, 4, 72), Note(48, 4, 71, ("好",))]
    return Song("t2", 4, 100, "G major", [4, 4, 3, 4, 4], [Section("verse", 0, 2), Section("chorus", 2, 3)],
                notes, [Chord(0, 16, "G"), Chord(16, 12, "C"), Chord(28, 20, "D")])


def test_v2_header_and_countdown_are_written_per_section_and_bar():
    text = song_to_abc_v2(two_section_song())
    lines = text.split("\n")
    assert lines[lines.index("P:verse") + 1] == "% section 1/2 | 2 bars"
    assert lines[lines.index("P:chorus") + 1] == "% section 2/2 | 3 bars"
    assert "[r:2] " in text and "[r:1] " in text and "[r:3] [M:3/4]" in text
    assert text.count("[r:") == 5


def test_v2_parses_to_the_same_song_and_strips_to_v1_exactly():
    song = two_section_song()
    v2, v1 = song_to_abc_v2(song), song_to_abc(song)
    assert strip_v2(v2) == v1
    res = parse_abc(v2)
    assert res.strict_ok, dict(res.errors)
    assert comparable(res.song) == comparable(song)
    assert comparable(parse_abc(v1).song) == comparable(res.song)


def test_v2_lyric_alignment_and_midi_roundtrip_are_preserved(tmp_path):
    song = two_section_song()
    res = parse_abc(song_to_abc_v2(song))
    assert [n.lyric for n in res.song.notes] == [n.lyric for n in song.notes]
    assert [n.melisma for n in res.song.notes] == [n.melisma for n in song.notes]
    path = str(tmp_path / "x.mid")
    song_to_midi(res.song, path)
    assert [(p) for _, _, p in midi_melody(path)] == [n.pitch for n in song.notes]


def test_counter_report_is_perfect_on_reference_and_detects_errors():
    song = two_section_song()
    spec = song_to_spec(song)
    text = song_to_abc_v2(song)
    rep = counter_report(text, spec)
    assert rep["counter_bars"] == 5
    for k in ("counter_present_frac", "counter_self_consistent_frac", "counter_plan_frac", "header_plan_frac",
              "sections_ending_on_counter_1"):
        assert rep[k] == 1.0, k
    # the model ends the chorus one bar early but keeps counting down from 3: state says 2 bars remain, reality 1
    broken = text.replace("[r:3] [M:3/4]", "[r:3] [M:3/4]", 1)
    lines = broken.split("\n")
    last_music = max(i for i, l in enumerate(lines) if l.startswith("[r:"))
    lines[last_music] = lines[last_music].split(" | [r:1]")[0] + " |"
    bad = counter_report("\n".join(lines), spec)
    assert bad["counter_self_consistent_frac"] < 1.0
    assert bad["sections_ending_on_counter_1"] < 1.0
    # a header that disagrees with the plan
    wrong = text.replace("% section 2/2 | 3 bars", "% section 2/2 | 4 bars")
    assert counter_report(wrong, spec)["header_plan_frac"] == 0.5


def test_v2_prompt_adds_section_index_and_keeps_lyrics():
    song = two_section_song()
    spec = song_to_spec(song)
    prompt = spec_to_prompt_v2(spec)
    assert "P:verse | 2 bars | section 1/2" in prompt
    assert "P:chorus | 3 bars | beats 3 4x2 | section 2/2" in prompt
    assert prompt.endswith("ABC:\n")
    assert spec_syllables(spec) == ["我", "爱", "你", "啊", "好"]


def test_v1_output_is_unchanged_without_hooks():
    song = two_section_song()
    assert "[r:" not in song_to_abc(song) and "% section" not in song_to_abc(song)
