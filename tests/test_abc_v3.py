"""ABC-v3 prompt: syllable budgets, and the marks on lines a section boundary splits."""
from qwen_abc.abc_v2 import spec_to_prompt_v2
from qwen_abc.abc_v3 import CONTINUES, section_syllables, spec_to_prompt_v3, strip_v3_marks
from qwen_abc.canonical import Chord, Note, Section, Song
from qwen_abc.prompt import song_to_spec, spec_syllables, split_syllables


def pickup_song():
    """A verse whose first phrase starts on the last note of the intro."""
    notes = [
        Note(12, 4, 67, ("无",), line=0),                  # pickup, still in the intro
        Note(16, 4, 69, ("法",), line=0), Note(20, 4, 71, ("阻",), line=0),
        Note(24, 4, 72, ("止",), line=0), Note(28, 4, 74, ("心",), line=0),
        Note(32, 4, 72, ("流",), line=1), Note(36, 4, 71, ("感",), line=1),
    ]
    return Song("p", 4, 100, "C major", [4] * 3,
                [Section("intro", 0, 1), Section("verse", 1, 1), Section("chorus", 2, 1)],
                notes, [Chord(0, 48, "C")])


def lines_of(prompt):
    return [l for l in prompt.split("\n")
            if l and not l.startswith(("P:", "Task", "Language", "Meter", "Tempo", "Key", "Structure", "ABC"))]


def test_every_section_header_states_its_syllable_count():
    spec = song_to_spec(pickup_song())
    heads = [l for l in spec_to_prompt_v3(spec).split("\n") if l.startswith("P:")]
    assert heads[0].endswith("| 1 syllable")     # the pickup, singular
    assert heads[1].endswith("| 4 syllables")
    assert heads[2].endswith("| 2 syllables")
    assert [section_syllables(s) for s in spec["sections"]] == [1, 4, 2]


def test_a_line_split_by_a_boundary_is_marked_on_both_sides():
    prompt = spec_to_prompt_v3(song_to_spec(pickup_song()))
    body = lines_of(prompt)
    assert body[0] == f"无{CONTINUES}"
    assert body[1] == f"{CONTINUES}法阻止心"
    assert CONTINUES not in body[2]  # a line that starts and ends inside its section


def test_an_instrumental_between_the_pickup_and_its_phrase_is_stepped_over():
    song = pickup_song()
    song.bar_beats = [4] * 4
    song.sections = [Section("intro", 0, 1), Section("instrumental", 1, 1),
                     Section("verse", 2, 1), Section("chorus", 3, 1)]
    for n in song.notes:  # push the sung notes past the empty instrumental bar
        n.onset += 16
    body = lines_of(spec_to_prompt_v3(song_to_spec(song)))
    assert body[0] == f"无{CONTINUES}" and body[1] == f"{CONTINUES}法阻止心"


def test_the_marks_are_notation_and_never_change_the_lyrics():
    spec = song_to_spec(pickup_song())
    written = [s for line in lines_of(spec_to_prompt_v3(spec)) for s in split_syllables(strip_v3_marks(line))]
    assert written == spec_syllables(spec)


def test_v3_only_adds_to_v2_and_keeps_the_plan_lines():
    spec = song_to_spec(pickup_song())
    v2, v3 = spec_to_prompt_v2(spec), spec_to_prompt_v3(spec)
    assert v3.startswith(v2.split("\n")[0])  # same task header
    for a, b in zip([l for l in v2.split("\n") if l.startswith("P:")],
                    [l for l in v3.split("\n") if l.startswith("P:")]):
        assert b.startswith(a) and "syllable" in b[len(a):]
    assert strip_v3_marks(v3).replace(" | 1 syllable", "").replace(" | 4 syllables", "") \
        .replace(" | 2 syllables", "") == v2


def test_a_song_with_no_lyrics_still_gets_a_budget_of_zero():
    song = pickup_song()
    for n in song.notes:
        n.lyric = None
    heads = [l for l in spec_to_prompt_v3(song_to_spec(song)).split("\n") if l.startswith("P:")]
    assert all(h.endswith("| 0 syllables") for h in heads)
