"""Conservative section-boundary cleaning."""
from qwen_abc.abc_v2 import song_to_abc_v2
from qwen_abc.canonical import Chord, Note, Section, Song, comparable
from qwen_abc.cleaning import clean_sections, crossings_at, cut_count, pathology_flags, CleanConfig, _line_events


def song(sections, lyric_notes, n_bars=8):
    """lyric_notes: (onset_tick, line, text or None for melisma)."""
    notes = []
    for onset, line, text in lyric_notes:
        notes.append(Note(onset, 2, 67, (text,) if text else None, melisma=text is None, line=line))
    return Song("c", 4, 100, "C major", [4] * n_bars, sections, notes, [Chord(0, 16 * n_bars, "C")])


def bar(b, beat=0):
    return 16 * b + 4 * beat


def test_tail_spill_moves_boundary_later_and_music_is_unchanged():
    # line 0 = 5 syllables in bars 0-1, last syllable spills onto bar 2's downbeat; line 1 starts in bar 3
    notes = [(bar(0, i), 0, "abcde"[i]) for i in range(4)] + [(bar(1, 0), 0, "e"), (bar(2, 0), 0, "f")] + \
            [(bar(3, i), 1, "wxyz"[i]) for i in range(4)] + [(bar(4, 0), 1, "v"), (bar(5, 0), 1, "u")]
    s = song([Section("verse", 0, 2), Section("chorus", 2, 6)], notes)
    assert cut_count(s)[0] == 1
    secs, dec = clean_sections(s)
    assert [(x.label, x.start_bar, x.num_bars) for x in secs] == [("verse", 0, 3), ("chorus", 3, 5)]
    assert dec[0].action == "moved" and dec[0].new_bar - dec[0].old_bar == 1
    s2 = song(secs, notes)
    assert cut_count(s2)[0] == 0
    a, b = comparable(s), comparable(s2)
    assert all(a[k] == b[k] for k in ("notes", "chords", "bar_beats", "tempo_bpm", "key"))


def test_pickup_is_tolerated():
    # line 1 starts with 2 syllables on beats 3-4 of bar 1, then continues in bar 2 (the section start)
    notes = [(bar(0, 0), 0, "a"), (bar(0, 1), 0, "b"), (bar(1, 2), 1, "c"), (bar(1, 3), 1, "d")] + \
            [(bar(2, i), 1, "efgh"[i]) for i in range(4)]
    s = song([Section("intro", 0, 2), Section("verse", 2, 6)], notes)
    cr = crossings_at(s, 2, _line_events(s), CleanConfig())
    assert [c["kind"] for c in cr] == ["pickup"]
    secs, dec = clean_sections(s)
    assert dec[0].action == "kept_clean" and secs[1].start_bar == 2


def test_balanced_split_is_left_alone():
    notes = [(bar(1, i), 0, "abcd"[i]) for i in range(4)] + [(bar(2, i), 0, "efgh"[i]) for i in range(4)]
    s = song([Section("verse", 0, 2), Section("chorus", 2, 6)], notes)
    secs, dec = clean_sections(s)
    assert dec[0].action == "balanced_split_kept" and secs[1].start_bar == 2


def test_melisma_spill_is_not_a_cut():
    notes = [(bar(1, i), 0, "abcd"[i]) for i in range(4)] + [(bar(2, 0), 0, None)]
    s = song([Section("verse", 0, 2), Section("chorus", 2, 6)], notes)
    assert cut_count(s)[0] == 0


def test_no_move_when_it_would_split_another_line():
    # tail spill of line 0 into bar 2, but line 1 already starts in bar 2 beat 2 and runs into bar 3
    notes = [(bar(1, i), 0, "abcd"[i]) for i in range(4)] + [(bar(1, 3) + 2, 0, "e"), (bar(1, 3) + 3, 0, "f"),
                                                            (bar(2, 0), 0, "g")] + \
            [(bar(2, 2), 1, "h"), (bar(2, 3), 1, "i"), (bar(3, 0), 1, "j"), (bar(3, 1), 1, "k"), (bar(3, 2), 1, "l")]
    s = song([Section("verse", 0, 2), Section("chorus", 2, 6)], notes)
    secs, dec = clean_sections(s)
    assert dec[0].action != "moved" and secs[1].start_bar == 2


def test_cleaning_is_deterministic_and_v2_counters_follow_the_cleaned_plan():
    notes = [(bar(0, i), 0, "abcd"[i]) for i in range(4)] + [(bar(1, 0), 0, "e"), (bar(2, 0), 0, "f")] + \
            [(bar(3, i), 1, "wxyz"[i]) for i in range(4)]
    s = song([Section("verse", 0, 2), Section("chorus", 2, 6)], notes)
    a, b = clean_sections(s), clean_sections(s)
    assert [(x.start_bar, x.num_bars) for x in a[0]] == [(x.start_bar, x.num_bars) for x in b[0]]
    s.sections = a[0]
    text = song_to_abc_v2(s)
    assert "% section 1/2 | 3 bars" in text and "% section 2/2 | 5 bars" in text


def test_pathology_flags_detect_double_time():
    s = song([Section("verse", 0, 8)], [(bar(0, 0), 0, "a"), (bar(4, 0), 0, "b")])
    s.tempo_bpm = 230
    assert pathology_flags(s)["double_time_suspect"]
