import json

from qwen_abc.abc import parse_abc, song_to_abc
from qwen_abc.canonical import Chord, Note, Section, Song, comparable
from qwen_abc.source import song_from_leadsheet
from qwen_abc.splits import load_inherited_split, lyric_shingles, lyrics_exact_hash, melody_shingles, resolve_leakage


def song_with(lyrics, pitches, sid="s"):
    notes = [Note(i * 4, 4, p, (l,)) for i, (l, p) in enumerate(zip(lyrics, pitches))]
    bars = -(-len(notes) // 4)
    return Song(sid, 4, 100, "C major", [4] * bars, [Section("verse", 0, bars)], notes, [Chord(0, 16 * bars, "C")])


def rows_for(songs_by_split):
    out = {}
    for split, songs in songs_by_split.items():
        for s in songs:
            out[s.song_id] = {"split": split, "lyric_sh": lyric_shingles(s), "melody_sh": melody_shingles(s),
                              "lyrics_hash": lyrics_exact_hash(s)}
    return out


LYR_A = list("月台火车在开动祝你一路顺风甜酸苦辣在无言中")
LYR_B = list("城市的灯光照亮了回家的路我们一起走过的日子")
PITCH_A = [60, 62, 64, 65, 67, 65, 64, 62, 60, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 59, 60]
PITCH_B = [55, 57, 59, 55, 64, 62, 60, 59, 57, 55, 67, 65, 64, 62, 65, 64, 62, 60, 59, 57, 55, 53]


def test_exact_lyric_duplicate_is_removed_from_test_not_train():
    tr = song_with(LYR_A, PITCH_A, "train1")
    te = song_with(LYR_A, PITCH_B, "test1")  # re-upload with a different transcription
    final, ex = resolve_leakage(rows_for({"train": [tr], "test": [te]}), 0.3, 0.3)
    assert final == {"train1": "train"}
    assert ex[0]["song_id"] == "test1" and ex[0]["reason"].startswith("lyrics_exact")


def test_melody_near_duplicate_with_new_lyrics_is_removed():
    tr = song_with(LYR_A, PITCH_A, "train1")
    te = song_with(LYR_B, [p + 5 for p in PITCH_A], "test1")  # transposed, other language version
    final, ex = resolve_leakage(rows_for({"train": [tr], "test": [te]}), 0.3, 0.3)
    assert "test1" not in final and ex[0]["reason"].startswith("melody_near")


def test_unrelated_song_survives_and_validation_yields_to_test():
    tr = song_with(LYR_A, PITCH_A, "train1")
    te = song_with(LYR_B, PITCH_B, "test1")
    va = song_with(LYR_B, PITCH_B, "val1")
    final, ex = resolve_leakage(rows_for({"train": [tr], "test": [te], "validation": [va]}), 0.3, 0.3)
    assert final == {"train1": "train", "test1": "test"}
    assert ex[0]["song_id"] == "val1" and ex[0]["reason"].endswith("_of_test")


def test_inherited_split_rejects_song_in_two_splits(tmp_path):
    for split, ids in (("train", ["a", "b"]), ("validation", ["c"]), ("test", ["b"])):
        (tmp_path / f"{split}_songs.jsonl").write_text("".join(json.dumps({"song_id": i}) + "\n" for i in ids))
    try:
        load_inherited_split(tmp_path)
    except ValueError:
        return
    raise AssertionError("overlap not detected")


def _mini_leadsheet():
    beats = [{"time_sec": 1.0 + 0.5 * i, "is_downbeat": i % 4 == 0} for i in range(17)]  # 4 bars @120
    sylls = [{"syllable_id": f"s{i}", "text": t, "normalized_text": t, "start_sec": 1.0 + i * 0.5,
              "end_sec": 1.0 + i * 0.5 + 0.45, "line_id": "l0"} for i, t in enumerate("我们一起走")]
    sylls.append({"syllable_id": "s5", "text": "吧", "normalized_text": "吧", "start_sec": 4.0, "end_sec": 4.2,
                  "line_id": "l0"})
    notes, items = [], []
    for i in range(5):
        on = 1.0 + i * 0.5
        notes.append({"note_id": f"n{i}", "onset_sec": on, "offset_sec": on + 0.5, "onset_grid_time_sec": on,
                      "offset_grid_time_sec": on + 0.5, "pitch_midi": 60 + i})
        items.append({"note_id": f"n{i}", "lyric_unit_id": f"s{i}", "is_lyric_continuation": False})
    # an unlinked note inside s4's window -> melisma recovery; s5 has no note -> joined
    notes.append({"note_id": "n5", "onset_sec": 3.25, "offset_sec": 3.5, "onset_grid_time_sec": 3.25,
                  "offset_grid_time_sec": 3.5, "pitch_midi": 67})
    notes.append({"note_id": "n6", "onset_sec": 5.0, "offset_sec": 6.0, "onset_grid_time_sec": 5.0,
                  "offset_grid_time_sec": 6.0, "pitch_midi": 62})
    return {
        "song_id": "mini", "global": {"meter": "4/4", "tempo_bpm": 120, "key": "C# major", "language": "zh"},
        "beats": beats, "sections": [{"label": "inst", "start_sec": 0.0}, {"label": "verse", "start_sec": 3.0}],
        "syllables": sylls, "lines": [{"line_id": "l0", "text": "我们一起走吧"}], "melody_notes": notes,
        "note_lyric_alignment": {"items": items},
        "recognized_chords": [{"symbol": "N", "root": None, "quality": "N", "bass": None, "start_sec": 0.0},
                              {"symbol": "C#:maj", "root": "C#", "quality": "maj", "bass": None, "start_sec": 1.0},
                              {"symbol": "F#:min7", "root": "F#", "quality": "min7", "bass": "b7", "start_sec": 3.0}],
    }


def test_source_conversion_semantics_and_determinism():
    data = _mini_leadsheet()
    s1, stats = song_from_leadsheet(data)
    s2, _ = song_from_leadsheet(json.loads(json.dumps(data)))
    assert song_to_abc(s1) == song_to_abc(s2)
    assert s1.key == "Db major" and s1.bar_beats == [4, 4, 4, 4]
    assert [(s.label, s.start_bar) for s in s1.sections] == [("instrumental", 0), ("verse", 1)]
    assert s1.notes[0].onset == 0 and s1.notes[0].duration == 4
    # n4 (走) at ticks 16..20; unlinked n5 at tick 18 lies in 走's window -> melisma, n4 truncated
    lyr = [(n.onset, n.duration, n.lyric, n.melisma) for n in s1.notes]
    assert lyr[5] == (18, 2, None, True) and lyr[4][1] == 2
    assert stats["melisma_inherited"] == 1
    # 吧 had no note: joined onto the previous attack of the same line, not dropped
    assert lyr[4][2] == ("走", "吧") and stats["unmatched_syllable_joined"] == 1
    # the note outside every syllable window stays wordless
    assert s1.notes[-1].lyric is None and not s1.notes[-1].melisma
    # N at 0.0 is overridden by C#:maj on the same (first) beat; roots spelled for Db
    assert [c.symbol for c in s1.chords] == ["Db", "Gbm7/E"]
    res = parse_abc(song_to_abc(s1))
    assert res.strict_ok and comparable(res.song) == comparable(s1)


def test_bar_regularization_keeps_anchors_and_beats():
    from collections import Counter

    from qwen_abc.source import _regularize_bars

    st = Counter()
    # bars: 4 | 2 2 | 4 | 8 | 4 | 5 3 | 3 | 4 | 2 2 (a section starts between the last two)
    b = [0, 4, 6, 8, 12, 20, 24, 29, 32, 35, 39, 41, 43]
    out = _regularize_bars(b, anchors={0, 41}, meter_num=4, stats=st)
    # 2+2 -> 4; 8 -> 4+4; 5+3+3 = 11 -> 4+4+3 (the run's anchored ends stay);
    # the final 2 | 2 straddles a section start -> kept
    assert out == [0, 4, 8, 12, 16, 20, 24, 28, 32, 35, 39, 41, 43]
    assert st["irregular_runs_regularized"] == 2 and st["irregular_runs_packed_with_remainder"] == 1
    # a single odd-length bar is left alone
    assert _regularize_bars([0, 4, 9, 13], {0}, 4, Counter()) == [0, 4, 9, 13]
