"""paper_eval.memorization: n-gram hashing, containment and run length on toy data."""

import numpy as np

from paper_eval.memorization import abc_tokens, longest_common_run, ngram_hashes, song_features
from qwen_abc.canonical import Chord, Note, Section, Song


def _song(pitches, key="C major", shift=0):
    notes = [Note(onset=4 * i, duration=4, pitch=p + shift) for i, p in enumerate(pitches)]
    nb = len(pitches) // 4 + 1
    return Song(song_id="t", meter_num=4, tempo_bpm=100, key=key, bar_beats=[4] * nb,
                sections=[Section("verse", 0, nb)], notes=notes, chords=[Chord(0, 16, "C"), Chord(16, 16, "G7")])


def test_ngram_hashes_deterministic_and_order_sensitive():
    a = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(ngram_hashes(a, 3), ngram_hashes(a.copy(), 3))
    assert set(ngram_hashes(a, 3).tolist()) != set(ngram_hashes(a[::-1].copy(), 3).tolist())
    assert len(ngram_hashes(a, 6)) == 0


def test_features_are_transposition_invariant():
    p = [60, 62, 64, 65, 67, 65, 64, 62, 60, 59, 60, 62]
    a, b = song_features(_song(p)), song_features(_song(p, key="D major", shift=2))
    for rep in ("interval", "contour", "rhythm", "motif", "chord"):
        if rep != "chord":
            assert a[rep] == b[rep]
    assert a["bars"] == b["bars"]           # key-relative bar signatures


def test_longest_common_run_and_abc_tokens():
    assert longest_common_run([1, 2, 3, 9, 4, 5], [0, 1, 2, 3, 4, 5]) == 3
    toks = abc_tokens('X:1\nK:C\n% c\n[r:2] "C"C2 D/ z | [r:1] E4- |\nw: 啊 啊\n')
    assert '"C"' in toks and "C2" in toks and "|" in toks and not any(t.startswith("[r") for t in toks)


def test_informative_run_ignores_repeated_pitches():
    from paper_eval.memorization import informative_run, longest_common_run_seq
    rep = ["0"] * 20
    assert longest_common_run_seq(rep, rep)[0] == 20
    assert informative_run(rep, rep)["nn_run_nz"] == 0
    tune = ["2", "2", "1", "-3", "5", "-2"]
    r = informative_run(["0"] + tune + ["0"], ["0", "0"] + tune)
    assert r["nn_run_nz"] == 6 and r["nn_run_nz_distinct"] == 5
