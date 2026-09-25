"""paper_eval: interventions change exactly one control, the paired-seed sampler is exact and
batch-invariant, and the pairwise distances behave on known cases."""

import copy
import math

import pytest
import torch

from paper_eval.interventions import (change_bars, change_label, control_family, controls_changed, pair_distances,
                                      replace_lyrics, scale_tempo, seq_distance, syllables_of, transpose_key)
from paper_eval.sampler import GumbelCRN
from qwen_abc.canonical import Chord, Note, Section, Song


def _spec():
    return {"language": "zh", "meter": "4/4", "tempo_bpm": 100, "key": "G major", "sections": [
        {"label": "intro", "bars": 4, "beats": [4, 4, 4, 4], "lines": [], "line_ids": []},
        {"label": "verse", "bars": 8, "beats": [4] * 7 + [2], "lines": ["我们一起走", "看月亮"], "line_ids": [0, 1]},
        {"label": "chorus", "bars": 8, "beats": [4] * 8, "lines": ["唱一首歌 hello"], "line_ids": [2]},
    ]}


@pytest.mark.parametrize("delta", [-4, -2, 2, 4])
def test_change_bars_is_one_control(delta):
    s = _spec()
    out = change_bars(s, 1, delta)
    assert out["sections"][1]["bars"] == 8 + delta == len(out["sections"][1]["beats"])
    assert out["sections"][1]["beats"][-1] == 2          # the irregular trailing bar survives
    assert controls_changed(s, out) == ["sec1.bars"]
    assert control_family(controls_changed(s, out)) == "bars"
    assert s["sections"][1]["bars"] == 8                  # input not mutated


def test_other_interventions_are_one_control():
    s = _spec()
    assert controls_changed(s, change_label(s, 2, "bridge")) == ["sec2.label"]
    assert controls_changed(s, transpose_key(s, 5)) == ["key"]
    assert transpose_key(s, 5)["key"] == "C major"
    assert transpose_key(s, -3)["key"] == "E major"
    assert controls_changed(s, scale_tempo(s, 1.25)) == ["tempo"]
    assert scale_tempo(s, 0.8)["tempo_bpm"] == 80
    donor = list("春风吹过山岗天边有云彩飘")
    out = replace_lyrics(s, donor)
    assert control_family(controls_changed(s, out)) == "lyrics"
    for a, b in zip(s["sections"], out["sections"]):   # syllable load per section preserved
        assert len(syllables_of(a)) == len(syllables_of(b))
    out1 = replace_lyrics(s, donor, 2)
    assert controls_changed(s, out1) == ["sec2.lyrics"]


def test_too_few_bars_to_remove():
    s = _spec()
    with pytest.raises(ValueError):
        change_bars(s, 0, -5)


def test_gumbel_crn_is_exact_top_p_sampling():
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0]])
    T, p = 1.0, 0.9
    probs = (logits / T).softmax(-1)[0]
    # expected truncated distribution: smallest prefix (by prob) with mass >= p
    order = probs.argsort(descending=True)
    keep, mass = [], 0.0
    for i in order.tolist():
        keep.append(i)
        mass += probs[i].item()
        if mass >= p:
            break
    want = torch.zeros(5)
    want[keep] = probs[keep] / probs[keep].sum()
    counts = torch.zeros(5)
    n = 20000
    for s in range(n):
        proc = GumbelCRN([s], prompt_len=3, temperature=T, top_p=p)
        out = proc(torch.zeros(1, 3, dtype=torch.long), logits.clone())
        counts[out.argmax(-1)] += 1
    got = counts / n
    assert torch.allclose(got, want, atol=0.015), (got, want)


def test_gumbel_crn_noise_depends_only_on_row_seed_and_step():
    V = 50
    scores = torch.randn(3, V)
    a = GumbelCRN([7, 8, 9], prompt_len=10, temperature=1.0, top_p=1.0)(torch.zeros(3, 12, dtype=torch.long), scores.clone())
    # same seed 8 in a different batch position, different batch size, different prompt length but same step (2)
    b = GumbelCRN([8, 1], prompt_len=4, temperature=1.0, top_p=1.0)(torch.zeros(2, 6, dtype=torch.long), scores[[1, 0]].clone())
    assert torch.equal(a[1], b[0])
    c = GumbelCRN([8], prompt_len=4, temperature=1.0, top_p=1.0)(torch.zeros(1, 7, dtype=torch.long), scores[[1]].clone())
    assert not torch.equal(a[1], c[0])           # step 3 != step 2


def _song(pitches, shift=0):
    notes = [Note(onset=4 * i, duration=4, pitch=p + shift, lyric=("啊",)) for i, p in enumerate(pitches)]
    return Song(song_id="t", meter_num=4, tempo_bpm=100, key="C major", bar_beats=[4] * (len(pitches) // 4 + 1),
                sections=[Section("verse", 0, len(pitches) // 4 + 1)], notes=notes,
                chords=[Chord(0, 16, "C"), Chord(16, 16, "G")])


def test_pair_distances():
    a = _song([60, 62, 64, 65, 67, 65, 64, 62])
    assert pair_distances(a, a)["d_melody"] == 0.0
    b = _song([60, 62, 64, 65, 67, 65, 64, 62], shift=5)
    d = pair_distances(a, b)
    assert d["d_melody"] == 1.0 and d["d_contour"] == 0.0 and d["d_rhythm"] == 0.0
    assert pair_distances(a, b, transpose=5)["d_melody"] == 0.0
    assert math.isclose(seq_distance(list("abcd"), list("abxd")), 0.25)


def test_lcs_fast_matches_reference():
    import random
    from paper_eval.seqsim import lcs_fast
    from qwen_abc.metrics import lcs_len
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.choice("abcde") for _ in range(rng.randint(0, 80))]
        b = [rng.choice("abcdef") for _ in range(rng.randint(0, 80))]
        assert lcs_fast(a, b) == lcs_len(a, b)


def test_lcs_pairs_and_tones():
    from paper_eval.seqsim import lcs_fast, lcs_pairs
    a, b = list("abcbdab"), list("bdcaba")
    pr = lcs_pairs(a, b)
    assert len(pr) == lcs_fast(a, b)
    assert all(a[i] == b[j] for i, j in pr)
    assert all(i1 < i2 and j1 < j2 for (i1, j1), (i2, j2) in zip(pr, pr[1:]))
    from paper_eval.tone_melody import tones_of
    assert tones_of(list("妈麻马骂")) == [1, 2, 3, 4]
    assert tones_of(list("你好")) == [2, 3]          # 3-3 sandhi
