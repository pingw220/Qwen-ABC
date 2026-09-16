"""Long-range task construction, round-2 metrics, and trainer update accounting."""
import random

from qwen_abc.abc_v2 import song_to_abc_v2
from qwen_abc.canonical import Chord, Note, Section, Song
from qwen_abc.longrange import (GAP_LINE, choose_infill_target, continuation_prompt, continuation_start, infill_example,
                                split_abc_sections)
from qwen_abc.metrics import lcs_len, lcs_matches, long_structure_metrics, motif_similarity, song_metrics
from qwen_abc.prompt import song_to_spec
from qwen_abc.train import TrainConfig, make_batches, update_plan


def pop_song():
    """intro(2) verse(4) chorus(4) verse(4) chorus(4); both choruses share a melody transposed by +2."""
    labels = [("intro", 2), ("verse", 4), ("chorus", 4), ("verse", 4), ("chorus", 4)]
    sections, notes, bar = [], [], 0
    motif = [0, 2, 4, 2, 0, 5, 4, 2]
    syll = iter([chr(0x4E00 + i) for i in range(1000)])
    for li, (label, n) in enumerate(labels):
        sections.append(Section(label, bar, n))
        if label != "intro":
            base = 60 + (2 if label == "chorus" and li == 4 else 0) + (7 if label == "chorus" else 0)
            for b in range(n):
                for k in range(4):
                    notes.append(Note((bar + b) * 16 + k * 4, 4, base + motif[(b * 4 + k) % 8], (next(syll),), line=li * 10 + b))
        bar += n
    return Song("p", 4, 100, "C major", [4] * bar, sections, notes, [Chord(0, bar * 16, "C")])


def test_split_abc_sections_is_lossless_and_infill_hides_exactly_one_section():
    song = pop_song()
    spec = song_to_spec(song)
    abc = song_to_abc_v2(song)
    header, blocks = split_abc_sections(abc)
    assert header + "".join(blocks) == abc and len(blocks) == 5
    t = choose_infill_target("p", spec)
    assert t in (3, 4)  # later-half sections whose label occurred before
    ex = infill_example(song, spec, t)
    assert ex["completion"] == blocks[t]
    assert ex["prompt"].count(GAP_LINE) == 1
    assert blocks[1] in ex["prompt"] and blocks[t] not in ex["prompt"]
    assert choose_infill_target("p", spec) == t  # deterministic


def test_continuation_prompt_contains_prefix_only():
    song = pop_song()
    spec = song_to_spec(song)
    k = continuation_start(spec)
    c = continuation_prompt(song, spec, k)
    header, blocks = split_abc_sections(song_to_abc_v2(song))
    assert k == 4 and c["prompt"].endswith("".join(blocks[:k])) and c["reference_rest"] == blocks[4]


def test_motif_similarity_is_transposition_invariant():
    song = pop_song()
    ch1 = [n for n in song.notes if 96 <= n.onset < 160]
    ch2 = [n for n in song.notes if 224 <= n.onset < 288]
    assert motif_similarity(ch1, ch2) == 1.0
    m = long_structure_metrics(song, song_to_spec(song))
    assert m["chorus_chorus_motif_sim"] == 1.0 and m["chorus_motif_preservation"] == 1.0


def test_structure_metrics_detect_early_end_and_late_lyrics():
    song = pop_song()
    spec = song_to_spec(song)
    full = song_metrics(song, spec)
    assert full["section_completion_ratio"] == 1.0 and full["late_lyric_recall"] == 1.0 and full["section_bars_seq_match"] == 1.0
    cut = Song("p", 4, 100, "C major", [4] * 14, song.sections[:4], [n for n in song.notes if n.onset < 14 * 16],
               [Chord(0, 14 * 16, "C")])
    m = song_metrics(cut, spec)
    assert m["fewer_sections_than_requested"] == 1.0 and m["late_section_present"] < 1.0
    assert m["late_lyric_recall"] < 0.5 and m["lyric_recall"] > 0.5


def test_repetition_metrics():
    song = pop_song()
    m = song_metrics(song, song_to_spec(song))
    assert 0 <= m["repeated_bar_frac"] <= 1 and 0 <= m["interval_4gram_repeat_frac"] <= 1
    assert m["repeated_bar_frac"] > 0  # the two verses repeat exactly


def test_lcs_matches_agrees_with_lcs_len():
    rng = random.Random(0)
    for _ in range(50):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        idx = lcs_matches(a, b)
        assert len(idx) == lcs_len(a, b) and idx == sorted(set(idx))


def test_update_plan_keeps_micro_batches_per_update_constant_across_world_sizes():
    cfg = TrainConfig(mode="sft", data_dir="x", output_dir="y", tokens_per_micro_batch=16384, tokens_per_update=524288, epochs=2)
    plans = [update_plan(1528, cfg, w) for w in (1, 2, 4, 8)]
    assert {p["micro_batches_per_update"] for p in plans} == {32}
    assert {p["steps_per_epoch"] for p in plans} == {48}
    assert plans[0]["total_steps"] == 96 and plans[2]["grad_accum"] == 8
    base = TrainConfig(mode="sft", data_dir="x", output_dir="y")
    assert update_plan(1528, base, 1)["total_steps"] == 382  # round-1 E0 accounting


def test_ddp_rank_partition_covers_each_update_exactly_once():
    ex = [{"input_ids": [0] * random.Random(i).randint(100, 4000)} for i in range(300)]
    batches = make_batches(ex, 16384, random.Random(3))
    per_update, world = 8, 4
    for start in range(0, len(batches), per_update):
        group = batches[start: start + per_update]
        parts = [group[r::world] for r in range(world)]
        assert sorted(i for p in parts for b in p for i in b) == sorted(i for b in group for i in b)
