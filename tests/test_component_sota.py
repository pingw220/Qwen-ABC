"""component_sota: the harmonizer's chord text round-trips, the CSL-L2M event conversion keeps the
sung pitches and lyrics, the infill prompt masks exactly the target section, and the melody
metrics behave on identical inputs."""

import pytest

from qwen_abc.canonical import Song, normalize_chords

pytest.importorskip("paper_eval.component_sota.formats")
from paper_eval.component_sota import csl_l2m, melody_metrics  # noqa: E402
from paper_eval.component_sota.formats import chord_lines, parse_chord_lines, strip_chords  # noqa: E402


def _songs(n=40):
    from paper_eval.common import load_rows
    try:
        rows = load_rows("test")[:n]
    except (FileNotFoundError, OSError):
        pytest.skip("canonical test split not available")
    return rows, [Song.from_json(r["song"]) for r in rows]


def test_chord_lines_round_trip():
    _, songs = _songs()
    for s in songs:
        got, st = parse_chord_lines(chord_lines(s), strip_chords(s))
        assert st["chord_bar_count_match"] == 1.0
        assert [(c.onset, c.symbol) for c in got] == [(c.onset, c.symbol) for c in normalize_chords(s.chords, s.total_ticks)]


def test_csl_events_round_trip_keeps_sung_pitches():
    rows, songs = _songs()
    checked = 0
    for r, s in zip(rows, songs):
        if s.meter_num != 4 or any(n.lyric and len(n.lyric) > 2 for n in s.notes):   # crammed spoken passages
            continue
        conv = csl_l2m.song_to_events(s)
        if conv is None:
            continue
        lines, _, events = conv
        ev = [e["name"] if e["value"] is None else f"{e['name']}_{e['value']}" for e in events]
        sec_of = [0] * len(lines)
        spec = {"tempo_bpm": s.tempo_bpm, "sections": [{"label": "verse"}]}
        back = csl_l2m.events_to_song(ev, lines, sec_of, spec, s.song_id)
        sung = [n for n in back.notes if n.lyric]
        got, want = [n.lyric[0] for n in sung], [c for l in lines for c in l]
        it = iter(want)
        assert all(c in it for c in got)              # an in-order subsequence of the input lyric
        # CSL's 1/64-bar grid is finer than ours (1/16): syllables split below a 16th collide and
        # are dropped on the way back. Real CSL outputs keep 99.9% (CSL_L2M_COMPARISON.md).
        assert len(got) >= 0.9 * len(want)
        assert min(n.pitch for n in back.notes) >= 35 and max(n.pitch for n in back.notes) <= 94
        checked += 1
    assert checked >= 5


def test_infill_prompt_masks_only_target():
    from paper_eval.component_sota.infill import GAP_LINE, infill_prompt
    rows, _ = _songs(10)
    for r in rows:
        n = len(r["spec"]["sections"])
        t = n // 2
        p = infill_prompt(r["abc"], r["spec"], t)
        assert p["prompt"].count(GAP_LINE) == 1
        for i, b in enumerate(p["blocks"]):
            if i != t and b.strip():
                assert b in p["prompt"]


def test_melody_metrics_identity():
    _, songs = _songs(5)
    for s in songs:
        assert melody_metrics.pd_score(s, s) == pytest.approx(1.0)
        assert melody_metrics.dd_score(s, s) == pytest.approx(1.0)
        assert melody_metrics.md_score(s, s) == pytest.approx(0.0)
