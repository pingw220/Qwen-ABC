"""Best-of-n selection: the rules, and the guard that keeps the reference out of them."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_best_of_n import rank_key, signal  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_best_of_n.py"


def row(strict=False, plan=0.0, recall=0.0, in_key=0.0, cram=0.0):
    return {
        "song_id": "s", "strict_ok": strict, "parse_ok": True,
        "metrics": {"section_plan_exact": plan, "lyric_recall": recall, "cram_syllable_frac": cram,
                    "melody_in_key_frac": in_key, "chord_root_in_key_frac": in_key,
                    "chord_tone_frac": 0.5, "early_eos": 0.0,
                    # present in real rows, and must never be used
                    "melody_distance_to_reference": 0.0},
    }


def choose(rows, rule):
    return max(enumerate(rows), key=lambda t: rank_key(t[1], rule, t[0]))[0]


def test_a_reference_derived_signal_is_refused_rather_than_used():
    with pytest.raises(SystemExit):
        signal(row(), "melody_distance_to_reference")


def test_seed0_rule_is_the_single_sample_baseline():
    rows = [row(strict=False), row(strict=True, plan=1.0, recall=1.0)]
    assert choose(rows, "seed0") == 0


def test_valid_rule_takes_the_first_strict_valid_sample():
    rows = [row(strict=False), row(strict=False), row(strict=True), row(strict=True)]
    assert choose(rows, "valid") == 2


def test_lex_rule_prefers_validity_then_plan_then_lyrics():
    rows = [
        row(strict=True, plan=0.0, recall=1.0),
        row(strict=True, plan=1.0, recall=0.5),
        row(strict=True, plan=1.0, recall=0.9),
        row(strict=False, plan=1.0, recall=1.0),
    ]
    assert choose(rows, "lex") == 2


def test_sum_rule_trades_cramming_against_the_rest():
    clean = row(strict=True, plan=1.0, recall=0.95, in_key=1.0, cram=0.02)
    crammed = row(strict=True, plan=1.0, recall=0.95, in_key=1.0, cram=0.40)
    assert choose([crammed, clean], "sum") == 1


def test_ties_fall_back_to_the_lowest_seed():
    same = [row(strict=True, plan=1.0, recall=1.0) for _ in range(3)]
    assert choose(same, "lex") == 0 and choose(same, "sum") == 0


def test_cli_writes_one_chosen_row_per_song_and_a_record(tmp_path):
    gen = tmp_path / "generations"
    gen.mkdir()
    for seed, r in enumerate([row(strict=False), row(strict=True, plan=1.0, recall=1.0, in_key=1.0)]):
        (gen / f"song1_s{seed}.json").write_text(json.dumps(r), encoding="utf-8")
    out = tmp_path / "picked"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--generations", str(gen), "--out", str(out),
                           "--rule", "sum"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "generations" / "song1_s0.json").exists()
    assert not (out / "generations" / "song1_s1.json").exists()
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert sel["rule"] == "sum" and sel["songs"] == 1
    assert sel["per_song"][0]["chosen_seed"] == 1
    assert sel["chosen_strict_valid"] == 1.0
    assert "melody_distance_to_reference" not in sel["safe_signals"]
