"""component_sota listening study: participant-facing files carry no system names, A/B sides are
balanced within every stratum across lists, and the analysis runs end-to-end on simulated data."""

import json
import re
from collections import Counter

import pytest

from paper_eval.component_sota import listening_study as ls

NAMES = ("qwen", "e3b", "mel", "csl", "am2", "accomontage", "fastsinger", "soulx", "ref", "infill", "regenerat", "relabel")


def _study():
    if not (ls.STUDY / "KEY_DO_NOT_SHARE.json").exists():
        pytest.skip("listening study not built")
    return json.loads((ls.STUDY / "KEY_DO_NOT_SHARE.json").read_text())


def test_participant_files_are_blind():
    _study()
    texts = [p.read_text(encoding="utf-8").lower() for p in (ls.STUDY / "lists").glob("list_*.csv")]
    texts.append((ls.STUDY / "questions.json").read_text(encoding="utf-8").lower())
    for t in texts:
        words = set(re.findall(r"[a-z0-9_]+", t))
        for n in NAMES:
            assert not any(w == n or (len(n) > 4 and w.startswith(n)) for w in words), n


def test_ab_balance_per_stratum():
    key = _study()
    for it in key["items"]:
        if it.get("fixed_order"):
            continue
        sides = Counter(it["A_is"].values())
        assert len(sides) == 2 and abs(sides[it["systems"][0]] - sides[it["systems"][1]]) <= 1


def test_simulated_analysis_runs(tmp_path):
    _study()
    csv_path = ls.simulate(ls.STUDY, tmp_path / "SIMULATED_responses.csv", n_raters=8)
    res = ls.analyze(csv_path, n_boot=200, simulated=True)
    assert "SIMULATED" in res["banner"]
