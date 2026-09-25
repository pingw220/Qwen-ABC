"""listening_study/analysis.py on SYNTHETIC responses (no listener data exists; generated here only)."""

import csv
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ls_analysis", ROOT / "reports/paper_final/listening_study/analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def _write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _study(tmp):
    items, lst = [], []
    for i in range(20):
        cond = "key_p5" if i < 10 else "bars_m4"
        items.append({"item_id": f"i{i}", "kind": "test", "condition": cond, "question": "higher_key" if i < 10 else "longer_section",
                      "song_id": f"s{i}", "realized_complied": "True", "realized_register_agrees_with_request": "True" if i < 10 else ""})
        lst.append({"list_id": 0, "item_id": f"i{i}", "edited_is": "A" if i % 2 else "B"})
    items.append({"item_id": "c0", "kind": "catch", "condition": "catch_identical", "question": "longer_section", "song_id": "c",
                  "realized_complied": "False", "realized_register_agrees_with_request": ""})
    lst.append({"list_id": 0, "item_id": "c0", "edited_is": "A"})
    (tmp / "lists").mkdir()
    _write(tmp / "items.csv", items)
    _write(tmp / "lists" / "list_0.csv", lst)
    return items, lst


def test_synthetic_accuracy_and_exclusion(tmp_path):
    items, lst = _study(tmp_path)
    rng = random.Random(0)
    edited = {r["item_id"]: r["edited_is"] for r in lst}
    rows = []
    for p in range(8):
        for it in items:
            if it["kind"] == "catch":
                resp = "A" if p == 7 else "none"          # participant 7 fails the catch item
            elif it["condition"] == "key_p5":
                resp = edited[it["item_id"]] if rng.random() < 0.95 else ("B" if edited[it["item_id"]] == "A" else "A")
            else:  # bars_m4: correct answer is the ORIGINAL clip; synthetic listeners guess
                resp = rng.choice(["A", "B"])
            rows.append({"participant_id": f"p{p}", "list_id": 0, "item_id": it["item_id"],
                         "question_id": it["question"], "response": resp, "rt_ms": 1000, "timestamp": ""})
            if it["kind"] == "test":
                rows.append({"participant_id": f"p{p}", "list_id": 0, "item_id": it["item_id"], "question_id": "edit_happened",
                             "response": "yes", "rt_ms": 1, "timestamp": ""})
    (tmp_path / "responses").mkdir()
    _write(tmp_path / "responses" / "synthetic.csv", rows)
    res = A.analyse(tmp_path, tmp_path / "responses", max_catch_errors=0, n_boot=500)
    assert res["excluded"] == ["p7"]
    key = res["conditions"]["key_p5"]
    assert key["accuracy_3way"]["mean"] > 0.85 and key["accuracy_3way"]["lo"] > 0.5
    bars = res["conditions"]["bars_m4"]
    assert 0.25 < bars["accuracy_forced_AB"]["mean"] < 0.75
    assert key["edit_happened_agrees_with_automatic"] == 1.0
    assert key["fleiss_kappa_2afc"] > bars["fleiss_kappa_2afc"]


def test_agreement_statistics():
    perfect = {"a": ["x", "x", "x"], "b": ["y", "y", "y"]}
    assert abs(A.fleiss_kappa(perfect) - 1.0) < 1e-9
    assert abs(A.krippendorff_alpha(perfect) - 1.0) < 1e-9
    assert abs(A.krippendorff_alpha({"a": [1, 1], "b": [5, 5], "c": [3, 3]}, "interval") - 1.0) < 1e-9
    # Krippendorff (2011) nominal example, 4 coders x 12 units (missing values omitted): alpha = 0.743
    data = {1: [1, 1, 1], 2: [2, 2, 3, 2], 3: [3, 3, 3, 3], 4: [3, 3, 3, 3], 5: [2, 2, 2, 2], 6: [1, 2, 3, 4],
            7: [4, 4, 4, 4], 8: [1, 1, 2, 1], 9: [2, 2, 2, 2], 10: [5, 5, 5], 11: [1, 1], 12: [3]}
    assert abs(A.krippendorff_alpha(data) - 0.743) < 0.002
