"""annotation_rescore on SYNTHETIC annotations (no human data exists; these records are made up
for the test and never written to the package)."""

import json

from paper_eval.annotation_rescore import human_plan, load_annotations, rescore


def _spec():
    return {"meter": "4/4", "tempo_bpm": 100, "key": "C major", "language": "zh", "sections": [
        {"label": "intro", "bars": 4, "beats": [4] * 4, "lines": []},
        {"label": "verse", "bars": 8, "beats": [4] * 8, "lines": ["一二三四"]},
        {"label": "verse", "bars": 8, "beats": [4] * 8, "lines": ["五六七八"]},
    ]}


def _row(plan, recall=1.0):
    return {"song": {"sections": [{"label": l, "num_bars": b} for l, b in plan]}, "metrics": {"lyric_recall": recall}}


def _record(annotator="synthetic_A", confidence=4):
    # SYNTHETIC: the annotator says section 2 is really a chorus and has 2 wrong syllables
    return {"schema_version": "1.0", "song_id": "S", "annotator_id": annotator, "confidence": confidence, "sections": [
        {"index": 0, "pseudo_label": "intro", "pseudo_bars": 4, "label_correct": "yes", "bars_correct": "yes"},
        {"index": 1, "pseudo_label": "verse", "pseudo_bars": 8, "label_correct": "yes", "bars_correct": "yes"},
        {"index": 2, "pseudo_label": "verse", "pseudo_bars": 8, "label_correct": "no", "correct_label": "chorus",
         "bars_correct": "yes", "wrong_lyric_syllables": 2},
    ]}


def test_human_plan_applies_corrections():
    assert human_plan(_record()) == [("intro", 4), ("verse", 8), ("chorus", 8)]
    r = _record()
    r["corrected_sections"] = [{"label": "intro", "bars": 4}, {"label": "verse", "bars": 16}]
    assert human_plan(r) == [("intro", 4), ("verse", 16)]


def test_rescore_decomposes_deviations(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([_record()]))
    ann = load_annotations(tmp_path)
    systems = {"sys": {"S": [
        _row([("intro", 4), ("verse", 8), ("chorus", 8)], 0.75),   # deviates toward the human label
        _row([("intro", 4), ("verse", 8), ("verse", 8)], 0.75),    # follows the (wrong) pseudo label
        _row([("intro", 4), ("bridge", 8), ("verse", 8)], 0.75),   # a genuine model error
    ]}}
    res = rescore(ann, systems, {"S": _spec()}, n_boot=200)
    r = res["systems"]["sys"]
    d = r["deviation_decomposition"]
    assert d["label_error_sections"] == 1 and d["model_error_sections"] == 1 and d["followed_wrong_label_sections"] == 2  # samples 2 and 3 both keep section 2 as verse
    assert abs(r["structure_exact"]["b"] - 1 / 3) < 1e-9 and abs(r["structure_exact"]["a"] - 1 / 3) < 1e-9
    # 8 prompt syllables, 2 marked wrong: recall 0.75 of 8 = 6 of the 6 valid ones
    assert abs(r["lyric_recall"]["a"] - 1.0) < 1e-9
    assert res["label_noise"]["label_corrected"] == 1


def test_agreement_between_annotators(tmp_path):
    b = _record("synthetic_B", 2)
    b["sections"][2]["label_correct"] = "yes"
    (tmp_path / "a.json").write_text(json.dumps([_record(), b]))
    res = rescore(load_annotations(tmp_path), {}, {"S": _spec()}, n_boot=100)
    assert res["annotator_agreement"]["section_pairs"] == 3
    assert abs(res["annotator_agreement"]["label_agreement"] - 2 / 3) < 1e-9
