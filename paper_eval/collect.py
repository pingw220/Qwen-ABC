"""Where every system's generation rows live, and a flat per-sample table of their metrics.

A *system* is a model + decoding + prompt condition whose samples are exchangeable. Legacy
systems are the round-2/R3/R4 evaluation directories (read-only); paper-final systems are
``experiments/paper_final/gen/<model>/<condition>``; MIDI-LLM rows come from
``paper_eval.midi_llm convert``.

  python -m paper_eval.collect            # writes reports/paper_final/data/samples.parquet
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .common import DATA_OUT, OFFLOAD, OUT_ROOT, ROOT, load_rows

EV = OFFLOAD / "evals"
X = ROOT / "experiments"
ML_ROWS = OUT_ROOT / "midi_llm" / "rows"

# legacy: system -> (dir, file glob); sample id = the _s<k> suffix
LEGACY: Dict[str, Tuple[Path, str]] = {
    "E0@T0.8": (X / "r2eval_E0_20260915_115636/v1cleantest_T0.8/generations", "*_s0.json"),
    "E0s2@T0.8": (X / "r2eval_E0_20260915_115636/seed2345_v1cleantest_T0.8/generations", "*_s0.json"),
    "E1c@T0.8": (X / "e1c_v1clean_sft_20260915_120949/eval_test_T0.8/generations", "*_s0.json"),
    "E1@T0.8": (X / "e1_v2_sft_20260915_120949/eval_test_T0.8/generations", "*_s0.json"),
    "E1s2@T0.8": (EV / "e1s2_test_T0.8/generations", "*_s0.json"),
    "E1long@T0.8": (EV / "e1long_test_T0.8/generations", "*_s0.json"),
    "E2a@T0.8": (X / "e2a_v2_bigbatch_lr50e5_20260915_123850/eval_test_T0.8/generations", "*_s0.json"),
    "E2b@T0.8": (EV / "e2b_test_T0.8/generations", "*_s0.json"),
    "E3@T0.8": (X / "e3_v2_longrange_20260915_130350/eval_test_T0.8/generations", "*_s0.json"),
    "E3b@T0.8": (EV / "e3b_test_T0.8/generations", "*_s0.json"),
    "E0@T1.0": (X / "r2eval_E0_20260915_115636/v1cleantest_T1.0_p0.95/generations", "*_s0.json"),
    "E1@T1.0": (X / "e1_v2_sft_20260915_120949/eval_test_T1.0_p0.95/generations", "*_s0.json"),
    "E1s2@T1.0": (EV / "e1s2_test_T1.0/generations", "*_s0.json"),
    "E1long@T1.0": (EV / "e1long_test_T1.0/generations", "*_s0.json"),
    "E3b@T1.0": (EV / "e3b_test_T1.0/generations", "*_s0.json"),
    "E3b@T1.0x4": (EV / "bon_e3b_test_T1.0/generations", "*_s[0-3].json"),
    "MuPT@T1.0x4": (X / "evals_r4/r4_mupt_test_T1.0_bon/generations", "*_s[0-3].json"),
    "R3A@T1.0x4": (EV / "r3a_test_T1.0_bon/generations", "*_s[0-3].json"),
    "R3C@T1.0x4": (EV / "r3c_test_T1.0_bon/generations", "*_s[0-3].json"),
    "nocram@T1.0x4": (EV / "e3b_nocram_test_T1.0/generations", "*_s[0-3].json"),
    "R5@T1.0x4": (X / "evals_r5/r5_free_test_T1.0_bon/generations", "*_s[0-3].json"),
}

LEGACY_MODEL = {"E0": "qwen_e0", "E0s2": "qwen_e0s2", "E1c": "qwen_e1c", "E1": "qwen_e1", "E1s2": "qwen_e1s2",
                "E1long": "qwen_e1long", "E2a": "qwen_e2a", "E2b": "qwen_e2b", "E3": "qwen_e3", "E3b": "qwen_e3b",
                "MuPT": "mupt", "R3A": "qwen_r3a", "R3C": "qwen_r3c", "nocram": "qwen_e3b_nocram", "R5": "qwen_r5"}


def iter_legacy(system: str) -> Iterator[Tuple[str, str, dict]]:
    d, pat = LEGACY[system]
    for p in sorted(d.glob(pat)):
        song, _, k = p.stem.rpartition("_s")
        yield song, f"s{k}", json.loads(p.read_text(encoding="utf-8"))


def iter_new(model: str, condition: str) -> Iterator[Tuple[str, str, dict]]:
    d = OUT_ROOT / "gen" / model / condition
    for p in sorted(d.glob("*_S[0-9].json")):
        song, _, s = p.stem.rpartition("_")
        yield song, s, json.loads(p.read_text(encoding="utf-8"))


def iter_midi_llm(condition: str) -> Iterator[Tuple[str, str, dict]]:
    """condition 'orig' merges the round-2 seed-1000 run (orig_r2) with seeds 1001-1003."""
    dirs = [ML_ROWS / "orig_r2", ML_ROWS / "orig"] if condition == "orig" else [ML_ROWS / condition]
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            song, _, s = p.stem.rpartition("_")
            if condition == "orig" and d.name == "orig" and s == "m1000":
                continue  # the regenerated seed-1000 base of the control subset is used only as the paired base
            yield song, s, json.loads(p.read_text(encoding="utf-8"))


def iter_system(system: str):
    """'legacy:<name>', 'new:<model>/<condition>', 'midi:<condition>'."""
    kind, _, rest = system.partition(":")
    if kind == "legacy":
        return iter_legacy(rest)
    if kind == "new":
        m, _, c = rest.partition("/")
        return iter_new(m, c)
    if kind == "midi":
        return iter_midi_llm(rest)
    raise KeyError(system)


# ------------------------------------------------------------------ failure-aware metric extraction
RATE_ZERO_ON_FAIL = ("strict_valid", "section_plan_exact", "section_count_match", "section_label_seq_match",
                     "section_bars_seq_match", "section_bars_exact_frac", "bar_count_match", "lyric_recall",
                     "lyric_exact", "section_lyric_recall", "late_section_exact", "late_lyric_recall",
                     "lyric_alignment_valid", "tempo_match", "key_match", "section_completion_ratio")


def sample_record(row: dict) -> Dict[str, float]:
    """Metrics of one sample with failures made explicit: a failed generation scores 0 on every
    rate in RATE_ZERO_ON_FAIL and NaN on content metrics."""
    m = dict(row.get("metrics") or {})
    ok = bool(row.get("parse_ok"))
    rec: Dict[str, float] = {"gen_success": float(bool(row.get("generation")) or ok),
                             "parse_ok": float(ok), "strict_valid": float(bool(row.get("strict_ok")))}
    if not ok:
        for k in RATE_ZERO_ON_FAIL:
            rec[k] = 0.0 if k != "strict_valid" else rec["strict_valid"]
        rec["failure"] = row.get("failure", "parse_failure")
        return rec
    for k, v in m.items():
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            rec[k] = float(v)
    rec["strict_valid"] = float(bool(row.get("strict_ok")))
    for k in ("new_tokens", "seconds", "batch_seconds", "batch_size"):
        if isinstance(row.get(k), (int, float)):
            rec[f"gen_{k}"] = float(row[k])
    return rec


def build_sample_table(systems: List[str], out: Optional[Path] = None):
    import pandas as pd
    recs = []
    for sysname in systems:
        for song, sample, row in iter_system(sysname):
            r = sample_record(row)
            r.update(system=sysname, song_id=song, sample=sample, condition=row.get("condition", "orig"))
            recs.append(r)
    df = pd.DataFrame(recs)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df


DEFAULT_SYSTEMS = ([f"legacy:{k}" for k in LEGACY]
                   + [f"new:{m}/orig" for m in ("qwen_e3b", "qwen_e1", "qwen_e1long", "qwen_e0", "mupt")]
                   + ["midi:orig"])


def main():
    df = build_sample_table(DEFAULT_SYSTEMS, DATA_OUT / "samples.parquet")
    print(df.groupby("system").song_id.agg(["count", "nunique"]).to_string())


if __name__ == "__main__":
    main()
