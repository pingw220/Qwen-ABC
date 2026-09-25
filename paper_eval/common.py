"""Paths, the model registry, data loading and paired statistics shared by every paper_eval module."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The canonical dataset every paper number is scored against: ABC-v2 specs (cleaned
# section boundaries) over the de-duplicated 10,243 / 273 / 225 split. E0 (ABC-v1) is
# prompted in its own format from the same specs, as in round 2.
DATA_DIR = ROOT / "data/generated/abc_v2_20260915_120927"
TRAIN_DATA_DIR = DATA_DIR
OUT_ROOT = ROOT / "experiments/paper_final"          # raw generations (git-ignored: they contain lyrics)
REPORT_DIR = ROOT / "reports/paper_final"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"
DATA_OUT = REPORT_DIR / "data"
CLEAN_IDS = ROOT / "reports/clean_test_song_ids.txt"

OFFLOAD = Path("/gscratch/ark/pingw220/qwen_abc_r2_offload")

# name -> checkpoint, prompt format, family, what it is
MODELS: Dict[str, Dict[str, str]] = {
    "qwen_e3b": {"ckpt": str(OFFLOAD / "runs/e3b_201131/final_model"), "fmt": "v2", "family": "Qwen-ABC",
                 "desc": "Qwen3.5-0.8B, ABC-v2 (ESS) + late-section reconstruction, 758 updates (current best)"},
    "qwen_e1": {"ckpt": str(ROOT / "experiments/e1_v2_sft_20260915_120949/final_model"), "fmt": "v2", "family": "Qwen-ABC",
                "desc": "Qwen3.5-0.8B, ABC-v2 (ESS), whole-song SFT, 506 updates"},
    "qwen_e1long": {"ckpt": str(OFFLOAD / "runs/e1long_225021/final_model"), "fmt": "v2", "family": "Qwen-ABC",
                    "desc": "Qwen3.5-0.8B, ABC-v2 (ESS), whole-song SFT, 758 updates (matched-update control for E3b)"},
    "qwen_e0": {"ckpt": str(ROOT / "experiments/direct_sft_20260915_023500/final_model"), "fmt": "v1", "family": "Qwen-ABC",
                "desc": "Qwen3.5-0.8B, ABC-v1 (no ESS, uncleaned training boundaries), 382 updates"},
    "mupt": {"ckpt": str(ROOT / "experiments/r4_mupt_20260923_001821/final_model"), "fmt": "v2", "family": "MuPT",
             "desc": "MuPT-1.07B (ABC-pretrained), same data/recipe as E3b, 1070 updates"},
}

DISPLAY = {"qwen_e3b": "Qwen-ABC (E3b)", "qwen_e1": "Qwen E1 (ESS)", "qwen_e1long": "Qwen E1-long (ESS)",
           "qwen_e0": "Qwen E0 (no ESS)", "mupt": "MuPT", "midi_llm": "MIDI-LLM", "reference": "Pseudo-GT"}

# Canonical sampling for every new generation (round-2 operating point).
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOTAL = 9216        # prompt + completion; longest reference completion is 6,703 tokens
MAX_NEW = 8704
# Paired-seed ("common random numbers") seeds; see paper_eval/sampler.py.
SEEDS = {"S1": 101, "S2": 202, "S3": 303, "S4": 404}


def jload(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(obj, p, indent: Optional[int] = 1) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(p)


def load_rows(split: str = "test", data_dir: Path = DATA_DIR, ids: Optional[Iterable[str]] = None) -> List[dict]:
    """Rows of songs_<split>.jsonl sorted by song id (spec, canonical song, reference ABC)."""
    wanted = set(ids) if ids is not None else None
    rows = []
    with open(data_dir / f"songs_{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if wanted is not None and line[len('{"song_id": "'):].split('"', 1)[0] not in wanted:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["song_id"])
    return rows


def iter_train_rows(data_dir: Path = TRAIN_DATA_DIR):
    """Stream training rows (the file is ~440 MB)."""
    with open(data_dir / "songs_train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def clean_ids() -> List[str]:
    return [l.strip() for l in open(CLEAN_IDS, encoding="utf-8") if l.strip()]


def make_prompt(fmt: str, spec: dict) -> str:
    from qwen_abc.abc_v2 import spec_to_prompt_v2
    from qwen_abc.prompt import spec_to_prompt
    return {"v2": spec_to_prompt_v2, "v1": spec_to_prompt}[fmt](spec)


def load_gen_dir(d: Path, pattern: str = "*_s*.json") -> Dict[Tuple[str, int], dict]:
    """<song>_s<k>.json rows of an existing generate_eval run -> {(song, k): row}."""
    out = {}
    for p in sorted(Path(d).glob(pattern)):
        song, _, k = p.stem.rpartition("_s")
        if k.isdigit():
            out[(song, int(k))] = json.loads(p.read_text(encoding="utf-8"))
    return out


# ------------------------------------------------------------------ statistics
def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else None


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                      alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Mean and percentile CI over units (songs). Non-finite values are dropped and the n reported."""
    xs = [float(x) for x in values if _finite(x)]
    n = len(xs)
    if not n:
        return None, None, None, 0
    m = sum(xs) / n
    if n == 1:
        return m, m, m, 1
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return m, float(lo), float(hi), n


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], n_boot: int = 10000, seed: int = 0):
    """Mean of (a - b) over songs present (finite) in both, with a song-level percentile CI."""
    keys = sorted(k for k in a if k in b and _finite(a[k]) and _finite(b[k]))
    diffs = [a[k] - b[k] for k in keys]
    m, lo, hi, n = bootstrap_mean_ci(diffs, n_boot, seed)
    return {"diff": m, "lo": lo, "hi": hi, "n": n,
            "a": mean([a[k] for k in keys]), "b": mean([b[k] for k in keys])}


def cluster_bootstrap(values_by_song: Dict[str, List[float]], n_boot: int = 10000, seed: int = 0):
    """Mean over songs of the per-song mean (songs are the resampling unit; several samples per song)."""
    per_song = {k: mean(v) for k, v in values_by_song.items()}
    return bootstrap_mean_ci([v for v in per_song.values() if _finite(v)], n_boot, seed)


def fmt_ci(m, lo, hi, digits: int = 3, signed: bool = False) -> str:
    if m is None:
        return "–"
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


def seeded_choice(key: str, options: Sequence, salt: str = ""):
    """Deterministic pick independent of Python's hash randomization."""
    import hashlib
    h = int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), 16)
    return options[h % len(options)]


def write_table(rows: List[dict], stem: str, caption: str = "", columns: Optional[List[str]] = None,
                table_dir: Path = TABLE_DIR) -> None:
    """Write <stem>.csv, <stem>.md and <stem>.tex from a list of dicts (values already formatted)."""
    import csv
    table_dir.mkdir(parents=True, exist_ok=True)
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(table_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    md = ([f"<!-- {caption} -->"] if caption else []) + ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (table_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def esc(s):
        return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
                .replace("_", "\\_").replace("#", "\\#").replace("−", "$-$").replace("×", "$\\times$")
                .replace("→", "$\\rightarrow$").replace("≈", "$\\approx$").replace("±", "$\\pm$"))
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(esc(r.get(c, "")) for c in cols) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    if caption:
        tex.append(f"\\caption{{{esc(caption)}}}")
    tex.append(f"\\label{{tab:{stem}}}")
    tex.append("\\end{table}")
    (table_dir / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if either side is constant."""
    import numpy as np

    def ranks(v):
        v = np.asarray(v, float)
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):
            idx = v == val
            r[idx] = r[idx].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
