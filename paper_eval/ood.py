"""Training-distribution statistics of the control space, and OOD plan construction.

Regimes (seed S1; the target section is the one the interventions use). The corpus
decides what is rare (reports/paper_final/data/train_distribution.json):

* section length is nearly independent of label (P(bars|verse)/P(bars) is 0.7-1.3 for every
  common length), and every length 3-17 bars holds >=1.1% of sections, so single-section
  lengths in that range are in-distribution for any label;
* what *is* compositionally rare is the arrangement: labels and lengths that are each common,
  in an order the corpus almost never uses.

in-distribution   ``orig``               the held-out song's own plan
compositional     ``ood_chorus_first``   first verse and first chorus swap places (intro->chorus is 0.8% of songs)
                  ``ood_end_on_verse``   the last verse is moved to the end of the song (verse-final: 1.5% of songs)
extrapolative     ``ood_long20``         target section = max(bars+8, 20) bars (lengths >=20: <0.55% each)
                  ``ood_long28``         target section = max(2*bars, 28) bars (<0.1% each)
                  ``ood_sections_p3``    three extra repeats of the last chorus (section count +3)
                  ``ood_sections_p6``    six extra repeats (section counts beyond the corpus maximum of 17)
                  ``ood_tempo``          60 BPM for songs at/above the median tempo, 240 BPM below it

Requests stay musically plausible (repeated choruses, a chorus-first opening, a long
section); none shortens a section, so no lyric density is pushed past the corpus. A
request whose estimated completion would not fit the token budget is recorded as
ineligible, never truncated.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Dict

from .common import DATA_OUT, iter_train_rows, jdump, jload
from .interventions import change_bars, pick_target_section, syllables_of

STATS_PATH = DATA_OUT / "train_distribution.json"
OOD_CONDITIONS = ("ood_chorus_first", "ood_end_on_verse", "ood_long20", "ood_long28", "ood_sections_p3",
                  "ood_sections_p6", "ood_tempo")
REGIME = {"orig": "in-distribution", "ood_chorus_first": "compositional", "ood_end_on_verse": "compositional",
          "ood_long20": "extrapolative", "ood_long28": "extrapolative", "ood_sections_p3": "extrapolative",
          "ood_sections_p6": "extrapolative", "ood_tempo": "extrapolative"}


def compute_train_stats() -> dict:
    bars, label, label_bars, n_sections, tempo, keys, modes, meters = (Counter() for _ in range(8))
    bigram, song_bars, song_sylls, density, label_pos = Counter(), Counter(), [], [], Counter()
    n_songs = 0
    for r in iter_train_rows():
        spec = r["spec"]
        n_songs += 1
        secs = spec["sections"]
        n_sections[len(secs)] += 1
        tempo[spec["tempo_bpm"]] += 1
        keys[spec.get("key")] += 1
        modes[(spec.get("key") or "none").split()[-1]] += 1
        meters[spec["meter"]] += 1
        song_bars[sum(s["bars"] for s in secs)] += 1
        song_sylls.append(sum(len(syllables_of(s)) for s in secs))
        prev = "<s>"
        for i, s in enumerate(secs):
            bars[s["bars"]] += 1
            label[s["label"]] += 1
            label_bars[f"{s['label']}|{s['bars']}"] += 1
            bigram[f"{prev}>{s['label']}"] += 1
            label_pos[f"{s['label']}|{min(i, 12)}"] += 1
            prev = s["label"]
            k = len(syllables_of(s))
            if k:
                density.append(k / max(s["bars"], 1))
        bigram[f"{prev}></s>"] += 1
    density.sort()
    tempos = sorted(t for t, c in tempo.items() for _ in range(c))
    q = lambda xs, p: xs[min(int(p * (len(xs) - 1)), len(xs) - 1)]
    return {
        "n_songs": n_songs, "bars": dict(bars), "label": dict(label), "label_bars": dict(label_bars),
        "n_sections": dict(n_sections), "tempo": dict(tempo), "key": {str(k): v for k, v in keys.items()},
        "mode": dict(modes), "meter": dict(meters), "label_bigram": dict(bigram), "label_pos": dict(label_pos),
        "song_bars": dict(song_bars),
        "song_syllables_quantiles": {p: q(sorted(song_sylls), p) for p in (0.01, 0.05, 0.5, 0.95, 0.99)},
        "syll_per_bar_quantiles": {p: q(density, p) for p in (0.01, 0.05, 0.5, 0.95, 0.99)},
        "tempo_quantiles": {p: q(tempos, p) for p in (0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995)},
    }


def load_train_stats(recompute: bool = False) -> dict:
    if STATS_PATH.exists() and not recompute:
        d = jload(STATS_PATH)
    else:
        d = compute_train_stats()
        jdump(d, STATS_PATH)
    # json turns float keys into strings
    for k in ("song_syllables_quantiles", "syll_per_bar_quantiles", "tempo_quantiles"):
        d[k] = {float(p): v for p, v in d[k].items()}
    for k in ("bars", "n_sections", "tempo", "song_bars"):
        d[k] = {int(x): v for x, v in d[k].items()}
    return d


# ------------------------------------------------------------------ probabilities / distances
def p_bars_given_label(stats: dict, label: str, b: int, alpha: float = 0.5) -> float:
    tot = stats["label"].get(label, 0)
    c = stats["label_bars"].get(f"{label}|{b}", 0)
    support = 64
    return (c + alpha) / (tot + alpha * support)


def p_bars(stats: dict, b: int, alpha: float = 0.5) -> float:
    tot = sum(stats["bars"].values())
    return (stats["bars"].get(b, 0) + alpha) / (tot + alpha * 64)


def surprisal_label_bars(stats: dict, label: str, b: int) -> float:
    return -math.log2(p_bars_given_label(stats, label, b))


def plan_surprisal(stats: dict, spec: dict) -> Dict[str, float]:
    """Plan-level distance from the training distribution (bits)."""
    secs = spec["sections"]
    lb = [surprisal_label_bars(stats, s["label"], s["bars"]) for s in secs]
    tot_bi = sum(stats["label_bigram"].values())
    labels = ["<s>"] + [s["label"] for s in secs] + ["</s>"]
    bi = [-math.log2((stats["label_bigram"].get(f"{a}>{b}", 0) + 0.5) / (tot_bi + 0.5 * 100)) for a, b in zip(labels, labels[1:])]
    ns = stats["n_sections"]
    p_ns = (ns.get(len(secs), 0) + 0.5) / (sum(ns.values()) + 0.5 * 40)
    return {"max_label_bars_bits": max(lb) if lb else 0.0, "mean_label_bars_bits": sum(lb) / max(len(lb), 1),
            "sum_label_bars_bits": sum(lb), "mean_bigram_bits": sum(bi) / max(len(bi), 1),
            "n_sections_bits": -math.log2(p_ns)}


# ------------------------------------------------------------------ OOD specs
def _rebar(spec: dict, i: int, b: int) -> dict:
    return change_bars(spec, i, b - spec["sections"][i]["bars"])


def _est_tokens(row: dict, spec: dict) -> float:
    """Completion tokens scaled by the requested/original bar ratio."""
    b0 = sum(x["bars"] for x in row["spec"]["sections"])
    b1 = sum(x["bars"] for x in spec["sections"])
    return row.get("abc_tokens", 4000) * b1 / max(b0, 1)


def ood_tasks_for_row(row: dict, stats: dict, conditions=None, token_limit: int = 8400):
    from .common import MAX_NEW
    spec, sid = row["spec"], row["song_id"]
    conds = conditions or OOD_CONDITIONS
    tgt = pick_target_section(spec, sid)
    secs = spec["sections"]
    for cond in conds:
        meta = {"family": "ood", "regime": REGIME[cond], "target_section": tgt}
        out = None
        if cond in ("ood_long20", "ood_long28"):
            if tgt is None:
                yield cond, None, "no eligible target section"
                continue
            b0 = secs[tgt]["bars"]
            b = max(b0 + 8, 20) if cond == "ood_long20" else max(2 * b0, 28)
            out = _rebar(spec, tgt, b)
            meta.update(changes=[[tgt, b0, b]], requested_bars=b, original_bars=b0)
        elif cond in ("ood_sections_p3", "ood_sections_p6"):
            ch = [j for j, x in enumerate(secs) if x["label"] == "chorus"]
            if not ch:
                yield cond, None, "no chorus"
                continue
            k = 3 if cond.endswith("p3") else 6
            out = copy.deepcopy(spec)
            for _ in range(k):
                out["sections"].insert(ch[-1] + 1, copy.deepcopy(secs[ch[-1]]))
            meta.update(inserted_after=ch[-1], n_inserted=k, n_sections=len(out["sections"]),
                        original_n_sections=len(secs))
        elif cond == "ood_chorus_first":
            v = next((j for j, x in enumerate(secs) if x["label"] == "verse"), None)
            c = next((j for j, x in enumerate(secs) if x["label"] == "chorus"), None)
            if v is None or c is None or not v < c:
                yield cond, None, "needs a verse before the first chorus"
                continue
            out = copy.deepcopy(spec)
            out["sections"][v], out["sections"][c] = copy.deepcopy(secs[c]), copy.deepcopy(secs[v])
            meta.update(swapped=[v, c])
        elif cond == "ood_end_on_verse":
            vs = [j for j, x in enumerate(secs) if x["label"] == "verse" and syllables_of(x)]
            if not vs or vs[-1] == len(secs) - 1:
                yield cond, None, "no verse to move, or already verse-final"
                continue
            out = copy.deepcopy(spec)
            moved = out["sections"].pop(vs[-1])
            out["sections"].append(moved)
            meta.update(moved=vs[-1])
        elif cond == "ood_tempo":
            out = copy.deepcopy(spec)
            out["tempo_bpm"] = 60 if spec["tempo_bpm"] >= stats["tempo_quantiles"][0.5] else 240
            meta.update(original_tempo=spec["tempo_bpm"], requested_tempo=out["tempo_bpm"])
        else:
            raise KeyError(cond)
        est = _est_tokens(row, out)
        if est > min(token_limit, MAX_NEW - 300):
            yield cond, None, f"estimated completion {est:.0f} tokens exceeds the budget"
            continue
        meta["est_tokens"] = round(est)
        meta["surprisal"] = plan_surprisal(stats, out)
        meta["surprisal_orig"] = plan_surprisal(stats, spec)
        yield cond, out, meta
