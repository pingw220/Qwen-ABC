#!/usr/bin/env python
"""Musical fluency vs control adherence, on the same songs and the same symbolic statistics.

  python -m paper_eval.musicality

Per system: per-song scalar statistics (mean, and the share of songs inside the reference's
5-95% band) and pooled-distribution JS divergences to the held-out references (song-level
bootstrap: songs are resampled, histograms re-pooled), next to its control metrics.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from qwen_abc.canonical import TICKS_PER_BEAT, Song
from qwen_abc.theory import parse_chord_symbol, parse_key_name

from .collect import iter_system, sample_record
from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, load_rows, write_table

SYSTEMS = {"Qwen-ABC (E3b)": "new:qwen_e3b/orig", "MuPT": "new:mupt/orig", "MIDI-LLM": "midi:orig",
           "Qwen E0 (no ESS)": "new:qwen_e0/orig"}
QUAL = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus", "other"]


def _qual(info) -> str:
    rel = sorted((p - info["root_pc"]) % 12 for p in info["pcs"])
    return {(0, 4, 7): "maj", (0, 3, 7): "min", (0, 4, 7, 10): "7", (0, 4, 7, 11): "maj7", (0, 3, 7, 10): "min7",
            (0, 3, 6): "dim", (0, 4, 8): "aug", (0, 5, 7): "sus", (0, 2, 7): "sus"}.get(tuple(rel), "other")


def song_stats(s: Song) -> Dict:
    """Scalars and histogram counts for one song."""
    ns = s.notes
    key = parse_key_name(s.key)
    tonic = key[0] if key else 0
    starts = s.bar_starts()
    pos, toks = [], []
    for n in ns:
        i = max(bisect.bisect_right(starts, n.onset) - 1, 0)
        p = n.onset - starts[i]
        pos.append(p % TICKS_PER_BEAT)
        toks.append((p, min(n.duration, 16)))
    ints = [b.pitch - a.pitch for a, b in zip(ns, ns[1:])]
    h = {"pc_rel": Counter((n.pitch - tonic) % 12 for n in ns),
         "interval": Counter(max(-12, min(12, x)) for x in ints),
         "duration": Counter(min(n.duration, 16) for n in ns),
         "onset": Counter(pos)}
    chords = [(c, parse_chord_symbol(c.symbol)) for c in s.chords]
    chords = [(c, i) for c, i in chords if i and i["root_pc"] is not None]
    roots = [(i["root_pc"] - tonic) % 12 for _, i in chords]
    h["chord_root"] = Counter(roots)
    h["chord_qual"] = Counter(_qual(i) for _, i in chords)
    h["chord_trans"] = Counter((a, b) for a, b in zip(roots, roots[1:]) if a != b)
    rt = Counter(toks)
    tot = sum(rt.values()) or 1
    nb = max(len(s.bar_beats), 1)
    sc = {
        "pitch_range": (max(n.pitch for n in ns) - min(n.pitch for n in ns)) if ns else None,
        "notes_per_bar": len(ns) / nb,
        "mean_abs_interval": sum(abs(x) for x in ints) / len(ints) if ints else None,
        "step_frac": sum(1 for x in ints if 1 <= abs(x) <= 2) / len(ints) if ints else None,
        "repeat_pitch_frac": sum(1 for x in ints if x == 0) / len(ints) if ints else None,
        "contour_up_frac": sum(1 for x in ints if x > 0) / len(ints) if ints else None,
        "rhythm_entropy": -sum(c / tot * math.log2(c / tot) for c in rt.values()) if ns else None,
        "chords_per_bar": len(chords) / nb,
        "chord_vocab": len({(r, _qual(i)) for r, (_, i) in zip(roots, chords)}),
    }
    return {"scalars": sc, "hists": h}


def add_song_metrics(sc: dict, row_metrics: dict) -> dict:
    for k in ("distinct_bar_frac", "pitch_4gram_repeat_frac", "chord_tone_frac", "bar_duration_ok_frac",
              "syncopation_frac", "melody_in_key_frac"):
        if k in row_metrics:
            sc[k] = row_metrics[k]
    return sc


HIST_KEYS = {"pc_rel": list(range(12)), "interval": list(range(-12, 13)), "duration": list(range(1, 17)),
             "onset": list(range(4)), "chord_root": list(range(12)), "chord_qual": QUAL,
             "chord_trans": [(a, b) for a in range(12) for b in range(12) if a != b]}


def js(p, q) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum()
        b = np.where(q > 0, q * np.log2(q / m), 0).sum()
    return float(0.5 * a + 0.5 * b)


def main() -> None:
    from qwen_abc.metrics import song_metrics
    refs = {r["song_id"]: r for r in load_rows("test")}
    data: Dict[str, Dict[str, List[dict]]] = {}
    ref_stats = {}
    for sid, r in refs.items():
        s = Song.from_json(r["song"])
        st = song_stats(s)
        add_song_metrics(st["scalars"], song_metrics(s, r["spec"]))
        ref_stats[sid] = [st]
    data["Pseudo-GT reference"] = ref_stats
    control = {}
    for name, spec in SYSTEMS.items():
        per = defaultdict(list)
        ctl = defaultdict(lambda: defaultdict(list))
        for sid, sample, row in iter_system(spec):
            rec = sample_record(row)
            for k in ("section_plan_exact", "lyric_recall", "strict_valid"):
                ctl[k][sid].append(rec.get(k, 0.0))
            if row.get("song"):
                s = Song.from_json(row["song"])
                if s.notes:
                    st = song_stats(s)
                    add_song_metrics(st["scalars"], row.get("metrics") or {})
                    per[sid].append(st)
        data[name] = per
        control[name] = {k: {s: sum(v) / len(v) for s, v in d.items()} for k, d in ctl.items()}
        print(name, len(per))
    # ---- scalar table
    scal_keys = list(next(iter(ref_stats.values()))[0]["scalars"].keys())
    bands = {}
    for k in scal_keys:
        vals = sorted(x[0]["scalars"][k] for x in ref_stats.values() if x[0]["scalars"].get(k) is not None)
        bands[k] = (vals[int(0.05 * (len(vals) - 1))], vals[int(0.95 * (len(vals) - 1))])
    rows = []
    for name, per in data.items():
        row = {"system": name, "N songs": len(per)}
        for k in scal_keys:
            song_means = [np.mean([x["scalars"][k] for x in v if x["scalars"].get(k) is not None]) for v in per.values()
                          if any(x["scalars"].get(k) is not None for x in v)]
            m, lo, hi, n = bootstrap_mean_ci(song_means)
            inband = [float(bands[k][0] <= x <= bands[k][1]) for x in song_means]
            row[k] = fmt_ci(m, lo, hi, 2 if abs(m or 0) >= 10 else 3)
            row[f"{k} in ref 5-95%"] = f"{np.mean(inband):.2f}" if inband else "–"
        rows.append(row)
    write_table(rows, "musicality_scalars", "Per-song symbolic statistics (mean over songs of per-song means over samples; 95% song bootstrap)")
    # ---- JS divergences to the reference, song-level bootstrap
    rng = np.random.default_rng(0)
    ref_ids = sorted(ref_stats)

    def pooled(per, ids, hk):
        keys = HIST_KEYS[hk]
        idx = {k: i for i, k in enumerate(keys)}
        v = np.zeros(len(keys))
        for sid in ids:
            for st in per.get(sid, []):
                for k, c in st["hists"][hk].items():
                    if k in idx:
                        v[idx[k]] += c / max(len(per[sid]), 1)
        return v
    js_rows, js_raw = [], {}
    for name, per in data.items():
        if name == "Pseudo-GT reference":
            continue
        ids = [s for s in ref_ids if s in per]
        row = {"system": name, "N songs": len(ids)}
        for hk in HIST_KEYS:
            point = js(pooled(per, ids, hk), pooled(ref_stats, ids, hk))
            bs = []
            for _ in range(1000):
                samp = list(rng.choice(ids, size=len(ids), replace=True))
                bs.append(js(pooled(per, samp, hk), pooled(ref_stats, samp, hk)))
            lo, hi = np.quantile(bs, [0.025, 0.975])
            row[f"JS {hk}"] = fmt_ci(point, float(lo), float(hi), 4)
            js_raw.setdefault(name, {})[hk] = [point, float(lo), float(hi)]
        # reference split-half floor: JS between two random halves of the references
        js_rows.append(row)
    half = []
    for _ in range(200):
        perm = list(rng.permutation(ref_ids))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        half.append({hk: js(pooled(ref_stats, a, hk), pooled(ref_stats, b, hk)) for hk in HIST_KEYS})
    js_rows.append({"system": "reference split-half (noise floor)", "N songs": len(ref_ids) // 2,
                    **{f"JS {hk}": f"{np.mean([h[hk] for h in half]):.4f}" for hk in HIST_KEYS}})
    write_table(js_rows, "musicality_js", "Jensen-Shannon divergence (bits) of pooled distributions to the held-out references; lower = closer. Song-level bootstrap (1,000)")
    # ---- fluency vs control side by side
    side = []
    for name in SYSTEMS:
        ctl = control[name]
        row = {"system": name}
        for k in ("section_plan_exact", "lyric_recall", "strict_valid"):
            m, lo, hi, n = bootstrap_mean_ci(list(ctl[k].values()))
            row[k] = fmt_ci(m, lo, hi)
        for hk in ("pc_rel", "interval", "duration", "chord_trans"):
            p, lo, hi = js_raw[name][hk]
            row[f"JS {hk}"] = fmt_ci(p, lo, hi, 4)
        side.append(row)
    write_table(side, "musicality_vs_control", "Control adherence (single sample, mean of 4) next to distributional distance to the references")
    jdump({"js": js_raw, "bands": bands}, DATA_OUT / "musicality.json")
    print("MUSICALITY_DONE")


if __name__ == "__main__":
    main()
