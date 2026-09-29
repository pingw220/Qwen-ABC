#!/usr/bin/env python
"""Two-SVS cross-render evaluation (Table 6): melody source x renderer, same symbolic input.

  python -m paper_eval.component_sota.svs_eval list > wavs.txt     # wavs still needing analysis
  python -m paper_eval.component_sota.svs_eval evaluate

Reference = the item's target.json (the sung-note list both renderers were fed).
* PER: recognized text (Paraformer-zh) vs target syllables, compared as pinyin initials+finals
  (toneless) by edit distance / reference length; CER on characters.
* Pitch: per target note, median RMVPE F0 over the middle 60% of the note's span (voiced frames);
  cents error vs the target MIDI pitch. note pitch accuracy = |err| < 50 cents; octave error =
  |err| within 1200 +- 100 cents; mean |cents| on octave-folded error.
* Timing: frame voicing vs target note spans -> voicing precision/recall/F1 (10 ms frames);
  onset hit = a voiced onset within 100 ms of each phrase onset (a note after >= 150 ms rest);
  duration ratio = rendered length / score length.
* Quality: Audiobox-aesthetics PQ / CE on the vocal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..common import bootstrap_mean_ci, fmt_ci, mean, paired_bootstrap, write_table
from .audio import AUDIO
from .melody_eval import CS_REPORT, DISPLAY

RENDERERS = {"fastsinger": "fastsinger.wav", "soulx": "soulx/generated.wav"}


def items():
    for d in sorted((AUDIO / "svs").glob("*/*")):
        if (d / "target.json").exists():
            yield d


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def phones(chars):
    from pypinyin import Style, lazy_pinyin
    s = "".join(chars)
    ini = lazy_pinyin(s, style=Style.INITIALS, strict=False)
    fin = lazy_pinyin(s, style=Style.FINALS, strict=False)
    out = []
    for a, b in zip(ini, fin):
        if a:
            out.append(a)
        if b:
            out.append(b)
    return out


def evaluate_one(wav: Path, target: dict) -> dict:
    notes = target["notes"]
    rec = {"rendered": float(wav.exists())}
    if not wav.exists():
        return rec
    ref_chars = [n["symbol"] for n in notes if n["symbol"] != "#"]
    asr = wav.with_name(wav.name + ".asr.json")
    if asr.exists():
        hyp = [c for c in json.loads(asr.read_text())["text"] if "一" <= c <= "鿿"]
        rp, hp = phones(ref_chars), phones(hyp)
        rec["PER"] = edit(rp, hp) / max(len(rp), 1)
        rec["CER"] = edit(ref_chars, hyp) / max(len(ref_chars), 1)
    f0p = wav.with_name(wav.name + ".f0.npz")
    if f0p.exists():
        z = np.load(f0p)
        f0, hop = z["f0"], float(z["hop_s"])
        if f0.ndim == 2:   # early sidecars stored RMVPE's (f0 Hz, cents) pair
            f0 = f0[0]
        t = np.arange(len(f0)) * hop
        errs = []
        for n in notes:
            a, b = n["start"], n["end"]
            lo, hi = a + 0.2 * (b - a), b - 0.2 * (b - a)
            seg = f0[(t >= lo) & (t <= hi)]
            seg = seg[seg > 0]
            if len(seg):
                target_hz = 440.0 * 2 ** ((n["pitch"] - 69) / 12)
                errs.append(1200 * math.log2(float(np.median(seg)) / target_hz))
        if errs:
            e = np.array(errs)
            folded = (e + 600) % 1200 - 600
            rec["note_pitch_acc"] = float(np.mean(np.abs(e) < 50))
            rec["note_pitch_acc_octave_folded"] = float(np.mean(np.abs(folded) < 50))
            rec["octave_error_rate"] = float(np.mean(np.abs(np.abs(e) - 1200) < 100))
            rec["abs_cents_folded"] = float(np.mean(np.abs(folded)))
            rec["notes_voiced"] = len(errs) / max(len(notes), 1)
        target_on = np.zeros(len(f0), bool)
        for n in notes:
            target_on[(t >= n["start"]) & (t < n["end"])] = True
        voiced = f0 > 0
        tp = float((voiced & target_on).sum())
        p = tp / max(voiced.sum(), 1)
        r = tp / max(target_on.sum(), 1)
        rec["voicing_precision"], rec["voicing_recall"] = p, r
        rec["voicing_f1"] = 2 * p * r / max(p + r, 1e-9)
        onsets = [n["start"] for i, n in enumerate(notes) if i == 0 or n["start"] - notes[i - 1]["end"] >= 0.15]
        v_on = t[1:][(voiced[1:]) & (~voiced[:-1])]
        hits, offs = 0, []
        for o in onsets:
            if len(v_on):
                k = np.argmin(np.abs(v_on - o))
                if abs(v_on[k] - o) <= 0.1:
                    hits += 1
                    offs.append(abs(v_on[k] - o))
        rec["phrase_onset_hit"] = hits / max(len(onsets), 1)
        rec["onset_offset_ms"] = 1000 * float(np.mean(offs)) if offs else None
        rec["duration_ratio"] = (len(f0) * hop) / max(notes[-1]["end"], 1e-3) if notes else None
    aes = wav.with_name(wav.name + ".aes.json")
    if aes.exists():
        a = json.loads(aes.read_text())
        for k in ("PQ", "CE", "CU", "PC"):
            if k in a:
                rec[f"aes_{k}"] = float(a[k])
    return rec


COLS = [("rendered", "rendered", 3), ("PER", "PER ↓", 3), ("CER", "CER ↓", 3), ("note_pitch_acc", "note pitch acc. (±50c) ↑", 3),
        ("note_pitch_acc_octave_folded", "pitch acc., octave-folded ↑", 3), ("octave_error_rate", "octave errors ↓", 3),
        ("abs_cents_folded", "abs. cents (folded) ↓", 1), ("voicing_f1", "voicing F1 vs score ↑", 3),
        ("phrase_onset_hit", "phrase onsets within 100 ms ↑", 3), ("duration_ratio", "duration ratio", 3),
        ("aes_PQ", "Audiobox PQ ↑", 2), ("aes_CE", "Audiobox CE ↑", 2)]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "evaluate"])
    args = ap.parse_args()
    if args.cmd == "list":
        for d in items():
            for w in RENDERERS.values():
                if (d / w).exists():
                    print(d / w)
        return
    recs = []
    for d in items():
        tgt = json.loads((d / "target.json").read_text())
        for rname, w in RENDERERS.items():
            recs.append({"source": d.parent.name, "song_id": d.name, "renderer": rname, **evaluate_one(d / w, tgt)})
    df = pd.DataFrame(recs)
    df.to_parquet(CS_REPORT / "data" / "svs.parquet", index=False)
    table = []
    for (src, rn), g in df.groupby(["source", "renderer"]):
        row = {"melody source": DISPLAY.get(src, src), "SVS": rn, "N songs": len(g)}
        for c, lab, dg in COLS:
            if c in g and g[c].notna().any():
                m, lo, hi, n = bootstrap_mean_ci(g[c].dropna().tolist())
                row[lab] = fmt_ci(m, lo, hi, dg)
        table.append(row)
    write_table(table, "svs_cross_render", "Melody source x SVS renderer on the frozen 36-song audio subset (same symbolic input to both renderers)",
                table_dir=CS_REPORT / "tables")
    # main effects and interaction: paired over songs
    eff = []
    for c, lab, _ in COLS[1:9]:
        if c not in df:
            continue
        piv = df.pivot_table(index="song_id", columns=["source", "renderer"], values=c)
        srcs = sorted({s for s, _ in piv.columns})
        if ("fastsinger" in {r for _, r in piv.columns}) and ("soulx" in {r for _, r in piv.columns}):
            for s in srcs:
                if (s, "fastsinger") in piv and (s, "soulx") in piv:
                    d = paired_bootstrap(piv[(s, "soulx")].dropna().to_dict(), piv[(s, "fastsinger")].dropna().to_dict())
                    eff.append({"metric": lab, "contrast": f"SoulX − FastSinger | {DISPLAY.get(s, s)}", "estimate [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
        # ranking consistency: Spearman of per-source means across renderers
        means = {r: [piv[(s, r)].mean() for s in srcs if (s, r) in piv] for r in ("fastsinger", "soulx")}
        if len(means["fastsinger"]) == len(means["soulx"]) and len(srcs) > 2:
            from ..common import spearman
            eff.append({"metric": lab, "contrast": "rank agreement of melody sources across renderers (Spearman)",
                        "estimate [95% CI]": f"{spearman(means['fastsinger'], means['soulx']):+.2f}", "N": len(srcs)})
    write_table(eff, "svs_effects", "Renderer effect per melody source (paired over songs) and melody-ranking agreement across renderers",
                table_dir=CS_REPORT / "tables")
    print("SVS_EVAL_DONE")


if __name__ == "__main__":
    main()
