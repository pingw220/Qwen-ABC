#!/usr/bin/env python
"""Does a section-label intervention change the *music*, not only the printed label?

  python -m paper_eval.component_sota.section_semantics

Inputs: E3b paper-final samples -- ``orig`` (S1-S4) and ``label_swap`` (verse<->chorus, S1-S2),
``label_bridge`` (verse/chorus -> bridge, S1-S2) -- same songs, same target section, same bars
and lyrics (the label is the only changed control).

1. Descriptive: per-section features of the target section (register relative to the song,
   density, range, rhythmic activity, repetition, chord density, tonic stability, lyric density),
   intervened minus unmodified samples, paired over songs.
2. Classifier: logistic regression (numpy) on the same features, trained on TRAIN-split reference
   sections (verse vs chorus), validated on held-out TEST reference sections, then applied to the
   generated target section: P(chorus) when chorus was requested vs when verse was requested.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from qwen_abc.canonical import Song
from qwen_abc.metrics import chord_at
from qwen_abc.theory import parse_chord_symbol, parse_key_name

from ..common import PAPER_FINAL_GEN, bootstrap_mean_ci, fmt_ci, iter_train_rows, jdump, load_rows, paired_bootstrap, write_table
from ..interventions import section_notes
from .melody_eval import CS_REPORT

FEATS = ["rel_register", "rel_max_pitch", "notes_per_bar", "pitch_range", "mean_abs_interval", "sixteenth_frac",
         "distinct_bar_frac", "chords_per_bar", "tonic_frac", "syll_per_bar"]


def section_features(song: Song, i: int):
    sn = section_notes(song)
    if i >= len(sn) or len(sn[i]) < 4:
        return None
    notes = sn[i]
    allp = [n.pitch for n in song.notes]
    s = song.sections[i]
    bars = max(s.num_bars, 1)
    ps = [n.pitch for n in notes]
    ints = [abs(b - a) for a, b in zip(ps, ps[1:])]
    starts = song.bar_starts()
    sigs = []
    for b in range(s.start_bar, min(s.start_bar + s.num_bars, len(starts))):
        lo = starts[b]
        hi = starts[b + 1] if b + 1 < len(starts) else song.total_ticks
        sigs.append(tuple((n.onset - lo, n.pitch, n.duration) for n in notes if lo <= n.onset < hi))
    ne = [x for x in sigs if x]
    kp = parse_key_name(song.key)
    lo = starts[s.start_bar] if s.start_bar < len(starts) else 0
    e = s.start_bar + s.num_bars
    hi = starts[e] if e < len(starts) else song.total_ticks
    chords = [c for c in song.chords if lo <= c.onset < hi]
    roots = []
    for t in range(lo, hi, 4):
        c = chord_at(song, t)
        info = parse_chord_symbol(c.symbol) if c else None
        if info and info["root_pc"] is not None:
            roots.append(info["root_pc"])
    return {"rel_register": np.mean(ps) - np.mean(allp), "rel_max_pitch": max(ps) - max(allp),
            "notes_per_bar": len(notes) / bars, "pitch_range": max(ps) - min(ps),
            "mean_abs_interval": float(np.mean(ints)) if ints else 0.0,
            "sixteenth_frac": np.mean([n.onset % 2 == 1 for n in notes]),
            "distinct_bar_frac": len(set(ne)) / max(len(ne), 1), "chords_per_bar": len(chords) / bars,
            "tonic_frac": float(np.mean([r == kp[0] for r in roots])) if (roots and kp) else 0.0,
            "syll_per_bar": sum(len(n.lyric) for n in notes if n.lyric) / bars}


def fit_logreg(X, y, l2=1e-2, iters=3000, lr=0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def auc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    # ---- classifier on reference sections (train split -> test split)
    Xtr, ytr = [], []
    for k, r in enumerate(iter_train_rows()):
        if k % 2:   # every other train song keeps this CPU-light; 5k songs, ~40k sections
            continue
        s = Song.from_json(r["song"])
        for i, sec in enumerate(s.sections):
            if sec.label in ("verse", "chorus"):
                f = section_features(s, i)
                if f:
                    Xtr.append([f[c] for c in FEATS])
                    ytr.append(float(sec.label == "chorus"))
    Xtr, ytr = np.array(Xtr, float), np.array(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w, b = fit_logreg((Xtr - mu) / sd, ytr)
    prob = lambda f: float(1 / (1 + np.exp(-(((np.array([f[c] for c in FEATS]) - mu) / sd) @ w + b))))
    test = {r["song_id"]: r for r in load_rows("test")}
    Xte, yte = [], []
    for r in test.values():
        s = Song.from_json(r["song"])
        for i, sec in enumerate(s.sections):
            if sec.label in ("verse", "chorus"):
                f = section_features(s, i)
                if f:
                    Xte.append(prob(f))
                    yte.append(float(sec.label == "chorus"))
    Xte, yte = np.array(Xte), np.array(yte)
    clf = {"train_sections": int(len(ytr)), "test_sections": int(len(yte)), "test_auc": auc(Xte, yte),
           "test_acc": float(np.mean((Xte > 0.5) == (yte == 1))), "chorus_share_test": float(yte.mean()),
           "weights": dict(zip(FEATS, map(float, w)))}
    print(clf, flush=True)
    # ---- generated sections
    rows, deltas = [], defaultdict(lambda: defaultdict(dict))
    pchorus = defaultdict(dict)
    for sid, r in test.items():
        o = [PAPER_FINAL_GEN / "qwen_e3b/orig" / f"{sid}_{s}.json" for s in ("S1", "S2", "S3", "S4")]
        o = [json.loads(p.read_text()) for p in o if p.exists()]
        for cond in ("label_swap", "label_bridge"):
            iv = [PAPER_FINAL_GEN / "qwen_e3b" / cond / f"{sid}_{s}.json" for s in ("S1", "S2")]
            iv = [json.loads(p.read_text()) for p in iv if p.exists()]
            if not iv or not o:
                continue
            tgt = iv[0]["meta"]["target_section"]
            orig_label = iv[0]["meta"]["original_label"]
            new_label = iv[0]["meta"]["requested_label"]
            fo = [section_features(Song.from_json(x["song"]), tgt) for x in o if x.get("song")]
            fi = [section_features(Song.from_json(x["song"]), tgt) for x in iv if x.get("song")]
            fo, fi = [f for f in fo if f], [f for f in fi if f]
            if not fo or not fi:
                continue
            for c in FEATS:
                deltas[(cond, orig_label, new_label)][c][sid] = np.mean([f[c] for f in fi]) - np.mean([f[c] for f in fo])
            if cond == "label_swap":
                pchorus[("requested " + orig_label)][sid] = np.mean([prob(f) for f in fo])
                pchorus[("requested " + new_label, orig_label)][sid] = np.mean([prob(f) for f in fi])
    table = []
    for key, d in deltas.items():
        cond, a, bb = key
        row = {"intervention": f"{a} → {bb}", "N songs": len(d[FEATS[0]])}
        for c in FEATS:
            m, lo, hi, n = bootstrap_mean_ci(list(d[c].values()))
            row[c] = fmt_ci(m, lo, hi, 3, True)
        table.append(row)
    # reference: how do real choruses differ from verses in the same song?
    ref_d = defaultdict(list)
    for r in test.values():
        s = Song.from_json(r["song"])
        v = [section_features(s, i) for i, x in enumerate(s.sections) if x.label == "verse"]
        c = [section_features(s, i) for i, x in enumerate(s.sections) if x.label == "chorus"]
        v, c = [f for f in v if f], [f for f in c if f]
        if v and c:
            for k in FEATS:
                ref_d[k].append(np.mean([f[k] for f in c]) - np.mean([f[k] for f in v]))
    row = {"intervention": "reference: chorus − verse (same song)", "N songs": len(ref_d[FEATS[0]])}
    for k in FEATS:
        m, lo, hi, n = bootstrap_mean_ci(ref_d[k])
        row[k] = fmt_ci(m, lo, hi, 3, True)
    table.append(row)
    write_table(table, "section_semantics_features", "Target-section feature change after a label-only intervention (E3b; intervened − unmodified samples, paired over songs) vs how real choruses differ from verses",
                table_dir=CS_REPORT / "tables")
    crow = []
    for a in ("verse", "chorus"):
        other = "chorus" if a == "verse" else "verse"
        base = pchorus.get("requested " + a, {})
        swapped = pchorus.get(("requested " + other, a), {})
        d = paired_bootstrap(swapped, base)
        crow.append({"section originally": a, "relabelled to": other, "P(chorus) unmodified": f"{d['b']:.3f}" if d["b"] is not None else "–",
                     "P(chorus) relabelled": f"{d['a']:.3f}" if d["a"] is not None else "–",
                     "difference [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
    crow.append({"section originally": "classifier check", "relabelled to": f"held-out reference sections: AUC {clf['test_auc']:.3f}, acc {clf['test_acc']:.3f}",
                 "N": clf["test_sections"]})
    write_table(crow, "section_semantics_classifier", "Verse/chorus classifier (trained on train-split reference sections) applied to generated sections before and after relabelling",
                table_dir=CS_REPORT / "tables")
    jdump(clf, CS_REPORT / "data" / "section_classifier.json")
    print("SECTION_SEMANTICS_DONE")


if __name__ == "__main__":
    main()
