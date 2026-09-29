"""Melody->chord metrics: reference similarity, reference-free melody-harmony compatibility,
and distributional plausibility. Every definition is documented in MELODY_TO_CHORD.md.

Chords are read on a **beat grid**: the chord sounding at each beat of the melody's bar grid.
Qualities are mapped to MIREX-style classes (maj / min / other) from the chord's third.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from qwen_abc.canonical import TICKS_PER_BEAT, Song
from qwen_abc.metrics import chord_at
from qwen_abc.theory import parse_chord_symbol, parse_key_name

from ..seqsim import lcs_fast

SCALE = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}


def beat_chords(song: Song) -> List[Optional[dict]]:
    """Chord info (parse_chord_symbol) at every beat; None = no chord / N.C."""
    out = []
    for b0, beats in zip(song.bar_starts(), song.bar_beats):
        for k in range(beats):
            c = chord_at(song, b0 + k * TICKS_PER_BEAT)
            info = parse_chord_symbol(c.symbol) if c else None
            out.append(info if info and info["root_pc"] is not None else None)
    return out


def qclass(info) -> str:
    if info is None:
        return "N"
    rel = {(p - info["root_pc"]) % 12 for p in info["pcs"]}
    if 4 in rel:
        return "maj"
    if 3 in rel:
        return "min"
    return "other"


def chroma(info) -> np.ndarray:
    v = np.zeros(12)
    if info is not None:
        v[list(info["pcs"])] = 1
    return v


def change_seq(beats) -> List[str]:
    seq = []
    for i in beats:
        t = "N" if i is None else f"{i['root_pc']}:{qclass(i)}"
        if not seq or seq[-1] != t:
            seq.append(t)
    return seq


# ------------------------------------------------------------------ reference similarity
def reference_similarity(gen: Song, ref: Song) -> Dict[str, float]:
    g, r = beat_chords(gen), beat_chords(ref)
    n = min(len(g), len(r))
    g, r = g[:n], r[:n]
    if not n:
        return {}
    root = np.mean([(a is None and b is None) or (a is not None and b is not None and a["root_pc"] == b["root_pc"]) for a, b in zip(g, r)])
    mm = np.mean([qclass(a) == qclass(b) and (a is None or a["root_pc"] == b["root_pc"]) for a, b in zip(g, r)])
    G, R = np.stack([chroma(a) for a in g]), np.stack([chroma(b) for b in r])
    tp = float((G * R).sum())
    prec = tp / max(G.sum(), 1)
    rec = tp / max(R.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    num = (G * R).sum(1)
    den = np.linalg.norm(G, axis=1) * np.linalg.norm(R, axis=1)
    cos = np.where(den > 0, num / np.maximum(den, 1e-9), np.where((G.sum(1) == 0) & (R.sum(1) == 0), 1.0, 0.0))
    sg, sr = change_seq(g), change_seq(r)
    ed_sim = lcs_fast(sg, sr) / max(len(sg), len(sr), 1)
    # cadences: the chord on the last beat of each reference section
    cad = []
    starts = ref.bar_starts()
    beat_index = []
    for b, beats in enumerate(ref.bar_beats):
        beat_index.append(sum(ref.bar_beats[:b]))
    for s in ref.sections:
        last_bar = min(s.start_bar + s.num_bars, len(ref.bar_beats)) - 1
        if last_bar < 0:
            continue
        bi = beat_index[last_bar] + ref.bar_beats[last_bar] - 1
        if bi < n and r[bi] is not None:
            cad.append(float(g[bi] is not None and g[bi]["root_pc"] == r[bi]["root_pc"]))
    ch_g = sum(1 for a, b in zip(g, g[1:]) if a != b) / max(len(gen.bar_beats), 1)
    ch_r = sum(1 for a, b in zip(r, r[1:]) if a != b) / max(len(ref.bar_beats), 1)
    return {"ref_root_acc": float(root), "ref_majmin_acc": float(mm), "ref_chord_f1": f1, "ref_chroma_cos": float(cos.mean()),
            "ref_edit_sim": ed_sim, "ref_cadence_agree": float(np.mean(cad)) if cad else None,
            "ref_harmonic_rhythm_absdiff": abs(ch_g - ch_r)}


# ------------------------------------------------------------------ compatibility (reference-free)
def compatibility(song: Song, key: Optional[str] = None) -> Dict[str, float]:
    key = key or song.key
    kp = parse_key_name(key)
    starts = song.bar_starts()
    import bisect
    tone = dur = strong_t = strong_d = diss = 0.0
    for n in song.notes:
        c = chord_at(song, n.onset)
        info = parse_chord_symbol(c.symbol) if c else None
        if not info or info["root_pc"] is None:
            continue
        pc = n.pitch % 12
        in_ch = pc in info["pcs"]
        dur += n.duration
        tone += n.duration * in_ch
        b = max(bisect.bisect_right(starts, n.onset) - 1, 0)
        pos = n.onset - starts[b]
        if pos % (2 * TICKS_PER_BEAT) == 0:
            strong_d += n.duration
            strong_t += n.duration * in_ch
            # dissonance: a non-chord tone a semitone from a chord tone, sounding on a strong beat
            if not in_ch and any((pc - p) % 12 in (1, 11) for p in info["pcs"]):
                diss += n.duration
    out = {"chord_tone": tone / dur if dur else None, "strong_chord_tone": strong_t / strong_d if strong_d else None,
           "strong_dissonance": diss / strong_d if strong_d else None}
    # chroma compatibility per bar: cosine(melody pc duration histogram, chord chroma of that bar)
    cos = []
    for b0, beats in zip(starts, song.bar_beats):
        b1 = b0 + beats * TICKS_PER_BEAT
        m = np.zeros(12)
        for n in song.notes:
            if b0 <= n.onset < b1:
                m[n.pitch % 12] += n.duration
        cv = np.zeros(12)
        for k in range(beats):
            c = chord_at(song, b0 + k * TICKS_PER_BEAT)
            info = parse_chord_symbol(c.symbol) if c else None
            if info and info["root_pc"] is not None:
                cv += chroma(info)
        if m.sum() and cv.sum():
            cos.append(float(m @ cv / (np.linalg.norm(m) * np.linalg.norm(cv))))
    out["chroma_compat"] = float(np.mean(cos)) if cos else None
    beats = beat_chords(song)
    roots = [b for b in beats if b is not None]
    out["beats_with_chord"] = len(roots) / max(len(beats), 1)
    out["chords_per_bar"] = sum(1 for a, b in zip(beats, beats[1:]) if a != b) / max(len(song.bar_beats), 1) + 1 / max(len(song.bar_beats), 1)
    seq = change_seq(beats)
    seq_n = [t for t in seq if t != "N"]
    out["distinct_chords"] = len(set(seq_n))
    cnt = Counter(t for t in seq_n)
    tot = sum(cnt.values()) or 1
    out["chord_entropy"] = -sum(c / tot * math.log2(c / tot) for c in cnt.values())
    if kp:
        tonic, mode = kp
        steps = SCALE[mode]
        out["root_in_key"] = float(np.mean([(b["root_pc"] - tonic) % 12 in steps for b in roots])) if roots else None
        out["all_tones_in_key"] = float(np.mean([all((p - tonic) % 12 in steps for p in b["pcs"]) for b in roots])) if roots else None
        out["tonic_time"] = float(np.mean([b["root_pc"] == tonic for b in roots])) if roots else None
        # cadence plausibility: section-final chord is I or V (major) / i, v, V, III (minor)
        ok_set = {0, 7} if mode == "major" else {0, 7, 3}
        bidx = [sum(song.bar_beats[:b]) for b in range(len(song.bar_beats))]
        cads = []
        for s in song.sections:
            lb = min(s.start_bar + s.num_bars, len(song.bar_beats)) - 1
            if lb < 0:
                continue
            bi = bidx[lb] + song.bar_beats[lb] - 1
            if bi < len(beats) and beats[bi] is not None:
                cads.append(float((beats[bi]["root_pc"] - tonic) % 12 in ok_set))
        out["cadence_plausible"] = float(np.mean(cads)) if cads else None
    rm = []
    prev = None
    for t in seq_n:
        r = int(t.split(":")[0])
        if prev is not None:
            rm.append((r - prev) % 12)
        prev = r
    out["root_motion_fifth"] = float(np.mean([x in (5, 7) for x in rm])) if rm else None
    # degeneracy: longest run of one chord, in bars
    run = best = 0
    last = None
    for b in beats:
        run = run + 1 if b == last else 1
        best = max(best, run)
        last = b
    out["longest_same_chord_bars"] = best / 4.0
    return out


# ------------------------------------------------------------------ distributions
def distribution_tokens(song: Song):
    kp = parse_key_name(song.key)
    tonic = kp[0] if kp else 0
    beats = beat_chords(song)
    seq = [t for t in change_seq(beats) if t != "N"]
    rel = [f"{(int(t.split(':')[0]) - tonic) % 12}:{t.split(':')[1]}" for t in seq]
    return {"root_rel": [x.split(":")[0] for x in rel], "quality": [x.split(":")[1] for x in rel],
            "bigram": [f"{a}>{b}" for a, b in zip(rel, rel[1:])]}


def js(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    P = np.array([p.get(k, 0) for k in keys], float)
    Q = np.array([q.get(k, 0) for k in keys], float)
    P, Q = P / max(P.sum(), 1), Q / max(Q.sum(), 1)
    M = (P + Q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / M), 0).sum()
        b = np.where(Q > 0, Q * np.log2(Q / M), 0).sum()
    return float(0.5 * a + 0.5 * b)
