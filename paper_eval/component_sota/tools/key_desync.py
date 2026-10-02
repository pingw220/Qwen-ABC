"""Melody/chord key desynchronization in lead sheets (the cause of audible vocal/backing clashes).

Per song and per section, over all sung time (duration-weighted, not only strong beats):
  clash    = share of melody time on a pitch class a semitone from a sounding chord tone and not itself a chord tone
  key_gap  = semitone distance between the melody's Krumhansl key and the chords' Krumhansl key (0..6),
             counting relative major/minor as the same key
A section is "desynced" when clash >= 0.25 and the clash drops below half if the melody is transposed by the
best shift; a song is flagged if any section with >= 2 bars of melody is desynced or the whole-song key_gap is
not 0 / 5 (fourth/fifth relations are ambiguous for short melodies and are not flagged).
  python -m paper_eval.component_sota.tools.key_desync    (E3b S1 and the reference, 225 test songs)
"""
import json
from collections import Counter

import numpy as np

from qwen_abc.canonical import Song
from qwen_abc.metrics import chord_at
from qwen_abc.theory import parse_chord_symbol

from paper_eval.common import PAPER_FINAL_GEN, bootstrap_mean_ci, fmt_ci, load_rows

MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def ks_tonic(h):
    """Major-key tonic pitch class (a minor key is mapped to its relative major)."""
    best = max(((np.corrcoef(h, np.roll(p, k))[0, 1], (k if m == "maj" else (k + 3) % 12)) for k in range(12)
                for p, m in ((MAJ, "maj"), (MIN, "min"))))
    return best[1]


def note_chords(song):
    out = []
    for n in song.notes:
        c = chord_at(song, n.onset)
        info = parse_chord_symbol(c.symbol) if c else None
        out.append(set(info["pcs"]) if info and info["root_pc"] is not None else None)
    return out


def clash_share(notes, pcsets, shift=0):
    tot = bad = 0.0
    for n, ch in zip(notes, pcsets):
        if not ch:
            continue
        pc = (n.pitch + shift) % 12
        tot += n.duration
        if pc not in ch and ({(pc + 1) % 12, (pc - 1) % 12} & ch):
            bad += n.duration
    return bad / tot if tot else None


def analyse(song):
    pcsets = note_chords(song)
    mh, chh = np.zeros(12), np.zeros(12)
    for n in song.notes:
        mh[n.pitch % 12] += n.duration
    for c in song.chords:
        info = parse_chord_symbol(c.symbol)
        if info and info["root_pc"] is not None:
            for p in info["pcs"]:
                chh[p] += c.duration
    if mh.sum() == 0 or chh.sum() == 0:
        return None
    gap = (ks_tonic(mh) - ks_tonic(chh)) % 12
    gap = min(gap, 12 - gap)
    starts = song.bar_starts()
    secs = []
    for s in song.sections:
        lo = starts[s.start_bar] if s.start_bar < len(starts) else song.total_ticks
        e = s.start_bar + s.num_bars
        hi = starts[e] if e < len(starts) else song.total_ticks
        idx = [i for i, n in enumerate(song.notes) if lo <= n.onset < hi]
        if sum(song.notes[i].duration for i in idx) < 2 * 16:
            continue
        ns, cs = [song.notes[i] for i in idx], [pcsets[i] for i in idx]
        c0 = clash_share(ns, cs)
        if c0 is None:
            continue
        best = min(range(1, 12), key=lambda k: clash_share(ns, cs, k))
        cb = clash_share(ns, cs, best)
        secs.append({"label": s.label, "bars": [s.start_bar, e], "clash": round(c0, 3), "best_shift": best if best <= 6 else best - 12,
                     "clash_at_best": round(cb, 3), "desync": c0 >= 0.25 and cb < c0 / 2})
    whole = clash_share(song.notes, pcsets)
    return {"clash": whole, "key_gap": int(gap), "sections": secs,
            "flag": any(s["desync"] for s in secs) or gap not in (0, 5)}


def main():
    rows = {r["song_id"]: r for r in load_rows("test")}
    res = {}
    for src in ("e3b", "ref"):
        out = {}
        for sid, r in rows.items():
            if src == "ref":
                song = Song.from_json(r["song"])
            else:
                g = json.loads((PAPER_FINAL_GEN / "qwen_e3b" / "orig" / f"{sid}_S1.json").read_text())
                if not g.get("song"):
                    continue
                song = Song.from_json(g["song"])
            a = analyse(song)
            if a:
                out[sid] = a
        res[src] = out
        flags = [a["flag"] for a in out.values()]
        secs = [s for a in out.values() for s in a["sections"]]
        m, lo, hi, n = bootstrap_mean_ci([float(f) for f in flags])
        cm, clo, chi, _ = bootstrap_mean_ci([a["clash"] for a in out.values()])
        print(f"{src}: songs flagged {fmt_ci(m, lo, hi)} (N={n}); desynced sections {sum(s['desync'] for s in secs)}/{len(secs)}; "
              f"mean clash share {fmt_ci(cm, clo, chi)}; key-gap distribution {dict(sorted(Counter(a['key_gap'] for a in out.values()).items()))}")
    both = sorted(set(res["e3b"]) & set(res["ref"]))
    from paper_eval.common import paired_bootstrap
    d = paired_bootstrap({k: float(res["e3b"][k]["flag"]) for k in both}, {k: float(res["ref"][k]["flag"]) for k in both})
    print(f"E3b - reference, flagged songs: {fmt_ci(d['diff'], d['lo'], d['hi'], 3, True)} (N={d['n']})")
    json.dump(res, open("experiments/component_sota/tmp/key_desync.json", "w"), indent=1)


if __name__ == "__main__":
    main()
