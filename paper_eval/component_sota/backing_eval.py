#!/usr/bin/env python
"""Backing-track audio evaluation (Table 7) and request -> symbol -> audio closed loop (Table 8).

  python -m paper_eval.component_sota.backing_eval list > mixes.txt
  python -m paper_eval.component_sota.backing_eval evaluate

Every mix is analysed by the same detectors (tools/): BTC large-vocabulary chords, BeatNet beats,
Krumhansl-Schmuckler chroma key, SongEval, Audiobox-aesthetics. Requested values come from the
lead sheet that was rendered (``<name>.song.json``: melody, chords, key, tempo of the excerpt).

* Chord F1 (MIDI-SAG definition, frame rate 10 Hz here): requested and detected chord labels ->
  12-bin chroma per frame, binarized, micro-F1 over (frame, pitch class); also frame root accuracy.
* Key accuracy: detected == requested; MIREX weighted score (fifth 0.5, relative 0.3, parallel 0.2).
* Rhythm F1 (MIDI-SAG): detected beats vs the score's beat times, 70 ms tolerance; BPM error from
  the median detected inter-beat interval (and octave-tolerant BPM accuracy within 4%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_abc.canonical import TICKS_PER_BEAT, Song
from qwen_abc.theory import corpus_chord_to_abc, parse_chord_symbol, parse_key_name

from ..common import bootstrap_mean_ci, fmt_ci, mean, paired_bootstrap, write_table
from .audio import AUDIO
from .melody_eval import CS_REPORT


def mixes():
    for grp in ("chords", "control"):
        for d in sorted((AUDIO / "backing" / grp / "render").glob("*")):
            if (d / "mix.wav").exists():
                yield grp, d


def req_segments(song: Song):
    spt = 60.0 / song.tempo_bpm / TICKS_PER_BEAT
    return [(c.onset * spt, (c.onset + c.duration) * spt, parse_chord_symbol(c.symbol)) for c in song.chords]


def lab_segments(p: Path):
    out = []
    for line in p.read_text().splitlines():
        a = line.split()
        if len(a) < 3:
            continue
        lab = a[2]
        if lab in ("N", "X"):
            info = None
        else:
            root, _, q = lab.partition(":")
            q, _, bass = (q or "maj").partition("/")
            sym, _ = corpus_chord_to_abc(lab, root, q or "maj", bass or None)
            info = parse_chord_symbol(sym)
        out.append((float(a[0]), float(a[1]), info))
    return out


def chroma_frames(segs, T, hz=10):
    F = np.zeros((int(T * hz) + 1, 12))
    R = np.full(len(F), -1)
    for a, b, info in segs:
        if info is None or info["root_pc"] is None:
            continue
        i0, i1 = int(a * hz), min(int(b * hz), len(F))
        F[i0:i1, list(info["pcs"])] = 1
        R[i0:i1] = info["root_pc"]
    return F, R


REL = {"major": {"minor": 9}, "minor": {"major": 3}}


def key_score(det: str, req: str) -> float:
    a, b = parse_key_name(det), parse_key_name(req)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a[1] == b[1] and (a[0] - b[0]) % 12 in (5, 7):
        return 0.5
    if a[1] != b[1] and (a[0] - b[0]) % 12 == REL[b[1]][a[1]]:
        return 0.3
    if a[1] != b[1] and a[0] == b[0]:
        return 0.2
    return 0.0


def evaluate_mix(d: Path, song: Song) -> dict:
    mix = d / "mix.wav"
    rec = {}
    dur = sum(song.bar_beats) * 60.0 / song.tempo_bpm
    lab = mix.with_name("mix.wav.lab")
    if lab.exists() and song.chords:
        R, Rr = chroma_frames(req_segments(song), dur)
        D, Dr = chroma_frames(lab_segments(lab), dur)
        n = min(len(R), len(D))
        R, D, Rr, Dr = R[:n], D[:n], Rr[:n], Dr[:n]
        tp = (R * D).sum()
        p, r = tp / max(D.sum(), 1), tp / max(R.sum(), 1)
        rec["chord_f1"] = 2 * p * r / max(p + r, 1e-9)
        m = Rr >= 0
        rec["chord_root_acc"] = float((Rr[m] == Dr[m]).mean()) if m.any() else None
    kj = mix.with_name("mix.wav.key.json")
    if kj.exists() and song.key:
        det = json.loads(kj.read_text())["key"]
        rec["detected_key"] = det
        rec["key_acc"] = float(parse_key_name(det) == parse_key_name(song.key))
        rec["key_weighted"] = key_score(det, song.key)
    bj = mix.with_name("mix.wav.beats.json")
    if bj.exists():
        det = np.array([b[0] for b in json.loads(bj.read_text())["beats"]])
        spb = 60.0 / song.tempo_bpm
        ref = np.arange(0, dur, spb)
        tol = 0.07
        hits = sum(1 for t in ref if len(det) and np.min(np.abs(det - t)) <= tol)
        p = hits / max(len(det), 1)
        r = hits / max(len(ref), 1)
        rec["rhythm_f1"] = 2 * p * r / max(p + r, 1e-9)
        if len(det) > 3:
            bpm = 60.0 / float(np.median(np.diff(det)))
            rec["detected_bpm"] = bpm
            rec["bpm_abs_err"] = abs(bpm - song.tempo_bpm)
            rec["bpm_acc_4pct"] = float(abs(bpm - song.tempo_bpm) <= 0.04 * song.tempo_bpm)
            rec["bpm_acc_octave"] = float(min(abs(bpm * f - song.tempo_bpm) for f in (0.5, 1, 2)) <= 0.04 * song.tempo_bpm)
        rec["audio_duration_ratio_to_score"] = None
    aes = mix.with_name("mix.wav.aes.json")
    if aes.exists():
        a = json.loads(aes.read_text())
        for k in ("PQ", "CE", "CU", "PC"):
            if k in a:
                rec[f"aes_{k}"] = float(a[k])
    se = mix.with_name("mix.wav.songeval.json")
    if se.exists():
        for k, v in json.loads(se.read_text()).items():
            if isinstance(v, (int, float)):
                rec[f"songeval_{k}"] = float(v)
    return rec


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "evaluate"])
    args = ap.parse_args()
    if args.cmd == "list":
        for _, d in mixes():
            print(d / "mix.wav")
        return
    recs = []
    for grp, d in mixes():
        sid, _, cond = d.name.partition("__")
        song = Song.from_json(json.loads((d.parent.parent / f"{d.name}.song.json").read_text()))
        recs.append({"group": grp, "song_id": sid, "condition": cond, "requested_key": song.key, "requested_bpm": song.tempo_bpm,
                     **evaluate_mix(d, song)})
    df = pd.DataFrame(recs)
    # closed loop: the *request* is derived from the draft (key + 5 semitones, tempo x 1.25), not from the
    # generated song's own declaration; symbolic success = the score declares the requested value
    from qwen_abc.theory import canonical_key
    names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    drafts = {r["song_id"]: r for r in recs if r["group"] == "control" and r["condition"] == "draft"}
    for i, r in df.iterrows():
        if r["group"] != "control" or r["song_id"] not in drafts:
            continue
        d = drafts[r["song_id"]]
        if r["condition"].startswith("key"):
            pc, mode = parse_key_name(d["requested_key"])
            want = canonical_key(f"{names[(pc + 5) % 12]} {mode}")
            df.loc[i, "request"] = want
            df.loc[i, "symbolic_success"] = float(parse_key_name(r["requested_key"]) == parse_key_name(want))
            df.loc[i, "audio_success"] = float(parse_key_name(r.get("detected_key")) == parse_key_name(want)) if isinstance(r.get("detected_key"), str) else None
        elif r["condition"].startswith("tempo"):
            want = int(round(d["requested_bpm"] * 1.25))
            df.loc[i, "request"] = str(want)
            df.loc[i, "symbolic_success"] = float(r["requested_bpm"] == want)
            det = r.get("detected_bpm")
            df.loc[i, "audio_success"] = float(min(abs(det * f - want) for f in (0.5, 1, 2)) <= 0.04 * want) if det == det and det is not None else None
            df.loc[i, "audio_abs_error"] = abs(det - want) if det == det and det is not None else None
    df.to_parquet(CS_REPORT / "data" / "backing.parquet", index=False)
    cols = [("chord_f1", "Chord F1 ↑", 3), ("chord_root_acc", "chord root acc. ↑", 3), ("key_acc", "Key accuracy ↑", 3),
            ("key_weighted", "Key (MIREX weighted) ↑", 3), ("rhythm_f1", "Rhythm F1 ↑", 3), ("bpm_abs_err", "abs. BPM error", 1),
            ("bpm_acc_4pct", "BPM within 4%", 3), ("bpm_acc_octave", "BPM within 4% (octave-tolerant)", 3),
            ("aes_PQ", "Audiobox PQ", 2), ("aes_CE", "Audiobox CE", 2),
            ("symbolic_success", "requested value in the score", 3), ("audio_success", "requested value detected in audio", 3),
            ("audio_abs_error", "abs. detected − requested BPM", 1)]
    se_cols = [c for c in df.columns if c.startswith("songeval_")]
    cols += [(c, c.replace("songeval_", "SongEval "), 2) for c in se_cols]
    for grp, stem, cap in (("chords", "backing_chord_audio", "Backing renders with the same reference melody and vocal; only the chord source changes (Table 7; 95.1 s excerpts, 36 songs)"),
                           ("control", "closed_loop_control", "Request -> symbolic score -> rendered audio -> detector (Table 8)")):
        g = df[df.group == grp]
        table = []
        for cond, x in g.groupby("condition"):
            row = {"condition": cond, "N songs": len(x)}
            for c, lab, dg in cols:
                if c in x and x[c].notna().any():
                    m, lo, hi, n = bootstrap_mean_ci(x[c].dropna().tolist())
                    row[lab] = fmt_ci(m, lo, hi, dg)
            table.append(row)
        write_table(table, stem, cap, table_dir=CS_REPORT / "tables")
    print("BACKING_EVAL_DONE")


if __name__ == "__main__":
    main()
