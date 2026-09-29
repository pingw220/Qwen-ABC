#!/usr/bin/env python
"""Melody->chord evaluation over every (melody source x chord source) combination (Tables 4 and 5).

  python -m paper_eval.component_sota.chord_eval

Chord sources: ``qwen`` (melody->chord harmonizer), ``qwen_lyr`` (with lyrics), ``am2``
(AccoMontage2), ``ref`` (pseudo-reference chords, reference melody only), ``joint`` (E3b's own
chords, E3b melody only), ``diatonic`` (cheap baseline: per bar the diatonic triad covering most
melody duration). Reference similarity is computed only on the reference melody, where the
pseudo-reference chords belong to the same melody.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from qwen_abc.canonical import Chord, Song, TICKS_PER_BEAT, normalize_chords
from qwen_abc.theory import parse_key_name

from ..common import bootstrap_mean_ci, fmt_ci, load_rows, mean, paired_bootstrap, write_table
from .chord_metrics import compatibility, distribution_tokens, js, reference_similarity
from .formats import with_chords
from .melody_eval import CS_REPORT
from .sources import CS_ROOT, SEEDS, iter_melodies

NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def diatonic(song: Song):
    kp = parse_key_name(song.key) or (0, "major")
    tonic, mode = kp
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    triads = []
    for i, s in enumerate(steps):
        third = steps[(i + 2) % 7] - s
        fifth = steps[(i + 4) % 7] - s
        root = (tonic + s) % 12
        q = "" if third % 12 == 4 else "m"
        if fifth % 12 == 6:
            q = "dim"
        triads.append((root, {root, (root + third) % 12, (root + fifth) % 12}, NAMES[root] + q))
    out, prev = [], None
    for b0, beats in zip(song.bar_starts(), song.bar_beats):
        b1 = b0 + beats * TICKS_PER_BEAT
        h = Counter()
        for n in song.notes:
            if b0 <= n.onset < b1:
                h[n.pitch % 12] += n.duration
        if not h:
            continue
        best = max(triads, key=lambda t: (sum(h[p] for p in t[1]), t[0] == tonic))
        if best[2] != prev:
            out.append(Chord(b0, 0, best[2]))
            prev = best[2]
    return normalize_chords(out, song.total_ticks)


def load_chords(h, msrc, mseed, sid, cseed="S1"):
    p = CS_ROOT / "chords" / h / f"{msrc}_{mseed}" / f"{sid}_{cseed}.json"
    if not p.exists():
        return None, "missing"
    r = json.loads(p.read_text())
    if not r.get("ok"):
        return None, r.get("failure") or "failed"
    return [Chord(o, d, s) for o, d, s in r["chords"]], None


COMBOS = [  # (melody source, melody seed, chord source, chord seeds)
    ("ref", "S1", "qwen", SEEDS), ("ref", "S1", "qwen_lyr", SEEDS), ("ref", "S1", "am2", ("S1",)), ("ref", "S1", "ref", ("S1",)),
    ("ref", "S1", "diatonic", ("S1",)),
    ("e3b", "S1", "joint", ("S1",)), ("e3b", "S1", "qwen", ("S1",)), ("e3b", "S1", "am2", ("S1",)),
    ("mel", "S1", "qwen", ("S1",)), ("mel", "S1", "am2", ("S1",)),
    ("csl_offc", "S1", "qwen", ("S1",)), ("csl_offc", "S1", "am2", ("S1",)),
    ("csl_rtc", "S1", "qwen", ("S1",)), ("csl_rtc", "S1", "am2", ("S1",)),
]


def main():
    import pandas as pd
    rows = {r["song_id"]: r for r in load_rows("test")}
    ref_songs = {sid: Song.from_json(r["song"]) for sid, r in rows.items()}
    recs, dist = [], defaultdict(lambda: defaultdict(Counter))
    for sid, s in ref_songs.items():
        for k, v in distribution_tokens(s).items():
            dist[("corpus", "")][k].update(v)
    for msrc, mseed, csrc, cseeds in COMBOS:
        mel = {r["song_id"]: r for r in iter_melodies(msrc, mseed, "orig", keep_chords=(csrc == "joint"))}
        if not mel:
            print("no melodies for", msrc)
            continue
        for sid in rows:
            m = mel.get(sid)
            for cs in cseeds:
                rec = {"melody": msrc, "chords": csrc, "song_id": sid, "chord_seed": cs, "ok": 0.0}
                if m is None or not m["ok"]:
                    rec["failure"] = "no melody"
                    recs.append(rec)
                    continue
                song = Song.from_json(m["song"])
                if csrc in ("joint", "ref"):
                    chords = song.chords if csrc == "joint" else ref_songs[sid].chords
                    fail = None if chords else "no chords"
                elif csrc == "diatonic":
                    chords, fail = diatonic(song), None
                else:
                    chords, fail = load_chords({"qwen": "chord", "qwen_lyr": "chord_lyr", "am2": "am2"}[csrc], msrc, mseed, sid, cs)
                if fail:
                    rec["failure"] = fail
                    recs.append(rec)
                    continue
                full = with_chords(song, chords)
                rec["ok"] = 1.0
                rec.update(compatibility(full))
                if msrc == "ref":
                    rec.update(reference_similarity(full, ref_songs[sid]))
                recs.append(rec)
                if cs == cseeds[0]:
                    for k, v in distribution_tokens(full).items():
                        dist[(msrc, csrc)][k].update(v)
        print(msrc, csrc, "done", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(CS_REPORT / "data" / "chords.parquet", index=False)
    cols = [("ok", "success", 3), ("ref_root_acc", "root acc. vs ref", 3), ("ref_majmin_acc", "maj/min acc. vs ref", 3),
            ("ref_chord_f1", "chord chroma F1 vs ref", 3), ("ref_edit_sim", "progression sim. vs ref", 3),
            ("ref_cadence_agree", "cadence agreement", 3), ("chord_tone", "chord-tone ratio", 3),
            ("strong_chord_tone", "strong-beat chord-tone", 3), ("strong_dissonance", "strong-beat dissonance", 3),
            ("chroma_compat", "chroma compatibility", 3), ("root_in_key", "roots in key", 3),
            ("cadence_plausible", "cadence I/V", 3), ("chords_per_bar", "chords/bar", 2), ("distinct_chords", "distinct chords", 1),
            ("chord_entropy", "chord entropy", 2), ("tonic_time", "time on tonic", 3), ("root_motion_fifth", "root motion by 4th/5th", 3)]
    table = []
    for (msrc, csrc), d in df.groupby(["melody", "chords"], sort=False):
        row = {"melody": msrc, "chords": csrc, "N songs": d.song_id.nunique()}
        for c, lab, dg in cols:
            if c in d and d[c].notna().any():
                per = d.groupby("song_id")[c].mean().dropna()
                m, lo, hi, n = bootstrap_mean_ci(per.tolist())
                row[lab] = fmt_ci(m, lo, hi, dg)
        for k in ("root_rel", "quality", "bigram"):
            if (msrc, csrc) in dist:
                row[f"JS {k} vs corpus"] = f"{js(dist[(msrc, csrc)][k], dist[('corpus', '')][k]):.4f}"
        table.append(row)
    write_table(table, "melody_to_chord", "Harmonization of the same melodies by each chord source (Qwen: mean of 4 samples on the reference melody). "
                "Reference-similarity columns only for the reference melody. JS vs the held-out corpus chord distributions",
                table_dir=CS_REPORT / "tables")
    drows = []
    for a, b in (("qwen", "am2"), ("qwen", "ref"), ("am2", "ref"), ("qwen_lyr", "qwen"), ("qwen", "diatonic"), ("am2", "diatonic")):
        da, db = df[(df.melody == "ref") & (df.chords == a)], df[(df.melody == "ref") & (df.chords == b)]
        for c, lab, _ in cols:
            if c in da and c in db:
                d = paired_bootstrap(da.groupby("song_id")[c].mean().dropna().to_dict(), db.groupby("song_id")[c].mean().dropna().to_dict())
                if d["n"]:
                    drows.append({"comparison (reference melody)": f"{a} − {b}", "metric": lab,
                                  "difference [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
    write_table(drows, "melody_to_chord_paired", "Paired harmonizer differences on the reference melodies", table_dir=CS_REPORT / "tables")
    print("CHORD_EVAL_DONE")


if __name__ == "__main__":
    main()
