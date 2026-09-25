#!/usr/bin/env python
"""Interventional controllability: does each control move the output the way it asks, and more
than reseeding does?

  python -m paper_eval.intervention_analysis --model qwen_e3b [--model mupt --model midi_llm]

Two kinds of estimand (EVALUATION_PROTOCOL.md §6):

* target-specific adherence per intervened sample (requested vs realized bars / label / key /
  tempo / lyrics), next to the same quantity on the song's unmodified samples;
* output change beyond sampling noise, per song: cross (orig_i vs intervened_j, all seed pairs)
  minus within (orig_i vs orig_j, i != j), for whole songs and for target vs non-target sections.

Writes reports/paper_final/data/interventions.parquet (sample level) and
interventions_effects.parquet (song level), no lyrics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from qwen_abc.canonical import TICKS_PER_BEAT, Song
from qwen_abc.theory import parse_chord_symbol, parse_key_name

from .common import DATA_OUT, OUT_ROOT, load_rows
from .interventions import (best_pc_shift, chord_seq_by_change, note_tokens, pair_distances, recall_of,
                            section_lyric_recall_of, section_notes, seq_distance, syllables_of)
from .midi_llm import ML_OUT

SCALE = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}


# ------------------------------------------------------------------ loading
def load_condition(model: str, cond: str) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    if model == "midi_llm":
        d = ML_OUT / "rows" / cond
        for p in sorted(d.glob("*_m*.json")):
            sid, _, s = p.stem.rpartition("_")
            out[sid][s] = json.loads(p.read_text(encoding="utf-8"))
        return out
    d = OUT_ROOT / "gen" / model / cond
    for p in sorted(d.glob("*_S[0-9].json")):
        sid, _, s = p.stem.rpartition("_")
        out[sid][s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def song_of(row) -> Optional[Song]:
    return Song.from_json(row["song"]) if row and row.get("song") else None


# ------------------------------------------------------------------ feature distances (non-saturating)
def _hist(xs, keys):
    c = Counter(xs)
    tot = sum(c[k] for k in keys) or 1
    return [c[k] / tot for k in keys]


def _js(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def features(s: Song) -> dict:
    ns = s.notes
    starts = s.bar_starts()
    import bisect
    pos = []
    for n in ns:
        i = max(bisect.bisect_right(starts, n.onset) - 1, 0)
        pos.append((n.onset - starts[i]) % TICKS_PER_BEAT)
    ints = [max(-12, min(12, b.pitch - a.pitch)) for a, b in zip(ns, ns[1:])]
    return {
        "pc": _hist([n.pitch % 12 for n in ns], range(12)),
        "int": _hist(ints, range(-12, 13)),
        "pos": _hist(pos, range(4)),
        "dur": _hist([min(n.duration, 17) for n in ns], range(1, 18)),
        "pitch_mean": sum(n.pitch for n in ns) / max(len(ns), 1),
        "density": len(ns) / max(len(s.bar_beats), 1),
    }


def feature_distances(fa: dict, fb: dict) -> Dict[str, float]:
    return {"f_pc_js": _js(fa["pc"], fb["pc"]), "f_interval_js": _js(fa["int"], fb["int"]),
            "f_rhythm_js": 0.5 * (_js(fa["pos"], fb["pos"]) + _js(fa["dur"], fb["dur"])),
            "f_pitch_mean": abs(fa["pitch_mean"] - fb["pitch_mean"]), "f_density": abs(fa["density"] - fb["density"])}


def all_distances(a: Song, b: Song, fa: dict, fb: dict, transpose: int = 0) -> Dict[str, float]:
    d = pair_distances(a, b, transpose=transpose)
    d.update(feature_distances(fa, fb))
    return d


def section_dist(a: Song, b: Song, idx: List[int], kind: str = "melody") -> Optional[float]:
    """Mean LCS distance of the notes a and b wrote in the listed section indices."""
    sa, sb = section_notes(a), section_notes(b)
    vals = []
    for i in idx:
        if i < len(sa) and i < len(sb) and (sa[i] or sb[i]):
            vals.append(seq_distance(note_tokens(a, kind, notes=sa[i]), note_tokens(b, kind, notes=sb[i])))
    return sum(vals) / len(vals) if vals else None


def section_chord_dist(a: Song, b: Song, idx: List[int]) -> Optional[float]:
    starts_a, starts_b = a.bar_starts(), b.bar_starts()

    def seq(s: Song, starts, i):
        if i >= len(s.sections):
            return None
        sec = s.sections[i]
        lo = starts[sec.start_bar] if sec.start_bar < len(starts) else s.total_ticks
        e = sec.start_bar + sec.num_bars
        hi = starts[e] if e < len(starts) else s.total_ticks
        out = []
        for c in s.chords:
            if lo <= c.onset < hi:
                info = parse_chord_symbol(c.symbol)
                if info and info["root_pc"] is not None:
                    out.append(f"{info['root_pc']}:{'.'.join(map(str, sorted((p - info['root_pc']) % 12 for p in info['pcs'])))}")
        return out
    vals = []
    for i in idx:
        x, y = seq(a, starts_a, i), seq(b, starts_b, i)
        if x is not None and y is not None and (x or y):
            vals.append(seq_distance(x, y))
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------ target adherence per sample
def target_metrics(cond: str, meta: dict, spec: dict, orig_spec: dict, s: Optional[Song], row: dict,
                   orig_songs: List[Song]) -> Dict[str, float]:
    fam = meta.get("family")
    tgt = meta.get("target_section")
    m = row.get("metrics") or {}
    r: Dict[str, float] = {"parse_ok": float(s is not None), "strict_valid": float(bool(row.get("strict_ok"))),
                           "plan_exact": float(m.get("section_plan_exact", 0.0)) if s else 0.0,
                           "lyric_recall": float(m.get("lyric_recall", 0.0)) if s else 0.0,
                           "cram": m.get("cram_syllable_frac") if s else None}
    if s is None:
        return r
    got = [(x.label, x.num_bars) for x in s.sections]
    want = [(x["label"], x["bars"]) for x in spec["sections"]]
    if tgt is not None and fam in ("bars", "label", "lyrics"):
        non = [i for i in range(len(want)) if i != tgt]
        r["nontarget_exact_frac"] = sum(1 for i in non if i < len(got) and got[i] == want[i]) / max(len(non), 1)
        r["target_present"] = float(tgt < len(got))
        if tgt < len(got):
            r["target_bars"] = got[tgt][1]
            r["target_label_ok"] = float(got[tgt][0] == want[tgt][0])
            r["target_bars_ok"] = float(got[tgt][1] == want[tgt][1])
            r["target_exact"] = float(got[tgt] == want[tgt])
            r["target_abs_bar_err"] = abs(got[tgt][1] - want[tgt][1])
        else:
            r.update(target_label_ok=0.0, target_bars_ok=0.0, target_exact=0.0)
    if fam == "bars":
        r["requested_delta"] = meta["arg"]
        if "target_bars" in r:
            r["realized_delta"] = r["target_bars"] - meta["original_bars"]
            r["direction_ok"] = float((r["realized_delta"] > 0) == (meta["arg"] > 0) and r["realized_delta"] != 0)
        r["total_bars_ok"] = float(m.get("bar_count_match", 0.0))
    if fam == "key":
        k = meta["semitones"]
        r["declared_key_ok"] = float(m.get("key_match", 0.0))
        tonic, mode = parse_key_name(spec["key"])
        otonic, _ = parse_key_name(orig_spec["key"])
        steps = SCALE[mode]
        dur = sum(n.duration for n in s.notes) or 1
        r["melody_in_new_key"] = sum(n.duration for n in s.notes if (n.pitch - tonic) % 12 in steps) / dur
        r["melody_in_old_key"] = sum(n.duration for n in s.notes if (n.pitch - otonic) % 12 in steps) / dur
        roots = [(c, parse_chord_symbol(c.symbol)) for c in s.chords]
        roots = [(c, i) for c, i in roots if i and i["root_pc"] is not None]
        tot = sum(c.duration for c, _ in roots) or 1
        r["chord_roots_in_new_key"] = sum(c.duration for c, i in roots if (i["root_pc"] - tonic) % 12 in steps) / tot
        shifts = [best_pc_shift(o, s)[0] for o in orig_songs]
        r["pc_shift_matches"] = sum(1 for x in shifts if x == k % 12) / max(len(shifts), 1)
        pm = sum(n.pitch for n in s.notes) / max(len(s.notes), 1)
        om = [sum(n.pitch for n in o.notes) / max(len(o.notes), 1) for o in orig_songs]
        r["pitch_mean_shift"] = pm - sum(om) / max(len(om), 1)
        r["pitch_mean_shift_err"] = min(abs(r["pitch_mean_shift"] - k), abs(r["pitch_mean_shift"] - (k - 12 if k > 0 else k + 12)))
        r["pitch_range"] = m.get("pitch_range")
    if fam == "tempo":
        r["tempo_ok"] = float(m.get("tempo_match", 0.0))
        r["bar_count_ok"] = float(m.get("bar_count_match", 0.0))
        beats = sum(s.bar_beats)
        dur_b = beats * 60.0 / max(s.tempo_bpm, 1)
        durs_o = [sum(o.bar_beats) * 60.0 / max(o.tempo_bpm, 1) for o in orig_songs]
        r["duration_ratio"] = dur_b / (sum(durs_o) / max(len(durs_o), 1))
        r["target_duration_ratio"] = orig_spec["tempo_bpm"] / spec["tempo_bpm"]
        r["notes_per_bar"] = m.get("notes_per_bar")
        r["notes_per_sec"] = len(s.notes) / max(dur_b, 1e-9)
    if fam == "label":
        r["requested_label"] = meta["requested_label"]
    if fam == "lyrics":
        new_sy = [x for sec in spec["sections"] for x in syllables_of(sec)]
        old_sy = [x for sec in orig_spec["sections"] for x in syllables_of(sec)]
        if cond == "lyrics_all":
            r["new_recall"] = recall_of(s, new_sy)
            r["old_leak_recall"] = recall_of(s, old_sy)
            # chance level: how much of the new lyric an unmodified sample "sings" by LCS coincidence
            r["chance_new_recall_in_orig"] = sum(recall_of(o, new_sy) for o in orig_songs) / max(len(orig_songs), 1)
        else:
            nsec = syllables_of(spec["sections"][tgt])
            osec = syllables_of(orig_spec["sections"][tgt])
            r["new_recall_target_section"] = section_lyric_recall_of(s, tgt, nsec)
            r["old_leak_target_section"] = section_lyric_recall_of(s, tgt, osec)
            r["chance_new_target_in_orig"] = sum(section_lyric_recall_of(o, tgt, nsec) for o in orig_songs) / max(len(orig_songs), 1)
    return r


# ------------------------------------------------------------------ main loop
def analyse(model: str, conditions: List[str]):
    import pandas as pd
    specs = {r["song_id"]: r["spec"] for r in load_rows("test")}
    orig = load_condition(model, "orig")
    replay = load_condition(model, "replay") if model != "midi_llm" else load_condition(model, "orig_r2")
    cond_rows = {c: load_condition(model, c) for c in conditions}
    samples, effects = [], []
    for sid in sorted(specs):
        o_rows = orig.get(sid, {})
        o_songs = {k: song_of(v) for k, v in o_rows.items()}
        o_ok = {k: v for k, v in o_songs.items() if v is not None}
        o_feat = {k: features(v) for k, v in o_ok.items()}
        o_keys = sorted(o_ok)
        # within-orig distances (reseed floor)
        within = [all_distances(o_ok[a], o_ok[b], o_feat[a], o_feat[b]) for a, b in itertools.combinations(o_keys, 2)]
        base_seed = "S1" if model != "midi_llm" else "m1000"
        rp = replay.get(sid, {}).get(base_seed)
        rp_song = song_of(rp)
        if rp_song is not None and base_seed in o_ok:
            d = all_distances(o_ok[base_seed], rp_song, o_feat[base_seed], features(rp_song))
            effects.append({"model": model, "song_id": sid, "condition": "replay", "kind": "whole",
                            **{f"cross_{k}": v for k, v in d.items()},
                            **({f"within_{k}": sum(w[k] for w in within) / len(within) for k in d} if within else {})})
        # orig-sample target metrics (baseline adherence) are computed per condition's target below
        for cond, rows in cond_rows.items():
            c_rows = rows.get(sid, {})
            if not c_rows:
                continue
            any_row = next(iter(c_rows.values()))
            meta, spec = any_row.get("meta") or {}, any_row.get("spec") or specs[sid]
            fam = meta.get("family")
            tgt = meta.get("target_section")
            c_songs = {k: song_of(v) for k, v in c_rows.items()}
            orig_list = list(o_ok.values())
            for k, row in c_rows.items():
                tm = target_metrics(cond, meta, spec, specs[sid], c_songs[k], row, orig_list)
                samples.append({"model": model, "song_id": sid, "condition": cond, "family": fam, "seed": k,
                                "role": "intervened", **tm})
            # the same target measured on the unmodified samples (what the section looked like before)
            if fam in ("bars", "label", "lyrics") and tgt is not None:
                for k, row in o_rows.items():
                    tm = target_metrics(cond, meta, specs[sid], specs[sid], o_songs.get(k), row, orig_list)
                    for key in ("new_recall", "old_leak_recall", "new_recall_target_section", "old_leak_target_section"):
                        tm.pop(key, None)
                    samples.append({"model": model, "song_id": sid, "condition": cond, "family": fam, "seed": k,
                                    "role": "orig_baseline", **tm})
            # distances
            c_ok = {k: v for k, v in c_songs.items() if v is not None}
            if not c_ok or len(o_ok) < 2:
                continue
            c_feat = {k: features(v) for k, v in c_ok.items()}
            tr = 0
            if fam == "key":
                k_ = meta["semitones"]
                tr = k_ if abs(k_) <= 6 else (k_ - 12 if k_ > 0 else k_ + 12)
            # cross pairs use DIFFERENT seeds only: a same-seed pair shares the paired sampler's noise and
            # hence an identical opening, which would make "cross" look smaller than "within" by construction
            cross = [all_distances(o_ok[a], c_ok[b], o_feat[a], c_feat[b]) for a in o_keys for b in sorted(c_ok) if a != b]
            same = [all_distances(o_ok[a], c_ok[a], o_feat[a], c_feat[a]) for a in o_keys if a in c_ok]
            if not cross:
                continue
            within_c = [all_distances(c_ok[a], c_ok[b], c_feat[a], c_feat[b]) for a, b in itertools.combinations(sorted(c_ok), 2)]
            rec = {"model": model, "song_id": sid, "condition": cond, "family": fam, "kind": "whole",
                   "n_orig": len(o_ok), "n_int": len(c_ok)}
            for key in cross[0]:
                rec[f"cross_{key}"] = sum(x[key] for x in cross) / len(cross)
                rec[f"within_{key}"] = sum(x[key] for x in within) / len(within)
                if within_c:
                    rec[f"withinint_{key}"] = sum(x[key] for x in within_c) / len(within_c)
                if same:
                    rec[f"sameseed_{key}"] = sum(x[key] for x in same) / len(same)
            # seed-paired: d(A_S1, B_S1) / d(A_S1, A_S2)
            s1, s2 = ("S1", "S2") if model != "midi_llm" else ("m1000", "m1001")
            if s1 in o_ok and s2 in o_ok and s1 in c_ok:
                dab = pair_distances(o_ok[s1], c_ok[s1])["d_melody"]
                dac = pair_distances(o_ok[s1], o_ok[s2])["d_melody"]
                rec["paired_d_AB_melody"], rec["paired_d_AC_melody"] = dab, dac
                rec["paired_ratio_melody"] = dab / (dac + 1e-6)
            if fam == "key":
                tcross = [pair_distances(o_ok[a], c_ok[b], transpose=tr) for a in o_keys for b in sorted(c_ok) if a != b]
                rec["cross_transposed_d_melody"] = sum(x["d_melody"] for x in tcross) / len(tcross)
                rec["cross_transposed_d_chord"] = sum(x["d_chord"] for x in tcross) / len(tcross)
            effects.append(rec)
            # section-level: target vs non-target (bars / label / lyrics_sec)
            if tgt is not None and fam in ("bars", "label", "lyrics"):
                n_sec = len(spec["sections"])
                for kind, idx in (("target", [tgt]), ("nontarget_before", list(range(0, tgt))),
                                  ("nontarget_after", list(range(tgt + 1, n_sec)))):
                    if not idx:
                        continue
                    srec = {"model": model, "song_id": sid, "condition": cond, "family": fam, "kind": kind}
                    for dk, fn in (("melody", lambda a, b: section_dist(a, b, idx, "melody")),
                                   ("contour", lambda a, b: section_dist(a, b, idx, "contour")),
                                   ("rhythm", lambda a, b: section_dist(a, b, idx, "rhythm")),
                                   ("chord", lambda a, b: section_chord_dist(a, b, idx))):
                        cr = [fn(o_ok[a], c_ok[b]) for a in o_keys for b in c_ok if a != b]
                        wi = [fn(o_ok[a], o_ok[b]) for a, b in itertools.combinations(o_keys, 2)]
                        cr = [x for x in cr if x is not None]
                        wi = [x for x in wi if x is not None]
                        if cr and wi:
                            srec[f"cross_d_{dk}"] = sum(cr) / len(cr)
                            srec[f"within_d_{dk}"] = sum(wi) / len(wi)
                    effects.append(srec)
    return pd.DataFrame(samples), pd.DataFrame(effects)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    args = ap.parse_args()
    import pandas as pd
    from .tasks import INTERVENTIONS
    frames_s, frames_e = [], []
    for model in args.model:
        conds = list(INTERVENTIONS) if model != "midi_llm" else ["bars_p4", "label_bridge", "key_p5", "tempo_x1.25", "lyrics_all"]
        s, e = analyse(model, conds)
        print(f"{model}: {len(s)} sample rows, {len(e)} effect rows")
        frames_s.append(s)
        frames_e.append(e)
    S = pd.concat(frames_s, ignore_index=True)
    E = pd.concat(frames_e, ignore_index=True)
    tag = "_".join(args.model)
    S.to_parquet(DATA_OUT / f"interventions_{tag}.parquet", index=False)
    E.to_parquet(DATA_OUT / f"interventions_effects_{tag}.parquet", index=False)
    print("INTERVENTION_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
