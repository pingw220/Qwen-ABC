#!/usr/bin/env python
"""Evaluate section infilling and compare local infill edits with whole-song regeneration (Table 3).

  python -m paper_eval.component_sota.infill_eval
"""

from __future__ import annotations

import json
from collections import defaultdict

from qwen_abc.abc import parse_abc
from qwen_abc.abc_v2 import song_to_abc_v2
from qwen_abc.canonical import Song, normalize_chords
from qwen_abc.longrange import split_abc_sections
from qwen_abc.theory import parse_key_name

from ..common import PAPER_FINAL_GEN, bootstrap_mean_ci, fmt_ci, load_rows, mean, paired_bootstrap, write_table
from ..interventions import note_tokens, section_lyric_recall_of, section_notes, syllables_of
from ..seqsim import lcs_sim
from .infill import KINDS, build_tasks, out_path
from .melody_eval import CS_REPORT

SCALE = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}


def sec_signature(song: Song, i: int):
    """(label, bars, notes rel. to section start, chords rel., lyrics) of section i."""
    if i >= len(song.sections):
        return None
    s = song.sections[i]
    st = song.bar_starts()
    lo = st[s.start_bar] if s.start_bar < len(st) else song.total_ticks
    e = s.start_bar + s.num_bars
    hi = st[e] if e < len(st) else song.total_ticks
    notes = tuple((n.onset - lo, n.duration, n.pitch, n.melisma) for n in song.notes if lo <= n.onset < hi)
    lyr = tuple(x for n in song.notes if lo <= n.onset < hi and n.lyric for x in n.lyric)
    ch = tuple((c.onset - lo, c.symbol) for c in normalize_chords(song.chords, song.total_ticks) if lo <= c.onset < hi)
    return {"struct": (s.label, s.num_bars), "notes": notes, "lyrics": lyr, "chords": ch}


def boundary(song: Song, i: int):
    """|pitch jump| and rest gap (beats) entering and leaving section i."""
    sn = section_notes(song)
    out = {}
    if i >= len(sn) or not sn[i]:
        return out
    prev = next((x for x in reversed(sn[:i]) if x), None)
    nxt = next((x for x in sn[i + 1:] if x), None)
    if prev:
        a, b = prev[-1], sn[i][0]
        out["jump_in"] = abs(b.pitch - a.pitch)
        out["gap_in_beats"] = max(b.onset - (a.onset + a.duration), 0) / 4
    if nxt:
        a, b = sn[i][-1], nxt[0]
        out["jump_out"] = abs(b.pitch - a.pitch)
        out["gap_out_beats"] = max(b.onset - (a.onset + a.duration), 0) / 4
    return out


def compare(base: Song, new: Song, target: int, spec: dict, base_text: str = None, new_text: str = None) -> dict:
    n = len(spec["sections"])
    rec = {}
    got = new.sections[target] if target < len(new.sections) else None
    want = spec["sections"][target]
    rec["target_struct_ok"] = float(got is not None and got.label == want["label"] and got.num_bars == want["bars"])
    rec["target_lyric_recall"] = section_lyric_recall_of(new, target, syllables_of(want)) if syllables_of(want) else None
    kp = parse_key_name(spec.get("key"))
    tn = section_notes(new)[target] if target < len(new.sections) else []
    if kp and tn:
        rec["target_in_key"] = sum(x.duration for x in tn if (x.pitch - kp[0]) % 12 in SCALE[kp[1]]) / max(sum(x.duration for x in tn), 1)
    rec.update(boundary(new, target))
    others = [i for i in range(n) if i != target]
    same = {"struct": [], "notes": [], "lyrics": [], "chords": [], "text": []}
    mel_d = []
    bt = split_abc_sections(base_text)[1] if base_text else None
    nt = split_abc_sections(new_text)[1] if new_text else None
    for i in others:
        a, b = sec_signature(base, i), sec_signature(new, i)
        if a is None or b is None:
            for k in same:
                same[k].append(0.0)
            continue
        for k in ("struct", "notes", "lyrics", "chords"):
            same[k].append(float(a[k] == b[k]))
        if bt is not None and nt is not None and i < len(bt) and i < len(nt):
            same["text"].append(float(bt[i] == nt[i]))
        sa, sb = section_notes(base), section_notes(new)
        if i < len(sa) and i < len(sb) and (sa[i] or sb[i]):
            mel_d.append(1 - lcs_sim(note_tokens(base, "melody", notes=sa[i]), note_tokens(new, "melody", notes=sb[i])))
    for k, v in same.items():
        if v:
            rec[f"nontarget_{k}_identical"] = sum(v) / len(v)
    rec["nontarget_melody_change"] = mean(mel_d)
    rec["plan_exact"] = float([(s.label, s.num_bars) for s in new.sections] == [(s["label"], s["bars"]) for s in spec["sections"]])
    return rec


def main():
    import pandas as pd
    rows = {r["song_id"]: r for r in load_rows("test")}
    recs = []
    # reference boundary statistics (what real section boundaries look like)
    ref_b = defaultdict(list)
    for r in rows.values():
        s = Song.from_json(r["song"])
        for i in range(len(s.sections)):
            for k, v in boundary(s, i).items():
                ref_b[k].append(v)
    for kind, sid, tgt, ctx_abc, spec, _ in build_tasks():
        ctx = parse_abc(ctx_abc).song
        for seed in ("S1", "S2"):
            p = out_path(kind, sid, seed)
            if not p.exists():
                recs.append({"method": "infill", "kind": kind, "song_id": sid, "seed": seed, "missing": 1.0})
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            pr = parse_abc(r["assembled"])
            rec = {"method": "infill", "kind": kind, "song_id": sid, "seed": seed, "parse_ok": float(pr.ok),
                   "strict_ok": float(pr.strict_ok)}
            if pr.ok:
                rec.update(compare(ctx, pr.song, tgt, spec, ctx_abc, r["assembled"]))
                if kind.startswith("recon"):
                    refsn, gsn = section_notes(ctx)[tgt], section_notes(pr.song)[tgt] if tgt < len(pr.song.sections) else []
                    rec["recon_interval_sim"] = lcs_sim(note_tokens(ctx, "contour", notes=refsn), note_tokens(pr.song, "contour", notes=gsn))
            recs.append(rec)
        # whole-song regeneration answering the same edit request
        if kind.startswith("edit"):
            cond, seeds = {"edit_resample": ("orig", ("S2", "S3")), "edit_lyrics": ("lyrics_sec", ("S1", "S2")),
                           "edit_extend": ("bars_p4", ("S1", "S2"))}[kind]
            for seed in seeds:
                q = PAPER_FINAL_GEN / "qwen_e3b" / cond / f"{sid}_{seed}.json"
                if not q.exists():
                    continue
                g = json.loads(q.read_text(encoding="utf-8"))
                rec = {"method": "regenerate", "kind": kind, "song_id": sid, "seed": seed, "parse_ok": float(bool(g.get("song"))),
                       "strict_ok": float(bool(g.get("strict_ok")))}
                if g.get("song"):
                    ns = Song.from_json(g["song"])
                    rec.update(compare(ctx, ns, tgt, spec, ctx_abc, song_to_abc_v2(ns)))
                recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_parquet(CS_REPORT / "data" / "infill.parquet", index=False)
    cols = [("target_struct_ok", "target label+bars as requested"), ("target_lyric_recall", "target lyric recall"),
            ("nontarget_text_identical", "untouched sections byte-identical"), ("nontarget_notes_identical", "untouched: notes identical"),
            ("nontarget_chords_identical", "untouched: chords identical"), ("nontarget_lyrics_identical", "untouched: lyrics identical"),
            ("nontarget_struct_identical", "untouched: label+bars identical"), ("nontarget_melody_change", "untouched: melody change (1−LCS)"),
            ("jump_in", "abs. pitch jump into target"), ("jump_out", "abs. pitch jump out of target"), ("gap_in_beats", "rest before target (beats)"),
            ("target_in_key", "target melody in key"), ("strict_ok", "strict-valid song"), ("recon_interval_sim", "interval sim. to reference section")]
    table = []
    for (method, kind), d in df.groupby(["method", "kind"]):
        row = {"method": method, "edit / task": kind, "N songs": d.song_id.nunique(), "samples": len(d)}
        for c, lab in cols:
            if c in d and d[c].notna().any():
                per = d.groupby("song_id")[c].mean().dropna()
                m, lo, hi, n = bootstrap_mean_ci(per.tolist())
                row[lab] = fmt_ci(m, lo, hi, 2 if "jump" in c or "gap" in c else 3)
        table.append(row)
    table.append({"method": "reference songs", "edit / task": "all section boundaries", "N songs": len(rows),
                  "abs. pitch jump into target": f"{mean(ref_b['jump_in']):.2f}", "abs. pitch jump out of target": f"{mean(ref_b['jump_out']):.2f}",
                  "rest before target (beats)": f"{mean(ref_b['gap_in_beats']):.2f}"})
    write_table(table, "infill", "Section infilling (E3b) vs whole-song regeneration for the same local edit; song-level means over 2 seeds, 95% bootstrap",
                table_dir=CS_REPORT / "tables")
    drows = []
    for kind in ("edit_resample", "edit_lyrics", "edit_extend"):
        a = df[(df.method == "infill") & (df.kind == kind)]
        b = df[(df.method == "regenerate") & (df.kind == kind)]
        for c, lab in cols[:8]:
            if c in a and c in b:
                d = paired_bootstrap(a.groupby("song_id")[c].mean().dropna().to_dict(), b.groupby("song_id")[c].mean().dropna().to_dict())
                if d["n"]:
                    drows.append({"edit": kind, "metric": lab, "infill − regenerate [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N": d["n"]})
    write_table(drows, "infill_vs_regen", "Paired infill − whole-song regeneration, same edit request, same songs", table_dir=CS_REPORT / "tables")
    print("INFILL_EVAL_DONE")


if __name__ == "__main__":
    main()
