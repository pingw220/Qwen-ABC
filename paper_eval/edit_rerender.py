#!/usr/bin/env python
"""Edit -> re-render demonstration: the generated lead sheet as an editable interface.

  python -m paper_eval.edit_rerender generate      # GPU: source songs, local (infill) and whole-song edits
  python -m paper_eval.edit_rerender build         # CPU: splice/transpose/retempo, verify, write scores + metadata
  python scripts/export_leadsheet_midi.py reports/paper_final/edit_rerender_demo/scores/*.abc \
      --outdir reports/paper_final/edit_rerender_demo/materials/leadsheets
  LEADSHEETS=... OUT=... sbatch scripts/slurm/midi_sag_render.sbatch      # FastSinger + MuseControlLite, unchanged
  python -m paper_eval.edit_rerender audio         # durations of the renders -> metadata
  python -m paper_eval.edit_rerender figure        # figures/edit_rerender_summary.{pdf,png}

Source: E3b (paper_eval.common.MODELS['qwen_e3b']), T=1.0 / top-p 0.95, paired-seed sampler,
four samples S1..S4 of the unedited plan, one chosen by the repo's inference-only ``sum`` selector
(scripts/select_best_of_n.py). Songs were chosen from reference features only (a later-half,
8-bar, all-4/4 chorus that the model's infill training covers, all-Chinese lyrics, 70-125 BPM,
no pathology flag), never from model output.

Edits (each changes one control):

A  structure  target chorus 8 -> 12 bars
   A_local   section infill: the rest of the score is kept byte-identical, E3b writes only the section
   A_whole   whole-song regeneration under the edited plan (same seed as the source)
B  key +5     B_direct: exact symbolic transposition of the score; B_model: regeneration under the new key
C  tempo 1.25x  C_direct: only the Q: field changes; C_model: regeneration under the new tempo
D  lyrics     one section's lyrics replaced (all-Chinese donor, same syllable count), by section infill

Scores, generations and audio contain lyrics and live in git-ignored ``scores/`` and ``materials/``;
the committed metadata carries measurements only.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

from .common import MODELS, REPORT_DIR, SEEDS, jdump, jload, load_rows
from .interventions import (best_pc_shift, change_bars, pick_donor, pick_target_section, replace_lyrics,
                            scale_tempo, section_notes, syllables_of, transpose_key)

DEMO = REPORT_DIR / "edit_rerender_demo"
MAT = DEMO / "materials"
GEN = MAT / "generations"
SCORES = DEMO / "scores"
SONGS = ["4VfHbfbzNPZmX5KI4iAOyT", "6coZcwGP3Cmhyqzvp6sEJS"]
T, P = 1.0, 0.95
AUDIO_ROOT = Path("/gscratch/ark/pingw220/qwen_abc_r2_offload/paper_final_edit_rerender")


def _rows():
    return {r["song_id"]: r for r in load_rows("test", ids=SONGS)}


def edited_specs(r: dict, pool: dict) -> dict:
    spec, sid = r["spec"], r["song_id"]
    t = pick_target_section(spec, sid)
    donor = pick_donor(sid, spec, pool)
    dsyl = [x for s in pool[donor]["sections"] for x in syllables_of(s)]
    return {"target": t, "donor": donor,
            "A": change_bars(spec, t, 12 - spec["sections"][t]["bars"]),
            "B": transpose_key(spec, 5), "C": scale_tempo(spec, 1.25), "D": replace_lyrics(spec, dsyl, t)}


def infill_prompt(spec_edit: dict, source_abc: str, target: int) -> str:
    """The masked-late-section prompt E3b was trained on (qwen_abc/longrange.py), built from the
    *edited* plan and the *generated* source score with the target section cut out."""
    from qwen_abc.abc_v2 import PROMPT_HEADER_V2, spec_to_prompt_v2
    from qwen_abc.longrange import GAP_LINE, INFILL_HEADER, INFILL_MARKER, split_abc_sections
    from qwen_abc.prompt import COMPLETION_MARKER
    header, blocks = split_abc_sections(source_abc)
    n = len(spec_edit["sections"])
    sec = spec_edit["sections"][target]
    head = f"P:{sec['label']}\n% section {target + 1}/{n} | {sec['bars']} bars\n"
    gapped = header + "".join(blocks[:target]) + head + GAP_LINE + "\n" + "".join(blocks[target + 1:])
    base = spec_to_prompt_v2(spec_edit)
    body = base[len(PROMPT_HEADER_V2): -len(COMPLETION_MARKER)]
    return INFILL_HEADER + body + "ABC with a gap:\n" + gapped + "\n" + INFILL_MARKER


def splice(source_abc: str, completion: str, target: int) -> str:
    from qwen_abc.longrange import split_abc_sections
    header, blocks = split_abc_sections(source_abc)
    comp = completion.strip("\n") + "\n"
    return header + "".join(blocks[:target]) + comp + "".join(blocks[target + 1:])


# ------------------------------------------------------------------ generation (GPU)
def cmd_generate(args) -> None:
    from qwen_abc.abc_v2 import spec_to_prompt_v2
    from qwen_abc.generate import load_for_generation
    from .sampler import generate_batch_crn
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_eval import score_row
    from select_best_of_n import rank_key

    rows = _rows()
    pool = {r["song_id"]: r["spec"] for r in load_rows("test")}
    model, tok = load_for_generation(MODELS["qwen_e3b"]["ckpt"])
    seeds = ["S1", "S2", "S3", "S4"]

    def run(prompts, seed_names, max_new, max_total):
        outs = []
        for k in range(0, len(prompts), 8):
            outs += generate_batch_crn(model, tok, prompts[k:k + 8], [SEEDS[s] for s in seed_names[k:k + 8]],
                                       max_new, T, P, max_total)
        return outs

    for sid in SONGS:
        r = rows[sid]
        # 1. source: four whole-song samples of the unedited plan, one chosen by the inference-only selector
        src_rows = []
        pending = [s for s in seeds if not (GEN / f"{sid}__source_{s}.json").exists()]
        if pending:
            outs = run([spec_to_prompt_v2(r["spec"])] * len(pending), pending, 8704, 9216)
            for s, g in zip(pending, outs):
                row = {"song_id": sid, "variant": "source", "seed_name": s, "seed": SEEDS[s], "generation": g["text"],
                       "hit_eos": g["hit_eos"], "new_tokens": g["new_tokens"], "seconds": g["seconds"]}
                score_row(row, r["spec"], "v2")
                jdump(row, GEN / f"{sid}__source_{s}.json", indent=None)
        for s in seeds:
            src_rows.append((s, jload(GEN / f"{sid}__source_{s}.json")))
        best_s, best = max(src_rows, key=lambda t: rank_key(t[1], "sum", seeds.index(t[0])))
        jdump({"chosen_seed": best_s, "rule": "sum (scripts/select_best_of_n.py, inference-only signals)",
               "per_seed": {s: {"strict_ok": x["strict_ok"], "section_plan_exact": x["metrics"].get("section_plan_exact"),
                                "lyric_recall": x["metrics"].get("lyric_recall")} for s, x in src_rows}},
              GEN / f"{sid}__source_choice.json")
        e = edited_specs(r, pool)
        t = e["target"]
        # 2. local edits by section infill (4 seeds each)
        jobs = []
        for v, spec in (("A_local", e["A"]), ("D_local", e["D"])):
            for s in seeds:
                if not (GEN / f"{sid}__{v}_{s}.json").exists():
                    jobs.append((v, s, infill_prompt(spec, best["generation"], t)))
        if jobs:
            outs = run([j[2] for j in jobs], [j[1] for j in jobs], 2048, 12288)
            for (v, s, pr), g in zip(jobs, outs):
                jdump({"song_id": sid, "variant": v, "seed_name": s, "seed": SEEDS[s], "target_section": t,
                       "completion": g["text"], "hit_eos": g["hit_eos"], "new_tokens": g["new_tokens"],
                       "prompt_tokens": g["prompt_tokens"]}, GEN / f"{sid}__{v}_{s}.json", indent=None)
        # 3. whole-song regenerations under each edited control, same seed as the chosen source
        jobs = [(v, spec) for v, spec in (("A_whole", e["A"]), ("B_model", e["B"]), ("C_model", e["C"]))
                if not (GEN / f"{sid}__{v}_{best_s}.json").exists()]
        if jobs:
            outs = run([spec_to_prompt_v2(sp) for _, sp in jobs], [best_s] * len(jobs), 8704, 9216)
            for (v, sp), g in zip(jobs, outs):
                row = {"song_id": sid, "variant": v, "seed_name": best_s, "seed": SEEDS[best_s], "generation": g["text"],
                       "hit_eos": g["hit_eos"], "new_tokens": g["new_tokens"]}
                score_row(row, sp, "v2")
                jdump(row, GEN / f"{sid}__{v}_{best_s}.json", indent=None)
    print("EDIT_GEN_DONE")


# ------------------------------------------------------------------ build + verify (CPU)
def _note_sig(song, notes, start_tick):
    return [(n.onset - start_tick, n.duration, n.pitch, tuple(n.lyric) if n.lyric else None, n.melisma) for n in notes]


def _sections_identical(a, b, skip: int):
    """Per non-target section: are notes (relative to section start), bars and label identical?"""
    sa, sb = section_notes(a), section_notes(b)
    starts_a, starts_b = a.bar_starts(), b.bar_starts()
    same, total = 0, 0
    for i in range(len(a.sections)):
        if i == skip:
            continue
        total += 1
        if i >= len(b.sections):
            continue
        x, y = a.sections[i], b.sections[i]
        if (x.label, x.num_bars) != (y.label, y.num_bars):
            continue
        if _note_sig(a, sa[i], starts_a[x.start_bar]) == _note_sig(b, sb[i], starts_b[y.start_bar]):
            same += 1
    return same, total


def _chords_in_section(song, i):
    starts = song.bar_starts()
    s = song.sections[i]
    a = starts[s.start_bar]
    e = s.start_bar + s.num_bars
    b = starts[e] if e < len(starts) else song.total_ticks
    return [c.symbol for c in song.chords if a <= c.onset < b]


def _in_scale(song, key: str) -> float:
    """Duration share of melody notes inside the diatonic scale of ``key``."""
    from qwen_abc.metrics import SCALE_STEPS
    from qwen_abc.theory import parse_key_name
    tonic, mode = parse_key_name(key)
    dur = sum(n.duration for n in song.notes) or 1
    return sum(n.duration for n in song.notes if (n.pitch - tonic) % 12 in SCALE_STEPS[mode]) / dur


def _texts_identical(src_abc: str, abc: str, skip: int) -> str:
    from qwen_abc.longrange import split_abc_sections
    _, a = split_abc_sections(src_abc)
    _, b = split_abc_sections(abc)
    same = sum(1 for i in range(len(a)) if i != skip and i < len(b) and a[i] == b[i])
    return f"{same}/{len(a) - 1}"


def transpose_song(song, k: int):
    import copy
    from qwen_abc.theory import canonical_key, parse_key_name, transpose_chord_symbol
    from .interventions import NAMES
    out = copy.deepcopy(song)
    pc, mode = parse_key_name(song.key)
    out.key = canonical_key(f"{NAMES[(pc + k) % 12]} {mode}")
    for n in out.notes:
        n.pitch += k
    for c in out.chords:
        c.symbol = transpose_chord_symbol(c.symbol, k, out.key)
    return out


def cmd_build(args) -> None:
    from qwen_abc.abc import parse_abc
    from qwen_abc.abc_v2 import song_to_abc_v2
    from qwen_abc.canonical import Song
    from qwen_abc.metrics import song_metrics
    from .interventions import section_lyric_recall_of

    rows = _rows()
    pool = {r["song_id"]: r["spec"] for r in load_rows("test")}
    SCORES.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid in SONGS:
        r = rows[sid]
        e = edited_specs(r, pool)
        t = e["target"]
        choice = jload(GEN / f"{sid}__source_choice.json")
        s0 = choice["chosen_seed"]
        src = jload(GEN / f"{sid}__source_{s0}.json")
        src_abc = src["generation"].rstrip("\n") + "\n"
        src_song = parse_abc(src_abc, sid).song
        (SCORES / f"{sid}__source.abc").write_text(src_abc, encoding="utf-8")
        meta = {"song_id": sid, "model": "qwen_e3b", "checkpoint": MODELS["qwen_e3b"]["ckpt"], "temperature": T, "top_p": P,
                "sampler": "gumbel_crn", "source": {"seed": s0, "selection": choice, "strict_ok": src["strict_ok"],
                                                     "plan_exact": src["metrics"]["section_plan_exact"],
                                                     "lyric_recall": round(src["metrics"]["lyric_recall"], 4),
                                                     "key": src_song.key, "tempo": src_song.tempo_bpm,
                                                     "target_section": t, "target_label": src_song.sections[t].label,
                                                     "target_bars": src_song.sections[t].num_bars,
                                                     "n_bars": len(src_song.bar_beats)},
                "donor_song_for_lyrics": e["donor"], "demos": {}}

        def local(variant, spec_edit, want_bars, new_sylls=None, old_sylls=None):
            tries = []
            for s in ("S1", "S2", "S3", "S4"):
                g = jload(GEN / f"{sid}__{variant}_{s}.json")
                abc = splice(src_abc, g["completion"], t)
                res = parse_abc(abc, sid)
                rec = {"seed": s, "hit_eos": g["hit_eos"], "parse_ok": res.ok, "strict_ok": bool(res.ok and res.strict_ok)}
                if res.ok and t < len(res.song.sections):
                    sec = res.song.sections[t]
                    same, tot = _sections_identical(src_song, res.song, t)
                    rec.update(target_label=sec.label, target_bars=sec.num_bars, n_sections=len(res.song.sections),
                               bars_ok=sec.num_bars == want_bars and sec.label == spec_edit["sections"][t]["label"],
                               other_sections_text_identical=_texts_identical(src_abc, abc, t),
                               other_sections_identical=f"{same}/{tot}",
                               plan_exact=song_metrics(res.song, spec_edit).get("section_plan_exact"))
                    if new_sylls is not None:
                        rec["target_new_lyric_recall"] = round(section_lyric_recall_of(res.song, t, new_sylls), 4)
                        rec["target_old_lyric_recall"] = round(section_lyric_recall_of(res.song, t, old_sylls), 4)
                        rec["source_target_new_lyric_recall"] = round(section_lyric_recall_of(src_song, t, new_sylls), 4)
                tries.append((rec, abc, res))
            # demo sample: first seed (S1..S4 order) that parses with the requested label and bar count;
            # every signal is available at inference (plan + output), none uses a reference
            ok = [x for x in tries if x[0].get("bars_ok") and x[0]["strict_ok"]] or [x for x in tries if x[0].get("bars_ok")] or tries
            rec, abc, res = ok[0]
            (SCORES / f"{sid}__{variant}.abc").write_text(abc, encoding="utf-8")
            return {"method": "section infill (qwen_abc/longrange.py prompt), rest of score byte-identical",
                    "chosen_seed": rec["seed"], "chosen": rec,
                    "all_seeds": [x[0] for x in tries],
                    "success_rate_bars_ok": sum(bool(x[0].get("bars_ok")) for x in tries) / len(tries)}

        def whole(variant, spec_edit):
            g = jload(GEN / f"{sid}__{variant}_{s0}.json")
            abc = g["generation"].rstrip("\n") + "\n"
            (SCORES / f"{sid}__{variant}.abc").write_text(abc, encoding="utf-8")
            return g, parse_abc(abc, sid)

        # A structure
        want = e["A"]["sections"][t]["bars"]
        dA = {"requested": {"target_section": t, "bars_from": r["spec"]["sections"][t]["bars"], "bars_to": want},
              "A_local": local("A_local", e["A"], want)}
        g, res = whole("A_whole", e["A"])
        if res.ok:
            same, tot = _sections_identical(src_song, res.song, t)
            dA["A_whole"] = {"method": "whole-song regeneration, same seed", "strict_ok": res.strict_ok,
                             "target_bars": res.song.sections[t].num_bars if t < len(res.song.sections) else None,
                             "plan_exact": g["metrics"]["section_plan_exact"], "lyric_recall": round(g["metrics"]["lyric_recall"], 4),
                             "other_sections_identical": f"{same}/{tot}"}
        meta["demos"]["A_structure"] = dA
        # B key +5
        tsong = transpose_song(src_song, 5)
        babc = song_to_abc_v2(tsong)
        (SCORES / f"{sid}__B_direct.abc").write_text(babc, encoding="utf-8")
        bres = parse_abc(babc, sid)
        bd = bres.song
        exact = (len(bd.notes) == len(src_song.notes) and all(
            (x.onset, x.duration, x.pitch + 5, x.lyric, x.melisma) == (y.onset, y.duration, y.pitch, y.lyric, y.melisma)
            for x, y in zip(src_song.notes, bd.notes)))
        dB = {"requested": {"key_from": src_song.key, "key_to": e["B"]["key"], "semitones": 5},
              "B_direct": {"method": "symbolic transposition of the parsed score (+5 semitones, chords respelled)",
                           "parse_ok": bres.ok, "strict_ok": bres.strict_ok, "declared_key": bd.key,
                           "every_note_exactly_+5": exact, "pc_shift_estimate": best_pc_shift(src_song, bd)[0],
                           "melody_in_new_key": round(_in_scale(bd, e["B"]["key"]), 3),
                           "pitch_mean_shift": round(sum(n.pitch for n in bd.notes) / len(bd.notes) - sum(n.pitch for n in src_song.notes) / len(src_song.notes), 3),
                           "chords_same_count": len(bd.chords) == len(src_song.chords)}}
        g, res = whole("B_model", e["B"])
        if res.ok:
            k, corr = best_pc_shift(src_song, res.song)
            dB["B_model"] = {"method": "whole-song regeneration under Key+5, same seed", "strict_ok": res.strict_ok,
                             "declared_key": res.song.key, "key_followed": res.song.key == e["B"]["key"],
                             "pc_shift_estimate": k, "pc_shift_correlation": round(corr, 3),
                             "pitch_mean_shift": round(g["metrics"]["pitch_mean"] - src["metrics"]["pitch_mean"], 3),
                             "melody_in_new_key": round(_in_scale(res.song, e["B"]["key"]), 3),
                             "melody_in_old_key": round(_in_scale(res.song, src_song.key), 3),
                             "source_melody_in_new_key": round(_in_scale(src_song, e["B"]["key"]), 3),
                             "source_melody_in_old_key": round(_in_scale(src_song, src_song.key), 3),
                             "plan_exact": g["metrics"]["section_plan_exact"], "lyric_recall": round(g["metrics"]["lyric_recall"], 4)}
        meta["demos"]["B_key"] = dB
        # C tempo 1.25x
        new_t = e["C"]["tempo_bpm"]
        lines = src_abc.split("\n")
        q = [i for i, l in enumerate(lines) if l.startswith("Q:")]
        lines[q[0]] = f"Q:1/4={new_t}"
        cabc = "\n".join(lines)
        (SCORES / f"{sid}__C_direct.abc").write_text(cabc, encoding="utf-8")
        cres = parse_abc(cabc, sid)
        beats = sum(src_song.bar_beats)
        dC = {"requested": {"tempo_from": src_song.tempo_bpm, "tempo_to": new_t, "factor": 1.25},
              "C_direct": {"method": "only the Q: field edited", "parse_ok": cres.ok, "tempo": cres.song.tempo_bpm,
                           "notes_identical": [(n.onset, n.duration, n.pitch) for n in cres.song.notes] == [(n.onset, n.duration, n.pitch) for n in src_song.notes],
                           "score_duration_s": round(beats * 60 / new_t, 2), "source_score_duration_s": round(beats * 60 / src_song.tempo_bpm, 2)}}
        g, res = whole("C_model", e["C"])
        if res.ok:
            b2 = sum(res.song.bar_beats)
            dC["C_model"] = {"method": "whole-song regeneration under the new tempo, same seed", "strict_ok": res.strict_ok,
                             "tempo": res.song.tempo_bpm, "tempo_followed": res.song.tempo_bpm == new_t,
                             "bars": len(res.song.bar_beats), "source_bars": len(src_song.bar_beats),
                             "plan_exact": g["metrics"]["section_plan_exact"], "score_duration_s": round(b2 * 60 / res.song.tempo_bpm, 2),
                             "notes_per_bar": round(g["metrics"]["notes_per_bar"], 3), "source_notes_per_bar": round(src["metrics"]["notes_per_bar"], 3),
                             "lyric_recall": round(g["metrics"]["lyric_recall"], 4)}
        meta["demos"]["C_tempo"] = dC
        # D lyrics
        new_s = syllables_of(e["D"]["sections"][t])
        old_s = syllables_of(r["spec"]["sections"][t])
        meta["demos"]["D_lyrics"] = {"requested": {"target_section": t, "n_syllables": len(new_s), "donor": e["donor"]},
                                     "D_local": local("D_local", e["D"], r["spec"]["sections"][t]["bars"], new_s, old_s)}
        jdump(meta, DEMO / f"demo_{sid}.json")
        summary.append(meta)
    jdump(summary, DEMO / "summary.json")
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:6000])


# ------------------------------------------------------------------ audio verification
def _wav_seconds(p: Path):
    try:
        with wave.open(str(p)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        try:
            import soundfile as sf
            return sf.info(str(p)).duration
        except Exception:  # noqa: BLE001
            return None


def _first_voice_s(p: Path, thresh_db: float = -35.0):
    """Seconds of the first 50-ms frame above thresh (vocal onset), for timing checks."""
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(str(p), always_2d=True)
        x = x.mean(1)
        hop = int(0.05 * sr)
        rms = [20 * np.log10(np.sqrt(np.mean(x[i:i + hop] ** 2)) + 1e-9) for i in range(0, len(x) - hop, hop)]
        for i, v in enumerate(rms):
            if v > thresh_db:
                return i * 0.05
    except Exception:  # noqa: BLE001
        return None
    return None


def cmd_audio(args) -> None:
    root = Path(args.render_dir)
    for sid in SONGS:
        p = DEMO / f"demo_{sid}.json"
        meta = jload(p)
        audio = {}
        for d in sorted(root.glob(f"{sid}__*")):
            v = d.name.split("__", 1)[1]
            audio[v] = {"mix_wav": str(d / "mix.wav") if (d / "mix.wav").exists() else None,
                        "mix_seconds": _wav_seconds(d / "mix.wav"),
                        "vocal_seconds": _wav_seconds(d / "vocal_fs.wav"),
                        "vocal_onset_s": _first_voice_s(d / "vocal_fs.wav"),
                        "rendered": (d / "mix.wav").exists()}
        meta["audio"] = audio
        src = audio.get("source", {}).get("mix_seconds")
        for v in ("C_direct", "C_model", "A_local", "A_whole", "B_direct", "B_model", "D_local"):
            if src and audio.get(v, {}).get("mix_seconds"):
                audio[v]["duration_ratio_vs_source"] = round(audio[v]["mix_seconds"] / src, 4)
        jdump(meta, p)
        print(sid, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mix_wav"} for k, v in audio.items()}, indent=1))


# ------------------------------------------------------------------ figure 6
def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qwen_abc.abc import parse_abc
    matplotlib.rcParams.update({"font.size": 8, "svg.hashsalt": "edit", "pdf.fonttype": 42})

    sid = SONGS[0]
    meta = jload(DEMO / f"demo_{sid}.json")
    t = meta["source"]["target_section"]
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.1, 1.1], hspace=0.75, wspace=0.45)
    # top row: requested vs measured per demo
    d = meta["demos"]
    panels = [
        ("A structure: target bars", [("req", d["A_structure"]["requested"]["bars_to"]),
                                      ("local", d["A_structure"]["A_local"]["chosen"].get("target_bars")),
                                      ("whole", (d["A_structure"].get("A_whole") or {}).get("target_bars")),
                                      ("src", d["A_structure"]["requested"]["bars_from"])]),
        ("B key +5: mean pitch shift (st)", [("req", 5), ("direct", d["B_key"]["B_direct"]["pitch_mean_shift"]),
                                             ("model", (d["B_key"].get("B_model") or {}).get("pitch_mean_shift"))]),
        ("C tempo: audio duration ratio", [("req", 0.8)] + [(v.split("_")[1], (meta.get("audio", {}).get(v) or {}).get("duration_ratio_vs_source"))
                                                            for v in ("C_direct", "C_model")]),
        ("D lyrics: target-section recall", [("new", d["D_lyrics"]["D_local"]["chosen"].get("target_new_lyric_recall")),
                                            ("old", d["D_lyrics"]["D_local"]["chosen"].get("target_old_lyric_recall")),
                                            ("src/new", d["D_lyrics"]["D_local"]["chosen"].get("source_target_new_lyric_recall"))]),
    ]
    for j, (title, bars) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        xs = [b[0] for b in bars]
        ys = [b[1] if isinstance(b[1], (int, float)) else 0 for b in bars]
        cols = ["#9aa0a6" if x in ("req", "src", "src/new", "old") else "#2a6fdb" for x in xs]
        ax.bar(range(len(xs)), ys, color=cols)
        for i, (x, y) in enumerate(zip(xs, [b[1] for b in bars])):
            ax.text(i, (y or 0), "n/a" if y is None else (f"{y:g}" if isinstance(y, int) else f"{y:.2f}"),
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(xs)), xs, fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    def roll(ax, variant, title):
        p = SCORES / f"{sid}__{variant}.abc"
        s = parse_abc(p.read_text(encoding="utf-8"), sid).song
        starts = s.bar_starts()
        sec = s.sections[t]
        lo = starts[max(sec.start_bar - 2, 0)]
        e = min(sec.start_bar + sec.num_bars + 2, len(starts) - 1)
        hi = starts[e]
        a0, a1 = starts[sec.start_bar], (starts[sec.start_bar + sec.num_bars] if sec.start_bar + sec.num_bars < len(starts) else s.total_ticks)
        ax.axvspan(a0 / 16, a1 / 16, color="#f4d03f", alpha=0.25, lw=0)
        for n in s.notes:
            if lo <= n.onset < hi:
                ax.broken_barh([(n.onset / 16, n.duration / 16)], (n.pitch - 0.4, 0.8), color="#1f3b73")
        ax.set_xlim(lo / 16, hi / 16)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("bar", fontsize=6)
        ax.set_ylabel("MIDI pitch", fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    roll(fig.add_subplot(gs[1, 0:2]), "source", f"source: section {t + 1} ({meta['source']['target_label']}, {meta['source']['target_bars']} bars) shaded")
    roll(fig.add_subplot(gs[1, 2:4]), "A_local", "A local edit: same section infilled at 12 bars")
    roll(fig.add_subplot(gs[2, 0:2]), "B_direct", "B direct: score transposed +5")
    roll(fig.add_subplot(gs[2, 2:4]), "D_local", "D local edit: target section re-sung to new lyrics")
    out = REPORT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "edit_rerender_summary.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / "edit_rerender_summary.png", dpi=200, bbox_inches="tight")
    print("FIGURE_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "build", "audio", "figure"])
    ap.add_argument("--render-dir", default=str(AUDIO_ROOT))
    args = ap.parse_args()
    {"generate": cmd_generate, "build": cmd_build, "audio": cmd_audio, "figure": cmd_figure}[args.command](args)


if __name__ == "__main__":
    main()
