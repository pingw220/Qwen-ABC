#!/usr/bin/env python
"""Audio side of the component round: frozen subset, two-SVS rendering, backing renders.

  python -m paper_eval.component_sota.audio subset                     # freeze the audio subset (reference features only)
  python -m paper_eval.component_sota.audio prep-svs --sources ref,e3b,mel,csl_offc,csl_rtc
  python -m paper_eval.component_sota.audio render-svs --shard k --num-shards n    # GPU
  python -m paper_eval.component_sota.audio prep-backing                # chord-source and closed-loop lead sheets
  python -m paper_eval.component_sota.audio render-backing --shard k --num-shards n  # GPU

**Identical symbolic input to both SVS.** One sung-note list per (source, song) is built by
``qwen_abc.fastsinger.sung_notes`` (wordless notes dropped, crammed notes divided, melisma held),
and both renderers are fed from it: FastSinger through its two files, SoulX-Singer through a MIDI
with one lyric event per note (melisma written as MIDI-SAG's adapter does: ``-`` alternating with a
repeat of the syllable, because SoulX turns two consecutive ``-`` into silence). The list itself
(``target.json``) is the reference for pitch / timing / lyric metrics.

Renderers are called exactly as their own scripts call them (FastSinger inference.py; SoulX-Singer
midi2json.py + cli.inference --control score with the shipped zh prompt; MIDI-SAG
run_midi_llm_to_midi_sag.py --harmonizer leadsheet for backings). Nothing upstream is modified.
Backings are rendered on the first <= 95.1 s of the song (two MuseControlLite windows) to bound cost.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from qwen_abc.canonical import Section, Song
from qwen_abc.fastsinger import sung_notes, write_fastsinger_inputs

from ..common import PAPER_FINAL_GEN, jdump, load_rows
from .sources import CS_ROOT, iter_melodies

AUDIO = Path("/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio")   # wavs (large, off-repo)
MANIFEST = CS_ROOT.parent.parent / "reports/component_sota/audio"
MUSIC = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc")
FS_PY = "/gscratch/ark/pingw220/miniconda3/envs/fastsinger/bin/python"
VPY = "/gscratch/ark/pingw220/midisag/.venv-midisag/bin/python"
SOULX = MUSIC / "MIDI-SAG/SoulX-Singer"
PROMPT = MUSIC / "MIDI-SAG/example_input"
EXCERPT_S = 95.1


# ------------------------------------------------------------------ subset (frozen before any audio metric)
def subset(args):
    import sys as _s
    _s.path.insert(0, str(CS_ROOT.parent.parent / "scripts"))
    from select_clean_subset import RULES
    rows = load_rows("test")
    cand = []
    for r in rows:
        spec = r["spec"]
        if spec["meter"] != "4/4":
            continue   # CSL-L2M, AccoMontage2 and the backing renderer are 4/4-only
        n_sec = len(spec["sections"])
        bars = sum(s["bars"] for s in spec["sections"])
        syl = sum(len([c for l in s["lines"] for c in l if not c.isspace()]) for s in spec["sections"])
        diff = sum(1 for fn in RULES.values() if not fn(r, r["flags"], r["cleaning"]))
        e3 = PAPER_FINAL_GEN / "qwen_e3b/orig" / f"{r['song_id']}_S1.json"
        agree = json.loads(e3.read_text()).get("metrics", {}).get("section_plan_exact", 0) if e3.exists() else 0
        cand.append({"song_id": r["song_id"], "sections": n_sec, "bars": bars, "syllables": syl,
                     "mode": (spec.get("key") or "x minor").split()[-1], "difficulty": diff, "e3b_plan_exact": agree})
    import numpy as np
    for k in ("bars", "syllables"):
        q = np.quantile([c[k] for c in cand], [1 / 3, 2 / 3])
        for c in cand:
            c[f"{k}_tercile"] = int(np.searchsorted(q, c[k], side="right"))
    strata = {}
    for c in cand:
        key = (c["bars_tercile"], c["syllables_tercile"], c["mode"], min(c["difficulty"], 2) >= 2)
        strata.setdefault(key, []).append(c)
    h = lambda s: hashlib.sha256(f"audio-subset:{s}".encode()).hexdigest()
    chosen = []
    # proportional allocation, deterministic by hash inside each stratum
    n = args.n
    for key, items in sorted(strata.items()):
        k = max(1, round(n * len(items) / len(cand)))
        chosen += sorted(items, key=lambda c: h(c["song_id"]))[:k]
    chosen = sorted(chosen, key=lambda c: h(c["song_id"]))[:n]
    MANIFEST.mkdir(parents=True, exist_ok=True)
    jdump({"frozen": time.strftime("%Y-%m-%d %H:%M"), "n": len(chosen), "pool": len(cand),
           "rule": "4/4 test songs; strata = song-length tercile x lyric-length tercile x mode x (>=2 pseudo-label quality rules failed); "
                   "proportional allocation, deterministic sha256 order within a stratum; selected before any audio was rendered",
           "songs": chosen}, MANIFEST / "subset.json")
    print(f"subset {len(chosen)} of {len(cand)}")


def subset_ids():
    return [c["song_id"] for c in json.loads((MANIFEST / "subset.json").read_text())["songs"]]


# ------------------------------------------------------------------ SVS inputs
def soulx_midi(rows, tempo_bpm, path):
    import mido
    ppq = 480
    mid = mido.MidiFile(type=0, ticks_per_beat=ppq, charset="utf-8")
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    spb = 60.0 / tempo_bpm
    ev = []
    last_char, run = None, 0
    for r in rows:
        on = int(round(r["start"] / spb * ppq))
        off = max(on + 1, int(round(r["end"] / spb * ppq)))
        if r["symbol"] == "#":
            run += 1
            lyr = "-" if run % 2 == 1 else (last_char or "-")
        else:
            lyr, last_char, run = r["symbol"], r["symbol"], 0
        ev.append((on, 0, mido.MetaMessage("lyrics", text=lyr)))
        ev.append((on, 1, mido.Message("note_on", note=r["pitch"], velocity=90)))
        ev.append((off, -1, mido.Message("note_off", note=r["pitch"], velocity=0)))
    t = 0
    for tick, _o, msg in sorted(ev, key=lambda e: (e[0], e[1])):
        tr.append(msg.copy(time=tick - t))
        t = tick
    mid.save(path)


def item_dir(kind, source, sid):
    return AUDIO / kind / source / sid


def prep_svs(args):
    ids = set(subset_ids())
    n = 0
    for src in args.sources.split(","):
        for r in iter_melodies(src, "S1", "orig"):
            if r["song_id"] not in ids or not r["ok"]:
                continue
            song = Song.from_json(r["song"])
            d = item_dir("svs", src, r["song_id"])
            d.mkdir(parents=True, exist_ok=True)
            st = write_fastsinger_inputs(song, str(d / "fs.mid"), str(d / "fs.txt"))
            rows, _ = sung_notes(song)
            soulx_midi(rows, song.tempo_bpm, d / "soulx.mid")
            jdump({"song_id": r["song_id"], "source": src, "tempo": song.tempo_bpm, "notes": rows, "stats": st}, d / "target.json")
            n += 1
    print(f"prepared {n} SVS items")


def run(cmd, cwd, log, timeout=3600, env=None):
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})
    return p.returncode, round(time.time() - t0, 1)


def render_svs(args):
    items = sorted(p for p in (AUDIO / "svs").glob("*/*") if (p / "target.json").exists())
    mine = items[args.shard::args.num_shards]
    for d in mine:
        rep = {}
        if not (d / "fastsinger.wav").exists():
            code, s = run([FS_PY, "inference.py", "--model_id", "suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag",
                           "--model_epoch", "400", "--spkr_ref", "6", "--pitch_shifts", "0", "--shift_consonant_forward_alignment",
                           "--midi_path", str(d / "fs.mid"), "--lyric_path", str(d / "fs.txt"), "--output_path", str(d / "fastsinger.wav")],
                          MUSIC / "fastsinger", d / "fastsinger.log")
            rep["fastsinger"] = {"exit": code, "seconds": s}
        if not (d / "soulx" / "generated.wav").exists():
            c1, s1 = run([VPY, "midi2json.py", str(d / "soulx.mid"), str(d / "soulx.json"), "--language", "Mandarin"], SOULX, d / "soulx_json.log")
            c2, s2 = run([VPY, "-m", "cli.inference", "--device", "cuda", "--model_path", "../MIDI-SAG_checkpoints/SoulX-Singer/model.pt",
                          "--config", "soulxsinger/config/soulxsinger.yaml", "--prompt_wav_path", str(PROMPT / "zh_prompt.mp3"),
                          "--prompt_metadata_path", str(PROMPT / "zh_prompt.json"), "--target_metadata_path", str(d / "soulx.json"),
                          "--phoneset_path", "soulxsinger/utils/phoneme/phone_set.json", "--save_dir", str(d / "soulx"), "--control", "score"],
                         SOULX, d / "soulx.log")
            rep["soulx"] = {"exit": [c1, c2], "seconds": s1 + s2}
        if rep:
            prev = json.loads((d / "render.json").read_text()) if (d / "render.json").exists() else {}
            jdump({**prev, **rep}, d / "render.json")
        print(d.parent.name, d.name, rep, flush=True)
    print("RENDER_SVS_DONE")


# ------------------------------------------------------------------ backing (full lead sheet) renders
def excerpt(song: Song, seconds: float = EXCERPT_S) -> Song:
    """The first bars of the song that fit in ``seconds`` (sections cut accordingly)."""
    spb = 60.0 / song.tempo_bpm
    t, nb = 0.0, 0
    for beats in song.bar_beats:
        if t + beats * spb > seconds:
            break
        t += beats * spb
        nb += 1
    s = copy.deepcopy(song)
    end = sum(song.bar_beats[:nb]) * 4
    s.bar_beats = song.bar_beats[:nb]
    s.notes = [n for n in s.notes if n.onset < end]
    for n in s.notes:
        n.duration = min(n.duration, end - n.onset)
    s.chords = [c for c in s.chords if c.onset < end]
    secs = []
    for x in song.sections:
        if x.start_bar < nb:
            secs.append(Section(x.label, x.start_bar, min(x.num_bars, nb - x.start_bar)))
    s.sections = secs
    return s


def write_leadsheet(song: Song, d: Path, name: str):
    """Lead-sheet MIDI + FastSinger inputs through the repo's exporter code path (scripts/export_leadsheet_midi.py)."""
    from qwen_abc.leadsheet_midi import renderable_problems, song_to_leadsheet_midi
    d.mkdir(parents=True, exist_ok=True)
    song.song_id = name
    audit = song_to_leadsheet_midi(song, str(d / f"{name}.mid"))
    audit["problems"] = renderable_problems(song, audit)
    audit["fastsinger"] = write_fastsinger_inputs(song, str(d / f"{name}.fs.mid"), str(d / f"{name}.fs.txt"))
    jdump(song.to_json(), d / f"{name}.song.json")
    jdump(audit, d / f"{name}.audit.json")


def prep_backing(args):
    from .formats import with_chords
    from qwen_abc.canonical import Chord
    from qwen_abc.abc import parse_abc
    from qwen_abc.theory import canonical_key, parse_key_name, transpose_chord_symbol
    ids = set(subset_ids())
    rows = {r["song_id"]: r for r in load_rows("test") if r["song_id"] in ids}
    n = 0
    for sid, r in rows.items():
        ref = Song.from_json(r["song"])
        # chord-source comparison: reference melody + {qwen, am2, reference} chords
        for csrc, h in (("qwen", "chord"), ("am2", "am2"), ("ref", None)):
            if h is None:
                chords = ref.chords
            else:
                p = CS_ROOT / "chords" / h / "ref_S1" / f"{sid}_S1.json"
                if not p.exists() or not json.loads(p.read_text()).get("ok"):
                    continue
                chords = [Chord(o, d, s) for o, d, s in json.loads(p.read_text())["chords"]]
            write_leadsheet(excerpt(with_chords(ref, chords)), AUDIO / "backing" / "chords", f"{sid}__{csrc}")
            n += 1
        # closed loop: E3b draft, E3b regenerated under key+5 / tempo x1.25, and deterministic edits of the draft
        g = {c: PAPER_FINAL_GEN / "qwen_e3b" / c / f"{sid}_S1.json" for c in ("orig", "key_p5", "tempo_x1.25")}
        if not all(p.exists() for p in g.values()):
            continue
        songs = {c: Song.from_json(json.loads(p.read_text())["song"]) for c, p in g.items() if json.loads(p.read_text()).get("song")}
        if "orig" not in songs:
            continue
        draft = songs["orig"]
        tr = copy.deepcopy(draft)
        for x in tr.notes:
            x.pitch += 5
        tr.chords = [Chord(c.onset, c.duration, transpose_chord_symbol(c.symbol, 5, None)) for c in tr.chords]
        pc, mode = parse_key_name(draft.key)
        tr.key = canonical_key(f"{['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B'][(pc + 5) % 12]} {mode}")
        tempo = copy.deepcopy(draft)
        tempo.tempo_bpm = int(round(draft.tempo_bpm * 1.25))
        variants = {"draft": draft, "key_gen": songs.get("key_p5"), "key_transpose": tr, "tempo_gen": songs.get("tempo_x1.25"),
                    "tempo_direct": tempo}
        for name, s in variants.items():
            if s is not None:
                write_leadsheet(excerpt(s), AUDIO / "backing" / "control", f"{sid}__{name}")
                n += 1
    print(f"prepared {n} backing lead sheets")


TAG_PROMPTS = ('{"intro": "Mandarin pop introduction with the full band already playing: drums, bass, piano and electric guitar", '
               '"outro": "Mandarin pop outro with the full band playing out: drums, bass, piano and guitar", '
               '"inst": "Mandarin pop instrumental break with the full band: drums, bass, piano and lead guitar"}')


def render_backing(args):
    items = sorted(p for p in (AUDIO / "backing").glob("*/*.mid") if not p.name.endswith(".fs.mid"))
    mine = items[args.shard::args.num_shards]
    for mid in mine:
        name = mid.stem
        out = mid.parent / "render" / name
        out.mkdir(parents=True, exist_ok=True)
        rep = {}
        if not (out / "vocal_fs.wav").exists():
            rep["fastsinger"] = run([FS_PY, "inference.py", "--model_id", "suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag",
                                     "--model_epoch", "400", "--spkr_ref", "6", "--pitch_shifts", "0", "--shift_consonant_forward_alignment",
                                     "--midi_path", str(mid.with_suffix(".fs.mid")), "--lyric_path", str(mid.with_suffix(".fs.txt")),
                                     "--output_path", str(out / "vocal_fs.wav")], MUSIC / "fastsinger", out / "fastsinger.log")
        if (out / "vocal_fs.wav").exists() and not (out / "mix.wav").exists():
            rep["midi_sag"] = run([VPY, "tools/midi_llm_adapter/run_midi_llm_to_midi_sag.py", "--midi", str(mid), "--output-dir", str(out),
                                   "--vocal-audio", str(out / "vocal_fs.wav"), "--gpu", "0", "--harmonizer", "leadsheet",
                                   "--tag-prompts", TAG_PROMPTS], MUSIC / "MIDI-SAG", out / "midi_sag.log", timeout=5400,
                                  env={"HF_HOME": "/gscratch/ark/pingw220/midisag/hf_home"})
        if rep:
            jdump(rep, out / "render.json")
        print(name, rep, (out / "mix.wav").exists(), flush=True)
    print("RENDER_BACKING_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subset", "prep-svs", "render-svs", "prep-backing", "render-backing"])
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--sources", default="ref,e3b,mel,csl_offc,csl_rtc")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    {"subset": subset, "prep-svs": prep_svs, "render-svs": render_svs, "prep-backing": prep_backing,
     "render-backing": render_backing}[args.cmd](args)


if __name__ == "__main__":
    main()
