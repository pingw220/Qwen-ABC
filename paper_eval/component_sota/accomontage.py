#!/usr/bin/env python
"""AccoMontage2 harmonization of any melody source, through its official entry point (unchanged).

  python -m paper_eval.component_sota.accomontage --melodies ref:S1,e3b:S1 --shard 0 --num-shards 16

For each melody: write a one-track melody MIDI (single tempo, 4/4, key signature), run
``MIDI-SAG/AccoMontage2/demo_SOME.py`` exactly as MIDI-SAG's ComposeFlow / ground-truth scripts do
(``--beat_subdivision 1 --chord_style POP_STANDARD --chords_per_bar 1 --key <key>``), and map its
per-beat BTC chord file (seconds) back onto the melody's tick grid. Only its melody->chord output is
used. AccoMontage2 assumes a regular 4/4 grid from the first beat: 3/4 songs are recorded as
unsupported; songs with irregular bars are harmonized on its regular grid (chord *times* map back
exactly, its phrase segmentation may straddle our irregular bars).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from qwen_abc.canonical import TICKS_PER_BEAT, Chord, Song, normalize_chords
from qwen_abc.theory import corpus_chord_to_abc, parse_key_name, respell_chord_symbol

from ..common import jdump
from .sources import CS_ROOT, iter_melodies

AM2 = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/AccoMontage2")
VPY = "/gscratch/ark/pingw220/midisag/.venv-midisag/bin/python"
NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def am2_key(key: str) -> str:
    pc, mode = parse_key_name(key) or (0, "major")
    flat = {1: "Db", 3: "Eb", 6: "F#", 8: "Ab", 10: "Bb"}
    name = flat.get(pc, NAMES_SHARP[pc])
    return name + ("m" if mode == "minor" else "")


def melody_midi(song: Song, path: Path):
    import mido
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(song.tempo_bpm), time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    ks = am2_key(song.key or "C major")
    try:
        tr.append(mido.MetaMessage("key_signature", key=ks, time=0))
    except Exception:  # noqa: BLE001
        pass
    evs = []
    k = tpb // TICKS_PER_BEAT
    for n in song.notes:
        evs.append((n.onset * k, 1, n.pitch))
        evs.append(((n.onset + n.duration) * k, 0, n.pitch))
    evs.sort(key=lambda e: (e[0], e[1]))
    t = 0
    for tick, on, p in evs:
        tr.append(mido.Message("note_on" if on else "note_off", note=p, velocity=100 if on else 0, time=tick - t))
        t = tick
    mid.save(path)


def btc_to_chords(txt: Path, song: Song):
    """``start end label`` (seconds, BTC labels) -> chords on the song's tick grid."""
    sec_per_tick = 60.0 / song.tempo_bpm / TICKS_PER_BEAT
    out = []
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        st, lab = float(parts[0]), parts[2]
        if lab in ("N", "X"):
            sym = "N.C."
        else:
            root, _, qual = lab.partition(":")
            qual, _, bass = (qual or "maj").partition("/")
            sym, _ = corpus_chord_to_abc(lab, root, qual or "maj", bass or None)
            sym = respell_chord_symbol(sym, song.key)
        out.append(Chord(int(round(st / sec_per_tick)), 0, sym))
    return normalize_chords(out, song.total_ticks)


def run_one(song: Song, key: str, workdir: Path, style: str = "POP_STANDARD"):
    workdir.mkdir(parents=True, exist_ok=True)
    mp = workdir / "melody.mid"
    melody_midi(song, mp)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [VPY, "demo_SOME.py", "--midi_path", str(mp), "--output_dir", str(workdir / "am2"), "--beat_subdivision", "1",
           "--chord_style", style, "--chords_per_bar", "1", "--key", am2_key(key)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=AM2, capture_output=True, text=True, env=env, timeout=1800)
    (workdir / "am2.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
    txt = workdir / "am2" / "btc_txt" / "melody_chord_gen.txt"
    return (btc_to_chords(txt, song) if txt.exists() else None), time.time() - t0, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--melodies", required=True)
    ap.add_argument("--condition", default="orig")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    tasks = []
    for ms in args.melodies.split(","):
        src, mseed = ms.split(":")
        for r in iter_melodies(src, mseed if src != "ref" else "S1", args.condition):
            tasks.append((src, mseed, r))
    mine = tasks[args.shard::args.num_shards]
    print(f"tasks {len(tasks)} shard {len(mine)}", flush=True)
    for src, mseed, r in mine:
        p = CS_ROOT / "chords" / "am2" / f"{src}_{mseed}" / f"{r['song_id']}_S1.json"
        if p.exists():
            continue
        row = {"song_id": r["song_id"], "melody_source": src, "melody_seed": mseed, "chord_seed": "S1",
               "harmonizer": "am2", "ok": False, "failure": None}
        if not r["ok"]:
            row["failure"] = f"no melody ({r.get('failure')})"
        else:
            song = Song.from_json(r["song"])
            if song.meter_num != 4:
                row["failure"] = "unsupported_meter"
            else:
                with tempfile.TemporaryDirectory(dir=str(CS_ROOT / "tmp") if (CS_ROOT / "tmp").exists() else None) as td:
                    try:
                        chords, secs, code = run_one(song, song.key or "C major", Path(td))
                        row.update(seconds=round(secs, 1), returncode=code)
                        if chords:
                            row.update(ok=True, chords=[[c.onset, c.duration, c.symbol] for c in chords])
                        else:
                            row["failure"] = f"no_chord_output (exit {code})"
                            row["log_tail"] = (Path(td) / "am2.log").read_text()[-1500:]
                    except subprocess.TimeoutExpired:
                        row["failure"] = "timeout"
        jdump(row, p, indent=None)
        print(r["song_id"], src, row["ok"], row.get("failure"), row.get("seconds"), flush=True)
    print("AM2_DONE")


if __name__ == "__main__":
    main()
