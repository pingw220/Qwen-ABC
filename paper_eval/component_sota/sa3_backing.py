#!/usr/bin/env python
"""Render the component round's backing lead sheets with the tuned SA3 MIDI-SAG model (the long-song
successor of MuseControlLite), from exactly the inputs the MuseControlLite renders used.

  source MIDI-SAG/SA3_MIDI_SAG/env.sh
  python paper_eval/component_sota/sa3_backing.py --group chords --shard k --num-shards n

For every existing MuseControlLite render dir ``backing/<group>/render/<name>/`` this reads the staged
conditions (``conditions/chord.txt`` in BTC labels, beat / downbeat times, ``structure.json``), the
fitted FastSinger vocal ``vocal.wav`` and the lead sheet's key / tempo, and writes
``backing/<group>/render_sa3/<name>/{backing.wav,mix.wav,conditions.json}``. The model is loaded once.
SA3 code is imported unchanged; nothing in MIDI-SAG is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SA3 = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/SA3_MIDI_SAG")
AUDIO = Path("/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio")
CKPT = Path(os.environ.get("SA3_WORK", "/mmfs1/gscratch/scrubbed/pingw220/music_acc/sa3_midi_sag")) / \
    "experiments/sa3_medium_midi_sag_long_v1/checkpoints/step_0030000"
GENRES = ["mandopop", "c-pop"]


def read_times(p: Path):
    """One time per line; '#' lines are comments (their text can contain numbers)."""
    return [float(l.split()[0]) for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def annotation(d: Path, song: dict) -> tuple[dict, float]:
    import soundfile as sf
    from sa3_midisag.alignment import parse_chord_lab
    c = d / "conditions"
    beats = read_times(next(p for p in c.glob("*_beat_times.txt") if not p.name.endswith("_downbeat_times.txt")))
    downbeats = read_times(next(c.glob("*_downbeat_times.txt")))
    st = json.loads((c / "structure.json").read_text())
    seconds = sf.info(str(d / "vocal.wav")).duration
    starts, tags = st["structure_starts"], st["structure_tags"]
    segs = [{"start": float(s), "end": float(starts[i + 1] if i + 1 < len(starts) else seconds), "label": str(t)}
            for i, (s, t) in enumerate(zip(starts, tags))]
    ann = {"beats": beats, "downbeats": downbeats, "segments": segs,
           "chords": parse_chord_lab((c / "chord.txt").read_text()),
           "key": song.get("key"), "bpm": float(song["tempo_bpm"]), "genres": GENRES, "caption": None}
    return ann, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="chords", choices=["chords", "control"])
    ap.add_argument("--only", default=None, help="comma list of render names (default: all)")
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, str(SA3))
    os.chdir(SA3)
    import soundfile as sf
    import yaml
    from sa3_midisag.constants import SAMPLE_RATE
    from sa3_midisag.data.songs import build_prompt
    from sa3_midisag.inference import build_condition_batch, extract_f0, generate_audio, load_for_inference, mix_with_vocal

    src = AUDIO / "backing" / a.group
    names = sorted(p.name for p in (src / "render").iterdir() if (p / "vocal.wav").exists() and (p / "conditions" / "chord.txt").exists())
    if a.only:
        keep = set(a.only.split(","))
        names = [n for n in names if n in keep]
    names = names[a.shard::a.num_shards]
    todo = [n for n in names if not (src / "render_sa3" / n / "mix.wav").exists()]
    print(f"{len(names)} items, {len(todo)} to render; checkpoint {a.checkpoint}", flush=True)
    if not todo:
        return
    # build every annotation first: a bad input must fail before the ~10-minute model load
    anns = {}
    for n in todo:
        song = json.loads((src / f"{n}.song.json").read_text())
        anns[n] = annotation(src / "render" / n, song)
        ann = anns[n][0]
        assert ann["beats"] and ann["downbeats"] and ann["chords"] and ann["segments"], n
    print(f"annotations ok for {len(anns)} items (e.g. {len(ann['beats'])} beats, {len(ann['chords'])} chord spans)", flush=True)
    cfg = yaml.safe_load(open("configs/sa3_medium_midi_sag_long_v1.yaml"))
    t0 = time.time()
    model, ae, step = load_for_inference(cfg, a.checkpoint)
    print(f"loaded step {step} in {time.time() - t0:.0f} s", flush=True)
    for n in todo:
        d = src / "render" / n
        ann, seconds = anns[n]
        prompt = build_prompt(ann, deterministic=True)
        f0 = extract_f0(str(d / "vocal.wav"))
        batch = build_condition_batch(ann, f0, seconds, prompt)
        t1 = time.time()
        audio = generate_audio(model, ae, batch, steps=a.steps, cfg_scale=a.cfg, seed=a.seed, use_controls=True)
        out = src / "render_sa3" / n
        out.mkdir(parents=True, exist_ok=True)
        sf.write(str(out / "backing.wav"), audio.T, SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(out / "mix.wav"), mix_with_vocal(audio, str(d / "vocal.wav")).T, SAMPLE_RATE, subtype="PCM_16")
        json.dump({"checkpoint": a.checkpoint, "step": step, "prompt": prompt, "seconds": seconds, "steps": a.steps, "cfg": a.cfg,
                   "seed": a.seed, "n_chords": len(ann["chords"]), "n_beats": len(ann["beats"]),
                   "segments": ann["segments"], "gen_seconds": round(time.time() - t1, 1)}, open(out / "conditions.json", "w"), indent=1)
        print(n, f"{seconds:.1f}s audio in {time.time() - t1:.0f}s |", prompt, flush=True)
    print("SA3_BACKING_DONE", flush=True)


if __name__ == "__main__":
    main()
