#!/usr/bin/env python
"""Qwen melody->chord harmonization of any melody source (resumable, shardable).

  python -m paper_eval.component_sota.chord_gen --harmonizer chord --melodies ref:S1,ref:S2,e3b:S1 [--chord-seeds S1,S2]

For every (melody source, melody seed, song) the melody is put read-only into the prompt
(``formats.chord_prompt``) and chord lines are sampled with the paired-seed sampler at the canonical
T=1.0 / top-p 0.95. Output rows: experiments/component_sota/chords/<harmonizer>/<melody>_<mseed>/<song>_<cseed>.json
holding the chord list on the melody's bar grid (the melody itself is never regenerated).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from qwen_abc.canonical import Song

from ..common import MAX_NEW, MAX_TOTAL, SEEDS, TEMPERATURE, TOP_P, jdump, load_rows
from .formats import chord_prompt, parse_chord_lines
from .sources import CS_ROOT, iter_melodies

HARMONIZERS = {"chord": ("experiments/component_sota/runs/chord_sft/final_model", False),
               "chord_lyr": ("experiments/component_sota/runs/chord_lyr_sft/final_model", True)}


def out_path(h, msrc, mseed, sid, cseed) -> Path:
    return CS_ROOT / "chords" / h / f"{msrc}_{mseed}" / f"{sid}_{cseed}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmonizer", default="chord", choices=sorted(HARMONIZERS))
    ap.add_argument("--melodies", required=True, help="comma list of source:seed, e.g. ref:S1,e3b:S1")
    ap.add_argument("--chord-seeds", default="S1")
    ap.add_argument("--condition", default="orig")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    ckpt_rel, with_lyr = HARMONIZERS[args.harmonizer]
    specs = {r["song_id"]: r["spec"] for r in load_rows("test")}
    tasks = []
    for ms in args.melodies.split(","):
        src, mseed = ms.split(":")
        for r in iter_melodies(src, mseed if src != "ref" else "S1", args.condition):
            for cs in args.chord_seeds.split(","):
                p = out_path(args.harmonizer, src, mseed, r["song_id"], cs)
                if not r["ok"]:
                    if not p.exists():
                        jdump({"song_id": r["song_id"], "melody_source": src, "melody_seed": mseed, "chord_seed": cs,
                               "ok": False, "failure": f"no melody ({r.get('failure')})"}, p, indent=None)
                    continue
                song = Song.from_json(r["song"])
                spec = (r.get("meta") or {}).get("spec") or specs[r["song_id"]]
                if src.startswith("csl"):   # no plan input: harmonize on its derived sections / estimated key
                    spec = {**spec, "key": song.key, "sections": [{"label": s.label, "bars": s.num_bars,
                                                                     "beats": song.bar_beats[s.start_bar:s.start_bar + s.num_bars],
                                                                     "lines": []} for s in song.sections]}
                tasks.append((p, song, spec, src, mseed, cs, r["song_id"]))
    tasks.sort(key=lambda t: (len(t[1].notes), str(t[0])))
    mine = [t for i, t in enumerate(tasks) if (i // args.batch_size) % args.num_shards == args.shard]
    todo = [t for t in mine if not t[0].exists()]
    print(f"tasks {len(tasks)} shard {len(mine)} todo {len(todo)}", flush=True)
    if not todo:
        return
    from qwen_abc.generate import load_for_generation
    from ..sampler import generate_batch_crn
    from ..common import ROOT
    model, tok = load_for_generation(str(ROOT / ckpt_rel))
    for k in range(0, len(todo), args.batch_size):
        batch = todo[k: k + args.batch_size]
        prompts = [chord_prompt(t[1], t[2], with_lyr) for t in batch]
        outs = generate_batch_crn(model, tok, prompts, [SEEDS[t[5]] for t in batch], MAX_NEW, TEMPERATURE, TOP_P, MAX_TOTAL)
        for t, g in zip(batch, outs):
            p, song, spec, src, mseed, cs, sid = t
            chords, st = parse_chord_lines(g["text"], song)
            jdump({"song_id": sid, "melody_source": src, "melody_seed": mseed, "chord_seed": cs, "ok": bool(chords),
                   "harmonizer": args.harmonizer, "chords": [[c.onset, c.duration, c.symbol] for c in chords],
                   "raw": g["text"], "hit_eos": g["hit_eos"], "new_tokens": g["new_tokens"], "seconds": g["seconds"],
                   "parse": st, "failure": None if chords else "no_chords_parsed"}, p, indent=None)
        print(f"[{k // args.batch_size + 1}/{-(-len(todo) // args.batch_size)}] {outs[0]['batch_seconds']}s", flush=True)
    print("CHORD_GEN_DONE")


if __name__ == "__main__":
    main()
