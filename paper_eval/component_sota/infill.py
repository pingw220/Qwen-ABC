#!/usr/bin/env python
"""Melody/section infilling benchmark and infill-vs-regeneration edit study (E3b, unchanged).

  python -m paper_eval.component_sota.infill generate --shard 0 --num-shards 4
  python -m paper_eval.component_sota.infill evaluate

E3b was trained on masked *later-half* lyric sections (qwen_abc/longrange.py). The same prompt
format is used here for any section:

* ``recon_late`` / ``recon_early`` -- reconstruction on the held-out reference with one section
  masked: the later-half target E3b was trained on (choose_infill_target) and the intervention
  target (pick_target_section) when it lies in the first half (never trained);
* ``edit_resample`` / ``edit_lyrics`` / ``edit_extend`` -- local edits of E3b's own draft song (the
  paper-final ``orig`` S1 sample): the target section is regenerated unchanged, with replaced lyrics
  (paper_eval.tasks lyrics_sec donor), or with 4 more bars (bars_p4). The same edit requests were
  answered by whole-song regeneration in the paper-final round (lyrics_sec / bars_p4 S1-S2, orig S2).

The assembled song = the untouched text blocks of the draft/reference + the generated block, so
non-target text is byte-identical by construction; the evaluation checks what survives parsing
(ties across boundaries, bar counts) and compares with regeneration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_abc.abc import parse_abc
from qwen_abc.abc_v2 import PROMPT_HEADER_V2, spec_to_prompt_v2, song_to_abc_v2
from qwen_abc.canonical import Song
from qwen_abc.longrange import GAP_LINE, INFILL_HEADER, INFILL_MARKER, choose_infill_target, split_abc_sections
from qwen_abc.prompt import COMPLETION_MARKER

from ..common import MAX_NEW, MAX_TOTAL, MODELS, PAPER_FINAL_GEN, SEEDS, TEMPERATURE, TOP_P, jdump, load_rows
from ..interventions import pick_target_section
from ..tasks import intervention_spec
from .sources import CS_ROOT

KINDS = ("recon_late", "recon_early", "edit_resample", "edit_lyrics", "edit_extend")


def infill_prompt(abc: str, spec: dict, target: int) -> dict:
    """E3b's infill prompt for ``target`` of an ABC-v2 text, the gap header written from ``spec``."""
    header, blocks = split_abc_sections(abc)
    n = len(spec["sections"])
    sec = spec["sections"][target]
    head = f"P:{sec['label']}\n% section {target + 1}/{n} | {sec['bars']} bars\n"
    gapped = header + "".join(blocks[:target]) + head + GAP_LINE + "\n" + "".join(blocks[target + 1:])
    base = spec_to_prompt_v2(spec)
    body = base[len(PROMPT_HEADER_V2): -len(COMPLETION_MARKER)]
    return {"prompt": INFILL_HEADER + body + "ABC with a gap:\n" + gapped + "\n" + INFILL_MARKER,
            "header": header, "blocks": blocks}


def build_tasks():
    rows = load_rows("test")
    pool = {r["song_id"]: r["spec"] for r in rows}
    tasks = []
    for r in rows:
        sid, spec = r["song_id"], r["spec"]
        n = len(spec["sections"])
        late = choose_infill_target(sid, spec)
        tgt = pick_target_section(spec, sid)
        if late is not None:
            tasks.append(("recon_late", sid, late, r["abc"], spec, {}))
        if tgt is not None and tgt < -(-n // 2):
            tasks.append(("recon_early", sid, tgt, r["abc"], spec, {}))
        dp = PAPER_FINAL_GEN / "qwen_e3b" / "orig" / f"{sid}_S1.json"
        if tgt is None or not dp.exists():
            continue
        draft = json.loads(dp.read_text(encoding="utf-8"))
        if not draft.get("song"):
            continue
        dsong = Song.from_json(draft["song"])
        if [(s.label, s.num_bars) for s in dsong.sections] != [(s["label"], s["bars"]) for s in spec["sections"]]:
            continue   # the draft does not follow the plan: section indices would not line up
        dabc = song_to_abc_v2(dsong)
        tasks.append(("edit_resample", sid, tgt, dabc, spec, {}))
        s_ly, _ = intervention_spec(r, "lyrics_sec", pool)
        if s_ly is not None:
            tasks.append(("edit_lyrics", sid, tgt, dabc, s_ly, {}))
        s_ex, _ = intervention_spec(r, "bars_p4", pool)
        if s_ex is not None:
            tasks.append(("edit_extend", sid, tgt, dabc, s_ex, {}))
    return tasks


def out_path(kind, sid, seed):
    return CS_ROOT / "infill" / "qwen_e3b" / kind / f"{sid}_{seed}.json"


def generate(args):
    tasks = [(k, s, t, a, sp, m, seed) for (k, s, t, a, sp, m) in build_tasks() for seed in args.seeds.split(",")]
    tasks.sort(key=lambda x: (len(x[3]), x[0], x[1], x[6]))
    mine = [t for i, t in enumerate(tasks) if (i // args.batch_size) % args.num_shards == args.shard]
    todo = [t for t in mine if not out_path(t[0], t[1], t[6]).exists()]
    print(f"tasks {len(tasks)} shard {len(mine)} todo {len(todo)}", flush=True)
    if not todo:
        return
    from qwen_abc.generate import load_for_generation
    from ..sampler import generate_batch_crn
    model, tok = load_for_generation(MODELS["qwen_e3b"]["ckpt"])
    for k in range(0, len(todo), args.batch_size):
        batch = todo[k:k + args.batch_size]
        built = [infill_prompt(t[3], t[4], t[2]) for t in batch]
        outs = generate_batch_crn(model, tok, [b["prompt"] for b in built], [SEEDS[t[6]] for t in batch],
                                  MAX_NEW, TEMPERATURE, TOP_P, MAX_TOTAL)
        for t, b, g in zip(batch, built, outs):
            kind, sid, tgt, abc, spec, _, seed = t
            text = g["text"].rstrip("\n") + "\n"
            assembled = b["header"] + "".join(b["blocks"][:tgt]) + text + "".join(b["blocks"][tgt + 1:])
            jdump({"song_id": sid, "kind": kind, "seed": seed, "target": tgt, "spec": spec, "generation": g["text"],
                   "assembled": assembled, "context_abc": abc, "hit_eos": g["hit_eos"], "new_tokens": g["new_tokens"],
                   "seconds": g["seconds"]}, out_path(kind, sid, seed), indent=None)
        print(f"[{k // args.batch_size + 1}] {outs[0]['batch_seconds']}s", flush=True)
    print("INFILL_GEN_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "tasks"])
    ap.add_argument("--seeds", default="S1,S2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "tasks":
        from collections import Counter
        print(Counter(t[0] for t in build_tasks()))
        return
    generate(args)


if __name__ == "__main__":
    main()
