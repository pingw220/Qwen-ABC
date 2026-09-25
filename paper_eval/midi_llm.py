#!/usr/bin/env python
"""MIDI-LLM (one-stage v4, mode A) for the paper-final round: extra seeds and a control subset.

  python -m paper_eval.midi_llm tasks      # write specs + the task list (idempotent)
  sbatch --array=0-7 scripts/paper_final/midi_llm_shard.sbatch   # runs MIDI-LLM's own CLI, read-only
  python -m paper_eval.midi_llm convert    # parsed_leadsheet.json -> canonical rows + metrics

MIDI-LLM runs from its own repository and environment, exactly as the round-2 baseline did
(scripts/slurm/midi_llm_r2_shard.sh, commit bea5597, recommended mode-A flags). Its spec
carries section count, bars per section, labels, lyric lines, key, tempo and meter; bars,
tempo and the section grid become a click track in the *input*, so structure and tempo
adherence are true by construction and only their effect on the content is measured.

Tasks:
* ``orig`` seeds 1001-1003 on all 225 songs -> with the round-2 seed-1000 run, 4 samples/song;
* a 60-song control subset (songs MIDI-LLM accepts, deterministic spread over ids):
  ``orig`` seed 1000 regenerated (the paired base), ``bars_p4 label_bridge key_p5 tempo_x1.25
  lyrics_all`` at seed 1000. The reseed floor is ``orig`` seed 1001.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import OUT_ROOT, ROOT, jdump, jload, load_rows
from .tasks import intervention_spec

ML_OUT = OUT_ROOT / "midi_llm"
R2 = ROOT / "experiments/midi_llm_r2_20260915_121309"
TRAINED = ("intro", "verse", "chorus", "bridge", "instrumental", "outro", "other")
SUBSET_CONDS = ("bars_p4", "label_bridge", "key_p5", "tempo_x1.25", "lyrics_all")
BON_SEEDS = (1001, 1002, 1003)


def to_midi_llm_spec(song_id: str, spec: dict) -> dict:
    sections = []
    for s in spec["sections"]:
        row = {"label": s["label"] if s["label"] in TRAINED else "verse", "bars": s["bars"]}
        if s["lines"]:
            row["lines"] = s["lines"]
        sections.append(row)
    return {"song_id": song_id, "task": "lyrics_to_leadsheet", "prompt": "", "tempo_bpm": spec["tempo_bpm"],
            "meter": spec["meter"], "key": spec["key"], "sections": sections}


def refused_ids() -> set:
    cov = jload(R2 / "coverage.json")["modeA"]["per_song"]
    return {sid for sid, v in cov.items() if (v.get("class") if isinstance(v, dict) else v) == "refused_input_g2p"}


def subset_ids(rows, n: int = 60):
    bad = refused_ids()
    ok = [r["song_id"] for r in rows if r["song_id"] not in bad]
    step = len(ok) / n
    return [ok[int(i * step)] for i in range(n)]


def cmd_tasks(args) -> None:
    rows = load_rows("test")
    pool = {r["song_id"]: r["spec"] for r in rows}
    by_id = {r["song_id"]: r for r in rows}
    tasks = []

    def add(cond, seed, sid, spec, meta):
        p = ML_OUT / "specs" / cond / f"{sid}.json"
        if not p.exists():
            jdump(to_midi_llm_spec(sid, spec), p)
        jdump(spec, ML_OUT / "specs_full" / cond / f"{sid}.json")
        tasks.append({"condition": cond, "seed": seed, "song_id": sid, "spec_path": str(p),
                      "out_dir": str(ML_OUT / "gen" / f"{cond}_{seed}"), "meta": meta})

    refused = refused_ids()   # rejected at G2P in round 2 (deterministic); not re-run, counted as failures
    for r in rows:
        if r["song_id"] in refused:
            continue
        for seed in BON_SEEDS:
            add("orig", seed, r["song_id"], r["spec"], {"family": "none"})
    sub = subset_ids(rows, args.subset)
    for sid in sub:
        add("orig", 1000, sid, by_id[sid]["spec"], {"family": "none"})
        for cond in SUBSET_CONDS:
            spec, meta = intervention_spec(by_id[sid], cond, pool)
            if spec is None:
                continue
            add(cond, 1000, sid, spec, meta)
    # longest-first within each seed so shards finish together
    with open(ML_OUT / "tasks.tsv", "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(f"{t['condition']}\t{t['seed']}\t{t['song_id']}\t{t['spec_path']}\t{t['out_dir']}\n")
    jdump({"subset": sub, "tasks": tasks}, ML_OUT / "tasks.json")
    print(f"{len(tasks)} tasks, subset {len(sub)} songs -> {ML_OUT / 'tasks.tsv'}")


def cmd_convert(args) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from midi_llm_baseline import to_song
    from qwen_abc.canonical import Song
    from qwen_abc.metrics import song_metrics

    rows = {r["song_id"]: r for r in load_rows("test")}
    t = jload(ML_OUT / "tasks.json")["tasks"]
    # the round-2 seed-1000 mode-A run is the fourth sample of 'orig'
    t = t + [{"condition": "orig_r2", "seed": 1000, "song_id": sid, "out_dir": str(R2 / "modeA/gen"),
              "spec_path": str(R2 / "specs" / f"{sid}.json"), "meta": {"family": "none"}} for sid in rows]
    refused = refused_ids()
    t = t + [{"condition": "orig", "seed": seed, "song_id": sid, "out_dir": str(ML_OUT / "gen" / f"orig_{seed}"),
              "spec_path": "", "meta": {"family": "none"}, "refused": True} for sid in sorted(refused) for seed in BON_SEEDS]
    ok = 0
    for task in t:
        sid = task["song_id"]
        cond = task["condition"]
        full_spec_p = ML_OUT / "specs_full" / ("orig" if cond == "orig_r2" else cond) / f"{sid}.json"
        spec = jload(full_spec_p) if full_spec_p.exists() else rows[sid]["spec"]
        gen_dir = Path(task["out_dir"]) / sid
        row = {"song_id": sid, "model": "midi_llm", "condition": cond, "seed": task["seed"], "seed_name": f"m{task['seed']}",
               "prompt": json.dumps(spec, ensure_ascii=False), "generation": "", "errors": {}, "hit_eos": False,
               "meta": task["meta"], "spec": spec}
        if (gen_dir / "parsed_leadsheet.json").exists():
            v = jload(gen_dir / "validation.json") if (gen_dir / "validation.json").exists() else {}
            song = to_song(gen_dir, spec)
            row.update(parse_ok=True, strict_ok=not v.get("errors"), midi_ok=bool(v.get("midi_valid")) and bool(song.notes),
                       errors={e if isinstance(e, str) else json.dumps(e): 1 for e in v.get("errors", [])},
                       hit_eos=True, song=song.to_json(), metrics=song_metrics(song, spec))
            row["metrics"]["strict_valid"] = float(row["strict_ok"])
            ok += 1
        else:
            row.update(parse_ok=False, strict_ok=False, midi_ok=False)
            log = ML_OUT / "logs" / f"{cond}_{task['seed']}" / f"{sid}.txt"
            if cond == "orig_r2":
                log = R2 / "modeA/logs" / f"{sid}.txt"
            txt = log.read_text(errors="replace") if log.exists() else ""
            if task.get("refused"):
                txt = "phoneme inventory (round-2 refusal, not re-run)"
            row["failure"] = ("refused_input_g2p" if ("phoneme inventory" in txt or "mandarin_g2p" in txt) else
                              "missing_output" if not txt else "generation_failed")
        out = ML_OUT / "rows" / cond / f"{sid}_m{task['seed']}.json"
        jdump(row, out, indent=None)
    print(f"converted {ok}/{len(t)} MIDI-LLM outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["tasks", "convert"])
    ap.add_argument("--subset", type=int, default=60)
    args = ap.parse_args()
    {"tasks": cmd_tasks, "convert": cmd_convert}[args.command](args)


if __name__ == "__main__":
    main()
