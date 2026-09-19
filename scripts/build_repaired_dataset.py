#!/usr/bin/env python
"""Training files whose targets never stack syllables on one note (R3-C).

  scripts/build_repaired_dataset.py --v2-dir data/generated/abc_v2_<ts> --output-dir data/generated/abc_v2r_<ts>

R3-A showed cramming is not an information problem: the model has the bar count,
the section plan, a per-bar countdown and a syllable budget and still crams,
because its targets do. This splits a crammed note into one note per syllable
(``qwen_abc/repair.py``) and rebuilds the ABC from the repaired song.

The prompt is unchanged — the repair keeps every syllable, its order and its
lyric line, so the prompt built from a repaired song is byte-identical to the
original's, and this build asserts that. The completions change, which is the
point and the one variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.abc_v2 import counter_report, song_to_abc_v2, spec_to_prompt_v2  # noqa: E402
from qwen_abc.canonical import Song, comparable  # noqa: E402
from qwen_abc.longrange import GAP_LINE, choose_infill_target, infill_example  # noqa: E402
from qwen_abc.prompt import song_to_spec  # noqa: E402
from qwen_abc.repair import check_repair, split_crammed_notes  # noqa: E402

COPY = ("split_manifest.json",)


def selected(song_id: str, fraction: float) -> bool:
    """The membership test build_longrange_tasks.py uses, so the mixture matches."""
    return int.from_bytes(hashlib.sha256(f"infill-select:{song_id}".encode()).digest()[:8], "big") / 2 ** 64 < fraction


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--infill-fraction", type=float, default=0.5)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    args = ap.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    report = {"v2_dir": str(args.v2_dir), "infill_fraction": args.infill_fraction, "splits": {}}
    failures: list = []
    for split in ("train", "validation", "test"):
        songs = [json.loads(l) for l in open(args.v2_dir / f"songs_{split}.jsonl", encoding="utf-8")]
        old_sft = [json.loads(l) for l in open(args.v2_dir / f"sft_{split}.jsonl", encoding="utf-8")]
        c = Counter()
        mix_path = out / f"sft_longrange_{split}.jsonl"
        with open(out / f"songs_{split}.jsonl", "w", encoding="utf-8") as f_song, \
                open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f_sft, \
                open(out / f"infill_{split}.jsonl", "w", encoding="utf-8") as f_inf, \
                (open(mix_path, "w", encoding="utf-8") if split != "test" else open("/dev/null", "w")) as f_mix:
            for r, old in zip(songs, old_sft):
                song = Song.from_json(r["song"])
                fixed, st = split_crammed_notes(song)
                checks = check_repair(song, fixed, st)
                if not all(checks.values()):
                    failures.append({"song_id": r["song_id"], "checks": {k: v for k, v in checks.items() if not v}})
                    continue
                for k, v in st.items():
                    c[k] += v
                spec = song_to_spec(fixed)
                prompt = spec_to_prompt_v2(spec)
                if prompt != old["prompt"]:
                    failures.append({"song_id": r["song_id"], "checks": {"prompt_identical": False}})
                    continue
                abc = song_to_abc_v2(fixed)
                p = parse_abc(abc, r["song_id"])
                if not (p.ok and comparable(p.song) == comparable(fixed)):
                    failures.append({"song_id": r["song_id"], "checks": {"abc_roundtrip": False}})
                    continue
                cr = counter_report(abc, spec)
                if not all(cr[k] == 1.0 for k in cr if k.endswith("_frac")):
                    failures.append({"song_id": r["song_id"], "checks": {"counters_correct": False}})
                    continue
                c["songs"] += 1
                c["changed"] += bool(st["notes_added"])
                ct = ntok(abc)
                f_song.write(json.dumps({**r, "song": fixed.to_json(), "abc": abc, "spec": spec,
                                         "repair": st}, ensure_ascii=False) + "\n")
                row = {"id": f"sft:{r['song_id']}", "song_id": r["song_id"], "prompt": prompt,
                       "completion": abc, "prompt_tokens": old["prompt_tokens"], "completion_tokens": ct}
                f_sft.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_mix.write(json.dumps({**row, "task": "whole_song"}, ensure_ascii=False) + "\n")

                t = choose_infill_target(r["song_id"], spec)
                if t is None:
                    continue
                ex = infill_example(fixed, spec, t)
                assert ex["prompt"].count(GAP_LINE) == 1
                inf = {"id": f"infill:{r['song_id']}", "song_id": r["song_id"], "task": "infill",
                       "prompt": ex["prompt"], "completion": ex["completion"], "target_section": t,
                       "target_label": spec["sections"][t]["label"],
                       "prompt_tokens": ntok(ex["prompt"]), "completion_tokens": ntok(ex["completion"])}
                f_inf.write(json.dumps(inf, ensure_ascii=False) + "\n")
                c["infill_eligible"] += 1
                if split != "test" and selected(r["song_id"], args.infill_fraction):
                    f_mix.write(json.dumps(inf, ensure_ascii=False) + "\n")
                    c["infill_in_mix"] += 1
        report["splits"][split] = dict(c)
        print(f"{split}: {c['songs']} songs, {c['crammed_notes']} crammed notes -> +{c['notes_added']} notes, "
              f"{c['notes_too_short_to_split']} too short, infill {c['infill_in_mix']}/{c['infill_eligible']} in mix",
              flush=True)

    for name in COPY:
        if (args.v2_dir / name).exists():
            shutil.copy2(args.v2_dir / name, out / name)
    report["failures"] = failures[:50]
    report["n_failures"] = len(failures)
    report["files_sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))}
    (out / "repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if failures:
        raise SystemExit(f"FAILED on {len(failures)} songs, e.g. {failures[0]}")
    print(f"REPAIR_OK -> {out}")


if __name__ == "__main__":
    main()
