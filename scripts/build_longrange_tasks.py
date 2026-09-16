#!/usr/bin/env python
"""E3 training/eval files: whole-song ABC-v2 SFT + masked late-section reconstruction.

  scripts/build_longrange_tasks.py --v2-dir data/generated/abc_v2_<ts> [--infill-fraction 0.5]

Writes into the v2 directory (never overwrites):
  sft_longrange_{train,validation}.jsonl  every whole-song example + infill examples for a
                                          deterministic ``--infill-fraction`` of songs
  infill_{split}.jsonl                    one infill example per eligible song (evaluation)
  longrange_report.json                   counts, token lengths, sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_abc_v2_dataset import ntok, sha256_file  # noqa: E402
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.longrange import GAP_LINE, choose_infill_target, infill_example, split_abc_sections  # noqa: E402


def selected(song_id: str, fraction: float) -> bool:
    return int.from_bytes(hashlib.sha256(f"infill-select:{song_id}".encode()).digest()[:8], "big") / 2 ** 64 < fraction


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)] if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--infill-fraction", type=float, default=0.5)
    args = ap.parse_args()
    d = args.v2_dir
    rep = {"infill_fraction": args.infill_fraction, "splits": {}}
    files = {}
    for split in ("train", "validation", "test"):
        out_mix = d / f"sft_longrange_{split}.jsonl"
        out_inf = d / f"infill_{split}.jsonl"
        for p in (out_mix, out_inf):
            if p.exists():
                raise SystemExit(f"refusing to overwrite {p}")
        c = Counter()
        lens, sup = [], []
        with open(d / f"songs_{split}.jsonl", encoding="utf-8") as f_song, open(d / f"sft_{split}.jsonl", encoding="utf-8") as f_sft, \
                open(out_inf, "w", encoding="utf-8") as f_inf, \
                (open(out_mix, "w", encoding="utf-8") if split != "test" else open("/dev/null", "w")) as f_mix:
            for song_line, sft_line in zip(f_song, f_sft):
                r, sft = json.loads(song_line), json.loads(sft_line)
                assert r["song_id"] == sft["song_id"]
                f_mix.write(json.dumps({**sft, "task": "whole_song"}, ensure_ascii=False) + "\n")
                c["whole_song"] += 1
                song = Song.from_json(r["song"])
                header, blocks = split_abc_sections(r["abc"])
                assert header + "".join(blocks) == r["abc"], "section split must be lossless"
                t = choose_infill_target(r["song_id"], r["spec"])
                if t is None:
                    c["no_infill_candidate"] += 1
                    continue
                ex = infill_example(song, r["spec"], t)
                # the gap plus the target reproduces the song exactly
                assert ex["prompt"].count(GAP_LINE) == 1
                pt, ct = ntok(ex["prompt"]), ntok(ex["completion"]) + 1
                row = {"id": f"infill:{r['song_id']}", "song_id": r["song_id"], "task": "infill", "prompt": ex["prompt"],
                       "completion": ex["completion"], "target_section": t,
                       "target_label": r["spec"]["sections"][t]["label"], "prompt_tokens": pt, "completion_tokens": ct - 1}
                f_inf.write(json.dumps(row, ensure_ascii=False) + "\n")
                c["infill_eligible"] += 1
                c[f"target_label_{row['target_label']}"] += 1
                if split != "test" and selected(r["song_id"], args.infill_fraction):
                    f_mix.write(json.dumps(row, ensure_ascii=False) + "\n")
                    c["infill_in_mix"] += 1
                    lens.append(pt + ct)
                    sup.append(ct)
        rep["splits"][split] = {**dict(c), "infill_tokens": {"median": pct(lens, .5), "p99": pct(lens, .99), "max": max(lens, default=0),
                                                             "sum": sum(lens), "supervised_sum": sum(sup)}}
        for p in (out_inf,) + ((out_mix,) if split != "test" else ()):
            files[p.name] = sha256_file(p)
        print(split, rep["splits"][split], flush=True)
    rep["files_sha256"] = files
    (d / "longrange_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("LONGRANGE_OK")


if __name__ == "__main__":
    main()
