#!/usr/bin/env python
"""Dump N SFT examples (prompt + ABC, optionally truncated) for manual inspection.

Selection is deterministic: every k-th song of the split, sorted by song_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-abc-lines", type=int, default=40)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data_dir / f"sft_{args.split}.jsonl", encoding="utf-8")]
    songs = {json.loads(l)["song_id"]: json.loads(l) for l in open(args.data_dir / f"songs_{args.split}.jsonl", encoding="utf-8")}
    step = max(len(rows) // args.n, 1)
    picked = rows[::step][: args.n]
    out = []
    for r in picked:
        s = songs[r["song_id"]]
        abc_lines = r["completion"].splitlines()
        shown = "\n".join(abc_lines[: args.max_abc_lines])
        more = len(abc_lines) - args.max_abc_lines
        out.append(
            f"## {r['song_id']} ({args.split})\n\n"
            f"tokens: prompt {r['prompt_tokens']}, completion {r['completion_tokens']}; "
            f"conversion stats: {json.dumps(s['stats'], ensure_ascii=False)}\n\n"
            f"### prompt\n```text\n{r['prompt']}```\n\n### ABC\n```abc\n{shown}\n"
            + (f"% ... {more} more lines\n" if more > 0 else "") + "```\n"
        )
    text = "\n".join(out)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {len(picked)} examples to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
