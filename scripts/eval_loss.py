#!/usr/bin/env python
"""Teacher-forced SFT loss / perplexity on a split, broken down by ABC token category.

  scripts/eval_loss.py --checkpoint CKPT --data-dir DATA --split test --output OUT.json

Categories are assigned per completion character (w: lyric text, chord symbol,
pitch, duration, bar line, rest, tie/accidental, structure P:/[M:], header,
whitespace) and a token takes the category of its first non-space character.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.train import load_model, read_examples  # noqa: E402


def char_categories(abc: str):
    cats = []
    for line in abc.splitlines(keepends=True):
        if line.startswith("w:"):
            for i, ch in enumerate(line):
                if i < 2:
                    cats.append("lyric_marker")
                elif ch in " \n":
                    cats.append("space")
                elif ch in "|":
                    cats.append("lyric_bar")
                elif ch in "_*~":
                    cats.append("lyric_marker")
                else:
                    cats.append("lyric_text")
            continue
        if line[:2] in ("X:", "M:", "L:", "Q:", "K:"):
            cats.extend(["header"] * len(line))
            continue
        if line.startswith("% section") or line.startswith("P:"):
            cats.extend(["structure"] * len(line))
            continue
        in_chord = in_inline = False
        for ch in line:
            if in_chord:
                cats.append("chord")
                in_chord = ch != '"'
                continue
            if in_inline:
                cats.append("structure")
                in_inline = ch != "]"
                continue
            if ch == '"':
                cats.append("chord")
                in_chord = True
            elif ch == "[":
                cats.append("structure")
                in_inline = True
            elif ch in "ABCDEFGabcdefg,'^_=":
                cats.append("pitch")
            elif ch in "0123456789/":
                cats.append("duration")
            elif ch == "z":
                cats.append("rest")
            elif ch == "|":
                cats.append("bar")
            elif ch == "-":
                cats.append("tie")
            elif ch in " \n":
                cats.append("space")
            else:
                cats.append("other")
    return cats


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sft-file", type=Path, default=None, help="default: DATA/sft_{split}.jsonl")
    args = ap.parse_args()

    model, tok = load_model(args.checkpoint, "sdpa", dtype=torch.bfloat16)
    model.cuda().eval()
    rows = read_examples(args.sft_file or args.data_dir / f"sft_{args.split}.jsonl")
    nll_by = defaultdict(float)
    n_by = defaultdict(int)
    per_song = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        enc = tok(r["completion"], add_special_tokens=False, return_offsets_mapping=True)
        c = enc["input_ids"] + [tok.eos_token_id]
        offsets = enc["offset_mapping"] + [(len(r["completion"]), len(r["completion"]))]
        ids = torch.tensor([p + c], device="cuda")
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(p) - 1: -1]
        logits = model.get_output_embeddings()(hidden).float()
        nll = torch.nn.functional.cross_entropy(logits, torch.tensor(c, device="cuda"), reduction="none").tolist()
        cats = char_categories(r["completion"])
        song_nll = 0.0
        for (a, b), x in zip(offsets, nll):
            if a >= len(cats):
                cat = "eos"
            else:
                span = [cats[i] for i in range(a, min(b, len(cats))) if cats[i] != "space"] or [cats[a]]
                cat = span[0]
            nll_by[cat] += x
            n_by[cat] += 1
            song_nll += x
        per_song.append({"song_id": r["song_id"], "loss": song_nll / len(c), "tokens": len(c)})
    total_nll, total_n = sum(nll_by.values()), sum(n_by.values())
    out = {
        "checkpoint": args.checkpoint, "split": args.split, "songs": len(rows),
        "loss": total_nll / total_n, "ppl": math.exp(total_nll / total_n), "tokens": total_n,
        "by_category": {k: {"loss": nll_by[k] / n_by[k], "tokens": n_by[k], "share_of_nll": nll_by[k] / total_nll}
                        for k in sorted(n_by, key=lambda k: -nll_by[k])},
        "per_song": per_song,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_song"}, indent=1))
    print("LOSS_EVAL_DONE")


if __name__ == "__main__":
    main()
