#!/usr/bin/env python
"""Make MuPT usable as a base model for lyrics-conditioned ABC (R4).

  scripts/build_mupt_base.py --data-dir data/generated/abc_v2_<ts> --output-dir models/mupt_zh_<ts>

MuPT (`m-a-p/MuPT-v1-8192-1.07B`, Apache 2.0) is a LLaMA-2-architecture model
pretrained on 10B tokens of ABC notation — the closest thing to a domain prior
for this task. Its 50k music BPE, however, **cannot represent Chinese**: `我`
tokenizes to `['æ', 'Ī', '<unk>']`, because the vocabulary is missing most of the
byte alphabet. A lyrics-conditioned model that cannot read or write the lyrics is
not a comparison, so the vocabulary is extended before anything is trained:

1. every byte token the BPE is missing, so UTF-8 is representable at all;
2. every Chinese character in the training split, so a syllable is one token
   rather than three bytes.

Both are appended, never reordered, so **every existing token keeps its id and
every ASCII string — all of the ABC — tokenizes exactly as MuPT pretrained it.**
The script asserts that. New embedding rows start at the mean of the pretrained
ones with small noise, the usual initialisation for added vocabulary.

This is surgery, and it has to be read into the result: the new rows are random
where Qwen's Chinese embeddings were pretrained, so a MuPT *loss* on lyric
metrics is ambiguous between "the ABC prior does not transfer" and "the lyric
embeddings are fresh". A MuPT *win* is not ambiguous.

The tokenizer is saved as a plain `GPT2Tokenizer`: MuPT's custom class is a
byte-level BPE with no behaviour of its own (verified here token-for-token), so
nothing downstream needs `trust_remote_code`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MUPT = "m-a-p/MuPT-v1-8192-1.07B"
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
ASCII_PROBES = [
    "X:1\nM:4/4\nL:1/8\nQ:1/4=77\nK:E\n",
    "P:verse\n% section 3/10 | 8 bars\n[r:8] \"E\"G, G, G, G,/ G, G,/ F,/ F,3/ B,, |\n",
    "|: A2 B2 c2 d2 :| [K:Gmin] z4 |]\n",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default=MUPT)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty {args.output_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    before = len(tok)
    unk = tok.convert_tokens_to_ids(tok.unk_token)

    # 1. the byte alphabet the BPE is missing. MuPT's tokenizer carries the
    # byte<->unicode table itself, so it is read from the instance rather than
    # from transformers, where it has moved between versions.
    byte_tokens = list(tok.byte_encoder.values())
    vocab = tok.get_vocab()
    missing_bytes = [b for b in byte_tokens if b not in vocab]

    # 2. every Chinese character the training split uses
    chars = Counter()
    for line in open(args.data_dir / f"sft_{args.split}.jsonl", encoding="utf-8"):
        chars.update(CJK.findall(json.loads(line)["prompt"]))
    # Every character gets merge rules, including the ones already present as
    # vocabulary entries: MuPT's ABC corpus left some Chinese in the vocab, but
    # without the merges that build them the BPE can never reach those tokens
    # and falls back to bytes. A vocabulary entry without a path to it is not a
    # token.
    new_chars = sorted(chars)

    # The vocabulary is extended inside the BPE itself rather than through
    # `add_tokens`. MuPT decodes by byte-decoding every token string, and an
    # added token is not byte-encoded, so it raises a KeyError on the way back
    # out. Adding byte tokens and the merge rules that build a character out of
    # them keeps one code path for everything.
    enc, merges = tok.encoder, tok.bpe_ranks
    next_id = max(enc.values()) + 1
    next_rank = max(merges.values()) + 1
    for bt in missing_bytes:
        enc[bt] = next_id
        next_id += 1
    for ch in new_chars:
        pieces = [tok.byte_encoder[b] for b in ch.encode("utf-8")]
        cur = pieces[0]
        for nxt in pieces[1:]:
            if (cur, nxt) not in merges:
                merges[(cur, nxt)] = next_rank
                next_rank += 1
            cur += nxt
            if cur not in enc:
                enc[cur] = next_id
                next_id += 1
    tok.decoder = {v: k for k, v in enc.items()}
    tok.cache = {}
    print(f"vocabulary {before} -> {len(enc)}  (+{len(enc) - before}: {len(missing_bytes)} byte tokens, "
          f"{len(new_chars)} Chinese characters from {len(chars)} distinct, plus their merge prefixes)")

    # the ABC prior must be untouched: same ids for every ASCII probe
    probe_before = [AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)(p, add_special_tokens=False)["input_ids"]
                    for p in ASCII_PROBES]
    probe_after = [tok(p, add_special_tokens=False)["input_ids"] for p in ASCII_PROBES]
    assert probe_before == probe_after, "extending the vocabulary changed how ABC tokenizes"
    print("ABC probes tokenize identically to pretrained MuPT: OK")

    # Chinese must now round-trip exactly and cost one token per character
    multi = 0
    for probe in ["无法阻止心流感扩散", "喜怒哀乐会互相感染", "".join(sorted(chars)[:200])]:
        ids = tok(probe, add_special_tokens=False)["input_ids"]
        assert unk not in ids, f"{probe!r} still hits <unk>"
        assert tok.decode(ids) == probe, f"{probe!r} does not round-trip: {tok.decode(ids)!r}"
        multi += len(ids) - len(probe)
    weighted = sum(len(tok(c, add_special_tokens=False)["input_ids"]) * n for c, n in chars.items())
    per_char = weighted / max(sum(chars.values()), 1)
    print(f"Chinese round-trips exactly: OK ({multi} probe characters over one token; "
          f"{per_char:.3f} tokens per character over the corpus)")
    assert per_char < 1.01, f"the merges did not take: {per_char:.3f} tokens per Chinese character"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(enc))
    with torch.no_grad():
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            if emb is None:
                continue
            w = emb.weight
            mean = w[:old_rows].mean(0, keepdim=True)
            std = w[:old_rows].std().item() * 0.1
            w[old_rows:] = mean + torch.randn_like(w[old_rows:]) * std
    print(f"embeddings {old_rows} -> {model.get_input_embeddings().weight.shape[0]} rows, "
          f"new rows initialised at the pretrained mean")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    # save as a plain GPT-2 BPE so nothing downstream needs trust_remote_code
    cfg = json.loads((args.output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg.pop("auto_map", None)
    cfg["tokenizer_class"] = "GPT2Tokenizer"
    (args.output_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    for stray in ("tokenization_mupt.py",):
        (args.output_dir / stray).unlink(missing_ok=True)

    reloaded = GPT2Tokenizer.from_pretrained(args.output_dir)
    for p in ASCII_PROBES + ["无法阻止心流感扩散"]:
        assert reloaded(p, add_special_tokens=False)["input_ids"] == tok(p, add_special_tokens=False)["input_ids"], p
    print("reloads as a standard GPT2Tokenizer with identical ids: OK")

    (args.output_dir / "vocab_extension.json").write_text(json.dumps({
        "base_model": args.model, "vocab_before": before, "vocab_after": len(tok),
        "byte_tokens_added": len(missing_bytes), "chinese_characters_added": len(new_chars),
        "distinct_chinese_in_split": len(chars), "split": args.split,
        "data_dir": str(args.data_dir),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"MUPT_BASE_OK -> {args.output_dir}")


if __name__ == "__main__":
    main()
