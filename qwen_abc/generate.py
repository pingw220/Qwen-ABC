"""Sampling ABC lead sheets from a finetuned checkpoint.

Generation is one prompt at a time: left-padded batches would push pad
positions into the Gated-DeltaNet recurrent state, and bit-identical results
across batch compositions matter more here than throughput.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


def load_for_generation(path: str, attn_impl: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.to("cuda").eval()
    return model, tok


@torch.no_grad()
def generate_one(model, tok, prompt: str, seed: int, max_new_tokens: int, temperature: float,
                 top_p: float, max_total: int = 8192) -> Dict:
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    budget = min(max_new_tokens, max_total - ids.shape[1])
    torch.manual_seed(seed)
    t0 = time.time()
    kwargs = dict(max_new_tokens=budget, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
    new = out[0, ids.shape[1]:].tolist()
    hit_eos = bool(new) and new[-1] == tok.eos_token_id
    if hit_eos:
        new = new[:-1]
    return {
        "text": tok.decode(new, skip_special_tokens=True),
        "new_tokens": len(new) + int(hit_eos),
        "hit_eos": hit_eos,
        "truncated": not hit_eos,
        "seconds": round(time.time() - t0, 2),
    }


@torch.no_grad()
def generate_batch(model, tok, prompts: List[str], seed: int, max_new_tokens: int, temperature: float,
                   top_p: float, max_total: int = 8192) -> List[Dict]:
    """Left-padded batched sampling.

    Padding is safe for the hybrid model: Transformers zeroes padded inputs
    before the Gated-DeltaNet mixer during prefill (``create_recurrent_attention_mask``),
    so the recurrent and conv states stay exactly zero until the first real
    token, and full attention masks pads. ``scripts/check_infra.py`` verifies
    greedy batched output against batch-1 output. Random draws differ from
    batch-1 sampling with the same seed, so samples are statistically, not
    bitwise, equivalent.
    """
    side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
    tok.padding_side = side
    ids, mask = enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")
    budget = min(max_new_tokens, max_total - ids.shape[1])
    torch.manual_seed(seed)
    t0 = time.time()
    eos = tok.eos_token_id
    kwargs = dict(max_new_tokens=budget, eos_token_id=eos, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(ids, attention_mask=mask, **kwargs)
    seconds = time.time() - t0
    rows = []
    for r in range(len(prompts)):
        new = out[r, ids.shape[1]:].tolist()
        hit_eos = eos in new
        if hit_eos:
            new = new[: new.index(eos)]
        rows.append({
            "text": tok.decode(new, skip_special_tokens=True),
            "new_tokens": len(new) + int(hit_eos),
            "hit_eos": hit_eos,
            "truncated": not hit_eos,
            "seconds": round(seconds / len(prompts), 2),
            "batch_size": len(prompts),
            "batch_seconds": round(seconds, 2),
        })
    return rows
