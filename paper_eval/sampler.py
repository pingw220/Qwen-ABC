"""Paired-seed ("common random numbers") sampling for intervention experiments.

Ordinary ``torch.manual_seed`` + ``model.generate(do_sample=True)`` gives a row's
random draws that depend on the batch it sits in, so "same seed, one control
changed" is not the same randomness once prompts are batched differently.

Here each row owns a seed, and at generation step t the row's noise is a
Gumbel vector drawn from a generator seeded by (row seed, t) alone. Adding it to
the temperature-scaled, top-p-truncated logits and taking the argmax is an exact
sample from the same distribution ``do_sample=True, temperature=T, top_p=p,
top_k=0`` samples from (the Gumbel-max trick). Two prompts sampled with the same
row seed therefore see identical noise at every step, whatever their batch,
padding or length, and an intervention effect is not confounded with a change of
random stream. The only remaining source of divergence between two runs of the
*same* prompt and seed is bf16 numerics under different padding; the ``replay``
condition measures it.
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch
from transformers import LogitsProcessor, LogitsProcessorList

_MIX = 1_000_003


class GumbelCRN(LogitsProcessor):
    def __init__(self, seeds: List[int], prompt_len: int, temperature: float, top_p: float):
        self.seeds = [int(s) for s in seeds]
        self.prompt_len = prompt_len
        self.temperature = temperature
        self.top_p = top_p
        self._gens: Dict[torch.device, List[torch.Generator]] = {}

    def _generators(self, device):
        if device not in self._gens:
            self._gens[device] = [torch.Generator(device=device) for _ in self.seeds]
        return self._gens[device]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        logits = scores.float() / self.temperature
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = sorted_logits.softmax(dim=-1)
            before = probs.cumsum(dim=-1) - probs          # mass strictly above each token
            remove_sorted = before >= self.top_p            # keep the smallest set reaching top_p
            remove_sorted[..., 0] = False
            remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
            logits = logits.masked_fill(remove, float("-inf"))
        gens = self._generators(logits.device)
        noise = torch.empty_like(logits)
        for r, (g, s) in enumerate(zip(gens, self.seeds)):
            g.manual_seed(s * _MIX + step)
            u = torch.rand(logits.shape[-1], generator=g, device=logits.device, dtype=torch.float32)
            noise[r] = -torch.log(-torch.log(u.clamp_(1e-10, 1.0 - 1e-7)))
        return logits + noise


@torch.no_grad()
def generate_batch_crn(model, tok, prompts: List[str], seeds: List[int], max_new_tokens: int,
                       temperature: float, top_p: float, max_total: int) -> List[dict]:
    """Left-padded batch; row r is sampled with its own seed ``seeds[r]`` (see module doc)."""
    assert len(prompts) == len(seeds)
    side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
    tok.padding_side = side
    ids, mask = enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)
    budget = min(max_new_tokens, max_total - ids.shape[1])
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    t0 = time.time()
    if temperature > 0:
        procs = LogitsProcessorList([GumbelCRN(seeds, ids.shape[1], temperature, top_p)])
    else:
        procs = LogitsProcessorList([])
    out = model.generate(ids, attention_mask=mask, max_new_tokens=budget, eos_token_id=eos, pad_token_id=pad,
                         do_sample=False, logits_processor=procs)
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
            "prompt_tokens": int(mask[r].sum().item()),
            "token_budget": int(budget),
        })
    return rows
