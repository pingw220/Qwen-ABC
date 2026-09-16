#!/usr/bin/env python
"""Is batched left padding numerically equivalent to batch-1 decoding?

Scores the same sequences (prompt + a fixed continuation) three ways:
alone, left-padded in a batch, right-padded in a batch (the training layout,
already known to be exact up to kernel noise). If left-padding differences are
of the same size as right-padding differences, divergence of greedy decoding is
bf16 kernel noise at near-tie tokens, not a padding leak.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.generate import load_for_generation  # noqa: E402

ckpt, data, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
model, tok = load_for_generation(ckpt)
rows = [json.loads(l) for l in open(data / "sft_test.jsonl", encoding="utf-8")]
rows.sort(key=lambda r: r["song_id"])
rows = rows[::37][:6]
seqs = [tok(r["prompt"] + r["completion"][:1500], add_special_tokens=False)["input_ids"] for r in rows]
L = max(len(s) for s in seqs)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


@torch.no_grad()
def logp(ids, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return torch.log_softmax(model.lm_head(h).float(), -1)


alone = [logp(torch.tensor([s], device="cuda"), torch.ones(1, len(s), device="cuda", dtype=torch.long))[0] for s in seqs]
left_ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device="cuda")
left_mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device="cuda")
right_ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
right_mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
# left padding needs explicit positions (generate() derives them from the mask the same way)
pos = (left_mask.cumsum(-1) - 1).clamp(min=0)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    hl = model.model(input_ids=left_ids, attention_mask=left_mask, position_ids=pos, use_cache=False).last_hidden_state
    lp_left = torch.log_softmax(model.lm_head(hl).float(), -1)
lp_right = logp(right_ids, right_mask)
res = []
for i, s in enumerate(seqs):
    a = alone[i]
    le = lp_left[i, L - len(s):]
    ri = lp_right[i, : len(s)]
    top2 = a.topk(2, -1).values
    res.append({
        "tokens": len(s), "pad": L - len(s),
        "left_max_abs_dlogp_of_argmax": float((le.gather(-1, a.argmax(-1, keepdim=True)) - a.max(-1, keepdim=True).values).abs().max()),
        "right_max_abs_dlogp_of_argmax": float((ri.gather(-1, a.argmax(-1, keepdim=True)) - a.max(-1, keepdim=True).values).abs().max()),
        "left_argmax_agree": float((le.argmax(-1) == a.argmax(-1)).float().mean()),
        "right_argmax_agree": float((ri.argmax(-1) == a.argmax(-1)).float().mean()),
        "left_mean_abs_dlogp": float((le - a).abs().mean()),
        "right_mean_abs_dlogp": float((ri - a).abs().mean()),
        "near_tie_positions_frac(top2 gap<0.05)": float(((top2[:, 0] - top2[:, 1]) < 0.05).float().mean()),
    })
out_path.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
print("LEFT_PADDING_CHECK_DONE")
