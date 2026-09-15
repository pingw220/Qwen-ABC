#!/usr/bin/env python
"""Right padding must not change a sequence's loss (linear-attention layers keep recurrent state)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.train import collate, encode, load_model, read_examples, token_nll_sum  # noqa: E402

ckpt, data = sys.argv[1], Path(sys.argv[2])
model, tok = load_model(ckpt, "sdpa")
model.cuda().eval()
rows = read_examples(data / "sft_validation.jsonl")[:6]
ex = [encode(tok, r, "sft", 8192) for r in rows]
out = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    longest = max(range(len(ex)), key=lambda i: len(ex[i]["input_ids"]))
    for i in range(len(ex)):
        if i == longest:
            continue
        alone = {k: v.cuda() for k, v in collate(ex, [i], tok.pad_token_id).items()}
        padded = {k: v.cuda()[:1] for k, v in collate(ex, [i, longest], tok.pad_token_id).items()}
        a, n = token_nll_sum(model, alone)
        b, _ = token_nll_sum(model, padded)
        out.append({"tokens": len(ex[i]["input_ids"]), "pad": len(ex[longest]["input_ids"]) - len(ex[i]["input_ids"]),
                    "loss_alone": float(a) / n, "loss_padded": float(b) / n, "rel_diff": abs(float(a) - float(b)) / float(a)})
print(json.dumps(out, indent=1))
print("PADDING_CHECK_DONE max_rel_diff", max(o["rel_diff"] for o in out))
