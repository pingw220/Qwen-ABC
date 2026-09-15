#!/usr/bin/env python
"""Where does training memory go? Peak memory for one 2048-token sequence under variants."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.train import load_model, token_nll_sum  # noqa: E402

out = {}
try:
    import causal_conv1d  # noqa: F401
    out["causal_conv1d"] = "ok"
except Exception as exc:  # noqa: BLE001
    out["causal_conv1d"] = repr(exc)[:300]

model, tok = load_model("Qwen/Qwen3.5-0.8B-Base", "sdpa")
model.cuda()
model.train()  # checkpointing only engages in training mode
out["layer_classes"] = sorted({type(l).__name__ + ":" + ",".join(c.__name__ for c in type(l).__mro__[1:4]) for l in model.model.layers})
ids = torch.randint(1000, 50000, (1, 2048), device="cuda")
batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def peak(label):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        nll, n = token_nll_sum(model, batch)
    fwd = torch.cuda.max_memory_allocated() - base
    (nll / n).backward()
    out[label] = {"fwd_peak_gb": round(fwd / 2**30, 2),
                  "fwd+bwd_peak_gb": round((torch.cuda.max_memory_allocated() - base) / 2**30, 2),
                  "is_gradient_checkpointing": getattr(model, "is_gradient_checkpointing", None)}
    model.zero_grad(set_to_none=True)


peak("no_checkpointing")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
out["layer0_gradient_checkpointing_attr"] = getattr(model.model.layers[0], "gradient_checkpointing", None)
peak("checkpointing_nonreentrant")
model.config.use_cache = False
peak("checkpointing_nonreentrant_use_cache_false")
for L in (4096, 6144, 8192):
    ids = torch.randint(1000, 50000, (1, L), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    peak(f"checkpointing_len{L}")
print(json.dumps(out, indent=1))
