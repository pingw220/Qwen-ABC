"""Audiobox-aesthetics (CE, CU, PC, PQ) for a list of wavs (conda env music-rl). Writes <wav>.aes.json."""
import json, sys
from audiobox_aesthetics.infer import initialize_predictor
pred = initialize_predictor()
paths = open(sys.argv[1]).read().split()
for i in range(0, len(paths), 8):
    batch = paths[i:i + 8]
    out = pred.forward([{"path": p} for p in batch])
    for p, o in zip(batch, out):
        json.dump(o, open(p + ".aes.json", "w"))
    print("aes", i, flush=True)
