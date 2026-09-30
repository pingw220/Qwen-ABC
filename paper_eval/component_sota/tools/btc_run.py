"""Run BTC-ISMIR19's test.py unchanged under a NumPy that removed the deprecated aliases.

BTC uses ``np.float`` / ``np.int`` (removed in NumPy 1.24) and a pickled checkpoint that torch >= 2.6 refuses by
default; the sheetsagepp-btc env has both newer versions.
We restore the aliases (the builtin types they always meant) and execute test.py as __main__ from
its own directory, passing the command-line arguments through. Upstream code is not modified.
  python btc_run.py --voca True --audio_dir D --save_dir D      (cwd = BTC-ISMIR19)
"""
import os
import runpy
import sys

import numpy as np

for name, typ in (("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)):
    if not hasattr(np, name) or name in getattr(np, "__former_attrs__", {}):
        setattr(np, name, typ)
# torch >= 2.6 defaults torch.load(weights_only=True), which rejects BTC's own (trusted, local) checkpoint.
import torch  # noqa: E402

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})
BTC = "/mmfs1/gscratch/scrubbed/pingw220/music_acc/BTC-ISMIR19"
os.chdir(BTC)
sys.path.insert(0, BTC)
sys.argv = [os.path.join(BTC, "test.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
