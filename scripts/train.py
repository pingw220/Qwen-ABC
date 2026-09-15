#!/usr/bin/env python
"""Train CPT or SFT from a YAML config: scripts/train.py CONFIG [key=value ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.train import load_config, train  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    train(load_config(sys.argv[1], sys.argv[2:]))
