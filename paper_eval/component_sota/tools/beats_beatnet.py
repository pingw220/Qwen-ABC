"""BeatNet offline (DBN) beats, as MIDI-SAG's Rhythm F1 uses (conda env beatbk-beatnet). Writes <wav>.beats.json."""
import json, sys
from BeatNet.BeatNet import BeatNet
est = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
for p in open(sys.argv[1]).read().split():
    out = est.process(p)
    json.dump({"beats": [[float(t), int(b)] for t, b in out]}, open(p + ".beats.json", "w"))
    print("beats", p, len(out), flush=True)
