"""Audio key: Krumhansl-Schmuckler on the mean CQT chroma (librosa; .venv-midisag). Writes <wav>.key.json.
(MIDI-SAG's key-CNN environment does not exist locally; this detector is applied identically to every condition.)"""
import json, sys, numpy as np, librosa
MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
N = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
for p in open(sys.argv[1]).read().split():
    y, sr = librosa.load(p, sr=22050, mono=True)
    c = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    best = max(((np.corrcoef(c, np.roll(prof, t))[0, 1], f"{N[t]} {mode}") for t in range(12) for prof, mode in ((MAJ, "major"), (MIN, "minor"))))
    json.dump({"key": best[1], "r": float(best[0])}, open(p + ".key.json", "w"))
    print("key", p, best[1], flush=True)
