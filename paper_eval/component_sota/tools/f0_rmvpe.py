"""RMVPE F0 for a list of wavs (run with MIDI-SAG's .venv-midisag). Writes <wav>.f0.npz (hop 10 ms)."""
import sys, numpy as np, librosa, torch
sys.path.insert(0, "/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/MuseControlLite")
from rmvpe.inference import RMVPE
m = RMVPE("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/MIDI-SAG_checkpoints/rmvpe_model.pt", hop_length=160,
          device="cuda" if torch.cuda.is_available() else "cpu")
for p in open(sys.argv[1]).read().split():
    y, _ = librosa.load(p, sr=16000, mono=True)
    f0, _cents = m.infer_from_audio(y, sample_rate=16000, thred=0.03)   # returns (f0 Hz, cents)
    np.savez_compressed(p + ".f0.npz", f0=np.asarray(f0, dtype=np.float32), hop_s=0.01)
    print("f0", p, len(f0), flush=True)
