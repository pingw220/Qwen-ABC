"""Loudness-matched vocal+backing mixes for renderer A/B listening.

The SA3 / MIDI-SAG mix peak-normalizes each stem, so a backing with a high crest factor (MuseControlLite)
ends up 4-14 dB lower under the vocal than a dense one (SA3). Here every backing and the vocal are scaled to
the same RMS (backing-to-vocal ratio 0 dB for every renderer), and the summed mix to -22 dBFS RMS (no clipping-safety gain needed on any pair).
  python loudness_matched_mix.py <compare_dir> [<compare_dir> ...]     (writes <tag>_mix_lm.wav next to <tag>_backing.wav)
"""
import glob, os, sys
import librosa, numpy as np, soundfile as sf

def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)

for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        n = os.path.basename(d)
        base = os.path.dirname(os.path.dirname(os.path.realpath(os.path.join(d, "sa3_backing.wav"))))
        voc_path = os.path.join(os.path.dirname(base), "render", n, "vocal.wav")
        tags = [t[:-len("_backing.wav")] for t in sorted(os.listdir(d)) if t.endswith("_backing.wav")]
        v, sr = librosa.load(voc_path, sr=44100, mono=False)
        v = np.repeat(np.atleast_2d(v), 2, axis=0) if np.atleast_2d(v).shape[0] == 1 else np.atleast_2d(v)
        for t in tags:
            b, _ = sf.read(os.path.join(d, f"{t}_backing.wav"), always_2d=True)
            b = b.T
            k = min(b.shape[1], v.shape[1])
            bb, vv = b[:, :k], v[:, :k]
            mix = bb / rms(bb) + vv / rms(vv)            # equal RMS stems
            mix = mix * (10 ** (-22 / 20) / rms(mix))    # mix at -22 dBFS RMS
            peak = np.abs(mix).max()
            if peak > 0.98:                              # never clip; report any safety reduction
                mix *= 0.98 / peak
                print(f"  {n} {t}: peak safety gain {20*np.log10(0.98/peak):.1f} dB")
            sf.write(os.path.join(d, f"{t}_mix_lm.wav"), mix.T, 44100, subtype="PCM_16")
            print(n, t, f"backing/vocal 0 dB, mix RMS {20*np.log10(rms(mix)):.1f} dBFS, peak {20*np.log10(np.abs(mix).max()):.1f} dBFS")
