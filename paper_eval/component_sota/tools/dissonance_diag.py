"""Where does vocal/backing dissonance come from? Per song and renderer:
  score:   melody vs its own chords (strong-beat chord-tone share, semitone clashes), melody key vs declared key
  vocal:   sung pitch vs score (RMVPE F0 of the FastSinger render), global tuning offset
  backing: global tuning (librosa), alignment to the requested chords (best chroma lag), chord match at lag 0
  mix:     vocal pitch class vs backing chroma frame by frame (supported / semitone clash)
  python dissonance_diag.py <group>      group = long | chords   (run in the SA3 env: librosa)
"""
import json, os, sys
import numpy as np, librosa, soundfile as sf

QA = "/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component"
sys.path.insert(0, QA)
from qwen_abc.canonical import Song                                   # noqa: E402
from paper_eval.component_sota.chord_metrics import compatibility     # noqa: E402
from paper_eval.component_sota.csl_l2m import estimate_key            # noqa: E402

AUD = "/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio"
SR, HOP = 22050, 2205          # 10 Hz frames
PCS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "dim": (0, 3, 6),
       "aug": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9), "hdim7": (0, 3, 6, 10),
       "dim7": (0, 3, 6, 9), "minmaj7": (0, 3, 7, 11), "9": (0, 4, 7, 10, 2), "maj9": (0, 4, 7, 11, 2), "min9": (0, 3, 7, 10, 2)}
NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}


def chord_chroma(lab_path, n):
    C = np.zeros((12, n))
    for line in open(lab_path):
        a, b, lab = line.split()[:3]
        if ":" not in lab:
            continue
        r, q = lab.split(":", 1)
        q = q.split("/")[0]
        if r not in NOTE:
            continue
        pcs = PCS.get(q, PCS["min"] if q.startswith("min") else PCS["maj"])
        i0, i1 = int(float(a) * SR / HOP), min(int(float(b) * SR / HOP), n)
        for p in pcs:
            C[(NOTE[r] + p) % 12, i0:i1] = 1
    return C


def norm(X):
    return X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)


def backing_stats(wav, req, f0_frames):
    y, _ = librosa.load(wav, sr=SR, mono=True)
    tuning = float(librosa.estimate_tuning(y=y, sr=SR)) * 100   # cents
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, tuning=0.0)
    n = min(ch.shape[1], req.shape[1])
    ch, R = ch[:, :n], req[:, :n]
    on = R.sum(0) > 0
    Cn, Rn = norm(ch), norm(R)
    lags = range(-20, 21)   # +-2 s
    corr = []
    for L in lags:
        if L >= 0:
            a, b, m = Cn[:, L:], Rn[:, :n - L], on[:n - L]
        else:
            a, b, m = Cn[:, :n + L], Rn[:, -L:], on[-L:]
        corr.append(float((a * b).sum(0)[m].mean()))
    best = int(np.argmax(corr))
    out = {"tuning_cents": round(tuning, 1), "chord_corr_lag0": round(corr[20], 3), "best_lag_s": lags[best] / 10,
           "chord_corr_best": round(corr[best], 3)}
    # vocal vs backing
    if f0_frames is not None:
        k = min(n, len(f0_frames))
        sup = clash = cnt = 0
        for t in range(k):
            f = f0_frames[t]
            if f <= 0:
                continue
            pc = int(round(12 * np.log2(f / 440.0) + 69)) % 12
            c = ch[:, t] / (ch[:, t].max() + 1e-9)
            if ch[:, t].max() < 1e-3:
                continue
            cnt += 1
            sup += c[pc] >= 0.5
            clash += (c[pc] < 0.3) and (max(c[(pc + 1) % 12], c[(pc - 1) % 12]) >= 0.8)
        out["vocal_pc_supported"] = round(sup / max(cnt, 1), 3)
        out["vocal_semitone_clash"] = round(clash / max(cnt, 1), 3)
    return out


def vocal_stats(f0_npz, target):
    z = np.load(f0_npz)
    f0, hop = z["f0"], float(z["hop_s"])
    f0 = f0[0] if f0.ndim == 2 else f0
    t = np.arange(len(f0)) * hop
    errs = []
    for nt in target["notes"]:
        a, b = nt["start"], nt["end"]
        seg = f0[(t >= a + 0.2 * (b - a)) & (t <= b - 0.2 * (b - a))]
        seg = seg[seg > 0]
        if len(seg):
            errs.append(1200 * np.log2(np.median(seg) / (440 * 2 ** ((nt["pitch"] - 69) / 12))))
    e = np.array(errs)
    f10 = np.interp(np.arange(0, t[-1], HOP / SR), t, f0)  # resample to 10 Hz
    f10[np.interp(np.arange(0, t[-1], HOP / SR), t, (f0 > 0).astype(float)) < 0.5] = 0
    return {"vocal_pitch_acc50": round(float(np.mean(np.abs(e) < 50)), 3), "vocal_median_cents": round(float(np.median(e)), 1),
            "vocal_offkey_100c": round(float(np.mean(np.abs(e) > 100)), 3)}, f10


def main():
    group = sys.argv[1]
    base = f"{AUD}/backing/{group}"
    rows = []
    for d in sorted(os.listdir(f"{base}/compare_sameprompt")):
        n = d
        sid, src = n.split("__")
        song = Song.from_json(json.load(open(f"{base}/{n}.song.json")))
        comp = compatibility(song)
        mel = Song.from_json(json.load(open(f"{base}/{n}.song.json")))
        mel.chords = []
        row = {"song": sid[:8], "group": group, "key": song.key, "melody_key_est": estimate_key(mel),
               "score_strong_chord_tone": round(comp["strong_chord_tone"], 3), "score_strong_dissonance": round(comp["strong_dissonance"], 3),
               "score_chroma_compat": round(comp["chroma_compat"], 3)}
        # vocal: long group = whole-song SVS render (has f0 + target); excerpt group = render/<n>/vocal.wav (no f0 sidecar)
        f10 = None
        if group == "long":
            sv = f"{AUD}/svs/{src if src != 'qwen' else 'ref'}/{sid}"
            vs, f10 = vocal_stats(f"{sv}/fastsinger.wav.f0.npz", json.load(open(f"{sv}/target.json")))
            row.update(vs)
        lab = f"{base}/render/{n}/conditions/chord.txt"
        y_len = int(sf.info(f"{base}/render/{n}/vocal.wav").duration * SR / HOP) + 1
        req = chord_chroma(lab, y_len)
        for tag in ("mcl", "sa3"):
            files = [f for f in os.listdir(f"{base}/compare_sameprompt/{n}") if f.startswith(tag) and f.endswith("_backing.wav")]
            if not files:
                continue
            st = backing_stats(f"{base}/compare_sameprompt/{n}/{files[0]}", req, f10)
            row.update({f"{tag}_{k}": v for k, v in st.items()})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(rows, open(f"{QA}/experiments/component_sota/tmp/dissonance_{group}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
