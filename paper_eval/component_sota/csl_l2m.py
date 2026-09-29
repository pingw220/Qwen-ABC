#!/usr/bin/env python
"""CSL-L2M adapter: official pretrained inference, retraining data, and conversion to our format.

Runs in CSL-L2M's own environment (``envs/midi-sag``) with its code imported *unchanged* from
``MIDI-SAG/lyrics2melody_new`` (nothing there is modified; outputs go to our experiments dir).

  # official lyrics-only checkpoint / our retrain, on the 225 test songs (orig + lyric-swap, 4 seeds)
  python -m paper_eval.component_sota.csl_l2m infer --which official --shard 0 --num-shards 8
  python -m paper_eval.component_sota.csl_l2m infer --which retrain --ckpt <step_N.pt>
  # retraining data (REMI-aligned events) from our canonical split
  python -m paper_eval.component_sota.csl_l2m build-data

Facts the adapter handles (reports/component_sota/CSL_L2M_COMPARISON.md):
* The official ``pretrained_CSLL2M_onlyLyrics.pt`` has no POS/tone weights while the shipped
  ``CSLL2M_withOnlyLyrics.yaml`` enables them and loads non-strictly; we use f_pos=f_tone=False,
  which is what the checkpoint was trained with (missing/unexpected keys are logged).
* No seeding exists upstream; we seed torch/numpy/random per (song, seed) before sampling.
* 4/4 only (64 positions per bar); 3/4 songs are recorded as unsupported failures.
* The lyric vocabulary is Han characters only: Latin words are removed from the input (their
  syllables count as not sung), out-of-vocabulary characters use the upstream homophone fallback.
* Output events are converted directly to our canonical Song (no MIDI writer, so no fixed
  tempo-90 and no 8 padding bars); tempo is set to the requested tempo (CSL-L2M has no tempo
  control), sections are derived post hoc from which input line each note sings.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

CSL_DIR = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/lyrics2melody_new")
CKPT_DIR = Path("/gscratch/ark/pingw220/third_party/csl_l2m_ckpt")
QA = Path(__file__).resolve().parents[2]
OUT = QA / "experiments/component_sota"
CJK = re.compile(r"[㐀-鿿豈-﫿]")
TPB_CSL = 480                  # CSL-L2M ticks per beat
POS_PER_BAR = 64               # 4/4, 1/16-beat grid
TICK_CSL = TPB_CSL * 4 // POS_PER_BAR   # 30


# ------------------------------------------------------------------ shared helpers (no torch needed)
def spec_lines(spec: dict):
    """CSL input: every lyric line of the spec with non-Han characters removed; plus the section of each line."""
    lines, sec_of = [], []
    dropped = 0
    for si, sec in enumerate(spec["sections"]):
        for line in sec["lines"]:
            chars = [c for c in line if CJK.fullmatch(c)]
            dropped += sum(1 for t in re.findall(r"[^\s㐀-鿿豈-﫿]+", line))
            if chars:
                lines.append(chars)
                sec_of.append(si)
    return lines, sec_of, dropped


def chunk_lines(lines, sec_of, max_lines=24):
    """Group consecutive sections into chunks of <= max_lines lyric lines (a longer section is split at
    line boundaries). CSL-L2M's corpus has 16 lines per song at the median; whole-song generation fails
    on most songs above ~35 lines, so the chunked mode gives it inputs of the length it was trained on."""
    chunks, cur = [], []
    for i in range(len(lines)):
        new_sec = i > 0 and sec_of[i] != sec_of[i - 1]
        if cur and (len(cur) >= max_lines or (new_sec and len(cur) + sum(1 for j in range(i, len(lines)) if sec_of[j] == sec_of[i]) > max_lines)):
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


def events_to_song(events, lines, sec_of, spec, song_id):
    """CSL-L2M event strings -> canonical Song JSON (melody + lyrics + post-hoc sections)."""
    sys.path.insert(0, str(QA))
    from qwen_abc.canonical import Note, Section, Song
    bar, notes, groups = -1, [], []
    cur = None
    pending = {}
    line_i, char_i = 0, 0
    for ev in events:
        name, _, val = ev.rpartition("_")
        if ev.startswith("Bar"):
            bar += 1
        elif ev.startswith("Beat"):
            pending = {"pos": int(val)}
        elif ev.startswith("Note_Pitch"):
            pending["pitch"] = int(val)
        elif ev.startswith("Note_Duration"):
            b = max(bar, 0)
            onset = b * 16 + round(pending.get("pos", 0) / 4)           # our grid: 4 ticks/beat, 16/bar
            dur = max(round(int(val) / (TPB_CSL / 4)), 1)
            if cur is None:
                cur = []
            cur.append(Note(onset=onset, duration=dur, pitch=pending.get("pitch", 60)))
        elif ev.startswith("ALIGN"):
            if cur:
                groups.append((line_i, cur))
                cur = None
        elif ev.startswith("SEQ"):
            line_i += 1
    # characters in order over groups
    flat = [(li, c) for li, l in enumerate(lines) for c in l]
    out, last_on = [], -1
    for gi, (li, g) in enumerate(groups):
        ch = flat[gi][1] if gi < len(flat) else None
        for k, n in enumerate(sorted(g, key=lambda n: n.onset)):
            if n.onset <= last_on:
                continue
            if out and out[-1].onset + out[-1].duration > n.onset:
                out[-1].duration = n.onset - out[-1].onset
            n.lyric = (ch,) if (k == 0 and ch) else None
            n.melisma = bool(k > 0 and ch)
            n.line = flat[gi][0] if gi < len(flat) else None
            out.append(n)
            last_on = n.onset
    if not out:
        return None
    n_bars = max(n.onset + n.duration - 1 for n in out) // 16 + 1
    bar_beats = [4] * n_bars
    # sections: a spec section starts at the bar of its first sung note; the first starts at bar 0
    starts = {}
    for n in out:
        if n.line is not None and n.lyric:
            si = sec_of[n.line]
            starts.setdefault(si, n.onset // 16)
    order = sorted(starts.items(), key=lambda t: (t[1], t[0]))
    secs = []
    for k, (si, b) in enumerate(order):
        b = 0 if k == 0 else b
        if secs and b <= secs[-1][1]:
            continue
        secs.append((si, b))
    sections = []
    for k, (si, b) in enumerate(secs):
        e = secs[k + 1][1] if k + 1 < len(secs) else n_bars
        sections.append(Section(spec["sections"][si]["label"], b, e - b))
    total = n_bars * 16
    for n in out:
        n.duration = min(n.duration, total - n.onset)
    s = Song(song_id, 4, int(spec["tempo_bpm"]), None, bar_beats, sections, out, [])
    s.key = estimate_key(s)
    return s


KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(song) -> str:
    """Krumhansl-Schmuckler on duration-weighted pitch classes (for systems without key control)."""
    import numpy as np
    h = np.zeros(12)
    for n in song.notes:
        h[n.pitch % 12] += n.duration
    best = (-2, "C major")
    for t in range(12):
        for prof, mode in ((KS_MAJ, "major"), (KS_MIN, "minor")):
            r = np.corrcoef(h, np.roll(prof, t))[0, 1]
            if r > best[0]:
                best = (r, f"{NAMES[t]} {mode}")
    from qwen_abc.theory import canonical_key
    return canonical_key(best[1])


# ------------------------------------------------------------------ inference
def _load(which, ckpt):
    cfg_src = QA / "configs/component_sota" / ("csl_l2m_onlylyrics_official.yaml" if which == "official" else "csl_l2m_onlylyrics_retrain.yaml")
    ckpt = ckpt or str(CKPT_DIR / "pretrained_CSLL2M_onlyLyrics.pt")
    os.chdir(CSL_DIR)
    sys.path.insert(0, str(CSL_DIR))
    sys.argv = ["generate.py", str(cfg_src), ckpt, str(OUT / "csl/_scratch"), "1", "/dev/null"]
    import generate as G   # upstream module, unchanged: reads config/vocab from sys.argv at import
    src = (CSL_DIR / "generate.py").read_text(encoding="utf-8")
    block = src[src.index("  model = CSLL2M("): src.index(".to(device)", src.index("  model = CSLL2M(")) + len(".to(device)")]
    ns = {"CSLL2M": G.CSLL2M, "mconf": G.config["model"], "event2idx": G.event2idx, "lyric2idx": G.lyric2idx,
          "device": G.device}
    exec(block.strip(), ns)          # the exact upstream constructor call
    model = ns["model"]
    model.eval()
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    info = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "ckpt": ckpt, "config": str(cfg_src)}
    return G, model, info


def infer(args):
    sys.path.insert(0, str(QA))
    from paper_eval.common import load_rows, SEEDS
    from paper_eval.tasks import intervention_spec
    rows = load_rows("test")
    pool = {r["song_id"]: r["spec"] for r in rows}
    source = ("csl_off" if args.which == "official" else "csl_rt") + ("c" if args.chunked else "")
    tasks = []
    for r in rows:
        for cond in args.conditions.split(","):
            if cond == "orig":
                spec, meta = r["spec"], {"family": "none"}
            else:
                spec, meta = intervention_spec(r, cond, pool)
                if spec is None:
                    continue
            for s in args.seeds.split(","):
                tasks.append((r["song_id"], cond, s, spec, meta))
    mine = tasks[args.shard::args.num_shards]
    todo = [t for t in mine if not (OUT / "csl" / source / t[1] / f"{t[0]}_{t[2]}.json").exists()]
    print(f"{source}: tasks {len(tasks)} shard {len(mine)} todo {len(todo)}", flush=True)
    if not todo:
        return
    G, model, info = _load(args.which, args.ckpt)
    print("load:", json.dumps({k: (v if not isinstance(v, list) else v[:8] + ([f"... {len(v)}"] if len(v) > 8 else [])) for k, v in info.items()}), flush=True)
    (OUT / "csl" / source).mkdir(parents=True, exist_ok=True)
    (OUT / "csl" / source / "load_info.json").write_text(json.dumps(info, indent=1))
    import numpy as np
    import torch
    from paper_eval.common import SEEDS as SEEDMAP
    for sid, cond, s, spec, meta in todo:
        out_p = OUT / "csl" / source / cond / f"{sid}_{s}.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        row = {"song_id": sid, "source": source, "seed": s, "condition": cond, "ok": False, "song": None, "failure": None,
               "meta": {**meta, "spec": spec}}
        t0 = time.time()
        if spec["meter"] != "4/4":
            row["failure"] = "unsupported_meter"
        else:
            lines, sec_of, dropped = spec_lines(spec)
            row["meta"].update(latin_words_dropped=dropped, n_lines=len(lines))
            try:
                seed = SEEDMAP[s] * 1000   # same seed for a song's orig and lyric-swap samples
                groups = chunk_lines(lines, sec_of) if args.chunked else [list(range(len(lines)))]
                evs, ok, attempts = [], True, []
                for ci, g in enumerate(groups):
                    sub = [list(lines[i]) for i in g]
                    enc = G.convert_lyrics([list(l) for l in sub], G.lyric2idx)
                    e_inp, e_mask, _ = G.get_encoder_input_data(enc)
                    e_inp = G.numpy_to_tensor(e_inp, device=G.device)
                    e_mask = G.numpy_to_tensor(e_mask, device=G.device)
                    random.seed(seed + ci); np.random.seed(seed + ci); torch.manual_seed(seed + ci)
                    lat = G.get_semantic_embedding(model, e_inp, e_mask)
                    song_ev, c_ok = None, False
                    for attempt in range(5):
                        song_ev, c_ok = G.generate_on_latent_ctrl_vanilla_truncate(
                            model, lat, sub, None, None, None, None, None, None, None, None, None, None, None, None,
                            torch.tensor([0], device=G.device), torch.tensor([7], device=G.device), torch.tensor([0], device=G.device),
                            None, None, G.event2idx, G.idx2event,
                            nucleus_p=G.config["generate"]["nucleus_p"], temperature=G.config["generate"]["temperature"])
                        if c_ok:
                            break
                    attempts.append(attempt + 1)
                    if not c_ok:
                        ok = False
                        break
                    ce = G.word2event(song_ev, G.idx2event)
                    if evs:   # chunk boundary: close the previous line and start a fresh bar
                        if evs[-1] != "SEQ_None":
                            evs.append("SEQ_None")
                        evs.append("Bar_None")
                    evs += ce
                row["meta"]["attempts"] = attempts
                row["meta"]["chunks"] = len(groups)
                if not ok:
                    row["failure"] = "generation_failed_5_attempts"
                else:
                    song = events_to_song(evs, lines, sec_of, spec, sid)
                    if song is None:
                        row["failure"] = "empty_output"
                    else:
                        row.update(ok=True, song=song.to_json())
            except Exception as e:  # noqa: BLE001
                row["failure"] = f"exception: {type(e).__name__}: {str(e)[:200]}"
        row["meta"]["seconds"] = round(time.time() - t0, 2)
        tmp = out_p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False))
        tmp.replace(out_p)
        print(f"{sid} {cond} {s} ok={row['ok']} fail={row['failure']} {row['meta']['seconds']}s", flush=True)
    print("CSL_INFER_DONE")


# ------------------------------------------------------------------ retraining data
def song_to_events(song, max_dur=6240, durations=None):
    """Our canonical Song -> CSL-L2M (seq_lyrics, pos_seq, events), mirroring mid2events.py.

    Only Han-lyric notes and their melisma continuations are kept (wordless and Latin notes are
    dropped: the representation needs one lyric per note). A note carrying k syllables is split
    into k equal parts on the 1/16-beat grid. Positions are re-barred into 4/4 bars from the
    song's first beat.
    """
    items = []   # (abs_tick_csl, dur_csl, pitch, char or '*', line)
    notes = song.notes
    for n in notes:
        on = n.onset * (TPB_CSL // 4)
        du = n.duration * (TPB_CSL // 4)
        pitch = n.pitch
        while pitch < 35:
            pitch += 12
        while pitch > 94:
            pitch -= 12
        if n.lyric:
            sy = [c for c in n.lyric if CJK.fullmatch(c)]
            if not sy:
                items.append(None)  # Latin syllable: drop it and its melismas
                continue
            k = len(sy)
            part = max((du // k) // TICK_CSL * TICK_CSL, TICK_CSL)
            for j, c in enumerate(sy):
                items.append((on + j * part, part if j < k - 1 else max(du - part * (k - 1), TICK_CSL), pitch, c, n.line))
        elif n.melisma and items and items[-1] is not None:
            items.append((on, du, pitch, "*", items[-1][4]))
        elif n.melisma and items and items[-1] is None:
            items.append(None)
    items = [x for x in items if x is not None]
    if not items:
        return None
    events = [{"name": "SEQ", "value": None}]
    seq_lyrics, cur_line = [], []
    current_bar = None
    for i, (on, du, pitch, c, line) in enumerate(items):
        pos = on // TICK_CSL
        bar, beat = pos // POS_PER_BAR, pos % POS_PER_BAR
        if current_bar != bar:
            events.append({"name": "Bar", "value": None})
        events.append({"name": "Beat", "value": int(beat)})
        events.append({"name": "Note_Pitch", "value": int(pitch)})
        d = min(max(round(du / TICK_CSL) * TICK_CSL, TICK_CSL), max_dur)
        if durations:  # the melody dictionary holds 109 specific durations: snap to the nearest
            d = min(durations, key=lambda x: (abs(x - d), x))
        events.append({"name": "Note_Duration", "value": int(d)})
        current_bar = bar
        nxt = items[i + 1] if i + 1 < len(items) else None
        if nxt is None or nxt[3] != "*":
            events.append({"name": "ALIGN", "value": None})
        if c != "*":
            cur_line.append(c)
        line_end = nxt is None or (nxt[3] != "*" and nxt[4] != line)
        if line_end:
            seq_lyrics.append(cur_line)
            cur_line = []
            if nxt is not None:
                events.append({"name": "SEQ", "value": None})
    events += [{"name": "Bar", "value": None}, {"name": "SEQ", "value": None}, {"name": "EOS", "value": None}]
    pos_seq = [i for i, e in enumerate(events) if e["name"] == "SEQ"]
    seq_lyrics = [l for l in seq_lyrics if l]
    return seq_lyrics, pos_seq, events


def build_data(args):
    sys.path.insert(0, str(QA))
    from qwen_abc.canonical import Song
    from paper_eval.common import load_rows, iter_train_rows
    d = OUT / "csl_l2m_data"
    ev_dir = d / "REMIaligned_events"
    ev_dir.mkdir(parents=True, exist_ok=True)
    melody_dict = pickle.load(open(CSL_DIR / "data/dictionary_melody.pkl", "rb"))
    ev2idx = melody_dict[0]
    durations = sorted(int(k.split("_")[-1]) for k in ev2idx if k.startswith("Note_Duration_"))
    chars, splits, stats = set(), {"train": [], "validation": [], "test": []}, {}
    for split in ("train", "validation", "test"):
        it = iter_train_rows() if split == "train" else load_rows(split)
        c = {"songs": 0, "kept": 0, "not_4_4": 0, "no_notes": 0, "too_many_lines": 0}
        for r in it:
            c["songs"] += 1
            if r["spec"]["meter"] != "4/4":
                c["not_4_4"] += 1
                continue
            res = song_to_events(Song.from_json(r["song"]), durations=durations)
            if res is None:
                c["no_notes"] += 1
                continue
            seq_lyrics, pos_seq, events = res
            for e in events:
                k = f"{e['name']}_{e['value']}"
                assert k in ev2idx, k
            fn = f"{r['song_id']}.pkl"
            pickle.dump(res, open(ev_dir / fn, "wb"))
            splits[split].append(fn)
            if split == "train":
                chars.update(ch for l in seq_lyrics for ch in l)
            c["kept"] += 1
        stats[split] = c
        print(split, c, flush=True)
    lyr = ["K"] + sorted(chars)
    lyric2idx = {c: i for i, c in enumerate(lyr)}
    pickle.dump((lyric2idx, {i: c for c, i in lyric2idx.items()}), open(d / "dictionary_lyric.pkl", "wb"))
    for split, name in (("train", "train.pkl"), ("validation", "val.pkl"), ("test", "test.pkl")):
        pickle.dump(splits[split], open(d / name, "wb"))
    (d / "build_report.json").write_text(json.dumps({"splits": stats, "lyric_vocab": len(lyr)}, indent=1))
    print("CSL_DATA_OK", len(lyr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "build-data"])
    ap.add_argument("--which", choices=["official", "retrain"], default="official")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--conditions", default="orig,lyrics_all")
    ap.add_argument("--seeds", default="S1,S2,S3,S4")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunked", action="store_true", help="section-chunked generation (<= 24 lines per call)")
    args = ap.parse_args()
    {"infer": infer, "build-data": build_data}[args.cmd](args)


if __name__ == "__main__":
    main()
