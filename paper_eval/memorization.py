#!/usr/bin/env python
"""Memorization / copying analysis: how close is each generation to its nearest training song?

  python -m paper_eval.memorization build-index            # parse 10,243 train songs once (cached)
  python -m paper_eval.memorization query [--systems e3b,mupt] [--limit N]   # per-song nearest neighbours
  python -m paper_eval.memorization validate               # shortlist recall vs brute force
  python -m paper_eval.memorization report                 # parquet, table, figure, stats json

Representations (per song):

A  ``interval``  melodic intervals, clamped to +-12 (transposition invariant)
B  ``contour``   interval sign (-1/0/+1)
C  ``rhythm``    (onset-in-bar, duration) per note
D  ``motif``     (interval, inter-onset interval) per note transition
E  ``chord``     chord per change, root relative to the tonic + quality (key invariant)
F  ``abc``       ABC music tokens (notes, rests, chord symbols, bar lines; counters, comments, lyrics stripped)
G  ``plan``      (label, bars) per section
H  ``lyrics``    syllables (held-out references only: generations copy the prompt's lyrics)

Nearest neighbour search is exact. Every train song's n-grams go into a sorted (hash, song)
array per representation, so a query's n-gram *containment* in every train song is one bincount;
normalized LCS (``paper_eval.seqsim``, bit-parallel) is computed against all 10,243 train songs.
``validate`` documents why: a top-50-by-containment shortlist misses the true LCS neighbour on a
sample of queries (small but nonzero gaps), so the shortlist is not used for any reported number.

Exact bar copy: bars with >=3 notes as key-relative (onset, duration, pitch - tonic) signatures;
the rate is the share of a song's such bars that occur verbatim anywhere in training.

Outputs contain ids and numbers only (no lyrics, no notes).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .common import (DATA_DIR, DATA_OUT, DISPLAY, FIG_DIR, OFFLOAD, OUT_ROOT, REPORT_DIR, ROOT, bootstrap_mean_ci,
                     jdump, load_rows, write_table)
from .seqsim import lcs_fast

MEM_DIR = Path(os.environ.get("PF_MEM_DIR", str(OUT_ROOT / "memorization")))
INDEX_PATH = MEM_DIR / "train_index.pkl"
REPS = ("interval", "contour", "rhythm", "motif", "chord", "abc", "plan", "lyrics")
NGRAM = {"interval": 8, "contour": 12, "rhythm": 8, "motif": 6, "chord": 4, "abc": 10, "plan": 3, "lyrics": 6}
TOPK = 50
MELODIC = ("interval", "motif")
# v1 (first full run) lacked the informative-run fields; its files are kept, v2 is what the report reads
RESULTS_VERSION = "v2"

SYSTEMS = {
    "reference": None,
    "qwen_e3b": ("legacy", OFFLOAD / "evals/bon_e3b_test_T1.0/generations", "{sid}_s0.json"),
    "mupt": ("legacy", ROOT / "experiments/evals_r4/r4_mupt_test_T1.0_bon/generations", "{sid}_s0.json"),
    "midi_llm": ("legacy", ROOT / "experiments/midi_llm_r2_20260915_121309/modeA/eval_test/generations", "{sid}_s0.json"),
    "qwen_e3b:bars_p4": ("pf", OUT_ROOT / "gen/qwen_e3b/bars_p4", "{sid}_S1.json"),
    "qwen_e3b:ood_long20": ("pf", OUT_ROOT / "gen/qwen_e3b/ood_long20", "{sid}_S1.json"),
    "qwen_e3b:lyrics_all": ("pf", OUT_ROOT / "gen/qwen_e3b/lyrics_all", "{sid}_S1.json"),
}


# ------------------------------------------------------------------ features
def _tonic(song) -> int:
    from qwen_abc.theory import parse_key_name
    k = parse_key_name(song.key)
    return k[0] if k else 0


_ABC_TOK = re.compile(r'"[^"]*"|\[[A-Za-z]:[^\]]*\]|[=^_]*[A-Ga-gz][,\']*[0-9]*/*[0-9]*-?|\|')


def abc_tokens(text: str) -> List[str]:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("w:") or re.match(r"^[A-Za-z]:", s):
            continue
        s = re.sub(r"\[r:[^\]]*\]", " ", s)
        out.extend(t for t in _ABC_TOK.findall(s) if not t.startswith("[r:"))
    return out


def song_features(song, abc_text: Optional[str] = None, spec: Optional[dict] = None) -> Dict[str, list]:
    from qwen_abc.theory import parse_chord_symbol
    notes = song.notes
    starts = song.bar_starts()
    tonic = _tonic(song)
    iv = [max(-12, min(12, b.pitch - a.pitch)) for a, b in zip(notes, notes[1:])]
    bar_of = [max(bisect.bisect_right(starts, n.onset) - 1, 0) for n in notes]
    feats = {
        "interval": [str(x) for x in iv],
        "contour": [str((x > 0) - (x < 0)) for x in iv],
        "rhythm": [f"{n.onset - starts[b]}:{n.duration}" for n, b in zip(notes, bar_of)],
        "motif": [f"{max(-12, min(12, y.pitch - x.pitch))}:{min(y.onset - x.onset, 16)}" for x, y in zip(notes, notes[1:])],
    }
    ch = []
    for c in song.chords:
        info = parse_chord_symbol(c.symbol)
        if not info or info["root_pc"] is None:
            continue
        tok = f"{(info['root_pc'] - tonic) % 12}:{'.'.join(map(str, sorted((p - info['root_pc']) % 12 for p in info['pcs'])))}"
        if not ch or ch[-1] != tok:
            ch.append(tok)
    feats["chord"] = ch
    feats["abc"] = abc_tokens(abc_text) if abc_text else []
    feats["plan"] = [f"{s.label}:{s.num_bars}" for s in song.sections]
    if spec is not None:
        from qwen_abc.prompt import spec_syllables
        feats["lyrics"] = spec_syllables(spec)
    else:
        feats["lyrics"] = []
    # bar signatures (key relative), bars with >= 3 notes
    bars: Dict[int, list] = {}
    for n, b in zip(notes, bar_of):
        bars.setdefault(b, []).append((n.onset - starts[b], n.duration, n.pitch - tonic))
    feats["bars"] = [_h64(repr(tuple(v))) for v in bars.values() if len(v) >= 3]
    return feats


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


_P = np.uint64(1099511628211)


def ngram_hashes(tok_ids: np.ndarray, n: int) -> np.ndarray:
    """Deterministic polynomial hashes of all n-grams (unique)."""
    if len(tok_ids) < n:
        return np.zeros(0, dtype=np.uint64)
    t = tok_ids.astype(np.uint64)
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _P + t[j: len(t) - n + 1 + j] + np.uint64(1)
    return np.unique(h)


class Vocab:
    def __init__(self):
        self.ids: Dict[str, Dict[str, int]] = {r: {} for r in REPS}

    def encode(self, rep: str, toks: List[str], grow: bool) -> np.ndarray:
        d = self.ids[rep]
        out = []
        for t in toks:
            i = d.get(t)
            if i is None:
                if grow:
                    i = len(d)
                    d[t] = i
                else:
                    i = -1 - (abs(hash(t)) % 1000003) - 10_000_000  # unseen: never equals a train id
            out.append(i)
        return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------------ index
def build_index(limit: int = 0) -> dict:
    from qwen_abc.canonical import Song
    t0 = time.time()
    vocab = Vocab()
    ids, seqs = [], {r: [] for r in REPS}
    postings = {r: [] for r in REPS}
    bar_set = set()
    with open(DATA_DIR / "songs_train.jsonl", encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if limit and k >= limit:
                break
            row = json.loads(line)
            song = Song.from_json(row["song"])
            f = song_features(song, row["abc"], row["spec"])
            ids.append(row["song_id"])
            for r in REPS:
                enc = vocab.encode(r, f[r], grow=True)
                seqs[r].append(enc.astype(np.int32))
                h = ngram_hashes(enc, NGRAM[r])
                postings[r].append(np.stack([h, np.full(len(h), k, dtype=np.uint64)], 1) if len(h) else np.zeros((0, 2), np.uint64))
            bar_set.update(f["bars"])
            if k % 1000 == 0:
                print(f"indexed {k} songs {time.time() - t0:.0f}s", flush=True)
    idx = {}
    for r in REPS:
        a = np.concatenate(postings[r]) if postings[r] else np.zeros((0, 2), np.uint64)
        o = np.argsort(a[:, 0], kind="stable")
        a = a[o]
        df_h, df_c = np.unique(a[:, 0], return_counts=True)
        idx[r] = {"hash": a[:, 0].copy(), "song": a[:, 1].astype(np.int32), "df_hash": df_h, "df_count": df_c}
    out = {"ids": ids, "vocab": vocab.ids, "seqs": seqs, "index": idx, "bar_set": bar_set,
           "n_train": len(ids), "built_seconds": time.time() - t0}
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    jdump({"n_train": len(ids), "n_train_distinct_bars3": len(bar_set)}, MEM_DIR / "index_meta.json")
    print(f"index: {len(ids)} songs, {len(bar_set)} distinct bars, {time.time() - t0:.0f}s -> {INDEX_PATH}")
    return out


def load_index() -> dict:
    with open(INDEX_PATH, "rb") as fh:
        return pickle.load(fh)


def containment_all(idx: dict, rep: str, qh: np.ndarray, n_train: int, max_df_frac: float = 1.0) -> np.ndarray:
    """|query n-grams present in train song s| / |query n-grams| for every train song s (exact)."""
    if not len(qh):
        return np.zeros(n_train)
    I = idx["index"][rep]
    lo = np.searchsorted(I["hash"], qh, side="left")
    hi = np.searchsorted(I["hash"], qh, side="right")
    if max_df_frac < 1.0:
        keep = (hi - lo) <= max_df_frac * n_train
        lo, hi = lo[keep], hi[keep]
    if not len(lo) or not (hi - lo).sum():
        return np.zeros(n_train)
    sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])
    cnt = np.bincount(I["song"][sel], minlength=n_train)
    return cnt / len(qh)


def longest_common_run(a, b) -> int:
    """Longest common contiguous substring length (DP, O(len(a)*len(b)))."""
    if not len(a) or not len(b):
        return 0
    a = np.asarray(a)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    best = 0
    bb = np.asarray(b)
    for x in a:
        eq = (bb == x)
        cur = np.zeros(len(b) + 1, dtype=np.int32)
        cur[1:] = np.where(eq, prev[:-1] + 1, 0)
        m = int(cur.max())
        best = max(best, m)
        prev = cur
    return best


def longest_common_run_seq(a: list, b: list):
    """(length, the run itself) of the longest common contiguous run of two token lists."""
    if not a or not b:
        return 0, []
    best, end = 0, 0
    prev = [0] * (len(b) + 1)
    for i, x in enumerate(a):
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                v = prev[j] + 1
                cur[j + 1] = v
                if v > best:
                    best, end = v, i
        prev = cur
    return best, a[end - best + 1: end + 1]


def informative_run(gen_iv: list, train_iv: list) -> dict:
    """Exact common runs that are not repeated-pitch stretches: runs over the interval sequence with
    unisons (0) removed, plus how many distinct intervals the run contains (a -2/+2 trill has 2)."""
    a = [x for x in gen_iv if x != "0"]
    b = [x for x in train_iv if x != "0"]
    n, run = longest_common_run_seq(a, b)
    return {"nn_run_nz": n, "nn_run_nz_distinct": len(set(run))}


def nearest(idx: dict, vocab: Vocab, feats: dict, rep: str, topk: Optional[int] = None) -> dict:
    """Nearest train song by exact containment (all songs) and by normalized LCS.

    ``topk=None`` (default) computes LCS against every train song (exact); an integer restricts
    LCS to the top-k songs by containment (used only by ``validate`` to show why brute force is needed).
    """
    n_train = idx["n_train"]
    q = vocab.encode(rep, feats[rep], grow=False)
    if len(q) < max(NGRAM[rep], 3):
        return {"n_tokens": int(len(q))}
    qh = ngram_hashes(q, NGRAM[rep])
    cont = containment_all(idx, rep, qh, n_train)
    j_cont = int(np.argmax(cont))
    cand = np.argsort(-cont, kind="stable")[:topk] if topk else range(n_train)
    ql = q.tolist()
    from .seqsim import _masks
    masks = _masks(ql)
    seqs = idx["seqs_list"][rep] if "seqs_list" in idx else [t.tolist() for t in idx["seqs"][rep]]
    lq = len(ql)
    if topk:
        best, jb = -1.0, -1
        for j in cand:
            t = seqs[j]
            s_ = lcs_fast(ql, t, masks) / max(lq, len(t), 1)
            if s_ > best:
                best, jb = s_, int(j)
        return {"n_tokens": int(lq), "nn_lcs": float(best), "nn_lcs_id": idx["ids"][jb]}
    sims = np.fromiter((lcs_fast(ql, t, masks) / max(lq, len(t), 1) for t in seqs), dtype=np.float64, count=n_train)
    order = np.argsort(-sims, kind="stable")
    jb, j2 = int(order[0]), int(order[1])
    best = float(sims[jb])
    mu, sd = float(sims.mean()), float(sims.std())
    run = longest_common_run(q, idx["seqs"][rep][jb]) if rep in MELODIC else None
    return {"n_tokens": int(lq), "nn_contain": float(cont[j_cont]), "nn_contain_id": idx["ids"][j_cont],
            "nn_lcs": best, "nn_lcs_id": idx["ids"][jb], "nn2_lcs": float(sims[j2]),
            "bg_mean": mu, "bg_sd": sd, "nn_excess": best - mu, "nn_z": (best - mu) / sd if sd > 0 else None,
            "nn_run": run, "n_train_contain_ge_0.5": int((cont >= 0.5).sum())}


# ------------------------------------------------------------------ systems
def load_system(name: str, rows: List[dict]) -> Dict[str, dict]:
    """song_id -> {'song': Song, 'abc': text or None}; failures (no song) kept as None."""
    from qwen_abc.canonical import Song
    out = {}
    if name == "reference":
        for r in rows:
            out[r["song_id"]] = {"song": Song.from_json(r["song"]), "abc": r["abc"], "spec": r["spec"]}
        return out
    kind, d, pat = SYSTEMS[name]
    for r in rows:
        p = Path(d) / pat.format(sid=r["song_id"])
        if not p.exists():
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        if not g.get("song"):
            out[r["song_id"]] = None
            continue
        abc = g.get("generation") or None
        out[r["song_id"]] = {"song": Song.from_json(g["song"]), "abc": abc if name != "midi_llm" else None, "spec": None}
    return out


_G: dict = {}


def _query_one(task):
    name, sid, it_json = task
    from qwen_abc.canonical import Song
    idx, vocab, ref_feats = _G["idx"], _G["vocab"], _G["ref_feats"]
    rec = {"system": name, "song_id": sid, "generated": it_json is not None}
    if it_json is None:
        return rec
    song = Song.from_json(it_json["song"])
    f = song_features(song, it_json["abc"], it_json["spec"])
    for rep in REPS:
        if rep == "lyrics" and name != "reference":
            continue
        if rep == "abc" and not f["abc"]:
            continue
        for k, v in nearest(idx, vocab, f, rep).items():
            rec[f"{rep}.{k}"] = v
    if "interval.nn_lcs_id" in rec:
        inv = _G.setdefault("inv_interval", {v: k for k, v in idx["vocab"]["interval"].items()})
        t_iv = [inv[x] for x in idx["seqs_list"]["interval"][idx["pos"][rec["interval.nn_lcs_id"]]]]
        n_raw, raw = longest_common_run_seq(f["interval"], t_iv)
        rec["interval.nn_run_zero_frac"] = (sum(1 for x in raw if x == "0") / n_raw) if n_raw else None
        for k, v in informative_run(f["interval"], t_iv).items():
            rec[f"interval.{k}"] = v
    bars = f["bars"]
    rec["bar_copy_rate"] = (sum(1 for b in bars if b in idx["bar_set"]) / len(bars)) if bars else None
    rec["n_bars3"] = len(bars)
    if name != "reference":   # generated -> its own held-out reference
        rf = ref_feats[sid]
        for rep in ("interval", "contour", "rhythm", "motif", "chord", "abc", "plan"):
            a, b = f[rep], rf[rep]
            if a and b:
                rec[f"{rep}.self_lcs"] = lcs_fast(a, b) / max(len(a), len(b))
        a, b = f["interval"], rf["interval"]
        rec["interval.self_run"] = longest_common_run(np.asarray([int(x) for x in a]), np.asarray([int(x) for x in b])) if a and b else None
        rs = set(rf["bars"])
        rec["bar_copy_from_ref_rate"] = (sum(1 for x in bars if x in rs) / len(bars)) if bars else None
    return rec


def cmd_query(args) -> None:
    import multiprocessing as mp
    idx = load_index()
    idx["seqs_list"] = {r: [t.tolist() for t in idx["seqs"][r]] for r in REPS}
    idx["pos"] = {sid: i for i, sid in enumerate(idx["ids"])}
    vocab = Vocab()
    vocab.ids = idx["vocab"]
    rows = load_rows("test")
    if args.limit:
        rows = rows[: args.limit]
    ref_feats = {}
    for r in rows:
        from qwen_abc.canonical import Song
        ref_feats[r["song_id"]] = song_features(Song.from_json(r["song"]), r["abc"], r["spec"])
    _G.update(idx=idx, vocab=vocab, ref_feats=ref_feats)
    systems = args.systems.split(",") if args.systems else list(SYSTEMS)
    for name in systems:
        out_p = MEM_DIR / f"results_{RESULTS_VERSION}_{name.replace(':', '__')}.jsonl"
        done = set()
        if out_p.exists():
            done = {json.loads(l)["song_id"] for l in open(out_p, encoding="utf-8")}
        tasks = []
        for r in rows:
            sid = r["song_id"]
            if sid in done:
                continue
            if name == "reference":
                tasks.append((name, sid, {"song": r["song"], "abc": r["abc"], "spec": r["spec"]}))
                continue
            kind, d, pat = SYSTEMS[name]
            p = Path(d) / pat.format(sid=sid)
            if not p.exists():
                continue
            g = json.loads(p.read_text(encoding="utf-8"))
            tasks.append((name, sid, None if not g.get("song") else
                          {"song": g["song"], "abc": (g.get("generation") or None) if name != "midi_llm" else None, "spec": None}))
        print(f"{name}: {len(tasks)} songs to query ({len(done)} already done)", flush=True)
        if args.dry_run or not tasks:
            continue
        t0 = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool, open(out_p, "a", encoding="utf-8") as fh:
            for i, rec in enumerate(pool.imap_unordered(_query_one, tasks, chunksize=1)):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 25 == 0:
                    print(f"  {name} {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    print("QUERY_DONE")


def cmd_validate(args) -> None:
    """Shortlist recall: does the top-K-by-containment shortlist contain the brute-force LCS NN?"""
    idx = load_index()
    vocab = Vocab()
    vocab.ids = idx["vocab"]
    rows = load_rows("test")
    rng = random.Random(0)
    sample = rng.sample(rows, min(args.n_validate, len(rows)))
    res = []
    for rep in ("interval", "rhythm", "chord"):
        hits = 0
        gaps = []
        for r in sample:
            it = load_system("reference", [r])[r["song_id"]]
            f = song_features(it["song"], it["abc"], it["spec"])
            nn = nearest(idx, vocab, f, rep, topk=TOPK)
            if "nn_lcs" not in nn:
                continue
            q = vocab.encode(rep, f[rep], grow=False).tolist()
            from .seqsim import _masks
            m = _masks(q)
            brute = max(lcs_fast(q, t.tolist(), m) / max(len(q), len(t), 1) for t in idx["seqs"][rep])
            hits += abs(brute - nn["nn_lcs"]) < 1e-12
            gaps.append(brute - nn["nn_lcs"])
        res.append({"rep": rep, "n": len(gaps), "exact_nn_found": hits, "max_gap": max(gaps) if gaps else None,
                    "mean_gap": sum(gaps) / len(gaps) if gaps else None})
        print(res[-1], flush=True)
    jdump(res, DATA_OUT / "memorization_shortlist_validation.json")


# ------------------------------------------------------------------ report
def _load_results():
    import pandas as pd
    recs = []
    for p in sorted(MEM_DIR.glob(f"results_{RESULTS_VERSION}_*.jsonl")):
        recs += [json.loads(l) for l in open(p, encoding="utf-8")]
    return pd.DataFrame(recs)


def cmd_report(args) -> None:
    import pandas as pd
    df = _load_results()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df.columns if c.endswith("_id") and c != "song_id"] + ["plan_exact_in_train"], errors="ignore") \
      .to_parquet(DATA_OUT / "memorization.parquet", index=False)
    # nearest-neighbour ids are train/test song ids (no content); keep them separately for inspection
    df[["system", "song_id"] + [c for c in df.columns if c.endswith("_id") and c != "song_id"]] \
        .to_parquet(DATA_OUT / "memorization_nn_ids.parquet", index=False)
    ref = df[df.system == "reference"].set_index("song_id")
    stats = {"n_train": None, "systems": {}}
    reps = ("interval", "contour", "rhythm", "motif", "chord", "abc", "plan", "lyrics")
    # flags use the held-out references as the null: excess similarity (NN minus background) above the
    # references' 99th percentile in BOTH melodic representations, or a longer exact interval run than
    # 99% of references share with their own nearest training song
    thresholds = {rep: float(np.nanquantile(ref[f"{rep}.nn_excess"].astype(float), 0.99)) for rep in reps if f"{rep}.nn_excess" in ref}
    run_max = float(np.nanmax(ref["interval.nn_run"].astype(float)))
    run_p99 = float(np.nanquantile(ref["interval.nn_run"].astype(float), 0.99))
    nz_p99 = float(np.nanquantile(ref["interval.nn_run_nz"].astype(float), 0.99))
    nz_max = float(np.nanmax(ref["interval.nn_run_nz"].astype(float)))
    table = []
    order = [s for s in SYSTEMS if s in set(df.system)]
    for sysname in order:
        d = df[df.system == sysname].set_index("song_id")
        g = d[d.generated == True]  # noqa: E712
        st = {"n_rows": int(len(d)), "n_generated": int(len(g))}
        row = {"system": DISPLAY.get(sysname.split(":")[0], sysname) + (f" [{sysname.split(':')[1]}]" if ":" in sysname else ""),
               "N": int(len(g))}
        for rep in reps:
            for stat in ("nn_lcs", "nn_excess", "nn_z", "bg_mean", "nn_contain"):
                col = f"{rep}.{stat}"
                if col not in g or g[col].isna().all():
                    continue
                m, lo, hi, n = bootstrap_mean_ci(g[col].astype(float).tolist())
                st[col] = [m, lo, hi, n]
                if sysname != "reference" and col in ref:
                    common = g.index.intersection(ref.index)
                    diffs = (g.loc[common, col].astype(float) - ref.loc[common, col].astype(float)).dropna().tolist()
                    st[col + "_minus_ref"] = list(bootstrap_mean_ci(diffs))
                if stat == "nn_lcs" and rep in ("interval", "plan"):
                    row[f"{rep} NN sim"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
                if stat == "nn_excess" and rep in ("interval", "motif", "rhythm", "chord", "abc"):
                    row[f"{rep} NN excess"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
            sc = f"{rep}.self_lcs"
            if sysname != "reference" and sc in g and not g[sc].isna().all():
                st[sc] = list(bootstrap_mean_ci(g[sc].astype(float).tolist()))
        for col in ("bar_copy_rate", "bar_copy_from_ref_rate", "interval.nn_run", "interval.nn_run_nz", "interval.nn_run_zero_frac",
                    "interval.self_run", "interval.nn_contain", "chord.nn_contain"):
            if col in g and not g[col].isna().all():
                st[col] = list(bootstrap_mean_ci(g[col].astype(float).tolist()))
                if sysname != "reference" and col in ref:
                    common = g.index.intersection(ref.index)
                    st[col + "_minus_ref"] = list(bootstrap_mean_ci(
                        (g.loc[common, col].astype(float) - ref.loc[common, col].astype(float)).dropna().tolist()))
        if "bar_copy_rate" in st:
            m, lo, hi, _ = st["bar_copy_rate"]
            row["bar copy rate"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
        if sysname != "reference":
            if "interval.self_lcs" in st:
                m, lo, hi, _ = st["interval.self_lcs"]
                row["interval vs own ref"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
            flags, strict = [], []
            for sid, r in g.iterrows():
                nz = r.get("interval.nn_run_nz")
                if nz is not None and not (isinstance(nz, float) and math.isnan(nz)) and nz > nz_p99 \
                        and (r.get("interval.nn_run_nz_distinct") or 0) >= 3:
                    strict.append({"song_id": sid, "interval.nn_run_nz": nz, "distinct": r.get("interval.nn_run_nz_distinct"),
                                   "interval.nn_lcs": r.get("interval.nn_lcs"), "nn_train_id": r.get("interval.nn_lcs_id")})
                why = [rep for rep in MELODIC if f"{rep}.nn_excess" in r and r[f"{rep}.nn_excess"] is not None
                       and not (isinstance(r[f"{rep}.nn_excess"], float) and math.isnan(r[f"{rep}.nn_excess"]))
                       and r[f"{rep}.nn_excess"] > thresholds[rep]]
                long_run = r.get("interval.nn_run")
                if long_run is not None and not (isinstance(long_run, float) and math.isnan(long_run)):
                    if long_run > run_p99:
                        why.append("interval_run>ref_p99")
                    if long_run > run_max:
                        why.append("interval_run>ref_max")
                if len([w for w in why if w in MELODIC]) == 2 or "interval_run>ref_p99" in why:
                    flags.append({"song_id": sid, "why": why, "interval.nn_lcs": r.get("interval.nn_lcs"),
                                  "run_zero_frac": r.get("interval.nn_run_zero_frac"), "run_nz": r.get("interval.nn_run_nz"),
                                  "motif.nn_lcs": r.get("motif.nn_lcs"), "interval.nn_z": r.get("interval.nn_z"),
                                  "motif.nn_z": r.get("motif.nn_z"), "interval.nn_run": long_run,
                                  "nn_train_id": r.get("interval.nn_lcs_id")})
            st["flagged"] = flags
            st["flagged_informative"] = strict
            row["raw flags"] = f"{len(flags)}/{len(g)}"
            row["informative-run flags"] = f"{len(strict)}/{len(g)}"
        stats["systems"][sysname] = st
        table.append(row)
    stats["thresholds_ref_p99"] = thresholds
    stats["ref_interval_run_max"] = run_max
    stats["ref_interval_run_p99"] = run_p99
    stats["ref_interval_run_nz_p99"] = nz_p99
    stats["ref_interval_run_nz_max"] = nz_max
    val = DATA_OUT / "memorization_shortlist_validation.json"
    if val.exists():
        stats["shortlist_validation"] = json.loads(val.read_text())
    meta_p = MEM_DIR / "index_meta.json"
    if meta_p.exists():
        stats.update(json.loads(meta_p.read_text()))
    elif INDEX_PATH.exists() and not args.no_index_meta:
        with open(INDEX_PATH, "rb") as fh:
            meta = pickle.load(fh)
        stats["n_train"] = meta["n_train"]
        stats["n_train_distinct_bars3"] = len(meta["bar_set"])
    jdump(stats, DATA_OUT / "memorization_stats.json")
    cols = ["system", "N", "interval NN sim", "interval NN excess", "motif NN excess", "rhythm NN excess", "chord NN excess",
            "abc NN excess", "plan NN sim", "bar copy rate", "interval vs own ref", "raw flags", "informative-run flags"]
    write_table(table, "memorization", caption="Nearest-training-song similarity over all 10,243 training songs (normalized LCS, "
                "1 = identical). 'Excess' = nearest-neighbour similarity minus the song's mean similarity to all training songs. "
                "Mean [95% CI] over songs; Pseudo-GT = held-out reference, the baseline for ordinary stylistic overlap.", columns=cols)
    _figure(df, order)
    print(json.dumps({k: v for k, v in stats.items() if k != "systems"}, indent=1))
    print("REPORT_DONE")


def _figure(df, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = ["interval", "motif", "rhythm", "chord", "abc", "plan"]
    fig, axes = plt.subplots(1, len(reps), figsize=(3.0 * len(reps), 3.6), sharey=True)
    colors = {"reference": "#6b6b6b", "qwen_e3b": "#1f5fa8", "mupt": "#c2571a", "midi_llm": "#2e8b57"}
    labels = []
    for ax, rep in zip(axes, reps):
        col = f"{rep}.nn_excess" if rep != "plan" else f"{rep}.nn_lcs"
        xs, ys, los, his, cs = [], [], [], [], []
        for i, s in enumerate(order):
            d = df[(df.system == s) & (df.generated == True)]  # noqa: E712
            if col not in d or d[col].isna().all():
                continue
            m, lo, hi, n = bootstrap_mean_ci(d[col].astype(float).tolist(), n_boot=2000)
            xs.append(i)
            ys.append(m)
            los.append(m - lo)
            his.append(hi - m)
            cs.append(colors.get(s.split(":")[0], "#1f5fa8"))
            # jittered per-song points (deterministic)
            v = d[col].astype(float).dropna().values
            rng = np.random.default_rng(i)
            ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=2, alpha=0.18, color=cs[-1], linewidths=0)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color="black", ms=4, capsize=3, zorder=3)
        ax.set_title(rep if rep != "plan" else "plan (NN sim)")
        if rep != "plan" and "reference" in order:
            ax.axhline(ys[0] if order[0] == "reference" and ys else 0, color="#6b6b6b", lw=0.8, ls="--")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DISPLAY.get(s.split(":")[0], s) + ("\n" + s.split(":")[1] if ":" in s else "") for s in order],
                           rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1 if rep == "plan" else 0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("nearest-training-song similarity\nminus mean similarity to all training songs")
    fig.suptitle("Excess similarity to the nearest of 10,243 training songs (normalized LCS; mean, 95% CI; dots = songs; dashed = held-out references)", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "memorization_nn.pdf")
    fig.savefig(FIG_DIR / "memorization_nn.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build-index", "query", "validate", "report"])
    ap.add_argument("--systems", default=None, help="comma list of " + ",".join(SYSTEMS))
    ap.add_argument("--limit", type=int, default=0, help="first N test songs (query) / train songs (build-index)")
    ap.add_argument("--n-validate", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-index-meta", action="store_true")
    args = ap.parse_args()
    if args.command == "build-index":
        if args.dry_run:
            print(f"would index {DATA_DIR / 'songs_train.jsonl'} -> {INDEX_PATH}")
            return
        build_index(args.limit)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
