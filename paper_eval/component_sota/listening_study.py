#!/usr/bin/env python
"""Component-round listening study: blinded, counterbalanced package + analysis. NO human results.

  python -m paper_eval.component_sota.listening_study build [--lists 4]      # re-runnable as audio appears
  python -m paper_eval.component_sota.listening_study prep-render; ... render --shard k --num-shards n   # D/E vocals (GPU)
  python -m paper_eval.component_sota.listening_study stage                  # trimmed copies -> AUDIO/listening_study/stimuli
  python -m paper_eval.component_sota.listening_study analyze --responses R.csv [--out DIR]
  python -m paper_eval.component_sota.listening_study analyze --simulate --out <scratch dir>   # pipeline test only

Blocks (spec section 24):
  A  lyrics->melody: pairs of melody sources on the same song, both clips sung by the same SVS
     (the renderer alternates by a hash of the item, never chosen by quality);
  B  SVS robustness: FastSinger vs SoulX-Singer on the same melody (+ a piano render of the target);
  C  harmonization: backing mixes of the reference melody with Qwen / AccoMontage2 / reference chords;
  D  infilling locality: original draft vs an edited version (section infill vs whole-song
     regeneration; which method is blinded, "original" is labelled);
  E  section semantics: original vs label-intervened sample, "which sounds more chorus-/bridge-like?".
Songs are drawn from the frozen audio subset (reports/component_sota/audio/subset.json) by a salted
hash, BEFORE looking at whether audio exists; items whose audio is missing stay in the design as
``pending`` so availability cannot select items.

Repo (reports/component_sota/listening_study/): participant-facing ``lists/list_<k>.csv`` and
``questions.json`` (no system names, no song ids, opaque clip names), ``response_schema.json`` +
``response_template.csv``, ``design.json`` (counts only), and the non-participant files
``KEY_DO_NOT_SHARE.json`` (item -> systems/songs/A-B mapping) and ``STAGING_DO_NOT_SHARE.csv``
(blinded clip -> absolute source path). Audio never enters the repo: ``stage`` writes trimmed
copies, QC-degraded clips and piano target renders under AUDIO/listening_study/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..common import ROOT, jdump

AUDIO = Path("/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio")
STUDY = ROOT / "reports/component_sota/listening_study"
STAGE = AUDIO / "listening_study"
PF_GEN = ROOT / "experiments/paper_final/gen/qwen_e3b"
INFILL = ROOT / "experiments/component_sota/infill/qwen_e3b"
SALT = "qwenabc-component-listening-2026"
SEED = 20260927
CLIP_S = 30.0
FLUIDSYNTH = "/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/fluidsynth"
SOUNDFONT = "/gscratch/ark/pingw220/miniconda3/envs/beatbk-render/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2"

PAIRS_A = [("mel", "csl_offc"), ("mel", "e3b"), ("e3b", "csl_offc"), ("csl_offc", "csl_rtc"), ("mel", "ref")]
PAIRS_C = [("qwen", "am2"), ("qwen", "ref"), ("am2", "ref")]
SVS_WAV = {"fastsinger": "fastsinger.wav", "soulx": "soulx/generated.wav"}
AFC = "A | B | no_difference"
QUESTIONS = {
    "A": {"fits_lyrics": f"Which melody fits the lyrics better? ({AFC})",
          "coherent": f"Which melody sounds more coherent? ({AFC})",
          "natural_phrasing": f"Which melody has more natural phrasing? ({AFC})",
          "stronger_structure": f"Which has stronger phrase/section structure? ({AFC})"},
    "B": {"clearer": f"Which singing is clearer? ({AFC})",
          "more_natural": f"Which is more natural? ({AFC})",
          "follows_target": f"Which follows the target melody (the piano clip) better? ({AFC})"},
    "C": {"fits_melody": f"Which accompaniment fits the melody better? ({AFC})",
          "harmonically_coherent": f"Which sounds more harmonically coherent? ({AFC})",
          "more_natural": f"Which sounds more natural? ({AFC})"},
    "D": {"connects_naturally": "Does the inserted (edited) section connect naturally to what comes before and after? (1 = not at all ... 5 = seamlessly)",
          "unrelated_consistent": "Compared with the ORIGINAL, did the unrelated sections remain consistent? (1 = completely different ... 5 = identical)"},
    "E": {"more_chorus_like": f"Which version of the marked section sounds more chorus-like? ({AFC})",
          "more_bridge_like": f"Which version of the marked section sounds more bridge-like? ({AFC})"},
}
BLOCK_TITLE = {"A": "Lyrics→Melody", "B": "SVS renderer robustness", "C": "Melody→Chord harmonization",
               "D": "Edited-section continuity", "E": "Section semantic control"}


def h(*parts) -> str:
    return hashlib.sha1("|".join(map(str, (SALT,) + parts)).encode()).hexdigest()[:10]


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def subset_ids() -> List[str]:
    d = _load(ROOT / "reports/component_sota/audio/subset.json")
    return [s["song_id"] for s in d["songs"]] if d else []


def pick(songs, k, *tag):
    return sorted(songs, key=lambda s: h("pick", *tag, s))[:k]


def wav_clip(path: Path, start: float = 0.0) -> dict:
    return {"type": "wav", "path": str(path), "start": round(max(0.0, start), 2), "dur": CLIP_S, "ready": path.exists()}


def vocal_start(item_dir: Path) -> float:
    t = _load(item_dir / "target.json")
    return max(0.0, t["notes"][0]["start"] - 1.0) if t and t.get("notes") else 0.0


def song_start(song_json: Path) -> float:
    s = _load(song_json)
    if not s or not s.get("notes"):
        return 0.0
    on = min(n["onset"] if isinstance(n, dict) else n[0] for n in s["notes"])
    return max(0.0, on / 4 * 60.0 / float(s.get("tempo_bpm") or 120) - 2.0)


# ------------------------------------------------------------------ D/E renders (FastSinger vocal of a whole song)
RENDER = AUDIO / "listening_render"


def _song_of(path: str, field: str):
    from qwen_abc.canonical import Song
    r = _load(Path(path))
    if not r or not r.get(field):
        return None
    if field == "assembled":
        from qwen_abc.abc import parse_abc
        return parse_abc(r[field]).song
    return Song.from_json(r[field])


def rendered_clip(path: str, field: str, sections) -> dict:
    """Clip dict for the FastSinger render of ``path[field]``, windowed on ``sections`` = [first, last]."""
    key = h("render", path, field)
    wav = RENDER / key / "fastsinger.wav"
    c = {"type": "wav", "path": str(wav), "render_key": key, "render_src": path, "render_field": field,
         "sections": sections, "start": 0.0, "dur": CLIP_S, "ready": wav.exists()}
    song = _song_of(path, field)
    if song is not None and sections is not None and song.sections:
        spb = 60.0 / song.tempo_bpm
        starts = song.bar_starts() + [song.total_ticks]
        lo_s = max(0, min(sections[0], len(song.sections) - 1))
        hi_s = max(0, min(sections[1], len(song.sections) - 1))
        a = starts[song.sections[lo_s].start_bar] / 4 * spb
        e_bar = song.sections[hi_s].start_bar + song.sections[hi_s].num_bars
        b = starts[min(e_bar, len(starts) - 1)] / 4 * spb
        c["start"] = round(max(0.0, a - 1.0), 2)
        c["dur"] = round(min(max(b - a + 2.0, 10.0), 45.0), 2)
    return c


def prep_render() -> dict:
    """Write FastSinger inputs for every D/E clip in the key file (whole song; the clip is cut at staging)."""
    from qwen_abc.fastsinger import write_fastsinger_inputs
    key = _load(STUDY / "KEY_DO_NOT_SHARE.json") or {"items": []}
    n = Counter()
    for it in key["items"]:
        for c in it["clips"].values():
            if not c.get("render_key"):
                continue
            d = RENDER / c["render_key"]
            if (d / "fs.txt").exists():
                n["exists"] += 1
                continue
            song = _song_of(c["render_src"], c["render_field"])
            if song is None:
                n["unparsable"] += 1
                continue
            d.mkdir(parents=True, exist_ok=True)
            write_fastsinger_inputs(song, str(d / "fs.mid"), str(d / "fs.txt"))
            jdump({"src": c["render_src"], "field": c["render_field"]}, d / "source.json")
            n["prepared"] += 1
    return dict(n)


def render(shard=0, num_shards=1) -> dict:
    from .audio import FS_PY, MUSIC, run
    ds = sorted(d for d in RENDER.glob("*") if (d / "fs.txt").exists())[shard::num_shards]
    n = Counter()
    for d in ds:
        if (d / "fastsinger.wav").exists():
            continue
        code, _ = run([FS_PY, "inference.py", "--model_id", "suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag", "--model_epoch", "400",
                       "--spkr_ref", "6", "--pitch_shifts", "0", "--shift_consonant_forward_alignment", "--midi_path", str(d / "fs.mid"),
                       "--lyric_path", str(d / "fs.txt"), "--output_path", str(d / "fastsinger.wav")], MUSIC / "fastsinger", d / "fastsinger.log")
        n["ok" if code == 0 else "failed"] += 1
    return dict(n)


# ------------------------------------------------------------------ design
def items_a(songs, per_pair):
    out = []
    for a, b in PAIRS_A:
        for sid in pick(songs, per_pair, "A", a, b):
            rend = "fastsinger" if int(h("rend", a, b, sid), 16) % 2 == 0 else "soulx"
            clips = {}
            for s in (a, b):
                d = AUDIO / "svs" / s / sid
                clips[s] = wav_clip(d / SVS_WAV[rend], vocal_start(d))
            out.append({"block": "A", "song_id": sid, "systems": [a, b], "context": {"renderer": rend}, "clips": clips,
                        "questions": list(QUESTIONS["A"]), "lyrics_sheet": True})
    return out


def items_b(songs, per_source):
    out = []
    for src in ("ref", "e3b", "mel", "csl_offc", "csl_rtc"):
        for sid in pick(songs, per_source, "B", src):
            d = AUDIO / "svs" / src / sid
            st = vocal_start(d)
            clips = {r: wav_clip(d / w, st) for r, w in SVS_WAV.items()}
            ref = {"type": "midi_to_render", "path": str(d / "fs.mid"), "start": round(st, 2), "dur": CLIP_S, "ready": (d / "fs.mid").exists()}
            out.append({"block": "B", "song_id": sid, "systems": ["fastsinger", "soulx"], "context": {"melody_source": src},
                        "clips": clips, "reference_clip": ref, "questions": list(QUESTIONS["B"]), "lyrics_sheet": False})
    return out


def items_c(songs, per_pair):
    out = []
    for a, b in PAIRS_C:
        for sid in pick(songs, per_pair, "C", a, b):
            st = song_start(AUDIO / "backing" / "chords" / f"{sid}__ref.song.json")
            clips = {s: wav_clip(AUDIO / "backing" / "chords" / "render" / f"{sid}__{s}" / "mix.wav", st) for s in (a, b)}
            out.append({"block": "C", "song_id": sid, "systems": [a, b], "context": {"melody": "reference"}, "clips": clips,
                        "questions": list(QUESTIONS["C"]), "lyrics_sheet": False})
    return out


def items_d(songs, per_kind):
    """Original E3b draft vs (infill | whole-song regeneration) of one section, all rendered by FastSinger;
    the clip spans the section before, the target and the section after."""
    regen = {"edit_resample": ("orig", "S2"), "edit_lyrics": ("lyrics_sec", "S1")}
    out = []
    for kind, (rc, rs) in regen.items():
        for sid in pick(songs, per_kind, "D", kind):
            rec = _load(INFILL / kind / f"{sid}_S1.json")
            tgt = int(rec["target"]) if rec and rec.get("target") not in (None, "") else None
            secs = [max(tgt - 1, 0), tgt + 1] if tgt is not None else None
            orig = rendered_clip(str(PF_GEN / "orig" / f"{sid}_S1.json"), "song", secs)
            for meth, clip in (("infill", rendered_clip(str(INFILL / kind / f"{sid}_S1.json"), "assembled", secs)),
                               ("regenerate", rendered_clip(str(PF_GEN / rc / f"{sid}_{rs}.json"), "song", secs))):
                out.append({"block": "D", "song_id": sid, "systems": ["original", meth], "context": {"edit": kind, "target_section": tgt},
                            "clips": {"original": dict(orig), meth: clip}, "questions": list(QUESTIONS["D"]), "lyrics_sheet": kind == "edit_lyrics",
                            "fixed_order": True})
    return out


def items_e(songs, per_cond):
    out = []
    for cond, q in (("label_swap", "more_chorus_like"), ("label_bridge", "more_bridge_like")):
        for sid in pick(songs, per_cond, "E", cond):
            r = _load(PF_GEN / cond / f"{sid}_S1.json")
            meta = (r or {}).get("meta") or {}
            t = meta.get("target_section")
            clips = {s: rendered_clip(str(PF_GEN / c / f"{sid}_S1.json"), "song", [t, t] if t is not None else None)
                     for s, c in (("original_label", "orig"), ("relabelled", cond))}
            out.append({"block": "E", "song_id": sid, "systems": ["original_label", "relabelled"],
                        "context": {"condition": cond, "original_label": meta.get("original_label"), "requested_label": meta.get("requested_label"),
                                    "target_section": t},
                        "clips": clips, "questions": [q], "lyrics_sheet": False})
    return out


def qc_items(pool: List[dict]) -> List[dict]:
    """Two identical pairs (expected no_difference) and two degraded pairs (expected: the clean clip)."""
    ready = [i for i in pool if i["block"] in "ABC" and all(c["ready"] for c in i["clips"].values())]
    ready = sorted(ready, key=lambda i: h("qc", i["block"], i["song_id"], *i["systems"]))
    if len(ready) < 4:  # not enough audio yet: keep placeholder slots so the design shape is stable
        ready = sorted(pool, key=lambda i: h("qc", i["block"], i["song_id"], *i["systems"]))
    out = []
    for j, base in enumerate(ready[:4]):
        s = base["systems"][0]
        clip = dict(base["clips"][s])
        kind = "qc_identical" if j < 2 else "qc_degraded"
        other = dict(clip, transform="degrade") if kind == "qc_degraded" else dict(clip)
        out.append({"block": base["block"], "song_id": base["song_id"], "systems": ["clean", "copy" if kind == "qc_identical" else "degraded"],
                    "context": {"qc": kind, "source_block": base["block"]}, "clips": {"clean": clip, other.get("transform") and "degraded" or "copy": other},
                    "questions": list(QUESTIONS[base["block"]]), "lyrics_sheet": base["lyrics_sheet"], "kind": kind,
                    **({"reference_clip": dict(base["reference_clip"])} if base.get("reference_clip") else {})})
    return out


def build(n_lists=4, per_a=6, per_b=4, per_c=6, per_d=6, per_e=6) -> dict:
    songs = subset_ids()
    items = items_a(songs, per_a) + items_b(songs, per_b) + items_c(songs, per_c) + items_d(songs, per_d) + items_e(songs, per_e)
    for it in items:
        it.setdefault("kind", "test")
    items += qc_items([i for i in items if i["kind"] == "test"])
    for it in items:
        it["item_id"] = h("item", it["block"], it["kind"], it["song_id"], *it["systems"], json.dumps(it["context"], sort_keys=True))
        it["clip_ids"] = {s: h("clip", it["item_id"], s) for s in it["clips"]}
        if it.get("reference_clip"):
            it["clip_ids"]["_target"] = h("clip", it["item_id"], "target")
        it["status"] = "ready" if all(c["ready"] for c in it["clips"].values()) else "pending_audio"
    # counterbalancing: within each (block, pair) stratum the first system is on side A for items
    # ranked even on list 0, odd on list 1, ... so every item flips across lists and strata stay balanced
    rank = {}
    strata = defaultdict(list)
    for it in items:
        strata[(it["block"], it["kind"], tuple(it["systems"]))].append(it)
    for grp in strata.values():
        for r, it in enumerate(sorted(grp, key=lambda i: i["item_id"])):
            rank[it["item_id"]] = r
    (STUDY / "lists").mkdir(parents=True, exist_ok=True)
    lists = {}
    for k in range(n_lists):
        rng = random.Random(SEED + k)
        order = sorted(items, key=lambda i: i["item_id"])
        rng.shuffle(order)
        order.sort(key=lambda i: "ABCDE".index(i["block"]))  # blocks stay contiguous (instructions per block)
        rows = []
        for t, it in enumerate(order, 1):
            s1, s2 = it["systems"]
            first_a = it.get("fixed_order") or (rank[it["item_id"]] + k) % 2 == 0
            A, B = (s1, s2) if first_a else (s2, s1)
            it.setdefault("A_is", {})[str(k)] = A
            rows.append({"list_id": k, "trial": t, "item_id": it["item_id"], "block": it["block"],
                         "clip_A": it["clip_ids"][A] + ".wav", "clip_B": it["clip_ids"][B] + ".wav",
                         "clip_A_label": "ORIGINAL" if it["block"] == "D" else "A", "clip_B_label": "EDITED" if it["block"] == "D" else "B",
                         "target_clip": (it["clip_ids"]["_target"] + ".wav") if "_target" in it["clip_ids"] else "",
                         "question_ids": ";".join(it["questions"]), "lyrics_sheet": int(bool(it["lyrics_sheet"])),
                         "status": it["status"]})
        with open(STUDY / "lists" / f"list_{k}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lists[k] = len(rows)
    jdump({"questions": QUESTIONS, "block_titles": BLOCK_TITLE,
           "instructions": {"A": "Both clips sing the same lyrics (shown on the sheet). Compare the MELODIES.",
                            "B": "Both clips sing the same melody. First listen to the piano clip (the target melody).",
                            "C": "Both clips have the same vocal melody. Compare the ACCOMPANIMENT.",
                            "D": "The first clip is the ORIGINAL; in the second, one marked section was rewritten.",
                            "E": "Listen to the marked section in each clip."}}, STUDY / "questions.json")
    jdump({"note": "NOT PARTICIPANT-FACING", "seed": SEED, "salt": SALT, "items": items}, STUDY / "KEY_DO_NOT_SHARE.json")
    with open(STUDY / "STAGING_DO_NOT_SHARE.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "item_id", "block", "status", "type", "source_path", "field", "start_s", "dur_s", "sections", "transform"])
        for it in items:
            for s, c in list(it["clips"].items()) + ([("_target", it["reference_clip"])] if it.get("reference_clip") else []):
                w.writerow([it["clip_ids"][s], it["item_id"], it["block"], "ready" if c["ready"] else "pending", c["type"], c["path"],
                            c.get("field", ""), c.get("start", ""), c.get("dur", ""), json.dumps(c.get("sections")) if c.get("sections") is not None else "",
                            c.get("transform", "")])
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "component listening-study response",
              "type": "object", "required": ["participant_id", "list_id", "trial", "item_id", "question_id", "response"],
              "properties": {"participant_id": {"type": "string"}, "list_id": {"type": "integer", "minimum": 0, "maximum": n_lists - 1},
                             "trial": {"type": "integer", "minimum": 1}, "item_id": {"type": "string"},
                             "question_id": {"enum": sorted({q for b in QUESTIONS.values() for q in b})},
                             "response": {"type": "string", "description": "A|B|no_difference for 2AFC; 1-5 for block D"},
                             "rt_ms": {"type": "integer"}, "timestamp": {"type": "string", "format": "date-time"}}}
    jdump(schema, STUDY / "response_schema.json")
    with open(STUDY / "response_template.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(list(schema["properties"]))
    counts = {b: dict(Counter(i["status"] for i in items if i["block"] == b and i["kind"] == "test")) for b in "ABCDE"}
    design = {"n_items": len(items), "n_test": sum(i["kind"] == "test" for i in items), "n_qc": sum(i["kind"] != "test" for i in items),
              "per_block_status": counts, "qc_status": dict(Counter(i["status"] for i in items if i["kind"] != "test")),
              "lists": lists, "seed": SEED, "clip_seconds": CLIP_S,
              "parameters": {"per_pair_A": per_a, "per_source_B": per_b, "per_pair_C": per_c, "per_kind_D": per_d, "per_condition_E": per_e}}
    jdump(design, STUDY / "design.json")
    return design


# ------------------------------------------------------------------ staging (off-repo audio)
def _read_wav(p: Path):
    import numpy as np
    with wave.open(str(p)) as f:
        sr, ch, sw, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
        x = np.frombuffer(f.readframes(n), dtype={2: "<i2", 4: "<i4"}[sw]).astype("float32") / (2 ** (8 * sw - 1))
    return x.reshape(-1, ch), sr


def _write_wav(p: Path, x, sr):
    import numpy as np
    p.parent.mkdir(parents=True, exist_ok=True)
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(p), "w") as f:
        f.setnchannels(y.shape[1])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(y.tobytes())


def degrade(x, sr, seed=0):
    """Obvious degradation for catch trials: 4-bit quantization, 2 kHz sample-and-hold, white noise at -12 dB."""
    import numpy as np
    rng = np.random.default_rng(seed)
    hold = max(1, sr // 2000)
    y = np.repeat(x[::hold], hold, axis=0)[: len(x)]
    y = np.round(y * 8) / 8
    return y + rng.normal(0, 0.25 * (np.abs(x).max() + 1e-6), size=y.shape).astype("float32")


def stage() -> dict:
    import subprocess
    rows = list(csv.DictReader(open(STUDY / "STAGING_DO_NOT_SHARE.csv", encoding="utf-8")))
    out = STAGE / "stimuli"
    done = Counter()
    for r in rows:
        dst = out / f"{r['clip_id']}.wav"
        if dst.exists():
            done["exists"] += 1
            continue
        src = Path(r["source_path"])
        if r["type"] == "wav" and src.exists():
            x, sr = _read_wav(src)
            a = int(float(r["start_s"] or 0) * sr)
            y = x[a: a + int(float(r["dur_s"] or CLIP_S) * sr)]
            if r["transform"] == "degrade":
                y = degrade(y, sr, int(r["clip_id"], 16) % 2 ** 31)
            n = int(0.05 * sr)  # 50 ms fades
            if len(y) > 2 * n:
                import numpy as np
                ramp = np.linspace(0, 1, n, dtype="float32")[:, None]
                y = y.copy()
                y[:n] *= ramp
                y[-n:] *= ramp[::-1]
            _write_wav(dst, y, sr)
            done["staged"] += 1
        elif r["type"] == "midi_to_render" and src.exists() and Path(FLUIDSYNTH).exists():
            full = STAGE / "piano_full" / f"{r['clip_id']}.wav"
            full.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([FLUIDSYNTH, "-ni", "-g", "0.8", "-r", "44100", "-F", str(full), SOUNDFONT, str(src)],
                           check=False, capture_output=True, timeout=600)
            if full.exists():
                x, sr = _read_wav(full)
                a = int(float(r["start_s"] or 0) * sr)
                _write_wav(dst, x[a: a + int(float(r["dur_s"] or CLIP_S) * sr)], sr)
                done["piano_rendered"] += 1
            else:
                done["piano_failed"] += 1
        else:
            done["pending"] += 1
    jdump(dict(done), STAGE / "stage_report.json")
    return dict(done)


# ------------------------------------------------------------------ analysis
def fleiss_kappa(ratings: Dict[str, List[str]]) -> Optional[float]:
    items = {k: v for k, v in ratings.items() if len(v) >= 2}
    if not items:
        return None
    cats = sorted({c for v in items.values() for c in v})
    p_i, totals, N = [], Counter(), 0
    for v in items.values():
        n, c = len(v), Counter(v)
        p_i.append((sum(x * x for x in c.values()) - n) / (n * (n - 1)))
        totals.update(c)
        N += n
    P_e = sum((totals[c] / N) ** 2 for c in cats)
    return None if P_e == 1 else (sum(p_i) / len(p_i) - P_e) / (1 - P_e)


def krippendorff_alpha(ratings: Dict[str, List], metric: str = "nominal") -> Optional[float]:
    units = {k: v for k, v in ratings.items() if len(v) >= 2}
    if not units:
        return None
    d = (lambda a, b: (float(a) - float(b)) ** 2) if metric == "interval" else (lambda a, b: float(a != b))
    o = defaultdict(float)
    for v in units.values():
        m = len(v)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(v[i], v[j])] += 1.0 / (m - 1)
    n_c = defaultdict(float)
    for (a, _), w in o.items():
        n_c[a] += w
    n = sum(n_c.values())
    vals = list(n_c)
    D_o = sum(w * d(a, b) for (a, b), w in o.items()) / n
    D_e = sum(n_c[a] * n_c[b] * d(a, b) for a in vals for b in vals) / (n * (n - 1))
    return None if D_e == 0 else 1 - D_o / D_e


def cluster_boot(obs: List[tuple], cluster: int, n_boot=10000, seed=0) -> Optional[dict]:
    """obs = [(song, rater, value)]; resample clusters (0 = songs, 1 = raters) with replacement."""
    import numpy as np
    if not obs:
        return None
    by = defaultdict(list)
    for o in obs:
        by[o[cluster]].append(o[2])
    keys = sorted(by)
    sums = np.array([sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"mean": float(sums.sum() / cnts.sum()), "lo": float(lo), "hi": float(hi), "n_clusters": len(keys), "n_obs": int(cnts.sum())}


def analyze(responses: Path, study: Path = STUDY, max_catch_errors: int = 1, n_boot: int = 10000, simulated: bool = False) -> dict:
    key = {i["item_id"]: i for i in _load(study / "KEY_DO_NOT_SHARE.json")["items"]}
    resp = list(csv.DictReader(open(responses, encoding="utf-8")))
    # QC: a catch trial is failed if most of its answered questions are wrong
    errs, seen = Counter(), Counter()
    per_trial = defaultdict(list)
    for r in resp:
        it = key.get(r["item_id"])
        if it and it["kind"] != "test":
            per_trial[(r["participant_id"], r["item_id"], r["list_id"])].append(r)
    for (pid, iid, lid), rs in per_trial.items():
        it = key[iid]
        if it["kind"] == "qc_identical":
            wrong = sum(x["response"] != "no_difference" for x in rs)
        else:
            clean_letter = "A" if it["A_is"][str(lid)] == "clean" else "B"
            wrong = sum(x["response"] != clean_letter for x in rs)
        seen[pid] += 1
        errs[pid] += wrong * 2 > len(rs)
    raters = sorted({r["participant_id"] for r in resp})
    excluded = [p for p in raters if errs[p] > max_catch_errors]
    keep = [r for r in resp if r["participant_id"] not in excluded and key.get(r["item_id"], {}).get("kind") == "test"]
    results, irr = [], []
    groups = defaultdict(list)
    for r in keep:
        it = key[r["item_id"]]
        groups[(it["block"], tuple(it["systems"]), json.dumps({k: v for k, v in it["context"].items() if k in ("edit", "condition")}), r["question_id"])].append((it, r))
    for (block, systems, ctx, q), rows in sorted(groups.items()):
        s1, s2 = systems
        obs, cats = [], defaultdict(list)
        for it, r in rows:
            if block == "D":
                try:
                    v = float(r["response"])
                except ValueError:
                    continue
                cats[it["item_id"]].append(v)
            else:
                A = it["A_is"][str(r["list_id"])]
                pref = "none" if r["response"] == "no_difference" else (A if r["response"] == "A" else (s2 if A == s1 else s1))
                v = {s1: 1.0, s2: 0.0}.get(pref, 0.5)
                cats[it["item_id"]].append(pref)
            obs.append((it["song_id"], r["participant_id"], v))
        res = {"block": block, "comparison": f"{s1} vs {s2}", "context": json.loads(ctx), "question": q,
               "estimand": f"mean Likert (1-5) for {s2}" if block == "D" else f"P(prefer {s1}); no_difference = 0.5",
               "by_song": cluster_boot(obs, 0, n_boot), "by_rater": cluster_boot(obs, 1, n_boot, seed=1)}
        if block != "D":
            res["no_difference_rate"] = sum(v == 0.5 for *_, v in obs) / max(len(obs), 1)
        results.append(res)
        irr.append({"block": block, "comparison": f"{s1} vs {s2}", "question": q,
                    "krippendorff_alpha": krippendorff_alpha(cats, "interval" if block == "D" else "nominal"),
                    "fleiss_kappa": fleiss_kappa({k: [str(x) for x in v] for k, v in cats.items()}),
                    "items_with_2plus": sum(len(v) >= 2 for v in cats.values())})
    # D: paired infill - regeneration difference over songs
    d_diff = []
    for q in QUESTIONS["D"]:
        per = defaultdict(lambda: defaultdict(list))
        for r in keep:
            it = key[r["item_id"]]
            if it["block"] == "D" and r["question_id"] == q:
                try:
                    per[(it["song_id"], it["context"]["edit"])][it["systems"][1]].append(float(r["response"]))
                except ValueError:
                    pass
        diffs = [(s, "", sum(v["infill"]) / len(v["infill"]) - sum(v["regenerate"]) / len(v["regenerate"]))
                 for (s, _), v in per.items() if v["infill"] and v["regenerate"]]
        d_diff.append({"question": q, "infill_minus_regenerate": cluster_boot(diffs, 0, n_boot)})
    out = {"SIMULATED": simulated, "banner": "SIMULATED RESPONSES - NOT HUMAN DATA" if simulated else "collected human responses",
           "raters": len(raters), "excluded_raters": excluded, "catch_errors": dict(errs), "n_responses_kept": len(keep),
           "results": results, "infill_vs_regenerate": d_diff, "reliability": irr}
    return out


def simulate(study: Path, out_csv: Path, n_raters=12, seed=0) -> Path:
    """Synthetic responses for pipeline testing only (one careless rater fails QC)."""
    key = {i["item_id"]: i for i in _load(study / "KEY_DO_NOT_SHARE.json")["items"]}
    lists = sorted((study / "lists").glob("list_*.csv"))
    rng = random.Random(seed)
    rows = []
    for p in range(n_raters):
        lst = list(csv.DictReader(open(lists[p % len(lists)], encoding="utf-8")))
        careless = p == n_raters - 1
        for tr in lst:
            it = key[tr["item_id"]]
            A = it["A_is"][tr["list_id"]]
            for q in tr["question_ids"].split(";"):
                if it["block"] == "D":
                    resp = str(rng.randint(1, 5)) if careless else str(min(5, max(1, round(rng.gauss(4.2 if it["systems"][1] == "infill" else 2.8, 0.8)))))
                elif careless:
                    resp = rng.choice(["A", "B"])
                elif it["kind"] == "qc_identical":
                    resp = "no_difference" if rng.random() < 0.9 else "A"
                elif it["kind"] == "qc_degraded":
                    resp = "A" if A == "clean" else "B"
                else:
                    u = rng.random()
                    first = it["systems"][0]
                    resp = "no_difference" if u < 0.15 else (("A" if A == first else "B") if u < 0.65 else ("B" if A == first else "A"))
                rows.append({"participant_id": f"sim{p:02d}", "list_id": tr["list_id"], "trial": tr["trial"], "item_id": tr["item_id"],
                             "question_id": q, "response": resp, "rt_ms": rng.randint(800, 9000), "timestamp": ""})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def write_results(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    jdump(res, out / ("SIMULATED_results.json" if res["SIMULATED"] else "results.json"))
    lines = [f"# Listening study analysis — {res['banner']}", "",
             f"raters {res['raters']}, excluded by QC {len(res['excluded_raters'])}, responses kept {res['n_responses_kept']}", "",
             "| block | comparison | context | question | estimate | 95% CI (songs) | 95% CI (raters) | N songs | N raters |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        s, t = r["by_song"] or {}, r["by_rater"] or {}
        if not s:
            continue
        lines.append(f"| {r['block']} | {r['comparison']} | {r['context'] or ''} | {r['question']} | {s['mean']:.3f} | "
                     f"[{s['lo']:.3f}, {s['hi']:.3f}] | [{t.get('lo', float('nan')):.3f}, {t.get('hi', float('nan')):.3f}] | {s['n_clusters']} | {t.get('n_clusters', '')} |")
    lines += ["", "| block | comparison | question | Krippendorff α | Fleiss κ | items rated ≥2× |", "|---|---|---|---|---|---|"]
    fm = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
    for r in res["reliability"]:
        lines.append(f"| {r['block']} | {r['comparison']} | {r['question']} | {fm(r['krippendorff_alpha'])} | {fm(r['fleiss_kappa'])} | {r['items_with_2plus']} |")
    (out / ("SIMULATED_results.md" if res["SIMULATED"] else "results.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "prep-render", "render", "stage", "analyze"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--lists", type=int, default=4)
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--simulate", action="store_true", help="synthetic responses, pipeline testing only")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.lists), indent=1))
    elif a.cmd == "prep-render":
        print(prep_render())
    elif a.cmd == "render":
        print(render(a.shard, a.num_shards))
    elif a.cmd == "stage":
        print(stage())
    else:
        if a.simulate:
            out = a.out or ROOT / "experiments/component_sota/tmp/listening_SIMULATED"
            if (ROOT / "reports").resolve() in out.resolve().parents:
                raise SystemExit("refusing to write simulated listening results under reports/")
            responses = simulate(STUDY, out / "SIMULATED_responses.csv")
        else:
            if not a.responses:
                raise SystemExit("--responses is required (or --simulate)")
            responses, out = a.responses, a.out or STUDY / "results"
        res = analyze(responses, n_boot=a.n_boot, simulated=a.simulate)
        write_results(res, out)
        print(res["banner"], "->", out)


if __name__ == "__main__":
    main()
