#!/usr/bin/env python
"""Control-oriented listening study: build the blinded, counterbalanced package (no human results).

  python -m paper_eval.listening_study build [--per-condition 12] [--lists 4]
  python -m paper_eval.listening_study render-cpu --pilot 16    # fluidsynth backing + FastSinger inputs
  sbatch scripts/paper_final/listening_render.sbatch            # FastSinger vocals + mix (GPU, pilot only)

Items pair the E3b paper-final output for an intervention with the *original-control* output at
the same seed (``orig/<song>_S1.json`` vs ``<condition>/<song>_S1.json``). Items are chosen by a
deterministic hash of the song id among songs where both outputs parse -- never by whether the
model complied, so listener judgments of "did the edit happen" estimate compliance as heard.

Each item has two blocks:

1. **blinded 2AFC** on clips A and B (order counterbalanced across lists):
   ``longer_section`` (bars_p4 / bars_m4), ``higher_key`` (key_p5 / key_m3), ``matches_lyrics``
   (lyrics_sec, the replacement words shown on screen), each with a "no difference / neither"
   option; plus ``plausibility_A`` / ``plausibility_B`` (1-5);
2. **edit verification** with the original labelled as such: ``edit_happened`` (yes / partly / no)
   for the stated edit, and ``unrelated_consistent`` (1-5): did everything else stay the same?

Catch items (per list): an identical pair (correct: no difference), a symbolic +7-semitone
transposition of an original clip (correct: the transposed clip is higher), and a symbolic
doubling of the middle section (correct: the doubled clip is longer).

Committed (no lyrics): ``items.csv`` (answer key with realized measurements), ``lists/list_<k>.csv``
(what participants see, blinded), ``response_schema.csv``. Lyric-bearing texts, MIDI and audio go to
``materials/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .common import OUT_ROOT, REPORT_DIR, jdump

STUDY = REPORT_DIR / "listening_study"
MAT = STUDY / "materials"
GEN = OUT_ROOT / "gen" / "qwen_e3b"
SALT = "qwenabc-listening-2026"
SEED = 20260925
MAX_EXCERPT_BARS = 24

CONDITIONS = {  # condition -> (question, edit description shown in block 2)
    "bars_p4": ("longer_section", "the middle section should be 4 bars LONGER"),
    "bars_m4": ("longer_section", "the middle section should be 4 bars SHORTER"),
    "key_p5": ("higher_key", "the song should be transposed UP by 5 semitones (a perfect fourth)"),
    "key_m3": ("higher_key", "the song should be transposed DOWN by 3 semitones (a minor third)"),
    "lyrics_sec": ("matches_lyrics", "the middle section should sing the NEW words shown on screen"),
}
QUESTIONS = {
    "longer_section": "Both clips contain the same run of sections. In which clip is the MIDDLE section longer? (A / B / no difference)",
    "higher_key": "Which clip is in the higher key, i.e. sounds higher overall? (A / B / no difference)",
    "matches_lyrics": "Which clip sings the words shown on screen in its middle section? (A / B / neither)",
    "plausibility_A": "How musically plausible is clip A as a pop song excerpt? (1 = not at all ... 5 = completely)",
    "plausibility_B": "How musically plausible is clip B? (1-5)",
    "edit_happened": "Compared with the ORIGINAL, did the EDITED clip carry out the requested edit? (yes / partly / no)",
    "unrelated_consistent": "Apart from the requested edit, did the rest of the music stay the same? (1 = completely different ... 5 = identical)",
}


def h(*parts) -> str:
    return hashlib.sha1("|".join(map(str, (SALT,) + parts)).encode()).hexdigest()[:10]


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ excerpts and symbolic edits
def excerpt(song, s_from: int, s_to: int, max_bars: int = MAX_EXCERPT_BARS):
    """Sections s_from..s_to (inclusive) as a new Song starting at tick 0, capped at max_bars
    around the middle of the range."""
    from qwen_abc.canonical import Chord, Note, Section, Song, normalize_chords
    s_from = max(0, s_from)
    s_to = min(len(song.sections) - 1, s_to)
    b0 = song.sections[s_from].start_bar
    b1 = song.sections[s_to].start_bar + song.sections[s_to].num_bars
    if b1 - b0 > max_bars:   # keep the middle section whole, trim the neighbours
        mid = song.sections[(s_from + s_to) // 2] if s_to > s_from else song.sections[s_from]
        spare = max(max_bars - mid.num_bars, 0)
        b0 = max(b0, mid.start_bar - spare // 2)
        b1 = min(b1, max(mid.start_bar + mid.num_bars + (spare - spare // 2), b0 + 1))
    starts = song.bar_starts() + [song.total_ticks]
    t0, t1 = starts[b0], starts[min(b1, len(starts) - 1)]
    notes = [Note(n.onset - t0, min(n.duration, t1 - n.onset), n.pitch, n.lyric, n.melisma, n.line)
             for n in song.notes if t0 <= n.onset < t1]
    if notes and notes[0].melisma:
        notes[0].melisma = False
    chords = [Chord(max(c.onset, t0) - t0, 0, c.symbol) for c in song.chords if c.onset < t1 and c.offset > t0]
    secs = []
    for s in song.sections:
        a, e = max(s.start_bar, b0), min(s.start_bar + s.num_bars, b1)
        if e > a:
            secs.append(Section(s.label, a - b0, e - a))
    out = Song(song.song_id, song.meter_num, song.tempo_bpm, song.key, list(song.bar_beats[b0:b1]), secs, notes, [])
    out.chords = normalize_chords(chords, out.total_ticks)
    return out


def transpose_song(song, k: int):
    from qwen_abc.theory import canonical_key, parse_key_name, transpose_chord_symbol
    out = copy.deepcopy(song)
    for n in out.notes:
        n.pitch += k
    names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    pk = parse_key_name(song.key)
    if pk:
        out.key = canonical_key(f"{names[(pk[0] + k) % 12]} {pk[1]}")
    for c in out.chords:
        c.symbol = transpose_chord_symbol(c.symbol, k, out.key)
    return out


def double_middle(song):
    """Repeat the middle section once (catch item: unmistakably longer)."""
    from qwen_abc.canonical import Chord, Note, Section, Song, normalize_chords
    if len(song.sections) < 1:
        return copy.deepcopy(song)
    mi = len(song.sections) // 2
    m = song.sections[mi]
    starts = song.bar_starts() + [song.total_ticks]
    a, e = starts[m.start_bar], starts[m.start_bar + m.num_bars]
    L = e - a
    shift = lambda t: t if t < e else t + L  # noqa: E731
    notes = [Note(shift(n.onset), n.duration, n.pitch, n.lyric, n.melisma, n.line) for n in song.notes]
    notes += [Note(n.onset + L, n.duration, n.pitch, n.lyric, n.melisma, n.line) for n in song.notes if a <= n.onset < e]
    chords = [Chord(shift(c.onset), 0, c.symbol) for c in song.chords] + [Chord(c.onset + L, 0, c.symbol) for c in song.chords if a <= c.onset < e]
    secs = []
    for i, s in enumerate(song.sections):
        start = s.start_bar + (m.num_bars if i > mi else 0)
        secs.append(Section(s.label, start, s.num_bars + (m.num_bars if i == mi else 0)))
    bb = list(song.bar_beats[: m.start_bar + m.num_bars]) + list(song.bar_beats[m.start_bar: m.start_bar + m.num_bars]) + list(song.bar_beats[m.start_bar + m.num_bars:])
    out = Song(song.song_id, song.meter_num, song.tempo_bpm, song.key, bb, secs, sorted(notes, key=lambda n: n.onset), [])
    out.chords = normalize_chords(sorted(chords, key=lambda c: c.onset), out.total_ticks)
    return out


# ------------------------------------------------------------------ item selection
def eligible(cond: str) -> List[str]:
    out = []
    for p in sorted((GEN / cond).glob("*_S1.json")):
        sid = p.stem.rsplit("_", 1)[0]
        a, b = _load(GEN / "orig" / f"{sid}_S1.json"), _load(p)
        if a and b and a.get("song") and b.get("song") and b.get("meta", {}).get("target_section") is not None:
            out.append(sid)
    return out


def realized(cond: str, a: dict, b: dict) -> Dict[str, object]:
    """What the edit actually did, measured on the two generations (answer-key side only)."""
    from qwen_abc.canonical import Song
    sa, sb = Song.from_json(a["song"]), Song.from_json(b["song"])
    t = b["meta"]["target_section"]
    r = {"target_section": t}
    if cond.startswith("bars"):
        ba = sa.sections[t].num_bars if t < len(sa.sections) else None
        bb = sb.sections[t].num_bars if t < len(sb.sections) else None
        r.update(orig_bars=ba, edited_bars=bb, requested_bars=b["meta"].get("requested_bars"),
                 complied=bb == b["meta"].get("requested_bars"))
    elif cond.startswith("key"):
        pa = sum(n.pitch for n in sa.notes) / max(len(sa.notes), 1)
        pb = sum(n.pitch for n in sb.notes) / max(len(sb.notes), 1)
        k = b["meta"]["semitones"]
        r.update(orig_key=sa.key, edited_key=sb.key, requested_key=b["meta"]["requested_key"],
                 mean_pitch_shift=round(pb - pa, 2), complied=sb.key == b["meta"]["requested_key"],
                 register_agrees_with_request=(pb - pa) * k > 0)
    else:
        from .interventions import section_lyric_recall_of, syllables_of
        new = syllables_of(b["spec"]["sections"][t])
        old = syllables_of(a["spec"]["sections"][t])
        r.update(new_recall_edited=round(section_lyric_recall_of(sb, t, new), 3),
                 new_recall_orig=round(section_lyric_recall_of(sa, t, new), 3),
                 old_recall_edited=round(section_lyric_recall_of(sb, t, old), 3),
                 complied=section_lyric_recall_of(sb, t, new) >= 0.8)
    return r


def build(per_condition: int, n_lists: int) -> dict:
    from qwen_abc.canonical import Song
    MAT.mkdir(parents=True, exist_ok=True)
    (STUDY / "lists").mkdir(parents=True, exist_ok=True)
    items, stim = [], {}
    skipped = {}
    for cond, (question, edit_text) in CONDITIONS.items():
        pool = eligible(cond)
        skipped[cond] = {"eligible": len(pool)}
        chosen = sorted(pool, key=lambda s: h("pick", cond, s))[:per_condition]
        for sid in chosen:
            a = _load(GEN / "orig" / f"{sid}_S1.json")
            b = _load(GEN / cond / f"{sid}_S1.json")
            t = b["meta"]["target_section"]
            sa, sb = Song.from_json(a["song"]), Song.from_json(b["song"])
            lo, hi = (t - 1, t + 1) if question != "higher_key" else (t, t)
            ea, eb = excerpt(sa, lo, hi), excerpt(sb, lo, hi)
            iid = h("item", cond, sid)
            fa, fb = h("stim", iid, "orig"), h("stim", iid, "edit")
            stim[fa] = ea
            stim[fb] = eb
            text = None
            if question == "matches_lyrics":
                from .interventions import syllables_of
                text = "".join(syllables_of(b["spec"]["sections"][t]))
            items.append({"item_id": iid, "kind": "test", "condition": cond, "question": question, "song_id": sid,
                          "stim_orig": fa, "stim_edit": fb, "edit_text": edit_text, "lyrics_shown": text,
                          **{f"realized_{k}": v for k, v in realized(cond, a, b).items()}})
    # catch items from original clips of songs not used as test items (deterministic)
    used = {i["song_id"] for i in items}
    pool = [s for s in sorted({p.stem.rsplit("_", 1)[0] for p in (GEN / "orig").glob("*_S1.json")}) if s not in used]
    pool = sorted(pool, key=lambda s: h("catch", s))
    catch_kinds = ["catch_identical", "catch_transposed", "catch_doubled"]
    for j, kind in enumerate(catch_kinds * max(1, n_lists // 2)):
        if j >= len(pool):
            break
        sid = pool[j]
        a = _load(GEN / "orig" / f"{sid}_S1.json")
        if not a or not a.get("song"):
            continue
        s = Song.from_json(a["song"])
        mid = len(s.sections) // 2
        e = excerpt(s, mid - 1, mid + 1)
        other = {"catch_identical": e, "catch_transposed": transpose_song(e, 7), "catch_doubled": double_middle(e)}[kind]
        iid = h("item", kind, sid)
        fa, fb = h("stim", iid, "orig"), h("stim", iid, "edit")
        stim[fa], stim[fb] = e, other
        q = {"catch_identical": "longer_section", "catch_transposed": "higher_key", "catch_doubled": "longer_section"}[kind]
        items.append({"item_id": iid, "kind": "catch", "condition": kind, "question": q, "song_id": sid,
                      "stim_orig": fa, "stim_edit": fb, "edit_text": None, "lyrics_shown": None,
                      "realized_complied": kind != "catch_identical"})
    # write stimuli as canonical-song JSON (materials; they carry lyrics)
    (MAT / "stimuli").mkdir(parents=True, exist_ok=True)
    for name, song in stim.items():
        jdump(song.to_json(), MAT / "stimuli" / f"{name}.song.json", indent=None)
    jdump({i["item_id"]: i["lyrics_shown"] for i in items if i["lyrics_shown"]}, MAT / "lyrics_shown.json")
    # answer key (committed): no lyrics
    key_cols = ["item_id", "kind", "condition", "question", "song_id", "stim_orig", "stim_edit", "edit_text"]
    extra = sorted({k for i in items for k in i if k.startswith("realized_")})
    with open(STUDY / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=key_cols + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    # lists: order counterbalanced (item index + list parity) and presentation order shuffled per list
    lists = {}
    for k in range(n_lists):
        rng = random.Random(SEED + k)
        order = list(range(len(items)))
        rng.shuffle(order)
        rows = []
        for trial, idx in enumerate(order):
            it = items[idx]
            edit_first = (idx + k) % 2 == 0
            A, B = (it["stim_edit"], it["stim_orig"]) if edit_first else (it["stim_orig"], it["stim_edit"])
            rows.append({"list_id": k, "trial": trial + 1, "item_id": it["item_id"], "question": it["question"],
                         "clip_A": f"{A}.wav", "clip_B": f"{B}.wav", "edited_is": "A" if edit_first else "B",
                         "original_clip": f"{it['stim_orig']}.wav", "edited_clip": f"{it['stim_edit']}.wav",
                         "edit_text": it["edit_text"] or "", "shows_lyrics": bool(it["lyrics_shown"])})
        with open(STUDY / "lists" / f"list_{k}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lists[k] = len(rows)
    with open(STUDY / "response_schema.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["column", "type", "description"])
        for row in (("participant_id", "str", "pseudonymous id"), ("list_id", "int", "which lists/list_<k>.csv"),
                    ("trial", "int", "trial number within the list"), ("item_id", "str", "from the list"),
                    ("question_id", "str", "one of: " + ", ".join(QUESTIONS)),
                    ("response", "str", "A|B|none for 2AFC; yes|partly|no for edit_happened; 1-5 for Likert"),
                    ("rt_ms", "int", "response time"), ("timestamp", "str", "ISO-8601")):
            w.writerow(row)
    jdump({"questions": QUESTIONS, "conditions": {k: v[0] for k, v in CONDITIONS.items()},
           "eligible_songs": skipped, "n_items": len(items), "n_test": sum(i["kind"] == "test" for i in items),
           "n_catch": sum(i["kind"] == "catch" for i in items), "lists": lists, "seed": SEED},
          STUDY / "design.json")
    return {"items": len(items), "eligible": skipped, "lists": lists}


# ------------------------------------------------------------------ audio (pilot)
FLUIDSYNTH = "/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/fluidsynth"
SOUNDFONT = "/gscratch/ark/pingw220/miniconda3/envs/beatbk-render/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2"


def render_cpu(pilot: int) -> List[str]:
    """For the first ``pilot`` items of list 0: backing MIDI (chords + a soft piano doubling of the
    melody), a fluidsynth render of it, and the FastSinger input pair; returns the stimulus names."""
    import subprocess
    from qwen_abc.canonical import Song
    from qwen_abc.fastsinger import write_fastsinger_inputs
    from qwen_abc.midi import song_to_midi
    rows = list(csv.DictReader(open(STUDY / "lists" / "list_0.csv", encoding="utf-8")))
    items = {r["item_id"]: r for r in csv.DictReader(open(STUDY / "items.csv", encoding="utf-8"))}
    # a balanced pilot: round-robin over conditions (catch items included)
    by_cond: Dict[str, List[str]] = {}
    for r in rows:
        by_cond.setdefault(items[r["item_id"]]["condition"], []).append(r["item_id"])
    pick, i = [], 0
    while len(pick) < pilot and any(by_cond.values()):
        c = sorted(by_cond)[i % len(by_cond)]
        if by_cond[c]:
            pick.append(by_cond[c].pop(0))
        i += 1
        if i > 10 * pilot:
            break
    out = MAT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for iid in pick:
        for name in (items[iid]["stim_orig"], items[iid]["stim_edit"]):
            if name in names:
                continue
            song = Song.from_json(json.loads((MAT / "stimuli" / f"{name}.song.json").read_text(encoding="utf-8")))
            song_to_midi(song, str(out / f"{name}.backing.mid"), chord_velocity=60)
            subprocess.run([FLUIDSYNTH, "-ni", "-g", "0.6", "-r", "44100", "-F", str(out / f"{name}.backing.wav"),
                            SOUNDFONT, str(out / f"{name}.backing.mid")], check=True, capture_output=True, timeout=600)
            write_fastsinger_inputs(song, str(out / f"{name}.fs.mid"), str(out / f"{name}.fs.txt"))
            names.append(name)
    (out / "pilot_stimuli.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    jdump({"items": pick, "stimuli": names}, STUDY / "pilot.json")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "render-cpu"])
    ap.add_argument("--per-condition", type=int, default=12)
    ap.add_argument("--lists", type=int, default=4)
    ap.add_argument("--pilot", type=int, default=16)
    args = ap.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.per_condition, args.lists), indent=1))
    else:
        print(f"{len(render_cpu(args.pilot))} pilot stimuli prepared in {MAT / 'audio'}")


if __name__ == "__main__":
    main()
