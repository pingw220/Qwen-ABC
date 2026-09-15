#!/usr/bin/env python
"""MIDI-LLM (one-stage v4-cd) baseline on the same held-out songs, scored with the same metrics.

  specs:    write MIDI-LLM --from-spec JSON files for the chosen test songs
  convert:  parsed_leadsheet.json (+ song plan, request) -> canonical Song -> metrics,
            in the same row format as scripts/generate_eval.py (so compare_results.py works)

MIDI-LLM is run with its own CLI and its recommended v4 decode flags, from its
own repository and environment (read-only use). Its spec carries the same
condition as the Qwen-ABC prompt -- task, section labels, bars per section,
lyric lines per section, key, tempo, meter -- except per-bar beat counts: its
click grid is always regular.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.canonical import TICKS_PER_BEAT, Chord, Note, Section, Song, normalize_chords  # noqa: E402
from qwen_abc.metrics import song_metrics  # noqa: E402
from qwen_abc.theory import respell_chord_symbol, corpus_chord_to_abc  # noqa: E402

TRAINED = ("intro", "verse", "chorus", "bridge", "instrumental", "outro", "other")


def pick(data_dir: Path, split: str, n: int):
    rows = [json.loads(l) for l in open(data_dir / f"songs_{split}.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["song_id"])
    step = max(len(rows) // n, 1)
    return rows[::step][:n]


def cmd_specs(args) -> None:
    out = args.output_dir / "specs"
    out.mkdir(parents=True, exist_ok=True)
    notes = {}
    for r in pick(args.data_dir, args.split, args.num_songs):
        spec = r["spec"]
        sections = []
        for s in spec["sections"]:
            label = s["label"] if s["label"] in TRAINED else "verse"
            if label != s["label"]:
                notes[r["song_id"]] = notes.get(r["song_id"], []) + [f"{s['label']}->verse"]
            row = {"label": label, "bars": s["bars"]}
            if s["lines"]:
                row["lines"] = s["lines"]
            sections.append(row)
        mspec = {"song_id": r["song_id"], "task": "lyrics_to_leadsheet", "prompt": "",
                 "tempo_bpm": spec["tempo_bpm"], "meter": spec["meter"], "key": spec["key"], "sections": sections}
        (out / f"{r['song_id']}.json").write_text(json.dumps(mspec, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output_dir / "spec_label_substitutions.json").write_text(json.dumps(notes, indent=1))
    print(f"wrote {len(list(out.glob('*.json')))} specs; label substitutions in {len(notes)} songs")


def section_texts(plan, lyrics_lines):
    texts = []
    for row in plan:
        sec = []
        for idx in row.get("lyric_line_indices", []):
            if idx < len(lyrics_lines):
                line = lyrics_lines[idx].strip()
                sec.extend([x for x in line.split() if x] if " " in line else [x for x in line if not x.isspace()])
        texts.append(sec)
    return texts


def to_song(gen_dir: Path, ref_spec: dict) -> Song:
    parsed = json.loads((gen_dir / "parsed_leadsheet.json").read_text(encoding="utf-8"))
    plan = json.loads((gen_dir / "parsed_song_plan.json").read_text(encoding="utf-8"))
    request = json.loads((gen_dir / "request.json").read_text(encoding="utf-8"))
    lines = request["lyrics"].split("\n")
    texts = section_texts(plan, lines)
    bpm = float(ref_spec["tempo_bpm"])
    beats_per_bar = int(ref_spec["meter"].split("/")[0])
    ticks_per_sec = bpm / 60.0 * TICKS_PER_BEAT
    notes, chords, sections, bar_beats = [], [], [], []
    offset = 0
    for i, sec in enumerate(parsed):
        n_bars = len(sec.get("bar_markers") or []) or int(round(sec["duration_sec"] * bpm / 60.0 / beats_per_bar))
        sections.append(Section(sec.get("label") or "other", len(bar_beats), n_bars))
        bar_beats.extend([beats_per_bar] * n_bars)
        sec_ticks = n_bars * beats_per_bar * TICKS_PER_BEAT
        sylls = texts[i] if i < len(texts) else []
        align = {a["note_id"]: a for a in sec.get("alignment", [])}
        last_syll = None
        for m in sorted(sec.get("melody_notes", []), key=lambda m: m["onset_sec"]):
            on = offset + int(round(m["onset_sec"] * ticks_per_sec))
            dur = max(int(round(m["duration_sec"] * ticks_per_sec)), 1)
            if notes and on <= notes[-1].onset:
                continue
            if notes and on < notes[-1].offset:
                notes[-1].duration = on - notes[-1].onset
            a = align.get(m["note_id"], {})
            idx = a.get("syllable_index")
            note = Note(on, dur, int(m["pitch"]))
            kind = a.get("kind")
            if kind == "continuation" and idx is not None and idx == last_syll:
                note.melisma = True  # align:continuation:NNN continuing the syllable just sung
            elif kind == "syllable" and idx is not None and idx < len(sylls):
                note.lyric = (sylls[idx],)
                last_syll = idx
            elif kind == "continuation" and idx is not None and idx < len(sylls):
                note.lyric = (sylls[idx],)  # continuation of a syllable never attacked: an attack
                last_syll = idx
            else:
                last_syll = None  # no_lyric / no_lyric_continuation
            notes.append(note)
        for c in sec.get("chord_events", []):
            sym, _ = corpus_chord_to_abc(c.get("symbol", "N"), c.get("root"), c.get("quality"), c.get("bass"))
            chords.append(Chord(offset + int(round(c["onset_sec"] * ticks_per_sec)), 0,
                                respell_chord_symbol(sym, ref_spec.get("key"))))
        offset += sec_ticks
    total = sum(bar_beats) * TICKS_PER_BEAT
    notes = [n for n in notes if n.onset < total]
    for n in notes:
        n.duration = min(n.duration, total - n.onset)
    melisma_ok = []
    for n in notes:  # a melisma must follow a sung note
        if n.melisma and not (melisma_ok and (melisma_ok[-1].lyric or melisma_ok[-1].melisma)):
            n.melisma = False
        melisma_ok.append(n)
    return Song(gen_dir.name, beats_per_bar, int(round(bpm)), ref_spec.get("key"), bar_beats, sections, notes,
                normalize_chords(chords, total))


def cmd_convert(args) -> None:
    refs = {r["song_id"]: r for r in pick(args.data_dir, args.split, args.num_songs)}
    out = args.output_dir / "eval_test" / "generations"
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for sid, r in sorted(refs.items()):
        gen_dir = args.output_dir / "gen" / sid
        row = {"song_id": sid, "kind": "main", "seed": 0, "prompt": json.dumps(r["spec"], ensure_ascii=False),
               "generation": "", "errors": {}, "hit_eos": False, "new_tokens": None, "seconds": None}
        val = gen_dir / "validation.json"
        if (gen_dir / "parsed_leadsheet.json").exists():
            v = json.loads(val.read_text()) if val.exists() else {}
            song = to_song(gen_dir, r["spec"])
            row.update(parse_ok=True, strict_ok=not v.get("errors"), midi_ok=bool(v.get("midi_valid")) and bool(song.notes),
                       errors={e if isinstance(e, str) else json.dumps(e): 1 for e in v.get("errors", [])},
                       hit_eos=True, song=song.to_json(), metrics=song_metrics(song, r["spec"]))
            ok += 1
        else:
            row.update(parse_ok=False, strict_ok=False, midi_ok=False)
        (out / f"{sid}_s0.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    print(f"converted {ok}/{len(refs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["specs", "convert"])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-songs", type=int, default=30)
    args = ap.parse_args()
    {"specs": cmd_specs, "convert": cmd_convert}[args.command](args)


if __name__ == "__main__":
    main()
