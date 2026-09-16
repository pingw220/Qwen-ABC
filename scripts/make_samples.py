#!/usr/bin/env python
"""Export listenable / viewable samples from evaluation outputs.

  scripts/make_samples.py --data-dir DATA --eval name=experiments/<run>/eval_test [--eval ...] \
      --output-dir experiments/samples_<ts> [--num 8] [--audio]

For each chosen test song:
  <out>/<song_id>/input.txt, reference.abc, reference.mid[, reference.wav]
  <out>/<song_id>/<name>/generated.abc, generated.mid[, generated.wav], metrics.json
Songs are the first N (sorted by id) that every evaluated model generated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.midi import song_to_midi  # noqa: E402

FLUIDSYNTH = "/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/fluidsynth"
SOUNDFONT = "/gscratch/ark/pingw220/miniconda3/envs/beatbk-render/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2"


def render(mid: Path, wav: Path) -> bool:
    try:
        subprocess.run([FLUIDSYNTH, "-ni", "-g", "0.8", "-r", "22050", "-F", str(wav), SOUNDFONT, str(mid)],
                       check=True, capture_output=True, timeout=600)
        return wav.exists()
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--eval", action="append", required=True, help="name=path/to/eval_dir")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--song-ids", type=Path, default=None, help="json list from scripts/select_listening_songs.py, or a text file of ids")
    args = ap.parse_args()
    evals = dict(e.split("=", 1) for e in args.eval)
    available = None
    for name, path in evals.items():
        ids = {p.name.rsplit("_s", 1)[0] for p in (Path(path) / "generations").glob("*_s0.json")}
        available = ids if available is None else available & ids
    if args.song_ids:
        raw = args.song_ids.read_text(encoding="utf-8")
        wanted = [r["song_id"] for r in json.loads(raw)] if raw.lstrip().startswith("[") else [l.strip() for l in raw.split() if l.strip()]
        missing = [s for s in wanted if s not in available]
        chosen = [s for s in wanted if s in available]
        if missing:
            print(f"note: {len(missing)} requested songs are missing from some run: {missing}")
    else:
        chosen = sorted(available)[: args.num]
    songs = {}
    with open(args.data_dir / f"songs_{args.split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            sid = line[len('{"song_id": "'):].split('"', 1)[0]
            if sid in chosen:
                songs[sid] = json.loads(line)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    index = ["# Samples", "", f"data: `{args.data_dir}`", ""]
    header = "| song | " + " | ".join(f"{n}: strict / lyric recall / chord-tone" for n in evals) + " |"
    index += [header, "|" + "---|" * (len(evals) + 1)]
    for sid in chosen:
        d = out / sid
        d.mkdir(exist_ok=True)
        ref = songs[sid]
        prompt = next(json.loads((Path(p) / "generations" / f"{sid}_s0.json").read_text())["prompt"]
                      for p in evals.values())
        (d / "input.txt").write_text(prompt, encoding="utf-8")
        (d / "reference.abc").write_text(ref["abc"], encoding="utf-8")
        song_to_midi(Song.from_json(ref["song"]), str(d / "reference.mid"))
        if args.audio:
            render(d / "reference.mid", d / "reference.wav")
        cells = []
        for name, path in evals.items():
            row = json.loads((Path(path) / "generations" / f"{sid}_s0.json").read_text(encoding="utf-8"))
            md = d / name
            md.mkdir(exist_ok=True)
            (md / "generated.abc").write_text(row["generation"], encoding="utf-8")
            m = row.get("metrics", {})
            (md / "metrics.json").write_text(json.dumps({k: row[k] for k in ("parse_ok", "strict_ok", "hit_eos", "errors")}
                                                        | {"metrics": m}, indent=1, ensure_ascii=False), encoding="utf-8")
            if row.get("song"):
                song_to_midi(Song.from_json(row["song"]), str(md / "generated.mid"))
                if args.audio:
                    render(md / "generated.mid", md / "generated.wav")
            cells.append(f"{int(row['strict_ok'])} / {m.get('lyric_recall', float('nan')):.2f} / "
                         f"{m.get('chord_tone_frac', float('nan')):.2f}")
        index.append(f"| {sid} | " + " | ".join(cells) + " |")
    (out / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} samples to {out}")


if __name__ == "__main__":
    main()
