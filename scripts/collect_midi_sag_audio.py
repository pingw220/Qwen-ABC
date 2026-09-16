#!/usr/bin/env python
"""Collect a MIDI-SAG render into the listening directory, as mp3.

  scripts/collect_midi_sag_audio.py --render-dir /gscratch/ark/.../midi_sag_<ts> \
      --sample-dir experiments/listen_now_<ts>

The renderer writes ~200 MB of wav per song (mix, backing, vocal and one wav per
generation window). This copies only what there is to listen to, encoded as mp3,
into ``<sample-dir>/<song>/<model>/``, next to the score and the MIDI that
produced it. The wavs stay where they are; nothing is deleted.

Sample directories are named ``<song_id>__<model>`` by
scripts/export_leadsheet_midi.py, which is how a render is matched back to the
lead sheet it came from.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FFMPEG = "/gscratch/ark/pingw220/miniconda3/envs/ace-step-15-rl/bin/ffmpeg"

# stem in the render dir -> (name in the sample dir, bitrate)
WANTED = {
    "mix": ("full_band.mp3", "128k"),
    "backing": ("backing_only.mp3", "96k"),
    "vocal": ("vocal_only.mp3", "96k"),
}


def encode(src: Path, dst: Path, bitrate: str, ffmpeg: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"ffmpeg failed on {src}: {proc.stderr.decode()[:200]}")
        return False
    return dst.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", type=Path, required=True)
    ap.add_argument("--sample-dir", type=Path, required=True, help="listening dir with <song>/<model>/ subdirs")
    ap.add_argument("--ffmpeg", default=FFMPEG)
    args = ap.parse_args()

    rows, missing = [], []
    for run in sorted(p for p in args.render_dir.iterdir() if p.is_dir()):
        if "__" not in run.name:
            continue
        song, model = run.name.split("__", 1)
        result = run / "result.json"
        state = json.loads(result.read_text()) if result.is_file() else {}
        if not state.get("ok"):
            missing.append({"run": run.name, "error": (state.get("error") or {}).get("type", "no result.json"),
                            "stopped_after": state.get("stopped_after")})
            continue
        out_dir = args.sample_dir / song / model
        if not out_dir.is_dir():
            missing.append({"run": run.name, "error": f"no sample dir {out_dir}"})
            continue
        made = {}
        for stem, (name, bitrate) in WANTED.items():
            src = run / f"{stem}.wav"
            if src.is_file() and encode(src, out_dir / name, bitrate, args.ffmpeg):
                made[stem] = str(out_dir / name)
        rows.append({"song": song, "model": model, "render_dir": str(run), "audio": made})
        print(f"{song}/{model}: {', '.join(sorted(made)) or 'nothing'}")

    index = {"render_dir": str(args.render_dir), "sample_dir": str(args.sample_dir),
             "collected": rows, "missing": missing}
    out = args.sample_dir / "midi_sag_audio.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COLLECT_DONE collected={len(rows)} missing={len(missing)} -> {out}")


if __name__ == "__main__":
    main()
