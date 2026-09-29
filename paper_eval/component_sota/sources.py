"""Melody and chord sources as one row format, so every metric and pipeline treats them alike.

A *source row* is ``{"song_id", "source", "seed", "ok", "song": canonical Song JSON | None,
"failure": str | None, "meta": {...}}``. Melody sources carry melody + lyrics + sections (chords
stripped); a lead sheet = melody source + chord source on the melody's bar grid.

Melody sources:
  ref          pseudo-reference (held-out SheetSage-Pro label)
  e3b          Qwen full lead sheet (paper-final E3b paired-seed samples) -> melody
  mel          Qwen melody-only
  csl_off      CSL-L2M official pretrained (lyrics-only checkpoint)
  csl_off_full CSL-L2M official pretrained (full-control checkpoint, controls from the reference)
  csl_rt       CSL-L2M retrained on our split (lyrics-only config)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional

from qwen_abc.canonical import Song

from ..common import OUT_ROOT, PAPER_FINAL_GEN, load_rows
from .formats import strip_chords

SEEDS = ("S1", "S2", "S3", "S4")
CS_ROOT = OUT_ROOT  # experiments/component_sota when run through scripts/component_sota/env.sh


def _row(song_id, source, seed, song: Optional[Song], failure=None, meta=None):
    return {"song_id": song_id, "source": source, "seed": seed, "ok": song is not None and bool(song.notes),
            "song": song.to_json() if song is not None else None, "failure": failure if song is None or not song.notes else None,
            "meta": meta or {}}


def _from_gen(path: Path, source: str, seed: str, keep_chords=False):
    sid = path.stem.rsplit("_", 1)[0]
    r = json.loads(path.read_text(encoding="utf-8"))
    if not r.get("song"):
        return _row(sid, source, seed, None, "parse_failure", {"strict_ok": False})
    s = Song.from_json(r["song"])
    return _row(sid, source, seed, s if keep_chords else strip_chords(s), None,
                {"strict_ok": bool(r.get("strict_ok")), "metrics": r.get("metrics", {}), "spec": r.get("spec")})


def iter_melodies(source: str, seed: str = "S1", condition: str = "orig", keep_chords: bool = False) -> Iterator[dict]:
    if source == "ref":
        for r in load_rows("test"):
            s = Song.from_json(r["song"])
            yield _row(r["song_id"], "ref", "ref", s if keep_chords else strip_chords(s), None, {"strict_ok": True, "spec": r["spec"]})
        return
    if source in ("e3b", "mel"):
        d = (PAPER_FINAL_GEN / "qwen_e3b" if source == "e3b" else CS_ROOT / "gen" / "qwen_mel") / condition
        for p in sorted(d.glob(f"*_{seed}.json")):
            yield _from_gen(p, source, seed, keep_chords)
        return
    if source.startswith("csl"):
        d = CS_ROOT / "csl" / source / condition
        for p in sorted(d.glob(f"*_{seed}.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            yield r
        return
    raise KeyError(source)


def melody_table(source: str, seed: str = "S1", condition: str = "orig") -> Dict[str, dict]:
    return {r["song_id"]: r for r in iter_melodies(source, seed, condition)}
