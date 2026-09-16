#!/usr/bin/env python
"""Make a sample directory viewable: plain ABC, MusicXML and engraved SVG next to every .abc.

  scripts/build_viewable_samples.py experiments/listen_now_<ts> [--pages 2]

For each `*.abc` it writes, in the same folder:
  <name>.clean.abc   ABC-v1 text (ABC-v2 counters and section comments stripped) for ABC tools
  <name>.musicxml    melody + lyrics + chord symbols + per-bar meters + section marks (MuseScore opens this)
  <name>.p1.svg …    engraved pages (verovio), for looking at without any software

MuseScore reads the .musicxml and the .mid directly; .abc needs a plugin, so the
MusicXML is the file to open.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_musicxml import convert  # noqa: E402
from qwen_abc.abc_v2 import strip_v2  # noqa: E402

SVG_OPTIONS = {
    "pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": True,
    "footer": "none", "header": "none", "lyricSize": 4.5, "spacingStaff": 8,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--pages", type=int, default=2, help="engrave at most this many pages per score")
    args = ap.parse_args()
    import verovio

    tk = verovio.toolkit()
    tk.setOptions(SVG_OPTIONS)
    n_x = n_s = 0
    for abc in sorted(args.sample_dir.rglob("*.abc")):
        if abc.name.endswith(".clean.abc") or abc.stat().st_size == 0:
            continue
        text = abc.read_text(encoding="utf-8")
        clean = strip_v2(text)
        if clean != text:
            abc.with_suffix(".clean.abc").write_text(clean, encoding="utf-8")
        try:
            out = convert(abc, abc.parent)
        except SystemExit as exc:
            print(f"skip {abc}: {exc}")
            continue
        n_x += 1
        if tk.loadFile(str(out)):
            for page in range(1, min(tk.getPageCount(), args.pages) + 1):
                out.with_suffix(f".p{page}.svg").write_text(tk.renderToSVG(page), encoding="utf-8")
                n_s += 1
        else:
            print(f"verovio could not load {out}")
    print(f"VIEWABLE_DONE musicxml={n_x} svg_pages={n_s}")


if __name__ == "__main__":
    main()
