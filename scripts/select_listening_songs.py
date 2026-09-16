#!/usr/bin/env python
"""Pick representative test songs for listening, from reference features only (never model output).

  scripts/select_listening_songs.py --v2-dir data/generated/abc_v2_<ts> > reports/listening_song_ids.json

One song per category; ties broken by song id; a song is used once. Categories:
simple verse/chorus, long multi-section, has bridge, repeated chorus (>=3), high lyric density,
melisma-heavy, code-switched (latin syllables), plus one random song (sha256 order).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_abc.canonical import Song  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--v2-dir", type=Path, required=True)
ap.add_argument("--clean", type=Path, default=Path("reports/clean_test_song_ids.txt"))
args = ap.parse_args()
clean = {l.strip() for l in open(args.clean) if l.strip()}
rows = [json.loads(l) for l in open(args.v2_dir / "songs_test.jsonl", encoding="utf-8")]
feat = {}
for r in rows:
    s = Song.from_json(r["song"])
    labels = [x["label"] for x in r["spec"]["sections"]]
    sylls = sum(len(n.lyric) for n in s.notes if n.lyric)
    latin = any(re.search(r"[a-z]", x) for n in s.notes if n.lyric for x in n.lyric)
    feat[r["song_id"]] = {
        "clean": r["song_id"] in clean, "pathological": bool(r["pathologies"]), "n_sections": len(labels), "labels": labels,
        "simple": set(labels) <= {"intro", "verse", "chorus", "outro", "instrumental"} and len(labels) <= 7,
        "bridge": "bridge" in labels, "choruses": labels.count("chorus"),
        "syll_per_bar": sylls / max(len(s.bar_beats), 1),
        "melisma": sum(1 for n in s.notes if n.melisma) / max(len(s.notes), 1), "latin": latin,
    }
ok = lambda sid: not feat[sid]["pathological"]  # noqa: E731
cats = [
    ("simple_verse_chorus", lambda f: f["simple"], lambda f: (not f["clean"], f["n_sections"])),
    ("long_multi_section", lambda f: True, lambda f: (-f["n_sections"],)),
    ("bridge", lambda f: f["bridge"], lambda f: (not f["clean"], -f["n_sections"])),
    ("repeated_chorus", lambda f: f["choruses"] >= 3, lambda f: (not f["clean"], -f["choruses"])),
    ("high_lyric_density", lambda f: True, lambda f: (-f["syll_per_bar"],)),
    ("melisma_heavy", lambda f: True, lambda f: (-f["melisma"],)),
    ("code_switching", lambda f: f["latin"], lambda f: (not f["clean"], -f["n_sections"])),
]
used, out = set(), []
for name, cond, key in cats:
    cands = sorted((sid for sid in feat if ok(sid) and sid not in used and cond(feat[sid])), key=lambda s: (key(feat[s]), s))
    if cands:
        used.add(cands[0])
        out.append({"category": name, "song_id": cands[0], **{k: feat[cands[0]][k] for k in ("clean", "n_sections", "choruses", "latin")},
                    "syll_per_bar": round(feat[cands[0]]["syll_per_bar"], 2), "melisma": round(feat[cands[0]]["melisma"], 3)})
rand = sorted((s for s in feat if ok(s) and s not in used), key=lambda s: hashlib.sha256(s.encode()).hexdigest())[0]
out.append({"category": "random", "song_id": rand, "clean": feat[rand]["clean"], "n_sections": feat[rand]["n_sections"]})
print(json.dumps(out, indent=1))
