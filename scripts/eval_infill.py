#!/usr/bin/env python
"""Masked late-section reconstruction on held-out songs (E3's second task).

  scripts/eval_infill.py --checkpoint CKPT --data-dir data/generated/abc_v2_<ts> --output-dir OUT

For each test song's infill example (infill_test.jsonl): generate the missing section, parse it under
the song's ABC header, and score
* bars exact (label + bar count of the missing section), counter consistency,
* lyric recall of that section's syllables,
* motif similarity (transposition-invariant) of the generated section to (a) the reference target and
  (b) the nearest earlier section with the same label (visible in the prompt), next to the same
  similarity for the reference target itself (how much the real song repeats).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc import parse_abc  # noqa: E402
from qwen_abc.abc_v2 import counter_report  # noqa: E402
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.generate import generate_batch, load_for_generation  # noqa: E402
from qwen_abc.longrange import split_abc_sections  # noqa: E402
from qwen_abc.metrics import aggregate, lcs_len, motif_similarity  # noqa: E402
from qwen_abc.prompt import split_syllables  # noqa: E402


def section_notes(song: Song, idx: int):
    starts = song.bar_starts()
    s = song.sections[idx]
    a = starts[s.start_bar]
    e = s.start_bar + s.num_bars
    b = starts[e] if e < len(starts) else song.total_ticks
    return [n for n in song.notes if a <= n.onset < b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    songs = {json.loads(l)["song_id"]: json.loads(l) for l in open(args.data_dir / "songs_test.jsonl", encoding="utf-8")}
    ex = [json.loads(l) for l in open(args.data_dir / "infill_test.jsonl", encoding="utf-8")]
    gen_dir = args.output_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    pending = [e for e in ex if not (gen_dir / f"{e['song_id']}.json").exists()]
    if pending:
        model, tok = load_for_generation(args.checkpoint)
        pending.sort(key=lambda e: e["prompt_tokens"])
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i: i + args.batch_size]
            outs = generate_batch(model, tok, [e["prompt"] for e in chunk], 1000 + i, 4096, args.temperature, args.top_p, 12288)
            for e, g in zip(chunk, outs):
                (gen_dir / f"{e['song_id']}.json").write_text(json.dumps({"song_id": e["song_id"], "target_section": e["target_section"],
                                                                           "generation": g["text"], "hit_eos": g["hit_eos"]}, ensure_ascii=False), encoding="utf-8")
    rows = []
    for e in ex:
        g = json.loads((gen_dir / f"{e['song_id']}.json").read_text(encoding="utf-8"))
        r = songs[e["song_id"]]
        t = e["target_section"]
        ref_song = Song.from_json(r["song"])
        spec_sec = r["spec"]["sections"][t]
        header, blocks = split_abc_sections(r["abc"])
        res = parse_abc(header + g["generation"], e["song_id"])
        m = {"parse_ok": float(res.ok), "strict_ok": float(res.strict_ok), "hit_eos": float(g["hit_eos"])}
        earlier = [i for i in range(t) if r["spec"]["sections"][i]["label"] == spec_sec["label"]]
        ref_target = section_notes(ref_song, t)
        ref_prev = section_notes(ref_song, earlier[-1]) if earlier else None
        if res.ok and res.song.sections:
            gs = res.song
            m["one_section"] = float(len(gs.sections) == 1)
            m["label_match"] = float(gs.sections[0].label == spec_sec["label"])
            m["bars_exact"] = float(len(gs.bar_beats) == spec_sec["bars"])
            m["bars_abs_error"] = abs(len(gs.bar_beats) - spec_sec["bars"])
            want = [x for line in spec_sec["lines"] for x in split_syllables(line)]
            sung = [x for n in gs.notes if n.lyric for x in n.lyric]
            m["section_lyric_recall"] = lcs_len(want, sung) / max(len(want), 1)
            cr = counter_report(header + g["generation"])
            m["counter_self_consistent_frac"] = cr["counter_self_consistent_frac"]
            notes = gs.notes
            sim_t = motif_similarity(notes, ref_target)
            if sim_t is not None:
                m["motif_sim_to_reference_target"] = sim_t
            if ref_prev is not None:
                s1 = motif_similarity(notes, ref_prev)
                s0 = motif_similarity(ref_target, ref_prev)
                if s1 is not None:
                    m["motif_sim_to_earlier_same_label"] = s1
                if s0 is not None:
                    m["reference_target_sim_to_earlier_same_label"] = s0
            if notes:
                m["pitch_range"] = max(n.pitch for n in notes) - min(n.pitch for n in notes)
        rows.append(m)
    summary = {"checkpoint": args.checkpoint, "n": len(rows), "metrics": aggregate(rows)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: round(v["mean"], 4) for k, v in summary["metrics"].items()}, indent=1))
    print("INFILL_DONE")


if __name__ == "__main__":
    main()
