#!/usr/bin/env python
"""Coverage and validity accounting for a round-2 MIDI-LLM baseline run (both modes).

  scripts/midi_llm_r2_report.py --base experiments/midi_llm_r2_<ts> --data-dir data/generated/abc_v2_<ts> [--json OUT]

Classifies every requested song from the per-song logs:
  generated_valid / generated_invalid (its own validator rejected it) / refused_input (G2P: code-switched
  English, unsupported characters) / context_overflow / timeout_or_other,
and reports seconds per song and the validator error histogram. Conversion of the generated songs into
the shared metric format is done by scripts/midi_llm_baseline.py convert.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def classify(log_text: str, exit_code: int, gen_dir: Path) -> str:
    if "outside the frozen phoneme inventory" in log_text or "no phonemes produced" in log_text:
        return "refused_input_g2p"
    if "exceeds checkpoint context" in log_text:
        return "context_overflow"
    if "syllables in a section" in log_text:
        return "refused_input_section_too_long"
    if (gen_dir / "parsed_leadsheet.json").exists():
        v = gen_dir / "validation.json"
        ok = False
        if v.exists():
            d = json.loads(v.read_text())
            ok = not d.get("errors")
        return "generated_valid" if ok else "generated_invalid"
    if exit_code == 124:
        return "timeout"
    return "failed_other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    out = {}
    specs = sorted(p.stem for p in (args.base / "specs").glob("*.json"))
    for mode in ("A", "B"):
        d = args.base / f"mode{mode}"
        if not d.exists():
            continue
        cls, errs, secs, retried = Counter(), Counter(), [], 0
        per_song = {}
        for sid in specs:
            done = d / "logs" / f"{sid}.done"
            log = d / "logs" / f"{sid}.txt"
            if not done.exists():
                cls["not_run"] += 1
                per_song[sid] = "not_run"
                continue
            text = done.read_text()
            code = int(re.search(r"exit=(\d+)", text).group(1))
            secs.append(int(re.search(r"seconds=(\d+)", text).group(1)))
            retried += "retry_max_new" in text
            c = classify(log.read_text(errors="ignore") if log.exists() else "", code, d / "gen" / sid)
            cls[c] += 1
            per_song[sid] = c
            v = d / "gen" / sid / "validation.json"
            if v.exists():
                for e in json.loads(v.read_text()).get("errors", []):
                    errs[e if isinstance(e, str) else json.dumps(e)[:60]] += 1
        secs.sort()
        out[f"mode{mode}"] = {
            "songs_requested": len(specs), "classes": dict(cls), "retried_for_context": retried,
            "seconds_per_song": {"median": secs[len(secs) // 2] if secs else None, "mean": sum(secs) / len(secs) if secs else None,
                                 "max": max(secs, default=None), "total_gpu_hours": round(sum(secs) / 3600, 2)},
            "validator_errors": dict(errs.most_common(12)),
            "per_song": per_song,
        }
        print(f"mode{mode}: {dict(cls)}  retried={retried}  median {out[f'mode{mode}']['seconds_per_song']['median']}s")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
