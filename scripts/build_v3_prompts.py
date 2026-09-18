#!/usr/bin/env python
"""ABC-v3 training files, derived from an existing ABC-v2 dataset.

  scripts/build_v3_prompts.py --v2-dir data/generated/abc_v2_<ts> --output-dir data/generated/abc_v3_<ts>

v3 changes the prompt only (syllable budget per section, continuation marks on
lyric lines a section boundary splits — see ``qwen_abc/abc_v3.py``). The lead
sheets, the cleaning, the split and every completion stay exactly as they are,
so a v3 run against E3b isolates one variable.

Deriving the files rather than rebuilding the corpus is what makes that
guarantee checkable: every completion here is asserted byte-identical to the
one in the v2 dataset, and the infill targets and mixture membership are the
same deterministic hashes of the song id, so the same songs contribute the same
tasks.

Output file names match the v2 dataset, so a config only changes ``data_dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_abc.abc_v3 import ABC_V3_VERSION, budget_report, spec_to_prompt_v3, strip_v3_marks  # noqa: E402
from qwen_abc.canonical import Song  # noqa: E402
from qwen_abc.longrange import GAP_LINE, choose_infill_target, infill_example  # noqa: E402
from qwen_abc.prompt import spec_syllables, split_syllables  # noqa: E402

COPY = ("songs_train.jsonl", "songs_validation.jsonl", "songs_test.jsonl",
        "split_manifest.json", "validation.json", "build_report.json", "cleaning_log.jsonl")


def selected(song_id: str, fraction: float) -> bool:
    """The membership test build_longrange_tasks.py uses, copied exactly.

    Copied rather than imported because that module is a script, not a package;
    the mixture check at the end of this build fails loudly if the two ever
    disagree.
    """
    return int.from_bytes(hashlib.sha256(f"infill-select:{song_id}".encode()).digest()[:8], "big") / 2 ** 64 < fraction


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--infill-fraction", type=float, default=0.5)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    args = ap.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    report = {"v3_version": ABC_V3_VERSION, "v2_dir": str(args.v2_dir), "infill_fraction": args.infill_fraction,
              "splits": {}}
    for split in ("train", "validation", "test"):
        songs = [json.loads(l) for l in open(args.v2_dir / f"songs_{split}.jsonl", encoding="utf-8")]
        v2_sft = [json.loads(l) for l in open(args.v2_dir / f"sft_{split}.jsonl", encoding="utf-8")]
        assert len(songs) == len(v2_sft)
        stats = {"songs": len(songs), "prompt_tokens_v2": 0, "prompt_tokens_v3": 0,
                 "lines_marked_continuation": 0, "fragment_lines": 0, "infill_eligible": 0, "infill_in_mix": 0}
        mix_path = out / f"sft_longrange_{split}.jsonl"
        with open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f_sft, \
                open(out / f"infill_{split}.jsonl", "w", encoding="utf-8") as f_inf, \
                (open(mix_path, "w", encoding="utf-8") if split != "test" else open("/dev/null", "w")) as f_mix:
            for r, old in zip(songs, v2_sft):
                assert r["song_id"] == old["song_id"]
                spec = r["spec"]
                prompt = spec_to_prompt_v3(spec)
                # the marks are notation, not lyrics: stripping them from the
                # written prompt must give back exactly the requested syllables
                written = [line for line in prompt.split("\n")
                           if line and not line.startswith(("P:", "Task", "Language", "Meter", "Tempo", "Key",
                                                            "Structure", "ABC"))]
                assert [s for line in written for s in split_syllables(strip_v3_marks(line))] == spec_syllables(spec)
                assert old["completion"] == r["abc"], "v2 sft completion must be the song's ABC"
                pt = ntok(prompt)
                stats["prompt_tokens_v2"] += old["prompt_tokens"]
                stats["prompt_tokens_v3"] += pt
                b = budget_report(spec)
                stats["lines_marked_continuation"] += b["lines_split_by_a_boundary"]
                stats["fragment_lines"] += b["fragment_lines"]
                row = {"id": f"sft:{r['song_id']}", "song_id": r["song_id"], "prompt": prompt,
                       "completion": r["abc"], "prompt_tokens": pt, "completion_tokens": old["completion_tokens"]}
                f_sft.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_mix.write(json.dumps({**row, "task": "whole_song"}, ensure_ascii=False) + "\n")

                t = choose_infill_target(r["song_id"], spec)
                if t is None:
                    continue
                ex = infill_example(Song.from_json(r["song"]), spec, t, prompt_fn=spec_to_prompt_v3)
                assert ex["prompt"].count(GAP_LINE) == 1
                inf = {"id": f"infill:{r['song_id']}", "song_id": r["song_id"], "task": "infill",
                       "prompt": ex["prompt"], "completion": ex["completion"], "target_section": t,
                       "target_label": spec["sections"][t]["label"],
                       "prompt_tokens": ntok(ex["prompt"]), "completion_tokens": ntok(ex["completion"])}
                f_inf.write(json.dumps(inf, ensure_ascii=False) + "\n")
                stats["infill_eligible"] += 1
                if split != "test" and selected(r["song_id"], args.infill_fraction):
                    f_mix.write(json.dumps(inf, ensure_ascii=False) + "\n")
                    stats["infill_in_mix"] += 1
        stats["prompt_tokens_delta_pct"] = round(
            (stats["prompt_tokens_v3"] - stats["prompt_tokens_v2"]) / max(stats["prompt_tokens_v2"], 1) * 100, 2)
        report["splits"][split] = stats
        print(f"{split}: {stats['songs']} songs, infill {stats['infill_in_mix']}/{stats['infill_eligible']} in mix, "
              f"prompt tokens {stats['prompt_tokens_delta_pct']:+.2f}%", flush=True)

    for name in COPY:
        src = args.v2_dir / name
        if src.exists():
            shutil.copy2(src, out / name)

    # every completion must be the v2 one, byte for byte
    checks = {}
    for split in ("train", "validation", "test"):
        a = [json.loads(l)["completion"] for l in open(args.v2_dir / f"sft_{split}.jsonl", encoding="utf-8")]
        b = [json.loads(l)["completion"] for l in open(out / f"sft_{split}.jsonl", encoding="utf-8")]
        checks[f"completions_identical_{split}"] = a == b
        if split != "test":
            ia = [json.loads(l)["completion"] for l in open(args.v2_dir / f"sft_longrange_{split}.jsonl", encoding="utf-8")]
            ib = [json.loads(l)["completion"] for l in open(out / f"sft_longrange_{split}.jsonl", encoding="utf-8")]
            checks[f"mixture_completions_identical_{split}"] = ia == ib
            checks[f"mixture_size_identical_{split}"] = len(ia) == len(ib)
    report["checks"] = checks
    report["files_sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))}
    (out / "v3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [k for k, v in checks.items() if not v]
    print(json.dumps(checks, indent=1))
    if bad:
        raise SystemExit(f"FAILED: {bad}")
    print(f"V3_OK -> {out}")


if __name__ == "__main__":
    main()
