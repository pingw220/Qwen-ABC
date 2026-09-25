#!/usr/bin/env python
"""Build reports/paper_final/experiment_registry.json from the artifacts on disk.

Each entry is curated (what the experiment is, whether it is comparable) and completed from
the run's own files: resolved training config, eval args, number of generation files,
checkpoint size and a partial content hash (sha256 of the first 64 MiB + file size; a full
hash of 1.5-4 GB files per checkpoint was not worth the I/O).

  python -m paper_eval.registry
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import MODELS, OFFLOAD, OUT_ROOT, REPORT_DIR, ROOT, jdump

EV = OFFLOAD / "evals"
X = ROOT / "experiments"

# run dir (training), eval dir(s), metadata
ENTRIES = [
    # ---------------------------------------------------------------- ESS / long-structure progression (T0.8 and T1.0, 1 sample)
    dict(id="E0_T0.8", family="Qwen-ABC", run=X / "direct_sft_20260915_023500", eval=X / "r2eval_E0_20260915_115636/v1cleantest_T0.8",
         representation="ABC-v1 prompt on cleaned (v2) specs; no ESS", objective="whole-song SFT", seed_train=1234,
         role="ESS ablation: pre-ESS baseline", comparable=True),
    dict(id="E0_T1.0", family="Qwen-ABC", run=X / "direct_sft_20260915_023500", eval=X / "r2eval_E0_20260915_115636/v1cleantest_T1.0_p0.95",
         representation="ABC-v1; no ESS", objective="whole-song SFT", seed_train=1234, role="ESS ablation at canonical decoding", comparable=True),
    dict(id="E0s2_T0.8", family="Qwen-ABC", run=X / "direct_sft_seed2345_20260915_023500", eval=X / "r2eval_E0_20260915_115636/seed2345_v1cleantest_T0.8",
         representation="ABC-v1; no ESS", objective="whole-song SFT", seed_train=2345, role="training-seed replicate", comparable=True),
    dict(id="E1c_T0.8", family="Qwen-ABC", run=X / "e1c_v1clean_sft_20260915_120949", eval=X / "e1c_v1clean_sft_20260915_120949/eval_test_T0.8",
         representation="ABC-v1 on cleaned section boundaries; no ESS", objective="whole-song SFT", seed_train=1234,
         role="ESS ablation: cleaning only", comparable=True, caveats=["final_model deleted after the storage incident; only T0.8 generations exist"]),
    dict(id="E1_T0.8", family="Qwen-ABC", run=X / "e1_v2_sft_20260915_120949", eval=X / "e1_v2_sft_20260915_120949/eval_test_T0.8",
         representation="ABC-v2 (ESS: section header + per-bar [r:k])", objective="whole-song SFT", seed_train=1234, role="ESS ablation: + ESS", comparable=True),
    dict(id="E1_T1.0", family="Qwen-ABC", run=X / "e1_v2_sft_20260915_120949", eval=X / "e1_v2_sft_20260915_120949/eval_test_T1.0_p0.95",
         representation="ABC-v2 (ESS)", objective="whole-song SFT", seed_train=1234, role="ESS ablation at canonical decoding", comparable=True,
         caveats=["max_total 10240 (other T1.0 runs 8192); no output reached either limit"]),
    dict(id="E1s2_T0.8", family="Qwen-ABC", run=X / "e1_v2_sft_seed2345_20260915_143349", eval=EV / "e1s2_test_T0.8",
         representation="ABC-v2 (ESS)", objective="whole-song SFT", seed_train=2345, role="training-seed replicate", comparable=True),
    dict(id="E1s2_T1.0", family="Qwen-ABC", run=X / "e1_v2_sft_seed2345_20260915_143349", eval=EV / "e1s2_test_T1.0",
         representation="ABC-v2 (ESS)", objective="whole-song SFT", seed_train=2345, role="training-seed replicate", comparable=True),
    dict(id="E1long_T0.8", family="Qwen-ABC", run=OFFLOAD / "runs/e1long_225021", eval=EV / "e1long_test_T0.8",
         representation="ABC-v2 (ESS)", objective="whole-song SFT, 758 updates", seed_train=1234, role="matched-update control for E3b", comparable=True),
    dict(id="E1long_T1.0", family="Qwen-ABC", run=OFFLOAD / "runs/e1long_225021", eval=EV / "e1long_test_T1.0",
         representation="ABC-v2 (ESS)", objective="whole-song SFT, 758 updates", seed_train=1234, role="matched-update control", comparable=True),
    dict(id="E2a_T0.8", family="Qwen-ABC", run=X / "e2a_v2_bigbatch_lr50e5_20260915_123850", eval=X / "e2a_v2_bigbatch_lr50e5_20260915_123850/eval_test_T0.8",
         representation="ABC-v2 (ESS)", objective="whole-song SFT at 524K tokens/update (128 updates)", seed_train=1234, role="batch ablation", comparable=True),
    dict(id="E2b_T0.8", family="Qwen-ABC", run=OFFLOAD / "runs/e2b_201131", eval=EV / "e2b_test_T0.8",
         representation="ABC-v2 (ESS)", objective="whole-song SFT at 524K tokens/update (506 updates, ~8 epochs)", seed_train=1234, role="batch ablation", comparable=True),
    dict(id="E3_T0.8", family="Qwen-ABC", run=X / "e3_v2_longrange_20260915_130350", eval=X / "e3_v2_longrange_20260915_130350/eval_test_T0.8",
         representation="ABC-v2 (ESS)", objective="+ late-section reconstruction at 524K tokens/update", seed_train=1234, role="objective x batch ablation", comparable=True,
         caveats=["final_model deleted after the storage incident"]),
    dict(id="E3b_T0.8", family="Qwen-ABC", run=OFFLOAD / "runs/e3b_201131", eval=EV / "e3b_test_T0.8",
         representation="ABC-v2 (ESS)", objective="+ late-section reconstruction (infill mixture), 131K tokens/update, 758 updates", seed_train=1234,
         role="current best model (T0.8)", comparable=True),
    dict(id="E3b_T1.0", family="Qwen-ABC", run=OFFLOAD / "runs/e3b_201131", eval=EV / "e3b_test_T1.0",
         representation="ABC-v2 (ESS)", objective="+ late-section reconstruction", seed_train=1234, role="current best model, single sample", comparable=True),
    dict(id="E3b_T1.0_x4", family="Qwen-ABC", run=OFFLOAD / "runs/e3b_201131", eval=EV / "bon_e3b_test_T1.0",
         representation="ABC-v2 (ESS)", objective="+ late-section reconstruction", seed_train=1234, role="4 samples/song for selector@4 / oracle@4",
         comparable=True, selector="none (raw samples)"),
    dict(id="E3b_T1.0_sel4", family="Qwen-ABC", run=OFFLOAD / "runs/e3b_201131", eval=X / "bon_r3b_20260918_100234/test_sum",
         representation="ABC-v2 (ESS)", objective="+ late-section reconstruction", seed_train=1234, role="headline 0.996 structure: selector@4 ('sum' rule)",
         comparable=True, selector="sum rule (scripts/select_best_of_n.py), chosen on 64 validation songs",
         caveats=["selector ranks on section_plan_exact and lyric_recall, i.e. the evaluation metrics themselves (legitimate: computed from prompt + output only)"]),
    # ---------------------------------------------------------------- other model families
    dict(id="MuPT_T1.0_x4", family="MuPT", run=X / "r4_mupt_20260923_001821", eval=X / "evals_r4/r4_mupt_test_T1.0_bon",
         representation="ABC-v2 (ESS), vocabulary extended with corpus characters", objective="same infill mixture as E3b, 1070 updates",
         seed_train=1234, role="music-pretrained base", comparable=True,
         caveats=["tokenizer, architecture, pretraining corpus and update count all differ from Qwen (not a controlled causal comparison)",
                  "one generation file regenerated after a quota failure (R4 report)"]),
    dict(id="MuPT_T1.0_sel4", family="MuPT", run=X / "r4_mupt_20260923_001821", eval=X / "bon_r3b_20260918_100234/r4_sum",
         representation="ABC-v2 (ESS)", objective="infill mixture", seed_train=1234, role="MuPT selector@4", comparable=True, selector="sum rule"),
    dict(id="MIDI-LLM_A_s1000", family="MIDI-LLM", run=Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1/runs/original-midi-llm-onestage-multitask-v1-v4/full/final"),
         eval=X / "midi_llm_r2_20260915_121309/modeA/eval_test", representation="MIDI-LLM token stream (phonemes, click grid)",
         objective="MIDI-LLM one-stage multitask v4", seed_train=None, role="MIDI-LLM baseline, recommended constraints", comparable=False,
         caveats=["section count, bars/section, tempo and bar grid are INPUT (click grid) and pinned by the decoder: structure exact by construction",
                  "26/225 songs refused at G2P (code-switched English)", "validity is its own validator, not strict ABC",
                  "MIDI-LLM repo commit bea5597, clean tracked tree"]),
    dict(id="MIDI-LLM_B_s1000", family="MIDI-LLM", run=Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1/runs/original-midi-llm-onestage-multitask-v1-v4/full/final"),
         eval=X / "midi_llm_r2_20260915_121309/modeB/eval_test", representation="MIDI-LLM token stream",
         objective="MIDI-LLM one-stage multitask v4", seed_train=None, role="MIDI-LLM with free section labels / lyric assignment", comparable=False,
         caveats=["bars/section still input; labels free -> exact plan 0.027"]),
    # ---------------------------------------------------------------- negative / side results kept on record
    dict(id="R3A_T1.0_x4", family="Qwen-ABC", run=X / "r3a_v3_longrange_20260918_102828", eval=EV / "r3a_test_T1.0_bon",
         representation="ABC-v3 (syllable budget in prompt)", objective="infill mixture", seed_train=1234, role="cramming attack (negative)", comparable=True),
    dict(id="R3C_T1.0_x4", family="Qwen-ABC", run=X / "r3c_repaired_20260919_092032", eval=EV / "r3c_test_T1.0_bon",
         representation="ABC-v2 on repaired targets", objective="infill mixture", seed_train=1234, role="cramming attack (rejected by listening)", comparable=True),
    dict(id="E3b_nocram_T1.0_x4", family="Qwen-ABC", run=OFFLOAD / "runs/e3b_201131", eval=EV / "e3b_nocram_test_T1.0",
         representation="ABC-v2", objective="E3b + decode-time ban of '~' in w: lines", seed_train=1234, role="cramming attack (negative)", comparable=True),
    dict(id="R5_T1.0_x4", family="Qwen-ABC", run=X / "r5_free_20260924_100314", eval=X / "evals_r5/r5_free_test_T1.0_bon",
         representation="free lyric assignment prompt", objective="infill mixture", seed_train=1234, role="R5 free lyric assignment", comparable=True),
]

NEW = [
    ("PF_bon", "orig x S1..S4 (paired-seed sampler)", "single / selector@4 / oracle@4, reseed floor, long-range & musicality"),
    ("PF_replay", "orig S1 re-batched", "numeric noise floor of the paired-seed sampler"),
    ("PF_interventions", "13 one-control interventions x S1,S2", "interventional controllability"),
    ("PF_ood", "7 OOD plans x S1", "compositional / extrapolative generalization"),
]


def partial_hash(ckpt: Path):
    for name in ("model.safetensors",):
        f = ckpt / name
        if f.exists():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                h.update(fh.read(64 << 20))
            size = f.stat().st_size
            h.update(str(size).encode())
            return {"file": str(f), "bytes": size, "sha256_first64MiB+size": h.hexdigest()}
    return None


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def build() -> list:
    out = []
    for e in ENTRIES:
        run, ev = Path(e["run"]), Path(e["eval"])
        cfg = _json(run / "resolved_config.json") or {}
        args = _json(ev / "args.json") or {}
        gens = list((ev / "generations").glob("*.json")) if (ev / "generations").exists() else []
        ckpt = run / "final_model" if (run / "final_model").exists() else run
        summ = _json(ev / "summary.json") or {}
        seeds = int(args.get("seeds", 1)) if str(args.get("seeds", "1")).isdigit() else 1
        n_songs = len({p.stem.rpartition("_s")[0] for p in gens})
        rec = {
            "experiment_id": e["id"], "model_family": e["family"], "role": e["role"],
            "checkpoint_path": str(ckpt), "checkpoint_exists": (ckpt / "model.safetensors").exists(),
            "checkpoint_hash": partial_hash(ckpt),
            "git_commit": (run / "job_info.txt").read_text(errors="replace").strip().splitlines()[0][:300] if (run / "job_info.txt").exists() else None,
            "training_config": {k: cfg.get(k) for k in ("model_name_or_path", "train_file", "tokens_per_update", "epochs", "total_steps",
                                                         "learning_rate", "seed", "max_seq_len", "world_size", "train_examples")} if cfg else None,
            "dataset_split": "abc_v2_20260915_120927 test (225 de-duplicated songs)" if args.get("split", "test") == "test" else args.get("split"),
            "representation": e["representation"], "context_length": cfg.get("max_seq_len"),
            "batch_size_tokens_per_update": cfg.get("tokens_per_update"), "objective": e["objective"],
            "decoding": {"temperature": args.get("temperature"), "top_p": args.get("top_p"), "top_k": 0,
                         "max_total": args.get("max_total"), "batch_size": args.get("batch_size"), "sampler": "torch multinomial (seed 1000+batch offset)"},
            "candidate_count": seeds, "selector": e.get("selector", "none"),
            "random_seeds": {"training": e.get("seed_train"), "sampling": "1000 + batch index (scripts/generate_eval.py)"},
            "n_songs_evaluated": n_songs, "n_generation_files": len(gens),
            "output_path": str(ev / "generations"), "metric_path": str(ev / "summary.json") if (ev / "summary.json").exists() else None,
            "complete": n_songs == 225 or (e["family"] == "MIDI-LLM" and n_songs == 225),
            "comparable": e["comparable"], "caveats": e.get("caveats", []),
            "summary_headline": {k: summ.get(k) for k in ("parse_success", "strict_valid", "hit_eos")} if summ else None,
        }
        out.append(rec)
    for mid, cfg in MODELS.items():
        for eid, what, why in NEW:
            d = OUT_ROOT / "gen" / mid
            out.append({
                "experiment_id": f"{eid}_{mid}", "model_family": cfg["family"], "role": why, "checkpoint_path": cfg["ckpt"],
                "checkpoint_hash": partial_hash(Path(cfg["ckpt"])), "representation": cfg["fmt"], "objective": cfg["desc"],
                "decoding": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "max_total": 9216, "batch_size": 16,
                             "sampler": "Gumbel-max paired-seed (paper_eval/sampler.py)"},
                "conditions": what, "random_seeds": {"S1": 101, "S2": 202, "S3": 303, "S4": 404},
                "output_path": str(d), "task_manifests": sorted(str(p) for p in d.glob("tasks_*.jsonl")),
                "n_generation_files": sum(1 for _ in d.glob("*/*.json")) if d.exists() else 0,
                "comparable": True, "caveats": [], "planned": True,
            })
    return out


def main():
    reg = build()
    jdump(reg, REPORT_DIR / "experiment_registry.json")
    print(f"{len(reg)} entries -> {REPORT_DIR / 'experiment_registry.json'}")


if __name__ == "__main__":
    main()
