# Qwen-ABC

Proof of concept: can a general-purpose LLM (Qwen3.5-0.8B) learn **lyrics-aware lead-sheet generation** in human-readable **ABC notation**, without a MIDI-specific vocabulary?

Two recipes, identical except for one stage:

| | recipe |
|---|---|
| **A. Direct SFT** | Qwen3.5-0.8B-Base → SFT (lyrics + structure → ABC) |
| **B. CPT + SFT** | Qwen3.5-0.8B-Base → ABC continued pretraining → the same SFT |

Both use the same base model, SFT data, SFT hyper-parameters, test songs and evaluation.

Reports:

* `reports/DATA_AUDIT.md`: what data exists and why D1 was chosen
* `reports/ABC_SCHEMA.md`: the ABC format (lyrics, melisma, sections, round trip)
* `reports/DATASET_VALIDATION.md`: dataset sanity checks
* `reports/REPORT.md`: results and verdict

## Layout

```
qwen_abc/        library: canonical model, corpus adapter, ABC writer/parser, MIDI, prompt, splits, train, metrics
scripts/         build / validate / inspect dataset, train, generate+evaluate, samples
scripts/slurm/   Slurm launchers (L40S and L40 variants)
configs/         training configs (sft_direct, cpt_abc, sft_after_cpt)
tests/           pytest: round trip, lyric alignment, parser errors, splits/leakage, determinism
data/generated/  built datasets (git-ignored)
experiments/     one timestamped directory per run (logs and configs tracked, weights ignored)
```

## Environment

```bash
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
/usr/bin/python3.12 -m venv .venv && .venv/bin/pip install uv
.venv/bin/uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/uv pip install --python .venv/bin/python "transformers>=5.2" accelerate datasets safetensors mido pytest pyyaml flash-linear-attention
source scripts/env.sh   # PATH, PYTHONPATH, HF cache on scrubbed
```

## Reproduce

```bash
source scripts/env.sh
python -m pytest -q tests

# 1. dataset (CPU node, ~15 min); the split is fixed before examples are written
TS=$(date +%Y%m%d_%H%M%S)
OUT=data/generated/abc_v1_$TS sbatch scripts/slurm/build_dataset_cpu.sbatch
python scripts/show_dataset_examples.py --data-dir data/generated/abc_v1_$TS -n 20 --output /tmp/examples.md

# 2. Experiment A
CONFIG=configs/sft_direct.yaml RUN_DIR=experiments/direct_sft_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS" sbatch scripts/slurm/run_direct_sft_l40s.sbatch

# 3. Experiment B
CONFIG=configs/cpt_abc.yaml RUN_DIR=experiments/abc_cpt_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS" sbatch scripts/slurm/run_cpt_l40s.sbatch
CONFIG=configs/sft_after_cpt.yaml RUN_DIR=experiments/cpt_then_sft_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS model_name_or_path=experiments/abc_cpt_<ts>/final_model" \
  sbatch scripts/slurm/run_cpt_sft_l40s.sbatch

# 4. evaluation (same test songs, same seeds, same decoding for both)
python scripts/generate_eval.py --checkpoint experiments/<run>/final_model \
  --data-dir data/generated/abc_v1_$TS --output-dir experiments/<run>/eval_test --num-songs 100 --probes
```

The exact commands, job ids and dataset path used for the reported results are listed in `reports/REPORT.md`.
