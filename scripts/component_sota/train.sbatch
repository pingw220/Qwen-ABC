#!/bin/bash
# Qwen SFT in the component worktree (same trainer as E3b: scripts/train.py; resumes from RUN_DIR/resume_state).
#   CONFIG=configs/component_sota/mel_sft.yaml RUN_DIR=experiments/component_sota/runs/mel_<ts> sbatch --gpus=2 scripts/component_sota/train.sbatch
#SBATCH --job-name=cs-train
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -euo pipefail
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/scripts/component_sota/env.sh
: "${CONFIG:?}" "${RUN_DIR:?}"
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/config.yaml" ] || cp "$CONFIG" "$RUN_DIR/config.yaml"
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
{ echo "job=${SLURM_JOB_ID:-none} host=$(hostname) partition=${SLURM_JOB_PARTITION:-none} restart=${SLURM_RESTART_COUNT:-0} nproc=$NPROC date=$(date -Is)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "commit=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l) overrides=${OVERRIDES:-}"; } >> "$RUN_DIR/job_info.txt" || true
set +e
if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py "$CONFIG" output_dir="$RUN_DIR" ${OVERRIDES:-} 2>&1 | tee -a "$RUN_DIR/stdout.log"
else
  python scripts/train.py "$CONFIG" output_dir="$RUN_DIR" ${OVERRIDES:-} 2>&1 | tee -a "$RUN_DIR/stdout.log"
fi
code=${PIPESTATUS[0]}
set -e
[ "$code" -eq 0 ] || exit "$code"
echo TRAIN_OK
