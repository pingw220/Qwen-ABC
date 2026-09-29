#!/bin/bash
# Generic 1-GPU job in the component worktree:  CMD="..." [PYENV=midi-sag] sbatch [-p gpu-l40 -A gpu-l40-ark] --array=0-N scripts/component_sota/gpu.sbatch
# $SHARD / $NSHARDS are exported for CMD from the array index.
#SBATCH --job-name=cs-gpu
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/component_sota/jobs/%x-%A_%a.out
set -eo pipefail
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/scripts/component_sota/env.sh
export SHARD=${SLURM_ARRAY_TASK_ID:-0} NSHARDS=${NSHARDS:-1} PYTHONDONTWRITEBYTECODE=1
if [ -n "${PYENV:-}" ]; then PY=/gscratch/ark/pingw220/miniconda3/envs/$PYENV/bin/python; else PY=python; fi
export PY
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -Is) commit=$(git rev-parse HEAD) cmd=$CMD"
eval "$CMD"
echo "end=$(date -Is) GPU_OK"
