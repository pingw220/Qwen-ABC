#!/bin/bash
# Edit -> re-render demo, generation half (E3b, one GPU): source samples, infill edits, whole-song edits.
#   sbatch scripts/paper_final/edit_rerender_gen.sbatch
# Resumable: finished generations under reports/paper_final/edit_rerender_demo/materials/generations are skipped.
#SBATCH --job-name=pf-edit-gen
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/paper_final/job_logs/%x-%j.out
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -Is) commit=$(git rev-parse HEAD)"
python -m paper_eval.edit_rerender generate
echo "end=$(date -Is) PF_EDIT_GEN_OK"
