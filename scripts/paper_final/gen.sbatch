#!/bin/bash
# Paper-final generation shard(s): paired-seed sampling + scoring, resumable (existing outputs are skipped).
#
#   MODEL=qwen_e3b SUITES="bon replay" NSHARDS=4 sbatch --array=0-3 scripts/paper_final/gen.sbatch
#   MODEL=mupt SUITES=interventions CONDITIONS=key_p5,lyrics_all NSHARDS=2 \
#       sbatch -p gpu-l40 -A gpu-l40-ark --array=0-1 scripts/paper_final/gen.sbatch
#   MODEL=qwen_e3b SUITES="bon replay interventions ood" SONGS=3 sbatch scripts/paper_final/gen.sbatch   # smoke
#
# Every array task takes batches b with b % NSHARDS == SLURM_ARRAY_TASK_ID of each suite, so
# shards never write the same file and a requeued or re-submitted shard only fills gaps.
#SBATCH --job-name=pf-gen
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/paper_final/job_logs/%x-%A_%a.out
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
: "${MODEL:?}" "${SUITES:?}"
SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSHARDS=${NSHARDS:-1}
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} host=$(hostname) partition=${SLURM_JOB_PARTITION:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -Is) commit=$(git rev-parse HEAD) model=$MODEL suites=$SUITES conditions=${CONDITIONS:-all} songs=${SONGS:-all} shard=$SHARD/$NSHARDS"
for SUITE in $SUITES; do
  python -m paper_eval.generation --model "$MODEL" --suite "$SUITE" --songs "${SONGS:-all}" \
    ${CONDITIONS:+--conditions "$CONDITIONS"} --shard "$SHARD" --num-shards "$NSHARDS" --batch-size "${BATCH:-16}"
done
echo "end=$(date -Is) PF_GEN_OK"
