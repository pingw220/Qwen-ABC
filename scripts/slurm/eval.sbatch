#!/bin/bash
#SBATCH --job-name=qwenabc-eval
#SBATCH --partition=ckpt-all
#SBATCH --account=ckpt-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=g3121,g3108
#SBATCH --output=logs/slurm/%x-%j.out
# Batched generation + scoring; resumes (cached generations are skipped).
#   CKPT=... DATA=... OUT=... [FMT=v1|v2] [ARGS="--num-songs 0 --temperature 0.8 ..."] sbatch scripts/slurm/eval.sbatch
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
: "${CKPT:?}" "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
{ echo "job=${SLURM_JOB_ID:-none} host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} date=$(date -Is) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
  echo "ckpt=$CKPT data=$DATA fmt=${FMT:-v1} args=${ARGS:-}"; } >> "$OUT/job_info.txt"
python scripts/generate_eval.py --checkpoint "$CKPT" --data-dir "$DATA" --output-dir "$OUT" --format "${FMT:-v1}" \
  --batch-size ${BATCH:-16} --num-songs 0 ${ARGS:-}
echo EVAL_OK
