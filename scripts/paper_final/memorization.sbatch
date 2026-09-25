#!/bin/bash
# Memorization / copying analysis (CPU): exact nearest-training-song search for every system.
#   sbatch scripts/paper_final/memorization.sbatch                    # index (cached) + validate + query all + report
#   SYSTEMS=qwen_e3b:bars_p4,qwen_e3b:ood_long20 sbatch scripts/paper_final/memorization.sbatch   # add systems later
# Resumable: the train index is cached and each system's results file is appended per song.
#SBATCH --job-name=pf-memorization
#SBATCH --account=ckpt-ark
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=reports/paper_final/job_logs/%x-%j.out
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is) commit=$(git rev-parse HEAD) systems=${SYSTEMS:-all}"
[ -f experiments/paper_final/memorization/train_index.pkl ] || python -m paper_eval.memorization build-index
[ "${VALIDATE:-0}" = 1 ] && python -m paper_eval.memorization validate --n-validate 20
python -m paper_eval.memorization query --workers 31 ${SYSTEMS:+--systems "$SYSTEMS"}
python -m paper_eval.memorization report
echo "end=$(date -Is) PF_MEM_OK"
