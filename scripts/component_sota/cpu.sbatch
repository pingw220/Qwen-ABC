#!/bin/bash
# Generic CPU job in the component worktree:  CMD="python -m ..." sbatch scripts/component_sota/cpu.sbatch
#SBATCH --job-name=cs-cpu
#SBATCH --partition=ckpt-all
#SBATCH --account=ckpt-ark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -eo pipefail
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/scripts/component_sota/env.sh
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is) commit=$(git rev-parse HEAD) cmd=$CMD"
eval "$CMD"
echo "end=$(date -Is) CPU_OK"
