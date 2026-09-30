#!/bin/bash
# Backing renders of the component-round lead sheets with the tuned SA3 MIDI-SAG model (step 30k).
#   GROUP=chords sbatch --array=0-2 scripts/component_sota/sa3_backing.sbatch
#SBATCH -A ark
#SBATCH -p ckpt-all
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -J cs-sa3-backing
#SBATCH -o /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/reports/component_sota/jobs/%x-%A_%a.out
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/SA3_MIDI_SAG/env.sh
N=${SLURM_ARRAY_TASK_COUNT:-1}
echo "job=$SLURM_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) start=$(date -Is) group=${GROUP:-chords}"
python /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/paper_eval/component_sota/sa3_backing.py \
    --group ${GROUP:-chords} --shard ${SLURM_ARRAY_TASK_ID:-0} --num-shards $N ${ONLY:+--only $ONLY} \
    2>&1 | grep --line-buffered -v -E "W[0-9]{4}|WeightNorm|Loading weights"
echo "end=$(date -Is)"
