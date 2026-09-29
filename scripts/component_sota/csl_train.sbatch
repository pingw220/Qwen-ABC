#!/bin/bash
# CSL-L2M retrain on our canonical split with the UNMODIFIED upstream trainer (MIDI-SAG/lyrics2melody_new/train_CSLL2M.py).
# Only the config differs (configs/component_sota/csl_l2m_onlylyrics_retrain.yaml: our data paths, our lyric vocab, our ckpt_dir).
#SBATCH --job-name=cs-csl-train
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -eo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
export PYTHONDONTWRITEBYTECODE=1
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG/lyrics2melody_new
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is) upstream_commit=$(git -C .. rev-parse HEAD) upstream_dirty=$(git -C .. status --porcelain --untracked-files=no | wc -l)"
/gscratch/ark/pingw220/miniconda3/envs/midi-sag/bin/python -u train_CSLL2M.py $QA/configs/component_sota/csl_l2m_onlylyrics_retrain.yaml
echo "end=$(date -Is) CSL_TRAIN_OK"
