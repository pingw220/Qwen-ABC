#!/bin/bash
# Final analysis pass: rebuild the vocal (SVS) and mix (backing) wav lists now, then run every analyzer
# that has not produced its sidecar yet (audio_tools.sbatch skips finished files).
#SBATCH --job-name=cs-audio-final
#SBATCH --partition=ckpt-all
#SBATCH --account=ckpt-ark
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -eo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
source $QA/scripts/component_sota/env.sh
L=$QA/experiments/component_sota/lists
python -m paper_eval.component_sota.svs_eval list > $L/vocal_final.txt
python -m paper_eval.component_sota.backing_eval list > $L/mix_final.txt
LIST=$L/vocal_final.txt MODE=vocal bash $QA/scripts/component_sota/audio_tools.sbatch
LIST=$L/mix_final.txt MODE=mix bash $QA/scripts/component_sota/audio_tools.sbatch
echo AUDIO_FINAL_OK
