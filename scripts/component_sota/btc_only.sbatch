#!/bin/bash
# BTC chord transcription only, over the backing-mix list (skips wavs that already have <wav>.lab).
#SBATCH --job-name=cs-btc
#SBATCH --partition=ckpt-all
#SBATCH --account=ckpt-ark
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
E=/gscratch/ark/pingw220/miniconda3/envs
LIST=$QA/experiments/component_sota/lists/mix_2.txt
D=$(mktemp -d -p $QA/experiments/component_sota/tmp btc.XXXX); i=0
while read -r p; do [ -f "$p.lab" ] && continue; ln -s "$p" "$D/$(printf %05d $i).wav"; echo "$(printf %05d $i) $p" >> $D/map.txt; i=$((i+1)); done < "$LIST"
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/BTC-ISMIR19 && LD_LIBRARY_PATH=$E/sheetsagepp-btc/lib $E/sheetsagepp-btc/bin/python $QA/paper_eval/component_sota/tools/btc_run.py --voca True --audio_dir $D --save_dir $D
while read -r k p; do [ -f "$D/$k.lab" ] && cp "$D/$k.lab" "$p.lab"; done < $D/map.txt
echo "BTC_DONE $(ls $D/*.lab | wc -l)"
