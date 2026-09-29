#!/bin/bash
# Run the audio analyzers over a list of wavs (one per line), each in its own environment.
#   LIST=<file> MODE=vocal|mix sbatch scripts/component_sota/audio_tools.sbatch
# vocal: RMVPE F0, Paraformer ASR, Audiobox;  mix: key (KS chroma), BeatNet beats, BTC chords, SongEval, Audiobox.
#SBATCH --job-name=cs-audio-tools
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-ark
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/component_sota/jobs/%x-%j.out
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
T=$QA/paper_eval/component_sota/tools
E=/gscratch/ark/pingw220/miniconda3/envs
VPY=/gscratch/ark/pingw220/midisag/.venv-midisag/bin/python
export PYTHONDONTWRITEBYTECODE=1
: "${LIST:?}" "${MODE:?}"
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is) list=$LIST mode=$MODE n=$(wc -l < $LIST)"
todo() { local suffix=$1; while read -r p; do [ -f "$p$suffix" ] || echo "$p"; done < "$LIST" > "$LIST.$suffix.todo"; echo "$LIST.$suffix.todo"; }
if [ "$MODE" = vocal ]; then
  L=$(todo .f0.npz); [ -s "$L" ] && $VPY $T/f0_rmvpe.py "$L"
  L=$(todo .asr.json); [ -s "$L" ] && $E/midi-sag/bin/python $T/asr_paraformer.py "$L"
else
  L=$(todo .key.json); [ -s "$L" ] && $VPY $T/key_chroma.py "$L"
  L=$(todo .beats.json); [ -s "$L" ] && $E/beatbk-beatnet/bin/python $T/beats_beatnet.py "$L"
  L=$(todo .lab)
  if [ -s "$L" ]; then   # BTC runs from its own directory over a directory of files: link each wav into a scratch dir
    D=$(mktemp -d -p $QA/experiments/component_sota/tmp btc.XXXX); i=0
    while read -r p; do ln -s "$p" "$D/$(printf %05d $i).wav"; echo "$(printf %05d $i) $p" >> $D/map.txt; i=$((i+1)); done < "$L"
    ( cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/BTC-ISMIR19 && LD_LIBRARY_PATH=$E/sheetsagepp-btc/lib $E/sheetsagepp-btc/bin/python test.py --voca True --audio_dir $D --save_dir $D )
    while read -r k p; do [ -f "$D/$k.lab" ] && cp "$D/$k.lab" "$p.lab"; done < $D/map.txt
  fi
  L=$(todo .songeval.json)
  if [ -s "$L" ]; then
    O=$(mktemp -d -p $QA/experiments/component_sota/tmp songeval.XXXX)
    ( cd /gscratch/ark/pingw220/andysu/SongEval && $E/audio_eval/bin/python eval.py -i "$L" -o $O ) && cp -r $O $LIST.songeval_out
  fi
fi
L=$(todo .aes.json); [ -s "$L" ] && $E/music-rl/bin/python $T/audiobox.py "$L"
echo "end=$(date -Is) AUDIO_TOOLS_OK"
