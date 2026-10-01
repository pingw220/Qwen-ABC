#!/bin/bash
# Full-length (> 180 s) backing comparison on E3b lead sheets: MuseControlLite (MIDI-SAG adapter, one global
# prompt) and SA3 MIDI-SAG step 30k from the SAME staged conditions and prompt, then an identical remix.
#SBATCH -J cs-long-compare
#SBATCH -p ckpt-all
#SBATCH -A ckpt-ark
#SBATCH --gpus=l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/reports/component_sota/jobs/%x-%j.out
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
A=/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio/backing/long
MS=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG
VPY=/gscratch/ark/pingw220/midisag/.venv-midisag/bin/python
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
# 1) MuseControlLite with one global prompt (identical to the SA3 prompt builder's output)
for m in $A/*__e3b.mid; do
  n=$(basename $m .mid); O=$A/render/$n
  P=$(python3 -c "import json;s=json.load(open('$A/$n.song.json'));print(f\"mandopop, c-pop, instrumental backing track, {s['key']}, {int(round(float(s['tempo_bpm'])))} BPM\")")
  echo "== MCL $n | $P"
  [ -f $O/backing.wav ] || (cd $MS && PYTHONDONTWRITEBYTECODE=1 HF_HOME=/gscratch/ark/pingw220/midisag/hf_home $VPY tools/midi_llm_adapter/run_midi_llm_to_midi_sag.py \
      --midi $m --output-dir $O --vocal-audio $O/vocal_fs.wav --gpu 0 --harmonizer leadsheet --prompt "$P" > $O/midi_sag.log 2>&1)
  ls -la $O/backing.wav || tail -n 20 $O/midi_sag.log
done
# 2) SA3 from the same staged conditions (conditions/, vocal.wav)
source $MS/SA3_MIDI_SAG/env.sh
python $QA/paper_eval/component_sota/sa3_backing.py --group long 2>&1 | grep --line-buffered -v -E "W[0-9]{4}|WeightNorm|Loading weights"
# 3) identical remix
python - <<PY
import os, glob, soundfile as sf, sys
sys.path.insert(0, "$MS/SA3_MIDI_SAG")
from sa3_midisag.inference import mix_with_vocal
A = "$A"
for d in sorted(glob.glob(f"{A}/render_sa3/*")):
    n = os.path.basename(d); voc = f"{A}/render/{n}/vocal.wav"; out = f"{A}/compare_sameprompt/{n}"
    os.makedirs(out, exist_ok=True)
    for tag, b in (("mcl", f"{A}/render/{n}/backing.wav"), ("sa3", f"{d}/backing.wav")):
        if not os.path.exists(b): print("missing", b); continue
        x, sr = sf.read(b, always_2d=True)
        sf.write(f"{out}/{tag}_mix.wav", mix_with_vocal(x.T, voc).T, sr, subtype="PCM_16")
        if not os.path.exists(f"{out}/{tag}_backing.wav"): os.symlink(b, f"{out}/{tag}_backing.wav")
        print("wrote", f"{out}/{tag}_mix.wav", round(x.shape[0] / sr, 1), "s")
PY
echo "end=$(date -Is) LONG_COMPARE_DONE"
