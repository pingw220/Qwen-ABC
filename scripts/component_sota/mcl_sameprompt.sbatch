#!/bin/bash
# Controlled renderer comparison: re-render MuseControlLite with the exact global text prompt the SA3 render
# used (applied to every section), then remix BOTH backings with the same vocal and the same mix function.
#   NAMES="a__qwen b__qwen" sbatch scripts/component_sota/mcl_sameprompt.sbatch
#SBATCH -J cs-mcl-sameprompt
#SBATCH -p ckpt-all
#SBATCH -A ckpt-ark
#SBATCH --gpus=l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH --requeue
#SBATCH -o /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component/reports/component_sota/jobs/%x-%j.out
set -uo pipefail
A=/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio/backing/chords
MS=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-SAG
VPY=/gscratch/ark/pingw220/midisag/.venv-midisag/bin/python
for n in $NAMES; do
  P=$(python3 -c "import json;print(json.load(open('$A/render_sa3/$n/conditions.json'))['prompt'])")
  O=$A/render_mcl_sameprompt/$n; mkdir -p $O
  echo "== $n prompt: $P"
  [ -f $O/backing.wav ] || (cd $MS && PYTHONDONTWRITEBYTECODE=1 HF_HOME=/gscratch/ark/pingw220/midisag/hf_home $VPY tools/midi_llm_adapter/run_midi_llm_to_midi_sag.py \
      --midi $A/$n.mid --output-dir $O --vocal-audio $A/render/$n/vocal_fs.wav --gpu 0 --harmonizer leadsheet --prompt "$P" > $O/midi_sag.log 2>&1)
  ls -la $O/backing.wav
done
# identical remix for both renderers
source $MS/SA3_MIDI_SAG/env.sh
python - <<PY
import soundfile as sf, sys
sys.path.insert(0, "$MS/SA3_MIDI_SAG")
from sa3_midisag.inference import mix_with_vocal
A = "$A"
for n in "$NAMES".split():
    voc = f"{A}/render/{n}/vocal.wav"
    for tag, b in (("mcl_sameprompt", f"{A}/render_mcl_sameprompt/{n}/backing.wav"), ("sa3", f"{A}/render_sa3/{n}/backing.wav")):
        x, sr = sf.read(b, always_2d=True)
        out = f"{A}/compare_sameprompt/{n}/{tag}_mix.wav"
        import os; os.makedirs(os.path.dirname(out), exist_ok=True)
        sf.write(out, mix_with_vocal(x.T, voc).T, sr, subtype="PCM_16")
        os.symlink(b, f"{A}/compare_sameprompt/{n}/{tag}_backing.wav") if not os.path.exists(f"{A}/compare_sameprompt/{n}/{tag}_backing.wav") else None
        print("wrote", out, sr, x.shape)
PY
echo COMPARE_DONE
