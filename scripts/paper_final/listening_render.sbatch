#!/bin/bash
# Listening-study pilot audio: FastSinger sings each stimulus excerpt, mixed over its fluidsynth backing.
#   python -m paper_eval.listening_study build && python -m paper_eval.listening_study render-cpu --pilot 16
#   sbatch scripts/paper_final/listening_render.sbatch
# Same singer/checkpoint/flags as scripts/slurm/midi_sag_render.sbatch. A stimulus whose vocal fails is
# written as its instrumental backing and listed in render_log.tsv (status=fallback_instrumental).
# Existing <name>.wav files are skipped, so the job can be resubmitted.
#SBATCH --job-name=pf-listen-render
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-ark
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/paper_final/job_logs/%x-%j.out
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
AUD=$QA/reports/paper_final/listening_study/materials/audio
FASTSINGER_ROOT=/mmfs1/gscratch/scrubbed/pingw220/music_acc/fastsinger
FSPY=/gscratch/ark/pingw220/miniconda3/envs/fastsinger/bin/python
FS_MODEL=suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag; FS_EPOCH=400; FS_SPKR=6
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is)"
while read -r name; do
  [ -z "$name" ] && continue
  [ -f "$AUD/$name.wav" ] && continue
  ( cd "$FASTSINGER_ROOT" && "$FSPY" inference.py --model_id "$FS_MODEL" --model_epoch "$FS_EPOCH" --spkr_ref "$FS_SPKR" \
      --pitch_shifts 0 --shift_consonant_forward_alignment \
      --midi_path "$AUD/$name.fs.mid" --lyric_path "$AUD/$name.fs.txt" --output_path "$AUD/$name.vocal.wav" ) > "$AUD/$name.fastsinger.log" 2>&1
  "$FSPY" - "$AUD" "$name" <<'EOF'
import sys, numpy as np, librosa, soundfile as sf
aud, name = sys.argv[1], sys.argv[2]
back, sr = librosa.load(f"{aud}/{name}.backing.wav", sr=44100, mono=True)
status = "ok"
try:
    voc, _ = librosa.load(f"{aud}/{name}.vocal.wav", sr=44100, mono=True)
except Exception:
    voc, status = np.zeros(1), "fallback_instrumental"
n = max(len(back), len(voc))
mix = np.zeros(n)
mix[:len(back)] += 0.5 * back / (np.max(np.abs(back)) + 1e-9)
if status == "ok":
    mix[:len(voc)] += 0.8 * voc / (np.max(np.abs(voc)) + 1e-9)
mix = 0.9 * mix / (np.max(np.abs(mix)) + 1e-9)
sf.write(f"{aud}/{name}.wav", mix, 44100)
open(f"{aud}/render_log.tsv", "a").write(f"{name}\t{status}\t{n / 44100:.1f}s\n")
EOF
done < "$AUD/pilot_stimuli.txt"
echo "end=$(date -Is) LISTEN_RENDER_OK"
