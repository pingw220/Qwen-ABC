#!/bin/bash
# Lead sheet -> sung vocal (FastSinger) -> full-band mix (MIDI-SAG / MuseControlLite).
#
#   LEADSHEETS=<dir of exported *.mid>  OUT=<dir>  sbatch scripts/slurm/midi_sag_render.sbatch
#
# Inputs come from scripts/export_leadsheet_midi.py: "<name>.mid" is the lead
# sheet MIDI-SAG conditions on, "<name>.fs.mid" + "<name>.fs.txt" are the
# note-aligned pair FastSinger sings (symbol k sings note k -- see
# qwen_abc/fastsinger.py). Passing the FastSinger vocal in with --vocal-audio
# replaces MIDI-SAG's own SoulX-Singer stage.
#
# Both stages are skipped per song when their output already exists, so the job
# can be requeued without redoing finished work. Audio lands on /gscratch/ark:
# a render is ~200 MB of wav per song and /gscratch/scrubbed has hit quota before.
#SBATCH --job-name=qwenabc-midisag
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=%x-%j.out

set -uo pipefail

MUSIC_ACC="${MUSIC_ACC:-/mmfs1/gscratch/scrubbed/pingw220/music_acc}"
MIDI_SAG_ROOT="${MIDI_SAG_ROOT:-$MUSIC_ACC/MIDI-SAG}"
FASTSINGER_ROOT="${FASTSINGER_ROOT:-$MUSIC_ACC/fastsinger}"
VPY="${MIDI_SAG_VENV:-/gscratch/ark/pingw220/midisag/.venv-midisag}/bin/python"
FSPY="${FASTSINGER_PY:-/gscratch/ark/pingw220/miniconda3/envs/fastsinger/bin/python}"
FS_MODEL="${FS_MODEL:-suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag}"
FS_EPOCH="${FS_EPOCH:-400}"
FS_SPKR="${FS_SPKR:-6}"
LEADSHEETS="${LEADSHEETS:?set LEADSHEETS to a directory of exported *.mid}"
OUT="${OUT:?set OUT to an output directory}"
# MIDI-SAG's default intro/outro prompts ask for "soft synthesizer pads,
# ethereal, fading slowly", which on these songs left the 11-28 s instrumental
# intro 13 dB below the verses and decaying to -46 dBFS just before the vocal
# enters -- it reads as a hole rather than an introduction. These override only
# the sections that were empty; every other tag keeps the repo's own prompt.
TAG_PROMPTS="${TAG_PROMPTS:-{\"intro\": \"Mandarin pop introduction with the full band already playing: drums, bass, piano and electric guitar\", \"outro\": \"Mandarin pop outro with the full band playing out: drums, bass, piano and guitar\", \"inst\": \"Mandarin pop instrumental break with the full band: drums, bass, piano and lead guitar\"}}"

export HF_HOME="${HF_HOME:-/gscratch/ark/pingw220/midisag/hf_home}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "leadsheets=$LEADSHEETS out=$OUT singer=$FS_MODEL@$FS_EPOCH spkr=$FS_SPKR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ok=0; failed=0
for mid in "$LEADSHEETS"/*.mid; do
    case "$mid" in *.fs.mid) continue ;; esac
    name=$(basename "$mid" .mid)
    dir="$OUT/$name"
    vocal="$dir/vocal_fs.wav"
    mkdir -p "$dir"

    if [ -f "$dir/mix.wav" ]; then
        echo "=== $name: already rendered, skipping"
        ok=$((ok + 1))
        continue
    fi

    if [ ! -s "$vocal" ]; then
        echo "=== $name: singing"
        ( cd "$FASTSINGER_ROOT" && "$FSPY" inference.py \
            --model_id "$FS_MODEL" --model_epoch "$FS_EPOCH" --spkr_ref "$FS_SPKR" \
            --pitch_shifts 0 --shift_consonant_forward_alignment \
            --midi_path "$LEADSHEETS/$name.fs.mid" \
            --lyric_path "$LEADSHEETS/$name.fs.txt" \
            --output_path "$vocal" ) > "$dir/fastsinger.log" 2>&1
        echo "    fastsinger exit=$? -> $vocal"
    fi
    if [ ! -s "$vocal" ]; then
        echo "    no vocal produced; see $dir/fastsinger.log"
        failed=$((failed + 1))
        continue
    fi

    # --harmonizer leadsheet keeps the chords the model composed; AccoMontage2
    # would re-harmonize from the melody and drop melody-free sections.
    ( cd "$MIDI_SAG_ROOT" && "$VPY" tools/midi_llm_adapter/run_midi_llm_to_midi_sag.py \
        --midi "$mid" --output-dir "$dir" --vocal-audio "$vocal" \
        --gpu 0 --harmonizer leadsheet --tag-prompts "$TAG_PROMPTS" ) > "$dir/midi_sag.log" 2>&1
    status=$?
    if [ -f "$dir/mix.wav" ]; then
        echo "    mix ok -> $dir/mix.wav"
        ok=$((ok + 1))
    else
        echo "    midi-sag exit=$status, no mix; see $dir/midi_sag.log"
        failed=$((failed + 1))
    fi
done

echo "RENDER_DONE ok=$ok failed=$failed out=$OUT"
