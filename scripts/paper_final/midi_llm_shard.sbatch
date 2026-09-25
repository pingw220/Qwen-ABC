#!/bin/bash
# MIDI-LLM (one-stage v4, mode A) paper-final tasks: its own repo + env, read-only use, one process per song,
# the same flags and checkpoint as the round-2 baseline (scripts/slurm/midi_llm_r2_shard.sh).
#   python -m paper_eval.midi_llm tasks
#   NSHARDS=8 sbatch --array=0-7 scripts/paper_final/midi_llm_shard.sbatch
# A song whose condition + max-new-tokens exceeds MIDI-LLM's 10,240-token context is retried once with
# max-new-tokens = 10240 - condition - 8 (the round-2 workaround). Finished tasks (.done) are skipped, and
# a task is claimed with an atomic mkdir lock, so a second array (e.g. REVERSE=1 on another partition)
# can share the same task list without running a song twice.
#SBATCH --job-name=pf-midillm
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3121,g3108
#SBATCH --output=reports/paper_final/job_logs/%x-%A_%a.out
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
ML=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1
CKPT=${CKPT:-runs/original-midi-llm-onestage-multitask-v1-v4/full/final}
FLAGS="--forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4 --require-section-syllables --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0"
PY=/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/python
SHARD=${SLURM_ARRAY_TASK_ID:-0}; NSHARDS=${NSHARDS:-1}
LOGS=$QA/experiments/paper_final/midi_llm/logs
cd "$ML"; export PYTHONPATH="$PWD:$PWD/scripts"
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_$SHARD host=$(hostname) start=$(date -Is) midi_llm_commit=$(git rev-parse HEAD) dirty_tracked=$(git diff --stat | tail -1) ckpt=$CKPT flags=$FLAGS"
i=0
LIST=$QA/experiments/paper_final/midi_llm/tasks.tsv
if [ "${REVERSE:-0}" = 1 ]; then
  LIST=$QA/experiments/paper_final/midi_llm/tasks_reversed.tsv
  tac "$QA/experiments/paper_final/midi_llm/tasks.tsv" > "$LIST.$SHARD.tmp" && mv "$LIST.$SHARD.tmp" "$LIST"
fi
while IFS=$'\t' read -r cond seed sid spec out; do
  if (( i % NSHARDS == SHARD )); then
    mkdir -p "$LOGS/${cond}_$seed" "$out"
    log="$LOGS/${cond}_$seed/$sid.txt"
    # skip finished, claimed, or started-by-an-unlocked-shard (log exists) tasks
    if [ ! -f "$LOGS/${cond}_$seed/$sid.done" ] && [ ! -f "$log" ] && mkdir "$LOGS/${cond}_$seed/$sid.lock" 2>/dev/null; then
      start=$(date +%s)
      timeout 1800 $PY scripts/generate_full_song_one_stage.py --checkpoint "$CKPT" --from-spec "$spec" \
        --output-root "$out" --overwrite --render-partial --seed "$seed" --max-new-tokens ${MAXNEW:-7800} $FLAGS > "$log" 2>&1
      code=$?
      c=$(grep -o "condition (\([0-9]*\)) + max-new-tokens" "$log" | grep -o "[0-9]*" | head -1)
      if [ -n "$c" ]; then
        mv "$log" "$log.first_attempt"
        timeout 1800 $PY scripts/generate_full_song_one_stage.py --checkpoint "$CKPT" --from-spec "$spec" \
          --output-root "$out" --overwrite --render-partial --seed "$seed" --max-new-tokens $((10240 - c - 8)) $FLAGS > "$log" 2>&1
        code=$?
      fi
      echo "$cond $seed $sid exit=$code seconds=$(( $(date +%s) - start )) retried=${c:+yes}" | tee "$LOGS/${cond}_$seed/$sid.done"
    fi
  fi
  i=$((i+1))
done < "$LIST"
echo "end=$(date -Is) PF_MIDILLM_OK"
