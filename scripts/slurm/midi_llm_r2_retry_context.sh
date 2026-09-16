#!/bin/bash
# Re-run songs whose first attempt failed only because condition + max-new-tokens exceeded MIDI-LLM's 10,240 context:
# max-new-tokens = 10240 - condition_tokens - 8 (the repo's documented workaround; no truncation of the condition).
#   BASE=experiments/midi_llm_r2_<ts> MODE=A|B [CKPT=...] bash scripts/slurm/midi_llm_r2_retry_context.sh
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
ML=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1
CKPT=${CKPT:-runs/original-midi-llm-onestage-multitask-v1-v4/full/final}
cd "$ML"; export PYTHONPATH="$PWD:$PWD/scripts"
PY=/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/python
OUT="$QA/$BASE/mode$MODE"
if [ "$MODE" = "A" ]; then
  FLAGS="--forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4 --require-section-syllables --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0"
else
  FLAGS="--free-lyric-assignment --free-section-labels"
fi
for log in "$OUT"/logs/*.txt; do
  sid=$(basename "$log" .txt)
  cond=$(grep -o "condition (\([0-9]*\)) + max-new-tokens" "$log" | grep -o "[0-9]*" | head -1)
  [ -z "$cond" ] && continue
  maxnew=$((10240 - cond - 8))
  mv "$log" "$OUT/logs/$sid.first_attempt.txt"; mv "$OUT/logs/$sid.done" "$OUT/logs/$sid.first_attempt.done"
  start=$(date +%s)
  timeout 1800 $PY scripts/generate_full_song_one_stage.py --checkpoint "$CKPT" --from-spec "$QA/$BASE/specs/$sid.json" \
    --output-root "$OUT/gen" --overwrite --render-partial --seed 1000 --max-new-tokens $maxnew $FLAGS > "$log" 2>&1
  echo "$sid exit=$? seconds=$(( $(date +%s) - start )) retry_max_new=$maxnew" | tee -a "$OUT/logs/$sid.done"
done
echo RETRY_DONE
