#!/bin/bash
# Round-2 MIDI-LLM baseline shard (its own repo and env, read-only use; one process per song).
#   BASE=experiments/midi_llm_r2_<ts> MODE=A|B SHARD=k NSHARDS=n [CKPT=runs/...] bash scripts/slurm/midi_llm_r2_shard.sh
# MODE A: the repo's recommended constraint flags. MODE B: every optional constraint off, plus
# --free-lyric-assignment --free-section-labels. In both modes the section count and bars per section
# are part of MIDI-LLM's input (click grid) and cannot deviate by construction.
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
ML=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1
CKPT=${CKPT:-runs/original-midi-llm-onestage-multitask-v1-v4/full/final}
cd "$ML"
export PYTHONPATH="$PWD:$PWD/scripts"
PY=/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/python
OUT="$QA/$BASE/mode$MODE"
mkdir -p "$OUT/gen" "$OUT/logs"
if [ "$MODE" = "A" ]; then
  FLAGS="--forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4 --require-section-syllables --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0"
else
  FLAGS="--free-lyric-assignment --free-section-labels"
fi
{ echo "job=${SLURM_JOB_ID:-none} host=$(hostname) midi_llm_commit=$(git rev-parse HEAD) dirty_tracked=$(git diff --stat | tail -1) ckpt=$CKPT mode=$MODE flags=$FLAGS"; } > "$OUT/provenance_shard$SHARD.txt"
i=0
for spec in $(ls "$QA/$BASE/specs"/*.json | sort); do
  if (( i % NSHARDS == SHARD )); then
    sid=$(basename "$spec" .json)
    if [ ! -f "$OUT/gen/$sid/validation.json" ] && [ ! -f "$OUT/logs/$sid.done" ]; then
      start=$(date +%s)
      timeout 1800 $PY scripts/generate_full_song_one_stage.py --checkpoint "$CKPT" \
        --from-spec "$spec" --output-root "$OUT/gen" --overwrite --render-partial \
        --seed 1000 --max-new-tokens ${MAXNEW:-7800} $FLAGS > "$OUT/logs/$sid.txt" 2>&1
      code=$?
      echo "$sid exit=$code seconds=$(( $(date +%s) - start ))" | tee -a "$OUT/logs/$sid.done"
    fi
  fi
  i=$((i+1))
done
echo MIDI_LLM_SHARD_DONE
