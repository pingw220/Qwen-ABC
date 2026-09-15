#!/bin/bash
# Run MIDI-LLM one-stage v4-cd (its own repo/env, read-only) on Qwen-ABC specs.
#   BASE=experiments/midi_llm_onestage_v4cd_baseline_<ts> SHARD=k NSHARDS=n bash scripts/slurm/midi_llm_baseline_shard.sh
set -uo pipefail
QA=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
ML=/mmfs1/gscratch/scrubbed/pingw220/music_acc/MIDI-LLM-phoneme-lyric-v1
cd "$ML"
export PYTHONPATH="$PWD:$PWD/scripts"
PY=/gscratch/ark/pingw220/miniconda3/envs/midi-llm/bin/python
{ echo "midi_llm_commit=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l)"; git diff --stat | tail -1; } > "$QA/$BASE/midi_llm_provenance_shard$SHARD.txt"
i=0
for spec in ${ONLY:-$(ls "$QA/$BASE/specs"/*.json | sort)}; do
  if (( i % NSHARDS == SHARD )); then
    sid=$(basename "$spec" .json)
    if [ ! -f "$QA/$BASE/gen/$sid/validation.json" ]; then
      timeout 1800 $PY scripts/generate_full_song_one_stage.py \
        --checkpoint runs/original-midi-llm-onestage-multitask-v1-v4-cd/full/final \
        --registry manifests/multitask_v2/token_registry.json \
        --from-spec "$spec" --output-root "$QA/$BASE/gen" --overwrite --render-partial \
        --seed 1000 --max-new-tokens ${MAXNEW:-7800} \
        --forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4 --require-section-syllables \
        --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0 > "$QA/$BASE/logs_$sid.txt" 2>&1
      echo "$sid exit=$?"
    fi
  fi
  i=$((i+1))
done
echo MIDI_LLM_SHARD_DONE
