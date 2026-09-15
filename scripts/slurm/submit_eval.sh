#!/bin/bash
# Submit sharded generation + probes + aggregation for one checkpoint.
#   CKPT=experiments/<run>/final_model OUT=experiments/<run>/eval_test [SHARDS=4] [NUM_SONGS=0 (all)] \
#   [SPLIT=test] [DEPEND=<jobid>] [PARTITION=ckpt-all ACCOUNT=ckpt-ark GPU=l40s] bash scripts/slurm/submit_eval.sh
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
: "${CKPT:?}" "${OUT:?}"
DATA=${DATA:-data/generated/abc_v1_20260915_012459}
SHARDS=${SHARDS:-4}; NUM_SONGS=${NUM_SONGS:-0}; SPLIT=${SPLIT:-test}
PARTITION=${PARTITION:-ckpt-all}; ACCOUNT=${ACCOUNT:-ckpt-ark}; GPU=${GPU:-l40s}
COMMON="--checkpoint $CKPT --data-dir $DATA --output-dir $OUT --split $SPLIT --num-songs $NUM_SONGS --num-probe-songs ${PROBE_SONGS:-20}"
DEP=${DEPEND:+--dependency=afterok:$DEPEND}
REQ="-p $PARTITION -A $ACCOUNT --gpus=$GPU:1 -c 8 --mem=64G --time=${TIME:-06:00:00} --requeue -o logs/slurm/%x-%j.out"
ids=()
for ((k=0; k<SHARDS; k++)); do
  ids+=($(sbatch --parsable $DEP $REQ -J qwenabc-gen$k --wrap "source scripts/env.sh; python scripts/generate_eval.py $COMMON --shard $k --num-shards $SHARDS"))
done
ids+=($(sbatch --parsable $DEP $REQ -J qwenabc-probes --wrap "source scripts/env.sh; python scripts/generate_eval.py $COMMON --probes-only"))
dep=$(IFS=:; echo "${ids[*]}")
agg=$(sbatch --parsable --dependency=afterok:$dep -p ${CPU_PARTITION:-ckpt-all} -A $ACCOUNT -c 4 --mem=32G --time=01:00:00 -J qwenabc-aggregate -o logs/slurm/%x-%j.out \
  --wrap "source scripts/env.sh; python scripts/generate_eval.py $COMMON --probes --aggregate-only")
echo "generation+probes: ${ids[*]}  aggregate: $agg"
