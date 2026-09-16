#!/bin/bash
# Evaluation bundle for a finished round-2 run.
#   RUN=experiments/<run> FMT=v1|v2 [DATA=data/generated/abc_v2_20260915_120927] [SFT_TEST=path] [WHAT="test ckpts cont loss"]
#   [MAXTOTAL=10240] [DEPEND=jobid] bash scripts/slurm/submit_run_evals.sh
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
: "${RUN:?}" "${FMT:?}"
DATA=${DATA:-data/generated/abc_v2_20260915_120927}
MAXTOTAL=${MAXTOTAL:-10240}
WHAT=${WHAT:-"test ckpts cont loss"}
DEP=${DEPEND:+--dependency=afterok:$DEPEND}
for w in $WHAT; do
  case $w in
    test)
      CKPT=$RUN/final_model DATA=$DATA OUT=$RUN/eval_test_T0.8 FMT=$FMT ARGS="--max-total $MAXTOTAL --max-new-tokens $MAXTOTAL" \
        sbatch --parsable $DEP -J qwenabc-evtest scripts/slurm/eval.sbatch ;;
    ckpts)
      for c in $(ls -d $RUN/step-* $RUN/epoch-* $RUN/final_model 2>/dev/null | grep -v '\.tmp$'); do
        CKPT=$c DATA=$DATA OUT=$RUN/valgen/$(basename $c) FMT=$FMT \
          ARGS="--split validation --song-id-list reports/decoding_sweep_validation_song_ids.txt --max-total $MAXTOTAL --max-new-tokens $MAXTOTAL" \
          sbatch --parsable $DEP -J qwenabc-evckpt scripts/slurm/eval.sbatch
      done ;;
    cont)
      sbatch --parsable $DEP -p ckpt-all -A ckpt-ark --gpus=l40s:1 -c 8 --mem=64G --time=06:00:00 --requeue --exclude=g3121 \
        -J qwenabc-evcont -o logs/slurm/%x-%j.out \
        --wrap "source scripts/env.sh; python scripts/eval_continuation.py --checkpoint $RUN/final_model --data-dir $DATA --output-dir $RUN/eval_continuation_T0.8 --format $FMT --max-total $MAXTOTAL" ;;
    loss)
      sbatch --parsable $DEP -p ckpt-all -A ckpt-ark --gpus=l40s:1 -c 8 --mem=64G --time=03:00:00 --requeue --exclude=g3121 \
        -J qwenabc-evloss -o logs/slurm/%x-%j.out \
        --wrap "source scripts/env.sh; python scripts/eval_loss.py --checkpoint $RUN/final_model --data-dir $DATA --split test ${SFT_TEST:+--sft-file $SFT_TEST} --output $RUN/loss_test.json" ;;
  esac
done
