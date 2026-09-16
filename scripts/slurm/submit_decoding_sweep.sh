#!/bin/bash
# Decoding sweep on the fixed 64-song validation subset (reports/decoding_sweep_validation_song_ids.txt).
#   CKPT=... DATA=... FMT=v1|v2 OUT=experiments/<sweep_dir> [MAXTOTAL=8192] bash scripts/slurm/submit_decoding_sweep.sh
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
: "${CKPT:?}" "${DATA:?}" "${FMT:?}" "${OUT:?}"
for setting in "0.0 1.0" "0.7 0.95" "0.8 0.95" "0.9 0.95" "1.0 0.95" "1.0 0.98" "1.1 0.98"; do
  set -- $setting
  name="T$1_p$2"
  CKPT=$CKPT DATA=$DATA OUT=$OUT/$name FMT=$FMT \
    ARGS="--split validation --song-id-list reports/decoding_sweep_validation_song_ids.txt --temperature $1 --top-p $2 --max-total ${MAXTOTAL:-8192} --max-new-tokens ${MAXTOTAL:-8192}" \
    sbatch --parsable ${DEPEND:+--dependency=afterok:$DEPEND} -J qwenabc-sweep-$name scripts/slurm/eval.sbatch
done
