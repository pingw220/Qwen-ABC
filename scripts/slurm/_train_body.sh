# Sourced by the run_*.sbatch wrappers (which carry the #SBATCH resource lines).
# Required env: CONFIG, RUN_DIR (experiments/<name>_<timestamp>); optional OVERRIDES ("k=v k=v").
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
: "${CONFIG:?}" "${RUN_DIR:?}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
{ echo "job=${SLURM_JOB_ID:-none} host=$(hostname) partition=${SLURM_JOB_PARTITION:-none}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo none) dirty_files=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "overrides=${OVERRIDES:-}"; } | tee -a "$RUN_DIR/job_info.txt"
python scripts/train.py "$CONFIG" output_dir="$RUN_DIR" ${OVERRIDES:-} 2>&1 | tee -a "$RUN_DIR/stdout.log"
echo TRAIN_OK
