# Sourced by the run_*.sbatch wrappers (which carry the #SBATCH resource lines).
# Required env: CONFIG, RUN_DIR (experiments/<name>_<timestamp>); optional OVERRIDES ("k=v k=v").
# NPROC>1 (or several GPUs allocated) runs data-parallel through torchrun; the optimizer step is unchanged.
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
: "${CONFIG:?}" "${RUN_DIR:?}"
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/config.yaml" ] || cp "$CONFIG" "$RUN_DIR/config.yaml" || true
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
# logging must never kill a run (a transient quota error once did): every write below tolerates failure
{ echo "job=${SLURM_JOB_ID:-none} host=$(hostname) partition=${SLURM_JOB_PARTITION:-none} restart=${SLURM_RESTART_COUNT:-0} nproc=$NPROC date=$(date -Is)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo none) dirty_files=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "overrides=${OVERRIDES:-}"; } >> "$RUN_DIR/job_info.txt" 2>/dev/null || true
git diff > "$RUN_DIR/git_diff_${SLURM_JOB_ID:-local}.patch" 2>/dev/null || true
set +e   # take the trainer's status, not tee's: a full log filesystem must not look like a training failure
if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py "$CONFIG" output_dir="$RUN_DIR" ${OVERRIDES:-} 2>&1 | tee -a "$RUN_DIR/stdout.log"
else
  python scripts/train.py "$CONFIG" output_dir="$RUN_DIR" ${OVERRIDES:-} 2>&1 | tee -a "$RUN_DIR/stdout.log"
fi
code=${PIPESTATUS[0]}
set -e
[ "$code" -eq 0 ] || exit "$code"
echo TRAIN_OK
