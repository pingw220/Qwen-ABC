#!/bin/bash
#SBATCH --job-name=qwenabc-build
#SBATCH --account=cpu-g2-stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
# Usage: OUT=data/generated/abc_v1_<ts> sbatch scripts/slurm/build_dataset_cpu.sbatch
set -euo pipefail
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
source scripts/env.sh
: "${OUT:?set OUT=data/generated/abc_v1_<timestamp>}"
echo "host=$(hostname) out=$OUT commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
python scripts/build_abc_dataset.py --output-dir "$OUT" --workers 30 ${EXTRA_ARGS:-}
python scripts/validate_abc_dataset.py --data-dir "$OUT" --report "$OUT/validation.json"
echo BUILD_OK
