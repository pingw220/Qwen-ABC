#!/bin/bash
#SBATCH --job-name=qwenabc-direct_sft
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
# Usage:
#   CONFIG=configs/sft_direct.yaml RUN_DIR=experiments/direct_sft_$(date +%Y%m%d_%H%M%S) \
#   OVERRIDES="data_dir=data/generated/<dataset>" sbatch scripts/slurm/run_direct_sft_l40s.sbatch
# (sbatch exports the caller's environment by default.)
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC/scripts/slurm/_train_body.sh
