#!/bin/bash
#SBATCH --job-name=qwenabc-train
#SBATCH --partition=ckpt-all
#SBATCH --account=ckpt-ark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=g3121,g3108
#SBATCH --output=logs/slurm/%x-%j.out
# Generic launcher; resources on the sbatch command line override the defaults above.
#   CONFIG=configs/<cfg>.yaml RUN_DIR=experiments/<name>_<ts> [OVERRIDES="k=v ..."] \
#     sbatch --gpus=l40s:4 -c 32 --mem=240G scripts/slurm/train.sbatch
# The trainer resumes from RUN_DIR/resume_state after preemption (--requeue).
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC/scripts/slurm/_train_body.sh
