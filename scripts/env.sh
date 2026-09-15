# Shared environment for Qwen-ABC scripts (sourced, never executed).
export QWEN_ABC_ROOT=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
export PATH="$QWEN_ABC_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$QWEN_ABC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# model weights go to scrubbed, not the 10 GB home quota; the token stays where huggingface-cli put it
export HF_HUB_CACHE="$QWEN_ABC_ROOT/.hf_cache/hub"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# compute nodes lack python3.12-devel; Triton (flash-linear-attention kernels) needs Python.h
export CPATH="$QWEN_ABC_ROOT/.venv/py312-include/python3.12${CPATH:+:$CPATH}"
export TRITON_CACHE_DIR="$QWEN_ABC_ROOT/.venv/triton-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
