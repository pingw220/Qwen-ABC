# Environment for the component-sota worktree (sourced). Reuses the main checkout's venv, but makes
# this worktree's code win on PYTHONPATH (scripts/env.sh hard-codes the main checkout).
source /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC/scripts/env.sh
export QWEN_ABC_ROOT=/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC-component
export PYTHONPATH="$QWEN_ABC_ROOT"
cd "$QWEN_ABC_ROOT"
export PAPER_EVAL_OUT_ROOT="$QWEN_ABC_ROOT/experiments/component_sota"
