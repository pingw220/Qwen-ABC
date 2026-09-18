# R3-A: a syllable budget in the prompt does not fix cramming

**Negative result on the main hypothesis.** Telling the model how many syllables
each section has does not stop it putting several of them on one note, and it
costs strict validity.

The experiment is single-variable by construction: `scripts/build_v3_prompts.py`
derives the training files from the v2 dataset, so every completion is asserted
byte-identical to E3b's and the infill targets and mixture membership are the
same deterministic hashes. Only the prompt changed — a syllable count in each
section header, and `…` on both ends of a lyric line that a section boundary
splits (`qwen_abc/abc_v3.py`). Same recipe, same seed, 774 updates against E3b's
758 (the longer prompts add 13.8% tokens, so two epochs is slightly more steps).
Teacher-forced loss is unchanged: 0.3773 against 0.3780 on identical supervised
tokens.

## What happened, at T=1.0, one sample per song, 225 test songs

| metric | corpus | E3b (v2) | R3-A (v3) | difference |
|---|---|---|---|---|
| **syllables sung crammed** | 0.057 | 0.163 | 0.173 | +0.011 [−0.005, +0.026] |
| notes with several syllables | 0.023 | 0.054 | 0.053 | −0.001 [−0.006, +0.005] |
| strict-valid ABC | 1.000 | 0.813 | 0.720 | **−0.093 [−0.164, −0.022]** |
| no lyric-alignment error | 1.000 | 0.920 | 0.804 | **−0.116 [−0.178, −0.053]** |
| lyrics exactly as requested | 1.000 | 0.267 | 0.351 | **+0.084 [+0.013, +0.160]** |
| lyric recall in late sections | 1.000 | 0.960 | 0.970 | **+0.011 [+0.002, +0.021]** |
| notes per syllable | 1.231 | 1.171 | 1.223 | **+0.051 [+0.022, +0.082]** |
| melisma note fraction | 0.161 | 0.189 | 0.212 | **+0.023 [+0.011, +0.035]** |
| exact structure | 1.000 | 0.982 | 0.973 | −0.009 [−0.036, +0.013] |
| lyric recall | 1.000 | 0.972 | 0.975 | +0.002 [−0.005, +0.010] |

Bold = 95% paired bootstrap CI excludes 0.

The budget did change how the model spends notes: **notes per syllable moves
from 1.171 to 1.223, almost exactly the corpus 1.231**, and it sings more of what
was asked (lyrics exactly as requested +8.4 points, late-section recall +0.011).
What it did not do is stop stacking syllables on a single note, which is what
cramming measures and what is audible.

It also doubled one error class: **`lyric_overflow` on 35 songs against 15** — a
`w:` line with more syllables than the notes it is written under. Those songs are
worse overall, not better (mean lyric recall 0.950 against 0.979 on the rest), so
this is not the budget forcing syllables in; the mechanism is not established.

## In the setting that would actually ship

With best-of-4 selection (R3-B) on top, the two are a wash — v3 is better on
lyric recall and nothing else clears the CI:

| metric | E3b + best-of-4 | R3-A + best-of-4 | difference |
|---|---|---|---|
| strict-valid ABC | 0.978 | 0.960 | −0.018 [−0.044, +0.009] |
| lyric recall | 0.982 | 0.987 | **+0.005 [+0.001, +0.009]** |
| lyrics exactly as requested | 0.364 | 0.413 | +0.049 [−0.018, +0.120] |
| syllables sung crammed | 0.114 | 0.117 | +0.003 [−0.008, +0.014] |

**So v3 is not adopted.** The one piece worth keeping is the continuation mark,
which is the plausible cause of the lyric-fidelity gains — but the two changes
shipped together, so which did what is untested. Separating them is two more
1.5 GPU-h runs.

## What this rules out, and where cramming can still be attacked

Cramming is not an information problem. The model knows the bar count, the
section count, a per-bar countdown and now the syllable count, and it still
writes two syllables on one note at three times the corpus rate. It is doing
what its training targets do — the corpus crams too, at 0.057, because the
note↔lyric matcher that produced the labels could not place every syllable.

Three routes remain, cheapest first:

1. **A cram-weighted selector** (free, no GPU). The oracle over the four samples
   already drawn is 0.086, and 0.099 if restricted to strict-valid samples,
   against the current rule's 0.114. Worth about 1.5 points, no more: reranking
   cannot beat the distribution it samples from.
2. **Repair the targets** (~5 GPU-h): split a note that carries several
   syllables into one note per syllable in the training data, so the model never
   sees the pathology. This attacks the cause rather than the request, and it
   changes the completions — no longer a single-variable comparison, which is
   why it was not the first thing tried.
3. **Constrained decoding**: forbid a second syllable on a note while generating.
   Takes cramming to zero by construction, but the syllables have to go
   somewhere, so it has to be measured against lyric recall.

## Reproduction

```bash
python scripts/build_v3_prompts.py --v2-dir data/generated/abc_v2_20260915_120927 \
    --output-dir data/generated/abc_v3_<ts>
CONFIG=configs/r3a_v3_longrange_sft.yaml RUN_DIR=experiments/r3a_v3_longrange_<ts> \
    sbatch --gpus=l40s:2 -c 16 --mem=120G scripts/slurm/train.sbatch
CKPT=<run>/final_model DATA=data/generated/abc_v3_<ts> OUT=<evals>/r3a_test_T1.0_bon FMT=v3 \
    ARGS="--split test --seeds 4 --temperature 1.0 --top-p 0.95" sbatch scripts/slurm/eval.sbatch
```

Run `experiments/r3a_v3_longrange_20260918_102828` (job 40295745, 1.5 GPU-h),
evaluation job 40303668 (~1.5 GPU-h), dataset
`data/generated/abc_v3_20260918_102229`.
