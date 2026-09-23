# R4: symbolic-music pretraining transfers musical competence, not instruction-following

MuPT-1.07B (`m-a-p/MuPT-v1-8192-1.07B`, Apache 2.0) is a LLaMA-2-architecture
model pretrained on 10B tokens of ABC notation. Qwen3.5-0.8B-Base has no music
pretraining at all. Same data, same recipe, same decoding, same best-of-4
selector: does the domain prior beat general language ability?

**It does not, and the way it loses is the result.** MuPT writes ABC that is as
well-formed and as musically corpus-like as Qwen's — and writes a different song
than the one that was asked for.

| | corpus | Qwen E3b | MuPT | |
|---|---|---|---|---|
| **what it writes** | | | | |
| parse success | 1.000 | 1.000 | 1.000 | — |
| bars with correct duration | 1.000 | 0.999 | 0.994 | −0.005 [−0.014, +0.002] |
| counter = bars actually left | 1.000 | 1.000 | 0.999 | −0.001 [−0.003, +0.000] |
| sections ending on `[r:1]` | 1.000 | 1.000 | 0.999 | −0.001 [−0.003, +0.000] |
| pitch range (semitones) | 22.01 | 17.56 | 18.01 | +0.46 [−0.27, +1.24] |
| mean \|interval\| | 2.394 | 2.042 | 2.050 | +0.008 [−0.071, +0.087] |
| chords per bar | 0.982 | 1.000 | 1.034 | +0.034 [−0.017, +0.085] |
| chord-tone agreement | 0.677 | 0.707 | 0.696 | −0.011 [−0.027, +0.005] |
| **whether it writes what was asked** | | | | |
| exact structure (labels + bars) | 1.000 | 0.996 | 0.689 | **−0.307** |
| exact section label sequence | 1.000 | 1.000 | 0.698 | **−0.302** |
| counter = the requested plan | 1.000 | 1.000 | 0.936 | **−0.064** |
| lyric recall | 1.000 | 0.982 | 0.539 | **−0.443** |
| lyrics exactly as requested | 1.000 | 0.364 | 0.000 | **−0.364** |
| strict-valid ABC | 1.000 | 0.978 | 0.467 | **−0.511** |
| early EOS | 0.000 | 0.000 | 0.093 | **+0.093** |

225 held-out songs, T=1.0 / top-p 0.95, best-of-4 with the `sum` selector, paired
bootstrap CIs. Bold = CI excludes zero.

Every metric in the top half is a tie. MuPT's bar durations are right, its
per-bar countdown is internally consistent to within 0.001 of Qwen's, its
melodies have the same range and the same interval distribution, its harmony is
as dense and as chord-tone-consistent. The ABC prior transferred.

Every metric in the bottom half is a rout. The song it writes is not the song the
prompt asked for: a third of the time the section labels are not the requested
ones, its countdown tracks its own bars (0.999) but not the requested plan
(0.936), and it sings about half the requested syllables.

## Why the vocabulary surgery does not explain it

MuPT's music BPE cannot represent Chinese, so the vocabulary was extended with
the corpus's characters before training (`scripts/build_mupt_base.py`, and the
report's own caveat was that a loss would then be ambiguous). It is not
ambiguous, because **the structural failures are on ASCII**:

* the requested plan is ASCII — `P:verse | 8 bars | section 3/10` — and MuPT gets
  the label sequence wrong 30% of the time;
* the countdown `[r:k]` is ASCII, and MuPT's is internally perfect but 6.4 points
  off the plan;
* early EOS is not a lyric problem, and MuPT stops early on 9.3% of songs.

Fresh Chinese embeddings explain the lyric gap. They do not explain a model that
writes a coherent eight-bar chorus when it was asked for a six-bar verse.

## What this says about the project

Round 1 found that continued pretraining on the same ABC corpus *hurt* prompt
adherence. R4 is the same finding at a much larger scale of music pretraining and
from a different direction: 10B tokens of ABC buys musical competence that a
general 0.8B model matches anyway after SFT, and costs the instruction-following
that the task is actually made of. For lyrics-conditioned, plan-conditioned
generation, **the general-purpose LLM is the better base**, and that is the
central claim of this project rather than an assumption behind it.

Two caveats kept on the record:

* MuPT is LLaMA-2-era. A 2026 music-pretrained model might not trade the same
  way, though there is no reason to expect ABC pretraining to teach plan
  adherence.
* On the same data MuPT sees 60.8M tokens to Qwen's 44.7M, so at a fixed token
  budget per update it got **1070 optimizer steps to Qwen's 758** — 41% more
  training, in its favour. It still lost.

## Cost and reproduction

9.5 GPU-h training (2×A40, 4.8 h) plus ~3 GPU-h evaluation.

```bash
python scripts/build_mupt_base.py --data-dir data/generated/abc_v2_20260915_120927 \
    --output-dir models/mupt_zh_<date>
CONFIG=configs/r4_mupt_longrange_sft.yaml RUN_DIR=experiments/r4_mupt_<ts> \
    sbatch --partition=gpu-a40 --account=ark --gpus=2 -c 8 --mem=120G scripts/slurm/train.sbatch
CKPT=<run>/final_model DATA=data/generated/abc_v2_20260915_120927 \
    OUT=experiments/evals_r4/r4_mupt_test_T1.0_bon FMT=v2 \
    ARGS="--split test --seeds 4 --temperature 1.0 --top-p 0.95" sbatch scripts/slurm/eval.sbatch
python scripts/select_best_of_n.py --generations <out>/generations --out <dir>/r4_sum --rule sum
```

Run `experiments/r4_mupt_20260923_001821` (job 40473949, 1070 updates, final
eval loss 0.3805 — **not comparable** to Qwen's 0.3780, different tokenizers and
889,745 against 672,601 supervised tokens). Evaluation jobs 40486016 (died when
`/gscratch/ark` hit its quota at sample 56), 40486073 and 40489515 on
`/gscratch/scrubbed`. One generation file was truncated by that quota failure and
was regenerated.
