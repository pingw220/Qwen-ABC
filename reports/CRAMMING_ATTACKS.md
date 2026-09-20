# Three attacks on syllable cramming, and which one worked

Cramming — several syllables sung on one note — is the defect listening exposed:
models put 16.3% of syllables on a note that already carries one against a corpus
5.7%, and in the worst bar measured, 19 syllables inside 0.19 s. R3-A had already
ruled out asking the model (a syllable budget in the prompt changed nothing).
All three remaining routes were run.

**Result: repairing the training targets works. The other two do not.**

| | crammed syllables | strict-valid | exact structure | lyric recall |
|---|---|---|---|---|
| corpus | 0.057 | 1.000 | 1.000 | 1.000 |
| E3b + best-of-4 *(previous best)* | 0.114 | 0.978 | 0.996 | 0.982 |
| **repaired targets + best-of-4** | **0.043** | 0.964 | 0.978 | 0.982 |
| cram-weighted reranking | 0.113 | 0.938 † | 1.000 † | 0.978 † |
| constrained decoding | **0.000** | 0.342 | 0.800 | 0.902 |

† validation subset, the others are the 225-song test set.

## 1. Repairing the targets — adopted

`qwen_abc/repair.py` splits a note carrying several syllables into one note per
syllable, same pitch, equal durations; `scripts/build_repaired_dataset.py`
rebuilds the ABC from the repaired songs and trains E3b's recipe on it. 61,621
crammed notes become +61,153 notes across the training split; 7,554 are too short
to subdivide and 468 span a section boundary and are left alone — splitting those
would move syllables into the next section and change the prompt.

The prompt does not change at all, and the build asserts it byte for byte, so the
completions are the single variable.

At best-of-4, against E3b (paired bootstrap, 225 songs):

| metric | E3b | repaired | difference |
|---|---|---|---|
| syllables sung crammed | 0.114 | **0.043** | **−0.071 [−0.081, −0.062]** |
| notes with several syllables | 0.037 | **0.010** | **−0.027 [−0.031, −0.024]** |
| strict-valid ABC | 0.978 | 0.964 | −0.013 [−0.040, +0.013] |
| exact structure | 0.996 | 0.978 | −0.018 [−0.040, +0.000] |
| lyric recall | 0.982 | 0.982 | +0.000 [−0.006, +0.006] |
| lyrics exactly as requested | 0.364 | 0.427 | +0.062 [−0.004, +0.129] |

Cramming lands **below** the corpus rate, at no significant cost to validity,
structure or lyrics. The one real cost is a rhythmic fingerprint: the model
learned the repair's equal subdivision, so notes per bar 3.49 → 3.68 (corpus
3.47), sixteenth durations 0.146 → 0.196 (corpus 0.167) and repeated-pitch
intervals 0.330 → 0.356 (corpus 0.260). Off-beat sixteenths, by contrast, land
exactly on the corpus (0.164 → 0.191 against 0.193). A repair that varied the
subdivision, or preferred a melisma where the melody already moves, would leave
less of a trace.

## 2. Cram-weighted reranking — at its ceiling

Two new rules in `scripts/select_best_of_n.py`, compared on the validation
subset:

| rule | crammed | pitch range | syncopation | chord-tone |
|---|---|---|---|---|
| `sum` (current) | 0.124 | 18.00 | **0.160** | 0.711 |
| `cram3` | 0.113 | 17.73 | 0.151 | 0.699 |
| `cram_lex` | 0.107 | 17.05 | 0.140 | 0.676 |
| corpus | 0.064 | 22.13 | 0.160 | 0.697 |

Every point of cramming comes out of musical fidelity — `sum` sits exactly on the
corpus syncopation and the cram-weighted rules walk away from it. This is what
the oracle said: the best of four samples is 0.086, so reranking never had more
than ~0.03 to give, and it cannot give even that for free. Not adopted.

## 3. Constrained decoding — fails, and the reason is structural

`qwen_abc/constrained.py` bans `~`, the only way this dialect writes a crammed
note, while a `w:` line is generated (37 of 248k tokens contain it; the ban lifts
inside music lines, where `~` is an ornament).

| metric | E3b | `~` banned | difference |
|---|---|---|---|
| syllables sung crammed | 0.163 | **0.000** | **−0.163** |
| strict-valid ABC | 0.813 | 0.342 | **−0.471** |
| no lyric-alignment error | 0.920 | 0.436 | **−0.484** |
| lyric recall | 0.972 | 0.902 | **−0.071** |
| lyric_overflow (songs) | 15 | **118** | |

**The constraint arrives too late.** In ABC the music line precedes its `w:`
line, so by the time the model writes the lyrics the bar's notes are already
committed. Banning the join cannot create a note — the syllable simply does not
fit, and the line overflows. Cramming is a decision made when the melody is
written, which is why the only thing that worked was changing what the model
learned to write.

## The recipe this leaves

ABC-v2 + infill mixture at 131K tokens/update, **trained on repaired targets**,
sampled at T=1.0 / top-p 0.95, **four samples with the `sum` selector**:

| | value | corpus |
|---|---|---|
| strict-valid ABC | 0.964 | 1.000 |
| exact structure | 0.978 | 1.000 |
| lyric recall | 0.982 | 1.000 |
| crammed syllables | 0.043 | 0.057 |
| pitch range | 18.2 | 22.0 |
| melody in declared key | 0.998 | 0.958 |

## Reproduction

```bash
python scripts/build_repaired_dataset.py --v2-dir data/generated/abc_v2_20260915_120927 \
    --output-dir data/generated/abc_v2r_<ts>
CONFIG=configs/r3c_repaired_longrange_sft.yaml RUN_DIR=experiments/r3c_repaired_<ts> \
    sbatch --gpus=l40s:2 -c 16 --mem=120G scripts/slurm/train.sbatch
CKPT=<run>/final_model DATA=data/generated/abc_v2r_<ts> OUT=<evals>/r3c_test_T1.0_bon FMT=v2 \
    ARGS="--split test --seeds 4 --temperature 1.0 --top-p 0.95" \
    sbatch --partition=gpu-a40 --account=ark --gpus=1 -c 4 --mem=48G scripts/slurm/eval.sbatch
python scripts/select_best_of_n.py --generations <evals>/r3c_test_T1.0_bon/generations \
    --out <out>/r3c_sum --rule sum
# constrained decoding, for the record: add --no-cram to the eval ARGS
```

Runs: training `experiments/r3c_repaired_20260919_092032` (job 40326311, 2.0 h on
2×L40S; an earlier attempt, job 40314140, died in NCCL setup on g3103 — an
infrastructure failure, not a code one), evaluation job 40342129 on gpu-a40,
constrained-decoding evaluation job 40314245. Dataset
`data/generated/abc_v2r_20260918_203632`.
