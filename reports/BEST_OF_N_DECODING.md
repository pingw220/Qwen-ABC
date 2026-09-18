# Best-of-n decoding (R3-B)

Sampling at T=1.0 / top-p 0.95 gives the best structure and the most corpus-like
melodies, but it costs strict validity against T=0.8 (0.813 vs 0.844 on the test
set) and occasionally writes a whole song out of key — one sample in the
listening set had 42% of its chord roots outside the declared key. Drawing four
samples and keeping the best one removes that tail without any training.

**Result: best-of-4 dominates both single-sample settings on every axis that
matters, at 4× inference cost and no training.**

| metric | corpus | T=0.8 | T=1.0 | **T=1.0 best-of-4** |
|---|---|---|---|---|
| strict-valid ABC | 1.000 | 0.844 | 0.813 | **0.978** |
| no lyric-alignment error | 1.000 | 0.911 | 0.920 | **0.982** |
| exact structure (labels + bars) | 1.000 | 0.978 | 0.982 | **0.996** |
| lyric recall | 1.000 | 0.967 | 0.972 | **0.982** |
| lyrics exactly as requested | 1.000 | 0.311 | 0.267 | **0.364** |
| crammed syllables | 0.057 | 0.144 | 0.163 | **0.114** |
| melody in declared key | 0.958 | 0.988 | 0.969 | **0.995** |
| chords entirely in key | 0.771 | 0.903 | 0.841 | **0.904** |
| chord-tone agreement | 0.677 | 0.726 | 0.671 | 0.707 |
| pitch range (semitones) | 22.0 | 13.8 | 17.7 | 17.6 |
| notes per bar | 3.47 | 3.24 | 3.49 | **3.49** |
| syncopation proxy | 0.176 | 0.123 | 0.164 | 0.144 |

225 de-duplicated held-out songs, paired bootstrap CIs. Every gain above except
structure is significant at 95%; the one regression is syncopation
(−0.020 [−0.034, −0.007]), which moves away from the corpus. Pitch range and the
interval distribution are unchanged, so the selection does **not** buy validity
by picking the dullest sample — the usual failure mode of reranking.

## The selector may only use what inference has

`scripts/select_best_of_n.py` scores each sample from the prompt and the
generation alone: the requested section plan, the requested lyrics, the declared
key, and the generation's own syntax. `SAFE_SIGNALS` is a whitelist and
`signal()` raises on anything outside it, so a reference-derived quantity such as
`melody_distance_to_reference` — which sits in the same metrics dict — cannot be
read by accident. A rule that wins here is one that would work on a new song
with no ground truth.

The winning rule is a weighted sum:

```
2.0·strict_valid + 1.5·section_plan_exact + 1.5·lyric_recall
  + 1.0·mean(melody_in_key, chord_roots_in_key) + 0.5·chord_tone
  − 1.0·crammed_syllables − 1.0·early_eos
```

## Four rules, chosen on validation

Rules were compared on the fixed 64-song validation subset (the same one the
round-2 decoding sweep used) and only the winner was run on the test set, so the
rule is not fitted to the test songs.

| rule | strict valid | structure | lyric recall | crammed | pitch range | distinct bars |
|---|---|---|---|---|---|---|
| `seed0` (single sample) | 0.766 | 0.984 | 0.964 | 0.177 | 17.45 | 0.825 |
| `valid` (first valid sample) | 0.938 | 0.984 | 0.966 | 0.166 | 17.08 | 0.820 |
| `lex` (lexicographic) | 0.938 | 1.000 | 0.992 | 0.175 | 17.08 | 0.831 |
| **`sum` (weighted)** | **0.938** | **1.000** | 0.979 | **0.124** | **18.00** | **0.850** |
| corpus | 1.000 | 1.000 | 1.000 | 0.064 | 22.13 | 0.893 |

`valid` recovers validity but nothing else; `lex` maximises lyric recall but
leaves cramming alone; `sum` is the only rule that improves cramming and
musical variety at the same time, so it was the one applied to the test set.

## Where the ceiling is

On the test set **98.2% of songs have at least one strict-valid sample among the
four**, and the selector achieves 97.8% — it is within 0.4 points of the ceiling
this sample budget allows. More samples would raise the ceiling slightly; a
better scorer would not. The chosen seed is spread evenly (55 / 60 / 41 / 69 of
225), so the gain is real sampling variance, not one lucky seed.

Cramming is the one defect this does not solve: 0.163 → 0.114 against a corpus
0.057. Reranking can only pick the least crammed of four samples that were all
drawn from the same distribution; making the model stop cramming is what R3-A
(syllable budget in the prompt) is for.

## Cost and reproduction

2.1 GPU-h on one L40S: 36 min for 64 validation songs × 4 samples, 91 min for
225 test songs × 4 samples, both at batch 16.

```bash
CKPT=/gscratch/ark/pingw220/qwen_abc_r2_offload/runs/e3b_201131/final_model
DATA=data/generated/abc_v2_20260915_120927
CKPT=$CKPT DATA=$DATA OUT=<evals>/bon_e3b_test_T1.0 FMT=v2 \
  ARGS="--split test --seeds 4 --temperature 1.0 --top-p 0.95" sbatch scripts/slurm/eval.sbatch

python scripts/select_best_of_n.py --generations <evals>/bon_e3b_test_T1.0/generations \
  --out experiments/bon_r3b_<ts>/test_sum --rule sum
python scripts/generate_eval.py --checkpoint $CKPT --data-dir $DATA --format v2 --split test \
  --num-songs 0 --seeds 1 --aggregate-only \
  --output-dir experiments/bon_r3b_<ts>/test_sum \
  --read-generations experiments/bon_r3b_<ts>/test_sum/generations
```

Run: `experiments/bon_r3b_20260918_100234`, jobs 40277675 / 40277676.
