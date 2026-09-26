# Pseudo-label (annotation) noise

Every target in this project is an automatic transcription. How much of the measured error is the
model's, and how much is the label's? Tables 7 = `tables/annotation_noise.*`,
`tables/annotation_noise_attribution.*`, `tables/annotation_noise_dose.*`; per-song indicators
`data/annotation_noise_songs.json`; code `paper_eval/annotation_noise.py`.

**Human data available: none beyond the existing 34-song clean subset**, which was *selected* by 12
rule filters plus a manual reading of each candidate (CLEAN_TEST_SUBSET.md) — it was checked, not
re-annotated. A 100-song re-annotation package is prepared in `human_annotation_package/`
(instructions, JSON schema, offline annotation UI, `paper_eval/annotation_rescore.py`); **no human
labels have been collected, and none are fabricated.**

## Table 7: all 225 pseudo-labelled test songs vs the 34-song clean subset (single sample)

| system | metric | all 225 | clean 34 | clean − other 191 |
|---|---|---|---|---|
| Qwen-ABC E3b | exact structure | 0.982 [0.973, 0.991] | **1.000** | +0.021 [+0.012, +0.031] |
| | strict validity | 0.777 [0.743, 0.809] | 0.868 [0.794, 0.934] | +0.107 [+0.024, +0.181] |
| | lyric recall | 0.970 [0.963, 0.976] | 0.977 [0.972, 0.983] | +0.008 [−0.001, +0.019] |
| | cramming | 0.173 | 0.154 | −0.022 [−0.041, −0.004] |
| Qwen-ABC E3b selector@4 | exact structure / validity | 0.996 / 0.978 | 1.000 / 1.000 | +0.005 / +0.026 |
| Qwen E0 (no ESS) | exact structure | 0.552 [0.511, 0.593] | 0.713 [0.625, 0.794] | +0.190 [+0.090, +0.281] |
| Qwen E1 (ESS) | exact structure | 0.844 | 0.868 | +0.027 [−0.041, +0.089] |
| MuPT | exact structure | 0.341 | 0.404 | +0.075 [−0.039, +0.188] |
| | lyric recall | 0.483 | 0.515 | +0.038 [−0.001, +0.079] |
| MIDI-LLM | exact structure (= generated) | 0.884 | 0.971 | +0.101 [+0.022, +0.170] * |
| | lyric recall | 0.734 | 0.832 | +0.115 [+0.043, +0.176] |

\* MIDI-LLM's shortfall is almost entirely G2P refusal of code-switched songs, and the clean subset
has almost no code-switching (2.9% vs 18.2%) — a front-end effect, not label noise.

## Dose-response: structure failure vs number of failed pseudo-label quality rules

| system | 0 rules (n=42) | 1 (n=84) | 2 (n=45) | 3+ (n=54) | Spearman ρ |
|---|---|---|---|---|---|
| Qwen-ABC E3b | **0.000** | 0.018 | 0.022 | 0.028 | +0.14 [+0.02, +0.26] |
| Qwen E0 | 0.351 | 0.432 | 0.461 | 0.537 | +0.19 [+0.06, +0.32] |
| Qwen E1 | 0.149 | 0.161 | 0.150 | 0.157 | +0.01 [−0.11, +0.14] |
| MuPT | 0.583 | 0.646 | 0.639 | 0.755 | +0.20 [+0.07, +0.33] |

The corpus' own confidence fields separate less: E3b structure failure is 0.024 for key-ensemble
agreement "high" vs 0.011 for "medium", and 0.016 (tier A, n=183) vs 0.024 (tier B, n=42) — no
resolved difference.

## Share of error that is absent on clean songs (upper bound on label-noise-attributable error)

| system | metric | error, all | error, clean | share absent on clean |
|---|---|---|---|---|
| E3b single | exact structure | 0.018 | 0.000 | 1.00 (degenerate CI: 0 errors on clean) |
| E3b single | strict validity | 0.223 | 0.132 | 0.41 [0.06, 0.70] |
| E3b single | lyric recall | 0.030 | 0.023 | 0.24 [−0.04, 0.46] |
| E0 | exact structure | 0.448 | 0.287 | 0.36 [0.15, 0.54] |
| MuPT | exact structure | 0.659 | 0.596 | 0.10 [−0.07, 0.27] |

## Measured facts

1. **All of E3b's residual single-sample structure error (1.8%) occurs on songs with at least one
   pseudo-label quality problem**; on the 42 songs that pass every rule and on the 34 clean songs it
   is 0.
2. Roughly 40% of E3b's strict-validity failures disappear on clean songs; lyric-recall loss barely
   changes (not resolved).
3. For weaker models label noise explains little of the error level: MuPT still fails 58% of the
   plans on songs that pass every quality rule (and 60% on clean songs), E0 35%, E1 15% — those are
   model errors. Noisy labels add failures on top (MuPT 0.58 → 0.76 from 0 to 3+ failed rules,
   ρ = +0.20; E0 0.35 → 0.54), but E1's failures do not depend on label quality at all (ρ = +0.01).
4. The clean subset is also **simpler** (shorter, fewer sections, narrower range, little
   code-switching), so every "share absent on clean songs" is an **upper bound** on label-noise-
   attributable error.

## What human annotation would add

Re-annotating the ~100 prioritized songs (34 anchors, 56 highest-priority, 10 random controls) would
turn the upper bound into an estimate: for each model deviation, whether the model or the label is
wrong. `paper_eval/annotation_rescore.py` computes pseudo-GT vs human-GT metrics with paired CIs
once `human_annotation_package/annotations/*.json` exist. Status: **PENDING (no annotations)**.
