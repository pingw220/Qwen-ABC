# Explicit Structural State (ESS) ablation

**ESS** = the ABC-v2 representation: after every `P:label` a comment `% section i/N | B bars`, and
before every bar a countdown remark `[r:k]` (bars left in the section). The prompt repeats
`section i/N` per section. Nothing else about the model changes.

This file separates **training / representation** changes from **inference-time selection**.
The historical sequence 0.529 → 0.658 → 0.858 → 0.978 → 0.996 mixes both; the last step
(selector@4) is not a model change and is reported only in `FAIR_DECODING.md`.

Tables: `tables/ess_ablation_T08.*`, `tables/ess_ablation_T10.*`,
`tables/ess_ablation_deltas_T0{8,10}.*`; figure `figures/fig2_ess_ablation.*`; code
`paper_eval/ess.py`. 225 test songs, song-level 95% bootstrap (10,000), paired where a
difference is shown.

## Table 4a — canonical decoding (T=1.0 / top-p 0.95), mean of 4 samples per song (new this round)

| model | change | exact structure | sections exact | \|total bars − req.\| | early EOS | late-section failure | lyric recall | strict validity | countdown = bars left |
|---|---|---|---|---|---|---|---|---|---|
| E0 | ABC-v1, raw boundaries | 0.552 [0.511, 0.593] | 0.913 | 2.82 | 0.012 | 0.080 | 0.975 | 0.609 | – |
| E1 | + cleaning + ESS | 0.844 [0.818, 0.870] | 0.975 | 1.22 | 0.001 | 0.017 | 0.971 | 0.574 | 0.980 |
| E1-long | ESS, 758 updates (matched control) | 0.853 [0.824, 0.880] | 0.977 | 0.75 | 0.004 | 0.021 | 0.980 | 0.700 | 0.979 |
| **E3b** | ESS + late-section reconstruction | **0.982 [0.973, 0.991]** | 0.995 | 0.21 | 0.001 | 0.005 | 0.970 | 0.777 | 0.999 |

Paired steps (N = 225):

| step | exact structure | strict validity | lyric recall | late-section failure |
|---|---|---|---|---|
| E1 − E0 (cleaning + ESS) | **+0.292 [+0.247, +0.338]** | −0.034 [−0.076, +0.007] | −0.004 [−0.009, +0.002] | **−0.063 [−0.079, −0.048]** |
| E1-long − E1 (+50% updates) | +0.009 [−0.022, +0.040] | **+0.126 [+0.080, +0.171]** | **+0.008 [+0.004, +0.012]** | +0.004 [−0.004, +0.013] |
| E3b − E1-long (reconstruction objective, matched updates) | **+0.129 [+0.101, +0.157]** | **+0.077 [+0.037, +0.117]** | **−0.009 [−0.013, −0.005]** | **−0.015 [−0.024, −0.007]** |

## Table 4b — round-2 decoding (T=0.8), one sample per song (existing, re-scored)

| model | change | exact structure | early EOS | late-section failure | lyric recall | strict validity |
|---|---|---|---|---|---|---|
| E0 | ABC-v1 | 0.529 [0.462, 0.596] | 0.040 | 0.097 | 0.955 | 0.711 |
| E1c | + boundary cleaning only | 0.658 [0.596, 0.720] | 0.036 | 0.083 | 0.955 | 0.640 |
| E1 | + ESS | 0.858 [0.809, 0.902] | 0.000 | 0.019 | 0.964 | 0.631 |
| E1-long | ESS, 758 updates | 0.809 [0.756, 0.858] | 0.009 | 0.032 | 0.978 | 0.724 |
| **E3b** | ESS + reconstruction | **0.978 [0.956, 0.996]** | 0.000 | 0.004 | 0.967 | 0.844 |
| E3 | ESS + reconstruction at 4× batch | 0.867 | 0.009 | 0.023 | 0.916 | 0.502 |
| E0 (seed 2) / E1 (seed 2) | training-seed replicates | 0.613 / 0.938 | 0.089 / 0.000 | | | |

(All T0.8 numbers reproduce `LONG_STRUCTURE_EXPERIMENTS.md` exactly. E1c and E3 exist only at
T0.8: their weights were deleted after the 09-15 storage incident.)

## Measured facts

1. **ESS is the largest single effect.** Cleaning + ESS raise exact structure by +0.29 at T1.0
   (+0.33 at T0.8); cleaning alone accounts for ~0.13 of it at T0.8. The effect exceeds the
   training-seed spread (E0 seeds 0.53/0.61, E1 seeds 0.86/0.94) in every pairing.
2. **The countdown is used, not decorative.** 98% of `[r:k]` counters equal the bars actually left
   in the written section (E1), 99.9% for E3b; E0 has no counter and its bar-count error is 2.8
   bars per song.
3. **The reconstruction objective buys structure at matched updates** (+0.129 over E1-long at
   T1.0; +0.169 at T0.8), while **more updates alone do not** (E1-long − E1: +0.009, n.s., at T1.0;
   the apparent −0.05 at T0.8 was single-sample noise). More updates buy validity and lyric recall.
4. **Costs.** The reconstruction objective costs a small but resolved amount of lyric recall
   (−0.009 vs E1-long). ESS itself does not measurably change validity at T1.0 (−0.034, CI crosses 0).
5. **Batch matters for the objective**: the same mixture at 4× tokens/update (E3) reaches only 0.867.

## Interpretation

Writing the structural state into the sequence converts a counting problem the model solves
unreliably into a copying problem it solves almost always; the late-section objective then trains
exactly the conditional ("write section *i* with *B* bars given everything around it") that
whole-song generation needs. Both are training/representation effects. Selection adds +0.016 on
top (FAIR_DECODING.md) and is not part of this ablation.
