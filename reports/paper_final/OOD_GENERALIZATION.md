# OOD / compositional generalization of control

Tables `tables/ood_generalization.*` (Table 6), `tables/ood_deltas.*`; per-sample
`data/ood_samples.parquet`; corpus statistics `data/train_distribution.json`; figure
`figures/fig5_ood_adherence.*`; code `paper_eval/ood.py`, `paper_eval/ood_analysis.py`. Seed S1, all
225 test songs (a condition skips songs where it is undefined, e.g. no verse before the first chorus;
every skip is logged in `experiments/paper_final/gen/<model>/skipped_ood.json`).

## What the training distribution makes rare

From the 10,243 training specs: section length is nearly independent of label (P(bars | verse) /
P(bars) = 0.7-1.3 for every common length) and every length from 3 to 17 bars holds ≥ 1.1% of
sections, so "a 5-, 7- or 13-bar section" is **in-distribution** in this pseudo-labelled corpus and
was *not* used as OOD. What is rare:

| regime | condition | why it is rare |
|---|---|---|
| compositional | chorus-first opening (first verse ↔ first chorus) | labels and lengths common; intro→chorus opens 84/10,243 songs |
| compositional | verse-final ending (last verse moved to the end) | verse→end in 155/10,243 songs |
| extrapolative | target section max(bars+8, 20) bars | each length ≥ 20 bars < 0.55% of sections |
| extrapolative | target section max(2×bars, 28) bars | < 0.1% each |
| extrapolative | +3 repeats of the last chorus | section counts up to 20 |
| extrapolative | +6 repeats | section counts up to 23; corpus maximum 17 |
| extrapolative | tempo 60 BPM (fast songs) / 240 BPM (slow songs) | outside the corpus 0.5-99.5% range 65-231 |

## Table 6 — exact whole-plan adherence by regime (seed S1, 95% CI)

| model | in-distribution | compositional (pooled) | extrapolative (pooled) |
|---|---|---|---|
| **Qwen-ABC E3b** (ESS + reconstruction) | 0.991 [0.978, 1.000] | 0.984 [0.969, 0.996] | **0.956 [0.940, 0.971]** |
| Qwen E1 (ESS) | 0.853 [0.804, 0.898] | 0.900 [0.871, 0.926] | 0.806 [0.780, 0.832] |
| Qwen E0 (no ESS) | 0.542 [0.476, 0.609] | 0.565 [0.513, 0.616] | **0.276 [0.248, 0.308]** |
| MuPT | 0.338 [0.276, 0.400] | 0.296 [0.235, 0.357] | 0.231 [0.197, 0.267] |

Per condition (exact plan; for the long-section conditions the target section's bars in brackets):

| condition | E3b | E1 | E0 | MuPT |
|---|---|---|---|---|
| own plan (ID) | 0.991 | 0.853 | 0.542 | 0.338 |
| chorus-first | 1.000 | 0.911 | 0.577 | 0.296 |
| verse-final | 0.973 | 0.891 | 0.566 | – |
| section ≥ 20 bars | 0.987 (target 1.000) | 0.839 (0.982) | 0.308 (0.670) | 0.290 (0.830) |
| section ≥ 28 bars | 0.973 (target 1.000) | 0.777 (0.987) | 0.134 (0.254) | – |
| +3 choruses | 0.973 | 0.839 | 0.277 | 0.089 |
| +6 choruses | **0.871 [0.826, 0.911]** | 0.741 | 0.062 | – |
| tempo 60/240 (header followed) | 0.978 (1.000) | 0.831 (1.000) | 0.604 (1.000) | 0.316 (1.000) |

Paired OOD − ID for E3b (exact plan): chorus-first +0.009, verse-final −0.018, ≥20 bars −0.004,
≥28 bars −0.018, +3 choruses −0.018, +6 choruses **−0.121 [−0.165, −0.080]**, tempo −0.013. For E0:
≥20 bars −0.237, ≥28 bars −0.411, +3 −0.268, +6 **−0.482**, compositional +0.02-0.03 (n.s.).

## Measured facts

1. **Compositional recombination is not a problem for any Qwen model**: chorus-first and verse-final
   plans are followed as well as the song's own plan (differences ≤ 0.03, n.s.). MuPT is poor
   everywhere.
2. **Extrapolation separates the models.** E3b loses 0-2 points on 20-28-bar sections, +3 choruses
   and extreme tempos; E0 (no ESS) loses 24-48 points on the same requests.
3. **The boundary for E3b is section count beyond the corpus.** On +6 choruses it is exact on 195/224
   songs. Plans with more sections than any training song (> 17) are exact 57/75 = 0.76, vs 138/149 =
   0.93 for ≤ 17. All 29 failures have the right number of sections and a wrong bar count or label;
   none terminates early or adds a section.
4. Adherence decreases smoothly with plan surprisal for every model (figure 5, right); the slope is
   shallow for ESS models and steep for E0 and MuPT.

## Interpretation

The explicit structural state generalizes: because the model copies `section i/N | B bars` and
counts down `[r:k]`, requests with unseen lengths or orderings do not need to be memorized plans.
The residual boundary (N > 17) is a count the model has never written; the counter itself survives
(no count errors), but bar/label fidelity starts to slip there. E0's collapse on long sections and
many sections is the same counting failure ESS was designed to remove, and it gets worse off
distribution.
