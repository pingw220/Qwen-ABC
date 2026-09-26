# Long-range failure analysis

Does generation degrade toward the end of a song, and does ESS fix long-range generation or only
the average? Each *requested* section is placed at its midpoint in normalized song progress
(requested bars), bucketed into fifths, so every model is measured on the same axis. Output section
*i* is compared with requested section *i*. Song-level means within a bucket, bootstrap over songs.
Code `paper_eval/sections.py`, `paper_eval/ess.py`; tables `tables/long_range_T08.*`,
`tables/long_range_T10.*`, `tables/long_range_trend.*`; figure `figures/fig3_failure_vs_position.*`.

## Section failure (label or bar count wrong) by position

| model | 0-20% | 20-40% | 40-60% | 60-80% | 80-100% |
|---|---|---|---|---|---|
| E0, T0.8 (pre-ESS) | 0.063 | 0.107 | 0.107 | 0.119 | 0.096 |
| E0, T1.0 ×4 | 0.079 | 0.110 | 0.090 | 0.117 | 0.070 |
| E1, T1.0 ×4 (ESS) | 0.018 | 0.032 | 0.032 | 0.032 | 0.010 |
| E1-long, T1.0 ×4 | 0.018 | 0.028 | 0.023 | 0.029 | 0.017 |
| **E3b, T1.0 ×4 (ESS + late reconstruction)** | **0.003** | **0.005** | **0.005** | **0.006** | **0.004** |

## Other failures by position (T1.0, mean of 4 samples; 95% CIs in the tables)

| model | metric | first 20% | last 20% |
|---|---|---|---|
| E0 | wrong label | 0.001 | 0.012 [0.005, 0.021] |
| E0 | premature EOS (section never written) | 0.001 | 0.009 [0.003, 0.017] |
| E0 | lyric omission | 0.028 | 0.061 [0.048, 0.075] |
| E1 / E3b | premature EOS | 0.000 | 0.000 / 0.000 |
| E1 / E3b | countdown ≠ bars left | 0.011 / 0.001 | 0.004 / 0.000 |
| E3b | lyric omission | 0.032 | 0.054 [0.043, 0.065] |
| E3b | cramming | 0.160 | 0.230 [0.214, 0.246] |
| all | bars with wrong duration (syntax/timing) | ≤0.003 | ≤0.003 |

Paired last-fifth − first-fifth changes (`tables/long_range_trend.*`), T0.8 single sample:
E0 lyric omission +0.115 [+0.077, +0.157], wrong label +0.036 [+0.015, +0.062], section failure
+0.035 [−0.002, +0.074]; E3b section failure −0.003 [−0.010, +0.003], lyric omission +0.053
[+0.025, +0.082].

## Measured facts

1. **Pre-ESS structural failures are spread over the whole song, not concentrated at the end.**
   E0's section failure is already 6-8% in the first fifth and peaks mid-to-late song
   (0.11-0.12 at 60-80%); the first-to-last change is not resolved (+0.035, CI crosses 0 at T0.8;
   −0.009 at T1.0).
2. **What *is* end-loaded before ESS**: wrong labels (+0.036), premature EOS (0 → 2.9% at T0.8,
   0.1 → 0.9% at T1.0) and lyric omission (+0.115 at T0.8). These are the "running out of plan"
   failures: the model loses track of where it is and either stops or drops the remaining lyrics.
3. **ESS removes the positional failures everywhere**: premature EOS is 0 in every bucket for E1,
   E1-long and E3b, wrong labels stay ≤0.5%, and E3b's section failure is flat at 0.3-0.6% from the
   first to the last fifth.
4. **What remains end-loaded after ESS is lyric realization, not structure**: lyric omission
   (0.03 → 0.05) and cramming (0.16 → 0.23) still rise toward the end for every model, ESS or not.
   The structural state tracks bars, not syllables.

## Interpretation

ESS improves long-range *structural* reliability rather than only the average: the failures that
grow with position without it (label drift, premature termination) disappear with it, and the
residual structure error of E3b does not depend on position. The claim "ESS fixes late-song
degradation" is too strong for lyrics: late-song lyric omission and cramming persist, consistent
with LYRIC_FAILURES.md (cramming is decided when the melody is written and ESS carries no
syllable budget).
