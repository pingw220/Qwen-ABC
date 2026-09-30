# Lyric conditioning: lyric intervention vs reseed (Table 2)

**Question:** when the lyrics are replaced, does the melody change beyond what resampling alone
changes? Does any model follow Mandarin tones?

**Code:** `paper_eval/component_sota/lyric_conditioning.py`. It reuses the paper-final estimators:
`paper_eval.intervention_analysis`, `interventions` and `tone_melody`.

**Tables:** `tables/lyric_conditioning.*`; the melody quality of the swapped samples is in
`tables/lyrics_to_melody_lyrics_all.*`.
**Data:** `data/lyric_conditioning.parquet`.
**Figure:** `figures/fig3_lyric_vs_reseed.*`.

## Design

- **Lyric swap:** for every test song, 4 samples with the original lyrics (S1–S4) and 4 with every
  section's lyrics replaced (`lyrics_all`).
  - The replacement comes from another held-out song, cut to each section's original syllable count.
  - The donor text, the seeds and all other controls are **identical across systems**.
- **Effect:** cross distance (orig_i vs swap_j, i ≠ j) minus within distance (orig_i vs orig_j, i ≠ j),
  per song, with a 10,000-resample song bootstrap.
  - The ratio is mean cross / mean within.
  - The seed-paired ratio d(orig_S1, swap_S1) / d(orig_S1, orig_S2) shows how much a shared seed dominates.
  - Melodies are compared with chords stripped.
- **Tone test:** agreement between Mandarin tone contours and pitch movement.
  - A model that uses tones should agree more with the lyric it was given than with the other lyric,
    in both directions.
  - "Own − other (orig)" and "own − other (swap)" should both be > 0.
- **CSL-L2M** uses a single seed per (song, seed) for both conditions. Its 3/4 songs are excluded.

## MEASURED FACT

| system | N | new-lyric recall | old-lyric leakage (chance) | melody: swap − reseed | contour | rhythm | ratio (melody) | tone own−other, orig / swap |
|---|---|---|---|---|---|---|---|---|
| Qwen Full (E3b) | 225 | 0.983 | 0.116 (0.115) | +0.003 [+0.000, +0.005] | +0.006 [+0.002, +0.010] | +0.007 [+0.003, +0.012] | 1.003 | −0.001 / −0.001 |
| Qwen Melody-Only | 225 | 0.989 | 0.117 (0.116) | +0.005 [+0.002, +0.007] | +0.009 [+0.005, +0.012] | **+0.015 [+0.010, +0.020]** | 1.005 | +0.000 / −0.003 |
| CSL-L2M official, chunked | 211 | 0.999 | 0.116 (0.116) | +0.001 [−0.002, +0.003] | +0.002 [−0.002, +0.006] | +0.000 [−0.008, +0.009] | 1.001 | +0.003 / −0.000 |
| CSL-L2M retrained, chunked | 211 | 1.000 | 0.116 (0.116) | +0.004 [+0.001, +0.007] | +0.009 [+0.004, +0.014] | +0.012 [+0.004, +0.021] | 1.004 | −0.001 / +0.003 |
| CSL-L2M official, whole | 149 | 0.999 | 0.116 (0.116) | −0.000 [−0.004, +0.003] | −0.001 | +0.001 | 1.000 | +0.007 [+0.000, +0.014] / −0.006 [−0.013, +0.001] |
| CSL-L2M retrained, whole | 209 | 1.000 | 0.116 (0.116) | +0.003 [−0.001, +0.007] | +0.009 [+0.003, +0.016] | +0.017 [+0.005, +0.031] | 1.003 | −0.003 / +0.000 |

- The largest ratio on any distance is **1.12**: Melody-Only's |Δ notes/bar|. On melody, contour
  and pitch class, every system is within 1.00–1.06 of its reseed floor.
- **Old-lyric leakage equals the chance level** for every system: 0.116 vs 0.116. After a swap, no
  model keeps singing the old words.
- Seed-paired ratios are 0.966–0.997. A shared seed makes the swapped sample *closer* to its original
  than an independent reseed, most visibly for CSL official (0.966 [0.953, 0.979]).
- **Tones:** no system shows the crossover pattern. The only CI that excludes 0 is CSL official
  whole-song "orig" (+0.007 [+0.000, +0.014]), and its swap counterpart has the opposite sign
  (−0.006 [−0.013, +0.001]).

## INTERPRETATION

- **Replacing the lyrics changes the melody barely more than drawing a new seed, for every system.**
  - This extends the paper-final result (lyric swap ≈ reseed) to the melody-only Qwen and to both
    CSL-L2M variants.
  - It is a property of these lyrics→melody models, not of joint lead-sheet training.
- The small real effects are on **rhythm and contour**, not pitch content:
  - Qwen Melody-Only (rhythm +0.015);
  - CSL-L2M retrained (+0.012 to +0.017).

  This fits syllable-count and phrase-length conditioning (how many notes and where), not semantic
  or tonal conditioning.
- **CSL-L2M does not condition more strongly on lyrics than Qwen.**
  - Official CSL-L2M shows **no** detectable lyric effect.
  - Retrained on our data it becomes similar to Qwen Melody-Only.
- **No model uses Mandarin tones measurably.** For CSL-L2M this is expected, since the released
  lyrics-only model has no tone input (the tone embeddings are absent from the checkpoint; see
  `AUDIT.md`). Qwen never saw tone labels either.
