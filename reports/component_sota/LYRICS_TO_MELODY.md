# Lyrics → melody (Table 1)

**Question:** is Qwen competitive as a lyrics→melody model, and does generating chords jointly
cost melody quality?

**Code:**
- `paper_eval/component_sota/melody_eval.py` (per-sample scoring; failures kept);
- `melody_metrics.py` (PD / DD / MD, following CSL-L2M's published definitions, reimplemented; plus DD_tb).

**Tables:** `tables/lyrics_to_melody.*` (Table 1) and `tables/lyrics_to_melody_paired.*`.
**Data:** `data/melody_samples_orig.parquet`.
**Figure:** `figures/fig2_lyrics_to_melody.{pdf,png}`.

## Systems and inputs

| system | receives | training |
|---|---|---|
| Qwen Full (E3b) → melody | plan (labels, bars, irregular beats), lyrics per section, key, meter, tempo | E3b, 758 updates; chords removed from its output before scoring |
| Qwen Melody-Only | the identical inputs (only the task sentence differs) | E3b's mixture without chords, the same 15,332 examples and the same recipe, **758 updates** |
| CSL-L2M official / retrained | Han lyric lines only | see `CSL_L2M_COMPARISON.md` |
| pseudo-reference | – | the transcribed test song itself |

## Metrics

- **PD / DD:** pitch and duration histogram overlap with the reference song.
- **DD_tb:** DD maximized over ×0.5/×1/×2 duration rescaling, to remove notation-timebase conventions.
- **MD:** DTW distance of mean-centred eighth-note pitch frames.
- **Generation success:** the song was produced and parsed.
- **Lyric recall:** the fraction of the input syllables sung, in order.
- **Crammed syllables:** syllables sharing a note with another syllable.
- **Exact structure:** labels and bar counts of every section as planned. Only for systems that receive the plan.
- **Distribution diagnostics:** pitch range, notes/bar, rhythm entropy, distinct bars, interval similarity.

## MEASURED FACT

Values are song-level means over 4 samples on 225 songs, with 95% bootstrap CIs.

| system | success | PD ↑ | DD ↑ | MD ↓ | lyric recall ↑ | crammed | exact structure |
|---|---|---|---|---|---|---|---|
| Qwen Melody-Only | 1.000 | **0.490** [0.473, 0.508] | **0.755** [0.745, 0.765] | **1.55** [1.51, 1.59] | **0.986** [0.982, 0.989] | 0.126 | 0.921 [0.899, 0.942] |
| Qwen Full (E3b) → melody | 0.999 | 0.466 [0.449, 0.483] | 0.740 [0.730, 0.750] | 1.58 [1.54, 1.62] | 0.970 [0.963, 0.976] | 0.173 | **0.982** [0.973, 0.991] |
| CSL-L2M retrained, chunked | 0.938 | 0.286 | 0.619 | 1.71 | 0.931 | 0.000 | n/a |
| CSL-L2M official, chunked | 0.934 | 0.230 | 0.568 | 1.96 | 0.927 | 0.000 | n/a |
| CSL-L2M official, whole song (as released) | 0.679 | 0.181 | 0.517 | 1.66 | 0.674 | 0.000 | n/a |
| pseudo-reference | – | 1 | 1 | 0 | 1 | 0.057 | 1 |

**Melody-Only − Qwen Full**, paired over 225 songs:

| metric | difference [95% CI] |
|---|---|
| PD | +0.024 [+0.007, +0.041] |
| DD | +0.015 [+0.006, +0.025] |
| MD | −0.036 [−0.057, −0.014] |
| lyric recall | +0.016 [+0.011, +0.021] |
| crammed syllables | −0.047 [−0.055, −0.038] |
| exact structure | **−0.061 [−0.084, −0.038]** |
| notes/bar | +0.209 [+0.154, +0.263] (closer to the reference's 3.47) |
| interval similarity to ref | +0.011 [+0.007, +0.016] |
| pitch range | −0.16 [−0.53, +0.21] (n.s.) |

Every paired Qwen-vs-CSL difference on success, PD, PD (pitch class), DD, DD_tb, MD, lyric
recall, rhythm entropy and distinct bars is resolved in Qwen's favour (`tables/lyrics_to_melody_paired.md`,
checked programmatically). The only unresolved comparisons are interval similarity to the reference
for E3b vs CSL official chunked (−0.005 [−0.010, +0.000]) and E3b vs CSL retrained chunked
(−0.003 [−0.008, +0.002]); Melody-Only is ahead on interval similarity against both.

## INTERPRETATION

- **Qwen is the strongest lyrics→melody system measured here,** by a wide margin over both CSL-L2M
  variants on the published metrics. See `CSL_L2M_COMPARISON.md` for the caveats: domain, timebase,
  lyric placement by construction.
- **Joint chord generation is not free, but the trade is small and two-sided.**
  - Dropping chords, with equal updates, gives slightly more reference-like melodies (PD +0.024,
    MD −0.036) and fewer crammed syllables (−4.7 points).
  - It loses 6.1 points of exact structure (0.921 vs 0.982).
  - A plausible reading: chord symbols at bar starts act as an extra bar-grid anchor that helps the
    model keep the planned bar counts, while spending capacity that would otherwise shape the melody.
    This mechanism is **not** tested here.
- All melody metrics compare against one transcribed reference per song, so they reward
  corpus-likeness. None of them measures listener preference: listening-study block A is pending,
  and no human data exists.
