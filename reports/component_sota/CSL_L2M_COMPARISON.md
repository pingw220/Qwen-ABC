# CSL-L2M comparison

CSL-L2M (AAAI-25) is the external lyrics→melody system. It is run in two ways:

- **EXTERNAL PRETRAINED SYSTEM COMPARISON:** the authors' released lyrics-only checkpoint,
  trained on *their* corpus;
- **controlled architecture comparison:** the same code, retrained on *our* split.

Neither variant can be claimed to be the "true" CSL-L2M score on our data. The official model
faces a domain shift. The retrained model uses the authors' trainer with hyper-parameters as
shipped, and was not tuned on our data.

**Code:** `paper_eval/component_sota/csl_l2m.py`. Upstream code is imported unchanged: the
adapter sets `sys.argv` and executes the exact constructor block of `generate.py`.
**Tables:** `tables/lyrics_to_melody.*` and `tables/lyrics_to_melody_paired.*`.
**Raw outputs:** `experiments/component_sota/csl/{csl_off,csl_offc,csl_rt,csl_rtc}/{orig,lyrics_all}/`.

## Variants

| name | checkpoint | input | runs |
|---|---|---|---|
| official, whole song ("as released") | `pretrained_CSLL2M_onlyLyrics.pt` (released; loaded with f_pos=f_tone=False, 0 missing keys) | all Han lyric lines of the song, one call | 225 songs × 4 seeds × {orig, lyrics_all} |
| official, section-chunked | same | consecutive sections grouped into chunks of ≤ 24 lines, one call per chunk, concatenated | same |
| retrained, whole song | `experiments/component_sota/runs/csl_l2m_retrain/params/step_29000-RC_0.581.pt` | as above | same |
| retrained, section-chunked | same | as above | same |

**Retraining:**
- trainer: upstream `train_CSLL2M.py`;
- config: `configs/csl_l2m_onlylyrics_retrain.yaml`;
- data: 9,681 training songs (our train split, 4/4 only); lyric vocabulary of 3,867 characters from our train split;
- length: 12 epochs, 29,052 steps, 5.2 h on one L40S;
- final training loss about 0.56; the last checkpoint is used.

Validation was disabled (`val_interval` 1e8) because of an upstream bug: `batch_struct` is
unbound when `use_musc_ctls=False`, so the first validation step crashed. No model selection on
test data took place.

## What the adapter has to do

These steps apply to both variants and are documented in `AUDIT.md`.

- CSL-L2M accepts Han characters only. Latin words are dropped from its input
  (`meta.latin_words_dropped`), and **lyric recall is reported on all syllables and on Han only**.
- 4/4 only. The 14 songs in 3/4 (6.2%) are failures: `unsupported_meter`.
- It gets no section plan, key or tempo.
  - Its sections are derived post hoc from where each lyric line's first note lands.
  - Its key is estimated from its notes (Krumhansl-Schmuckler) for the downstream harmonizers.
  - It uses the song's tempo for rendering.
  - Structure metrics are **n/a**.
- Upstream's rejection sampler gives up after 128 failed positions. We keep upstream's 5
  attempts; a song failing all 5 counts as a failure.
- Event → canonical Song conversion maps CSL's 1/64-bar grid onto our 16th-note grid.
  - Syllables CSL places closer than a 16th collide and are dropped.
  - In real CSL outputs this keeps **99.9%** of the characters (official chunked, first 449 successful samples: mean 0.9992;
    2% of songs below 0.99).
  - This is checked by `tests/test_component_sota.py::test_csl_events_round_trip_keeps_sung_pitches`.

## MEASURED FACT

Values are song-level means over 4 samples on the 225 test songs, with 95% bootstrap CIs.
Failures count as 0 for recall and success. PD, DD and MD are computed on successful samples.

| system | success | PD ↑ | DD ↑ | DD timebase-inv. ↑ | MD ↓ | lyric recall | notes/bar |
|---|---|---|---|---|---|---|---|
| Qwen Melody-Only | 1.000 | **0.490** [0.473, 0.508] | **0.755** | **0.769** | **1.55** [1.51, 1.59] | **0.986** | 3.45 |
| Qwen Full (E3b) → melody | 0.999 | 0.466 [0.449, 0.483] | 0.740 | 0.765 | 1.58 | 0.970 | 3.24 |
| CSL-L2M retrained, chunked | 0.938 | 0.286 [0.271, 0.300] | 0.619 | 0.664 | 1.71 [1.66, 1.76] | 0.931 | 5.46 |
| CSL-L2M retrained, whole | 0.926 | 0.231 | 0.584 | 0.633 | 1.68 | 0.919 | 5.75 |
| CSL-L2M official, chunked | 0.934 | 0.230 [0.218, 0.242] | 0.568 | 0.630 | 1.96 [1.90, 2.02] | 0.927 | 6.49 |
| CSL-L2M official, whole (as released) | 0.679 | 0.181 | 0.517 | 0.575 | 1.66 | 0.674 | 6.37 |
| pseudo-reference | – | 1 | 1 | 1 | 0 | 1 | 3.47 |

Paired differences, on songs where both systems produced output:

| comparison | PD | DD (timebase-inv.) | MD | lyric recall (N=225) |
|---|---|---|---|---|
| Melody-Only − CSL retrained chunked (N=211) | +0.203 [+0.181, +0.226] | +0.107 [+0.093, +0.122] | −0.154 [−0.188, −0.121] | +0.055 [+0.026, +0.088] |
| Melody-Only − CSL official chunked (N=211) | +0.259 [+0.238, +0.280] | – | −0.401 [−0.453, −0.350] | +0.059 [+0.029, +0.093] |
| CSL retrained − CSL official, both chunked (N=211) | +0.056 [+0.040, +0.072] | +0.034 [+0.025, +0.042] | −0.247 [−0.295, −0.197] | +0.004 [+0.000, +0.009] |

**Interval similarity to the reference** (a key- and register-free contour measure) is
indistinguishable across all systems: 0.441–0.457.

**Failure by lyric length.** Counts are samples; 3/4 songs are excluded.

| lyric lines (songs) | < 30 (86) | 30–39 (76) | 40–49 (44) | ≥ 50 (19) |
|---|---|---|---|---|
| official, whole | 3/316 | 76/288 | 97/172 | 57/68 |
| official, chunked | 2/316 | 1/288 | 0/172 | 0/68 |
| retrained, whole | 0/316 | 1/288 | 0/172 | 10/68 |
| retrained, chunked | 0/316 | 0/288 | 0/172 | 0/68 |

**Median seconds per successful song (L40S):** official 7.0 whole / 6.3 chunked; retrained
8.5 whole / 8.3 chunked.

**Density.** CSL's own 8 shipped corpus examples have 5.1–8.0 notes per bar; ours have 3.5.
- Retraining moves CSL from 6.49 to 5.46 notes/bar, and to 5.5 per *non-empty* bar vs the
  reference's 4.4.
- CSL's representation cannot express empty bars: one `Bar` event is written per bar *change*,
  as in upstream `mid2events.py`.
- Our converter splits a note carrying k syllables into k notes.

## INTERPRETATION

- **Qwen Melody-Only exceeds CSL-L2M on every reference-based metric, whichever CSL-L2M is used.**
  - Retraining CSL-L2M on our split improves it on all three published metrics.
  - It still closes only about 20% of the PD gap (0.230 → 0.286 vs Qwen 0.490).
  - The conclusion does not depend on which CSL-L2M is used; the size of the gap does.
- The official model *as released* cannot generate long songs: it fails 84% of samples of songs
  with ≥ 50 lines. Section chunking is our accommodation, needed for a meaningful comparison.
  Its whole-song row is the honest "as released" number.
- CSL-L2M places every character by construction (one ALIGN group per character), so its lyric
  recall on successful samples is about 1.0. Its lower recall comes entirely from failures.
  Qwen instead crams 12.6% of syllables (Melody-Only), against 5.7% in the reference.
  On lyric *placement* CSL-L2M is stricter; on melody *similarity to real songs* Qwen is far ahead.
- These are distributional and reference-similarity metrics against one pseudo-reference per song.
  They measure closeness to the corpus, not listener preference. Listening-study block A,
  prepared but **not run**, is needed for any claim about perceived quality.
