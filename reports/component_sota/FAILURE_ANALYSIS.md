# Failure analysis

Every failure is counted as a failure in the tables: success rates are 0, and recall is 0 for the
missing song. Nothing was re-drawn or dropped. The counts come from `data/*.parquet`, the raw
outputs and the render logs.

## 1. Generation failures (225 songs × 4 seeds = 900 samples per source)

| source | `orig` failures | `lyrics_all` failures | cause |
|---|---|---|---|
| Qwen Full (E3b) | 1 / 900 | – | degenerate lyric loop: output truncated at the 8,496-token budget with 8,379 overflow lyric tokens (song `27o7ErWiozSj47o3dnPvVf`, S3) |
| Qwen Melody-Only | 0 / 900 | 0 | – |
| CSL-L2M official, whole | 289 / 900 = 56 unsupported meter + **233 "stuck"** | 298 | upstream rejection sampler gives up after 5 attempts; failure rate grows with lyric length: 1% (< 30 lines) → 26% → 56% → **84% (≥ 50 lines)** |
| CSL-L2M official, chunked | 56 + 3 | 56 + 6 | 3/4 songs; rare stuck chunks |
| CSL-L2M retrained, whole | 56 + 11 (all on songs with ≥ 50 lines) | 56 + 17 | long songs |
| CSL-L2M retrained, chunked | 56 + 0 | 56 + 0 | 3/4 songs only |

CSL-L2M's 56 = 14 songs in 3/4 × 4 seeds. Its representation has 64 positions per 4/4 bar and
cannot express 3/4.

## 2. Content failures (successful samples)

| failure | E3b | Melody-Only | CSL-L2M | reference |
|---|---|---|---|---|
| crammed syllables (mean share) | 0.173 | 0.126 | 0.000 (by construction) | 0.057 |
| samples with > 30% crammed syllables | 8.7% | 3.7% | 0% | 1.3% |
| same, after a lyric swap | 12.1% | 6.9% | 0% | – |
| plan not followed exactly (labels or bar counts) | 1.8% | 7.9% | n/a (no plan input) | – |
| Latin words dropped from the input | 0 | 0 | 500 words in 40 of 225 songs (`meta.latin_words_dropped`); lyric recall is also reported on Han only | – |

Cramming is Qwen's main content failure, and it propagates to audio. Qwen melodies have higher PER
under both singing renderers than CSL-L2M-retrained melodies: +0.048 [+0.037, +0.060]
(`SVS_CROSS_RENDER.md`).

## 3. Section infill (E3b; 2 seeds)

Samples whose target section missed its label or bar count:

| task | infill | whole-song regeneration |
|---|---|---|
| recon_late (trained position) | 0 / 448 | – |
| recon_early (untrained position) | 9 / 258 | – |
| edit_resample | 5 / 444 | 3 / 444 |
| edit_lyrics | 7 / 444 | 3 / 444 |
| edit_extend (+4 bars) | **13 / 444** | 0 / 444 |

Extending a section while both neighbours are fixed is the hardest infill case. In untouched sections
the only note changes come from ties across the edited boundary: 61 of 2,850 sections, all adjacent
to the target.

## 4. Harmonizers

| harmonizer | failures | cause |
|---|---|---|
| Qwen Melody→Chord / +lyrics | 0 generation failures in 2,671 outputs | 6 of 2,671 outputs (0.2%) wrote a different number of bar lines than the melody has; they still parse, and the missing bars get no new chord |
| AccoMontage2 | 14 / 225 on every melody source | 3/4 meter unsupported |
| (both, on CSL-L2M melodies) | 15 / 14 | no melody to harmonize |

## 5. Audio

| stage | failures | cause |
|---|---|---|
| SoulX-Singer | 1 song × 5 sources | lyric character 揹 missing from its phoneme set (`KeyError: 'zh_揹'`) |
| FastSinger | 1 CSL-retrained melody | "Input length exceeds the maximum length": a long CSL phrase with no line break |
| backing (MIDI-SAG + MuseControlLite) | 0 / 288 | – |
| key detector (chroma KS) | – | recovers the draft's own key in only 53% of renders; absolute key accuracy is detector-limited (`AUDIO_CONTROL_EVALUATION.md`) |
| beat tracker (BeatNet) | – | 25–31% half/double-time readings on tempo-changed renders (strict vs octave-tolerant BPM) |

## 6. Measurement failures found and fixed during the round

All of these are also listed in `DATA_PROTOCOL.md`. None of them changed a system's output; each
would have corrupted a metric.

1. **ASR without VAD** transcribed only a fragment of each song (PER 0.91 for every system,
   including the reference). Fixed with fsmn-vad; the old outputs were kept as `*.asr_novad.json`.
2. **SongEval** keys results by file stem, so all 288 `mix.wav` results collided. Fixed with
   unique link names.
3. **BTC** crashed under the installed NumPy (`np.float`) and torch (`weights_only`). Fixed with a
   wrapper; upstream code is unchanged.
4. **RMVPE** returns `(f0, cents)`. Early sidecars stored both rows; the evaluator uses row 0.
5. **CSL structure metric:** failed CSL samples were first scored as "exact structure = 0". It is
   now n/a for systems without plan input.
6. **Table overwrite:** the `lyrics_all` melody evaluation would have overwritten Table 1. Tables are
   now suffixed by condition.
7. **Relative checkpoint path** made the first CSL-retrained inference jobs fail immediately
   (resubmitted; recorded in `COMPUTE_LEDGER.csv`).
8. **Upstream CSL-L2M validation bug** (`batch_struct` unbound when `use_musc_ctls=False`) crashed
   the first retraining run. Validation was disabled; no test-set model selection took place.
