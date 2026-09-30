# Component-SOTA + decomposition evaluation: final report

Branch `component-sota-eval-2026` of Qwen-ABC, 2026-09-29.

**Data:**
- **Symbolic evaluation:** all 225 held-out test songs (split 10,243 / 273 / 225).
- **Audio evaluation:** a frozen, stratified 36-song subset.

**Statistics:**
- CIs are song-level 95% percentile bootstraps with 10,000 resamples.
- Comparisons are paired over the songs where both systems produced output.
- Failures count as failures.

**Human results:** none. The listening study (`listening_study/`) is packaged and not run.

Each answer below gives **MEASURED FACT** (metric, N, estimate [95% CI], comparison, checkpoint,
artifact) and then **INTERPRETATION**.

## Checkpoints and artifacts

| key | path |
|---|---|
| E3b | `/gscratch/ark/pingw220/qwen_abc_r2_offload/runs/e3b_201131/final_model` |
| MEL | `experiments/component_sota/runs/mel_sft/final_model` |
| CH | `experiments/component_sota/runs/chord_sft/final_model` |
| CHL | `experiments/component_sota/runs/chord_lyr_sft/final_model` |
| CSL-O | `/gscratch/ark/pingw220/third_party/csl_l2m_ckpt/pretrained_CSLL2M_onlyLyrics.pt` |
| CSL-R | `experiments/component_sota/runs/csl_l2m_retrain/params/step_29000-RC_0.581.pt` |
| AM2 | `MIDI-SAG/AccoMontage2` |

Table and data paths are relative to `reports/component_sota/`.

---

### 1. Does Qwen Melody-Only match or exceed CSL-L2M?

**MEASURED FACT:** it exceeds it on every published metric.

| metric (N = 211 paired songs) | Melody-Only − CSL-L2M retrained (chunked) | Melody-Only − CSL-L2M official (chunked) |
|---|---|---|
| PD | +0.203 [+0.181, +0.226] | +0.259 [+0.238, +0.280] |
| DD (timebase-invariant) | +0.107 [+0.093, +0.122] | – |
| MD | −0.154 [−0.188, −0.121] | −0.401 [−0.453, −0.350] |
| lyric recall (N=225) | +0.055 [+0.026, +0.088] | +0.059 [+0.029, +0.093] |
| generation success (N=225) | +0.062 | – |

Levels: PD 0.490 vs 0.286 (retrained) vs 0.230 (official).
Checkpoints: MEL, CSL-R, CSL-O. Artifacts: `tables/lyrics_to_melody*.md`, `data/melody_samples_orig.parquet`.

**INTERPRETATION:** Qwen is the stronger lyrics→melody model against the reference-based metrics.
- CSL-L2M places every character on its own note, with zero cramming vs 12.6% for Qwen.
- Its melodies are more singable by SVS (see Q14–16).
- Perceived quality is untested: listening block A is pending.

### 2. Does the conclusion change with official vs retrained CSL-L2M?

**MEASURED FACT:** retraining on our split improves CSL-L2M on every published metric, N=211:

| metric | CSL-R − CSL-O (both chunked) |
|---|---|
| PD | +0.056 [+0.040, +0.072] |
| DD_tb | +0.034 [+0.025, +0.042] |
| MD | −0.247 [−0.295, −0.197] |

Retraining also removes whole-song failures on long songs: 11 vs 233 of 900. Qwen remains ahead of
both variants on all published metrics.

**INTERPRETATION:** the direction of the conclusion does not change; the gap narrows by about 20%
on PD. The official model *as released* cannot generate long songs (84% failure at ≥ 50 lyric lines).
Its whole-song numbers are the honest external baseline; chunked numbers are our accommodation.

### 3. Does full lead-sheet Qwen lose melody quality vs Melody-Only?

**MEASURED FACT:** yes, slightly. Melody-Only − E3b, N=225, equal updates (758):

| metric | difference [95% CI] |
|---|---|
| PD | +0.024 [+0.007, +0.041] |
| MD | −0.036 [−0.057, −0.014] |
| lyric recall | +0.016 [+0.011, +0.021] |
| crammed syllables | −0.047 [−0.055, −0.038] |
| **exact structure** | **−0.061 [−0.084, −0.038]** (Melody-Only is worse) |

Artifacts: `tables/lyrics_to_melody_paired.md`.

**INTERPRETATION:** a small, two-sided trade. Chords cost a little melody and lyric fidelity, and buy
structural adherence.

### 4. Is joint melody+chord generation essentially free?

**MEASURED FACT:** Joint (E3b) − Cascade-Qwen (Melody-Only → Qwen Melody→Chord), N=225:

| metric | difference [95% CI] |
|---|---|
| exact structure | +0.076 [+0.040, +0.116] |
| strong-beat chord-tone | +0.032 [+0.007, +0.057] |
| PD | −0.028 [−0.057, +0.001] |
| lyric recall | −0.015 [−0.024, −0.008] |
| cramming | +0.049 [+0.036, +0.063] |
| cadence I/V | −0.060 [−0.104, −0.018] |

Artifacts: `tables/joint_vs_cascade_paired.md`.

**INTERPRETATION:** not free, but a trade rather than a loss. Joint wins on structure and
chord–melody agreement; the cascade wins on melody and lyric fidelity and cadences.

### 5. Does CSL-L2M condition melody on lyrics more strongly than Qwen?

**MEASURED FACT:** no. Melody distance, lyric swap − reseed, 1 − LCS:

| system | N | swap − reseed [95% CI] | ratio |
|---|---|---|---|
| CSL-O (chunked) | 211 | +0.001 [−0.002, +0.003] | 1.001 |
| CSL-R (chunked) | 211 | +0.004 [+0.001, +0.007] | 1.004 |
| Melody-Only | 225 | +0.005 [+0.002, +0.007] | 1.005 |
| E3b | 225 | +0.003 [+0.000, +0.005] | 1.003 |

- The largest effects are on rhythm: Melody-Only +0.015 [+0.010, +0.020], CSL-R +0.012 [+0.004, +0.021].
- Old-lyric leakage equals chance (0.116) for all systems.

Artifacts: `tables/lyric_conditioning.md`, `data/lyric_conditioning.parquet`.

**INTERPRETATION:** for every system, replacing the lyrics changes the melody barely more than
reseeding. What lyrics do control is syllable count and phrasing (rhythm), not pitch content.

### 6. Does any model use Mandarin tones measurably?

**MEASURED FACT:** no. Tone–contour agreement, own lyric − other lyric, in the orig and swap
directions. Every system has at least one direction whose CI includes 0, and none is positive in both:

| system | orig | swap |
|---|---|---|
| E3b | −0.001 [−0.006, +0.004] | −0.001 [−0.006, +0.004] |
| Melody-Only | +0.000 | −0.003 |
| CSL-O chunked | +0.003 | −0.000 |
| CSL-R chunked | −0.001 | +0.003 |
| CSL-O whole | +0.007 [+0.000, +0.014] | −0.006 [−0.013, +0.001] (opposite sign) |

N = 149–225.

**INTERPRETATION:** no tonal conditioning is detectable. The released CSL-L2M lyrics-only checkpoint
has no tone embedding (`AUDIT.md`), and Qwen never saw tone labels.

### 7. Is section infilling better than whole-song regeneration for local edits?

**MEASURED FACT:** E3b, 222 songs × 2 seeds, paired infill − regeneration, same request:

| edit | untouched sections byte-identical | new-lyric recall | label + bars as requested |
|---|---|---|---|
| resample | +0.999 [+0.997, +1.000] | – | – |
| new lyrics | +0.970 [+0.964, +0.975] | −0.054 [−0.083, −0.028] | – |
| +4 bars | +0.948 [+0.940, +0.955] | – | −0.029 [−0.047, −0.014] |

Artifacts: `tables/infill*.md`, `data/infill.parquet`.

**INTERPRETATION:** yes, for locality, by a very large margin. Infill pays a small price in
target-section success. Boundaries are in the reference range on pitch-jump and rest proxies. Whether
the seams sound natural is listening block D, pending.

### 8. How much unrelated material does infilling preserve?

**MEASURED FACT:** untouched sections:

| | infill | regeneration |
|---|---|---|
| byte-identical | 0.997–1.000 | 0.001–0.050 |
| notes identical | 0.972–0.977 | 0.16–0.18 |
| chords identical | 0.980–0.993 | 0.005–0.059 |
| lyrics identical | 0.998 | 0.76–0.77 |

- In 2,850 untouched sections (edit_resample), every one of the 61 note differences with identical
  text sits next to the edited section: a tie across the boundary.
- recon_early (untrained positions) keeps structure 0.965 [0.930, 0.992].

**INTERPRETATION:** infilling is essentially lossless outside the target.

### 9. Does Qwen Melody→Chord match or exceed AccoMontage2?

**MEASURED FACT:** Qwen (CH) − AccoMontage2 (AM2), reference melodies, N=211:

| metric | difference [95% CI] |
|---|---|
| root accuracy | +0.113 [+0.098, +0.129] |
| chroma F1 | +0.090 [+0.078, +0.102] |
| strong-beat chord-tone | +0.115 [+0.096, +0.136] |
| cadence I/V | +0.139 [+0.103, +0.174] |
| roots in key | **−0.079 [−0.096, −0.063]** (AM2 wins) |

Success is 1.000 vs 0.938, since AM2 cannot handle 3/4.
Artifacts: `tables/melody_to_chord*.md`, `data/chords.parquet`.

**INTERPRETATION:** Qwen exceeds AccoMontage2 on all symbolic layers except diatonicity.

### 10. Is Qwen closer to the pseudo-reference chords?

**MEASURED FACT:** yes vs AccoMontage2: chroma F1 0.563 vs 0.471. But **not** vs a per-bar diatonic
heuristic: chroma F1 +0.002 [−0.007, +0.012], N=225. Qwen is ahead of the heuristic on:

| metric | Qwen − diatonic [95% CI] |
|---|---|
| progression similarity | +0.109 [+0.094, +0.123] |
| cadence agreement | +0.065 [+0.036, +0.095] |
| corpus bigram JS | 0.052 vs 0.367 |

**INTERPRETATION:** frame-level chord similarity to transcribed pseudo-labels is a weak discriminator.
Qwen's advantage lies in progressions and cadences. Part of its advantage over AccoMontage2 is
in-distribution: Qwen trained on labels of the same kind.

### 11. Are Qwen's chords compatible with the melody?

**MEASURED FACT:** strong-beat chord-tone ratio:

| chords | strong-beat chord-tone |
|---|---|
| Qwen (CH) | 0.660 [0.643, 0.675] |
| reference | 0.704 [0.686, 0.721] |
| AccoMontage2 | 0.544 |
| diatonic heuristic | 0.801 |

- Qwen − reference: −0.045 [−0.060, −0.030].
- Qwen is also between AccoMontage2 and the reference on strong-beat dissonance (0.153) and chroma
  compatibility (0.522).
- E3b's joint chords on its own melody reach 0.688, the same as re-harmonizing that melody with CH:
  +0.014 [−0.006, +0.035].

**INTERPRETATION:** Qwen's chords are compatible at close to reference level, and above AccoMontage2.
The compatibility metrics reward simplicity (the heuristic "beats" the reference), so they are a
floor, not a ranking.

### 12. Which harmonizer produces better downstream backing audio?

**MEASURED FACT:** same melody and vocal, 36 songs, Qwen − AM2:

| metric | difference [95% CI] |
|---|---|
| Chord F1 | +0.002 [−0.011, +0.015] |
| SongEval Musicality | +0.032 [−0.015, +0.076] |
| Rhythm F1 | +0.006 (n.s.) |
| key accuracy | **−0.194 [−0.361, −0.028]** (AM2 wins) |

Artifacts: `tables/backing_*.md`, `data/backing.parquet`.

**INTERPRETATION:** there is no measurable difference in audio quality or chord realization.
AccoMontage2's diatonic harmony makes the key easier to detect. The symbolic advantage of Qwen does
not show up in automatic audio metrics; listening block C is pending.

### 13. Which full system is best?

**MEASURED FACT:** `tables/joint_vs_cascade.md`, N=225.

| system | exact structure | PD | strong-beat chord-tone | cadence I/V | success | notes |
|---|---|---|---|---|---|---|
| Joint (E3b) | **0.991** | 0.473 | **0.688** | 0.676 | 1.000 | one model; infill measured |
| Cascade-Qwen | 0.916 | **0.501** | 0.656 | **0.736** | 1.000 | – |
| hybrid Qwen Mel → AM2 | 0.916 | 0.501 | 0.563 | 0.575 | 0.938 | – |
| hybrid CSL-R → Qwen Chord | n/a | 0.300 | 0.564 | 0.541 | 0.938 | – |
| Cascade-Specialized (CSL-O → AM2) | n/a | 0.233 | 0.553 | 0.560 | 0.933 | – |

Joint − Specialized:
- PD +0.239 [+0.202, +0.275];
- strong-beat chord-tone +0.131 [+0.104, +0.158];
- success +0.067.

Median inference cost per song: Joint 5.4 GPU-s; Cascade-Qwen 6.5 GPU-s; Specialized 5.6 GPU-s + 12.2 CPU-s.

**INTERPRETATION:** the two Qwen systems dominate every system with a specialized component. Between
them there is no single winner:
- Joint is best for structure, chord agreement and editability;
- Cascade-Qwen is best for melody and lyric fidelity and cadences.

### 14. Does melody ranking remain consistent under both SVS renderers?

**MEASURED FACT:** Spearman correlation of the 5 melody sources' means across FastSinger and SoulX:

| metric | Spearman |
|---|---|
| CER | 0.90 |
| voicing F1 | 0.90 |
| pitch accuracy | 0.60 |
| phrase onsets | 0.60 |
| PER | 0.50 |

N = 34–35 songs per cell. Artifacts: `tables/svs_*.md`, `data/svs.parquet`.

**INTERPRETATION:** only partly consistent. With 5 sources, these correlations are weak evidence.

### 15. Is there a strong Melody Source × SVS interaction?

**MEASURED FACT:** on PER, yes. (SoulX − FastSinger | Melody-Only) − (same | CSL-R) =
+0.038 [+0.021, +0.055], N=34. On pitch accuracy, E3b vs CSL-R: +0.060 [+0.037, +0.086]. On voicing
there is none: +0.001 [−0.004, +0.007].

**INTERPRETATION:** the renderer changes how intelligible Qwen melodies are relative to CSL melodies.
SoulX struggles more with Qwen's crammed syllables. Singability claims must name the renderer.

### 16. Which SVS follows pitch, timing and lyrics more accurately?

**MEASURED FACT:** SoulX − FastSinger:
- **voicing F1:** −0.034 to −0.039, and **phrase onsets within 100 ms:** −0.11 to −0.20. All 5
  sources have CIs excluding 0.
- **PER:** mixed. SoulX is better on CSL melodies (−0.009 [−0.018, −0.000] for CSL-R) and worse on
  Melody-Only (+0.028 [+0.010, +0.047]) and the reference (+0.016).
- **Pitch accuracy:** mixed, from −0.033 (CSL-R) to +0.031 (E3b).

Also: CSL-R melodies have the lowest PER under both renderers (0.034 / 0.027), and the melody main
effect Melody-Only − CSL-R on PER is +0.048 [+0.037, +0.060].

**INTERPRETATION:** FastSinger follows timing more faithfully. Neither renderer is uniformly better on
lyrics or pitch. CSL-L2M-retrained melodies are the most singable, a clear specialized-system win.

### 17. Does symbolic key control survive rendering?

**MEASURED FACT:** the requested key is in the score in 1.000 of songs for both implementations.
Detected in audio (N=36):

| check | regenerate | deterministic transpose |
|---|---|---|
| absolute key accuracy | 0.583 [0.417, 0.750] | 0.528 [0.361, 0.694] |
| **detected key moved +5 vs the draft's detected key** | **0.389 [0.222, 0.556]** | **0.778 [0.639, 0.917]** |

Paired transpose − regenerate on the relative test: +0.389 [+0.222, +0.556]. The detector recovers
the unedited draft's own key only 0.528 of the time.
Artifacts: `tables/closed_loop_control.md`, `tables/backing_paired.md`.

**INTERPRETATION:** partly. Symbolic transposition of a draft reliably shifts the audio's key.
Generative key conditioning is right in the score but not reliably audible, because it produces a
different song.

### 18. Does symbolic tempo control survive rendering?

**MEASURED FACT:** tempo ×1.25 is detected within ±4% up to octave in 1.000 of songs for both
implementations (N=36). Strictly, without octave tolerance:
- regenerate 0.694 [0.528, 0.833];
- edit `Q:` 0.750 [0.611, 0.889];
- difference +0.057 [−0.057, +0.171].

**INTERPRETATION:** yes. The remaining strict misses are beat-tracker half/double-time readings.

### 19. Do generated chords survive rendering as detected harmony?

**MEASURED FACT:** BTC Chord F1 vs the requested chords, N=36:

| chord source | matched | mismatched (a different source's chords, same song) |
|---|---|---|
| Qwen | 0.923 [0.909, 0.935] | 0.46–0.54 |
| AccoMontage2 | 0.921 | 0.46–0.54 |
| reference | 0.888 | 0.46–0.54 |
| E3b drafts | 0.87–0.90 | – |

Artifacts: `tables/backing_chord_f1_matrix.md`.

**INTERPRETATION:** yes, for every chord source. The mismatched control shows that the metric is
discriminative.

### 20. Does a section-label intervention cause actual musical-semantic change?

**MEASURED FACT:** E3b. The verse/chorus classifier (trained on 38,065 train-split reference
sections) scores **AUC 0.847 / accuracy 0.778 on 1,750 held-out test sections**.

| relabelling | N | Δ P(chorus) | Δ relative register |
|---|---|---|---|
| verse → chorus | 106 | +0.093 [+0.057, +0.129] | +0.91 semitones [+0.57, +1.25] |
| chorus → verse | 118 | −0.060 [−0.090, −0.030] | −0.54 |
| *real chorus − verse* | 223 | – | +2.53 [+2.22, +2.84] |

Artifacts: `tables/section_semantics_*.md`, `data/section_classifier.json`.

**INTERPRETATION:** partially. Labels move the music in the right direction, mostly by register, at
about 1/3 of the real verse–chorus gap. There is no harmonic or rhythmic differentiation. Block E
(listening) is pending.

### 21. Which findings are robust enough for the main paper?

These have large effects, tight CIs, N=211–225 and hold across metrics:

- **(a)** Qwen ≫ CSL-L2M on PD/DD/MD, against both the official and the retrained model (Q1–2).
- **(b)** Infill locality vs regeneration (Q7–8).
- **(c)** Qwen harmonizer ≫ AccoMontage2 symbolically (Q9), stated together with the diatonic-heuristic caveat (Q10).
- **(d)** Joint ≫ specialized cascade (Q13), and the joint-vs-Qwen-cascade trade: structure vs melody fidelity (Q3–4).
- **(e)** Lyric swap ≈ reseed for all lyrics→melody systems, including CSL-L2M (Q5).

### 22. Which belong in the appendix?

- SVS cross-render and its interaction (Q14–16), including the finding that CSL-retrained melodies
  are more singable.
- Backing audio by chord source (Q12).
- Closed-loop key and tempo (Q17–18) and chord survival (Q19).
- Section semantics (Q20).
- Tone analysis (Q6).
- DD_tb and the notation-timebase analysis.
- CSL-L2M adapter details and failure-by-length.
- The measurement fixes (`FAILURE_ANALYSIS.md` §6).

### 23. Which human-study results remain pending?

**All of them.** The package is ready: 108 items (102 of 104 test items with audio), 4 counterbalanced
lists, QC items, schema, and analysis code with song- and rater-cluster bootstraps, Krippendorff's α
and Fleiss' κ. No responses have been collected.

| block | question | status |
|---|---|---|
| A | lyrics→melody preference (fit, coherence, phrasing, structure) | pending |
| B | SVS clarity / naturalness / target-following | pending |
| C | harmonization fit / coherence / naturalness | pending |
| D | infill seam naturalness and preservation | pending |
| E | "which sounds more chorus-/bridge-like" | pending |

No claim of perceived quality in any report depends on them.
