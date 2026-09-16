# Clean held-out test subset (34 songs)

**Goal:** separate model limitations from pseudo-label noise.
**List:** `reports/clean_test_song_ids.txt`.
**Selection record:** `reports/clean_test_subset_selection.json` (rules, per-song rule failures, manual decisions).
**Scripts:** `scripts/select_clean_subset.py`; comparisons use `scripts/compare_r2.py --subset`.

## 1. Selection (never uses model output)

**Step 1: rule-based.** Applied to the 225 test songs of `abc_v2_20260915_120927`; a song must pass all rules.

| rule | songs failing |
|---|---|
| no pathology flag (`qwen_abc/cleaning.py:is_pathological`: double-time, >32-bar section, key instability, octave errors, aligner cramming, implausible density) | 20 |
| tempo 60–170 BPM (no half-/double-time estimate) | 10 |
| stable key: melody in the declared scale ≥85% of duration (≥92% when the key ensemble agreement is only "medium"), chord roots in key ≥75% | 45 |
| ≤10% irregular bars (reasonable downbeat structure) | 31 |
| no section/lyric-line cut left after cleaning | 122 |
| 2–6 notes per bar | 27 |
| ≤5% notes carrying several syllables (aligner cramming) | 14 |
| ≤10% wordless notes (melody extraction or alignment gaps) | 25 |
| chord-tone agreement ≥55% (chords consistent with melody) | 13 |
| no 1-bar section | 26 |
| ≥4 sections | – |
| pitch range ≤28 semitones (no octave jumps) | 35 |

42 songs pass every rule.

**Step 2: manual inspection.** Every candidate's plan and first 40 ABC lines were read (`experiments/clean_subset_20260915/manual_inspection.json`, git-ignored because the inspection sheet contains lyrics). **8 were dropped:**

* 4 for broken lyric alignment (long `_` runs, syllables piled on few notes, lyrics in rest bars);
* 2 for melody octave flips / repeated-pitch noise;
* 1 for an unstable bar grid;
* 1 for echoes crammed onto single notes.

**34 songs remain:** `0ZMWpv0aFBioxywoXsmaId`, `0ccEIUOa32bbAoGaPHGQuX`, `18wGt2QzrQO73VY4akreyx`, `1mcMqHzkg79tfjUqeLAzpI`, `1tKSEmbS7vmv7cFc24sjlI`, `255drLOJ94rmt1SDRj6H3N`, `28IzuspwIj6C8XAOoIJ1Wp`, `2doqLn2sfIhgoUB2DaLRiv`, `2fqUJToyhc6bzIXootn4nv`, `2y0wImEpj5p6SxBmY0opQN`, `3DtjJWwGp2qAVFkqeZ2Fgn`, `3JjXD9DVfxh001wI5IACZ1`, `3tOVrJlfy9g1dPBI8czFIy`, `499ghNme62G8hsotXXlAIE`, `49pMRkESXndNlp0jD1Vxnj`, `4VEbbZeJwrlqRBTTscq3m2`, `4VfHbfbzNPZmX5KI4iAOyT`, `4eMDRwO2dreLU0IvNoARUP`, `4eNxB89LADQeQnUvvIIuRG`, `53a8hVCKsw8NhGxmRH1YeZ`, `56EoTwd4LGjmEO7UzQrDoX`, `58ajbBxQUMMFpqHA62QNO0`, `5CbNIRIgmrh1W9xNnhtSz1`, `5K8AQ2dBiXolebEfawTrzv`, `5LlXalu5DZuQl7Q9CmNKp0`, `5i64hhiLZt1NWnRZiVubSL`, `6I3A7q18EJgbdZf2Zx1wxm`, `6NqPOID1RsXpj7IAEsgd2v`, `6ast0Gzj3H6MgDfkRHZxx1`, `6ousLSFH3faAPCokCk2gCm`, `72yPhoXJHKpAeCt99RLdK4`, `7IE2jtO8pkWtcSdj71OnlT`, `7omMiE6h9Js24Cr22A0Fbv`, `7qDiugtLVz4njwHDGCfdB5`.

The candidate list was recomputed after the cleaning fix (`abc_v2_20260915_113908` → `_120927`) and did not change.

## 2. How the clean subset differs from the full test set (reference side)

| | full (225) | clean (34) |
|---|---|---|
| tempo | 114 BPM | 116 BPM |
| sections per song | 10.6 | 9.8 |
| bars per song | 95.0 | 87.9 |
| ABC-v2 tokens | 3,089 | 2,817 |
| notes per bar | 3.47 | 3.28 |
| **pitch range** | **22.0** | **18.5** |
| mean abs interval | 2.39 | 2.31 |
| 16th off-beat onsets | 0.193 | 0.164 |
| distinct non-empty bars | 0.872 | 0.837 |
| chorus-chorus motif similarity | 0.358 | 0.400 |
| songs with code-switched latin syllables | 18.2% | 2.9% |

**Caveats.** The clean songs are also slightly *shorter and simpler*: fewer sections, less syncopation, almost no code-switching. The pitch-range rule removes octave errors, which also removes some wide real melodies. An improvement on this subset is therefore partly "easier songs", not only "cleaner labels". Every musical metric must be read against **the subset's own reference** (18.5 semitones, not 22).

## 3. Results: full test (225) vs clean subset (34)

Paired sampling, one sample per song. E0/E0s2 = round-1 direct SFT (seeds 1234 / 2345) on cleaned v1 prompts. E1 = ABC-v2. T0.8 = temperature 0.8, top-p 0.95; T1.0 = temperature 1.0, top-p 0.95.

| metric | E0 T0.8 full / clean | E0s2 T0.8 full / clean | E1 T0.8 full / clean | E1 T1.0 full / clean |
|---|---|---|---|---|
| strict-valid ABC | 0.711 / 0.824 | 0.640 / 0.706 | 0.631 / 0.824 | 0.591 / 0.735 |
| no lyric-alignment error | 0.782 / 0.912 | 0.724 / 0.794 | 0.653 / 0.853 | 0.689 / 0.824 |
| exact full structure | 0.529 / 0.765 | 0.613 / 0.735 | 0.858 / 0.882 | 0.836 / 0.941 |
| early EOS | 0.040 / 0.029 | 0.089 / 0.059 | 0.000 / 0.000 | 0.000 / 0.000 |
| lyric recall | 0.955 / 0.984 | 0.943 / 0.964 | 0.964 / 0.977 | 0.970 / 0.980 |
| notes with several syllables (ref 0.023 / 0.014) | 0.034 / 0.040 | 0.034 / 0.030 | 0.040 / 0.037 | 0.050 / 0.043 |
| pitch range (ref 22.0 / 18.5) | 14.07 / 14.03 | 13.71 / 12.18 | 13.30 / 13.18 | 18.40 / 17.29 |
| mean abs interval (ref 2.39 / 2.31) | 1.65 / 1.48 | 1.81 / 1.74 | 1.70 / 1.50 | 2.06 / 2.07 |
| 16th off-beat onsets (ref 0.193 / 0.164) | 0.113 / 0.087 | 0.111 / 0.080 | 0.129 / 0.108 | 0.182 / 0.151 |
| syncopation proxy (ref 0.176 / 0.157) | 0.112 / 0.107 | 0.115 / 0.143 | 0.126 / 0.127 | 0.165 / 0.140 |
| chord-tone agreement (ref 0.677 / 0.721) | 0.739 / 0.766 | 0.715 / 0.725 | 0.730 / 0.738 | 0.658 / 0.664 |
| distinct non-empty bars (ref 0.872 / 0.837) | 0.462 / 0.432 | 0.475 / 0.385 | 0.467 / 0.425 | 0.797 / 0.734 |
| chorus-chorus motif similarity (ref 0.358 / 0.400) | 0.584 / 0.624 | 0.597 / 0.653 | 0.579 / 0.630 | 0.409 / 0.489 |

Paired differences on the clean subset against E0 T0.8 (34 songs, 95% bootstrap CI):

| run | exact full structure | distinct bars | pitch range |
|---|---|---|---|
| E1 T0.8 | +0.118 [−0.059, +0.294] | – | – |
| E1 T1.0 | +0.176 [+0.029, +0.353] | +0.302 [+0.235, +0.370] | +3.3 [+1.3, +5.3] |

With 34 songs the intervals are wide; only large effects are resolvable.

## 4. Interpretation

1. **Structure and validity errors are concentrated in noisy songs, and E1 closes most of the gap on the full set.**
   * E0 goes from 0.53 exact structure on all songs to 0.77 on clean songs.
   * E1 is at 0.86 on all songs and 0.88–0.94 on clean songs.
   * Strict validity and alignment validity rise by 0.1–0.2 on clean songs for every model.
   * **Pseudo-label noise is therefore a major part of the residual structure/validity error.** Explicit counters remove most of the *counting* error that remained even on clean songs for E0.
2. **Musical conservatism is not a label-noise artifact.** At T0.8 every model has about 0.43 distinct bars on clean songs against a clean reference of 0.84, and 13–14 semitones of range against 18.5. That is the same relative gap as on the full set.
3. **At T1.0 the gap mostly closes on both sets** (distinct bars 0.73 vs 0.84, range 17.3 vs 18.5, syncopation 0.14 vs 0.16). The conservatism is a decoding effect (see LONG_STRUCTURE_EXPERIMENTS.md, decoding), not a data or capacity limit.
4. **Lyric following is near ceiling on clean songs** (recall 0.96–0.98). The 2–5% recall loss on the full set is also partly label noise, e.g. aligner-crammed references.
