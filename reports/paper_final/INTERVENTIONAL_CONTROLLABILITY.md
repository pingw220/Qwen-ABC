# Interventional controllability

Reproducing a song's own plan (0.98 exact structure) shows *adherence*; it does not show that the
model responds to a *change* of the request. Here each held-out song is generated from its original
spec and from specs that differ in exactly **one** control, and the output change is compared with
the change caused by reseeding alone.

Data: `data/interventions.parquet` (sample level: target adherence), `data/interventions_effects.parquet`
(song level: distances); tables `tables/interventions_adherence.*`, `tables/interventions_effects.*`,
`tables/interventions_effects_sections.*`, `tables/tone_melody.*` (Table 5); figure
`figures/fig4_intervention_vs_reseed.*`; code `paper_eval/{interventions,tasks,intervention_analysis,
intervention_report,tone_melody}.py`.

## Design

* **Models / N.** Qwen-ABC E3b: 13 interventions × 224-225 songs × seeds S1,S2 (`lyrics_all`
  S1-S4), 6,286 samples, plus `orig` S1-S4 and a `replay`. MuPT: 6 interventions (bars ±4, label →
  bridge, key +5, tempo ×1.25, lyrics_all). MIDI-LLM: 5 interventions on a deterministic 60-song
  subset of songs it accepts, seed 1000 (bars and tempo are its *inputs*, so their adherence is by
  construction).
* **One control only.** `controls_changed()` diffs every task against its original spec; the driver
  refuses any task that changes more than the intended control (verified for all 6,286 E3b tasks).
  Target section: a lyric-bearing verse/chorus chosen from the song id (same for every model).
  Replacement lyrics: another held-out, all-Chinese song, same syllable count per section.
* **Seeds.** A paired-seed Gumbel-max sampler gives two prompts with the same seed identical noise
  at every step. The *replay* check shows that coupling does not survive a full song: re-running the
  same prompt and seed in a different batch reproduces the opening and then diverges after a
  bf16-level near-tie (melody distance 0.68 vs 0.93 for a reseed). Therefore:
* **Estimands.** (i) target-specific adherence per sample; (ii) **effect = cross − within**, per song:
  mean distance between original and intervened samples over *different-seed* pairs, minus the
  mean distance between original samples of different seeds (the reseed floor). > 0 means the
  intervention moves the output beyond sampling noise. Also the seed-paired ratio d(A,B)/d(A,C).

## Table 5a — target adherence (Qwen-ABC E3b; song-level mean over seeds, 95% CI)

| control | requested change | adherence (intervened) | same measure, unmodified samples | collateral |
|---|---|---|---|---|
| structure length | target section −4 bars | 0.991 [0.982, 0.998] exact | 0.996 | non-target sections exact 0.998; plan exact 0.978 |
| | −2 bars | 0.998 [0.993, 1.000] | 0.996 | 0.997; 0.980 |
| | +2 bars | 1.000 [1.000, 1.000] | 0.996 | 0.998; 0.980 |
| | +4 bars | 1.000 [1.000, 1.000] | 0.996 | 0.998; 0.980 |
| section label | verse/chorus → bridge | 0.998 [0.993, 1.000] label; 0.996 duration kept | 0.998 | 0.996; 0.971 |
| | verse ↔ chorus | 1.000; 0.998 duration kept | 0.998 | 0.996; 0.980 |
| key | +2 / +5 / −3 semitones | declared key 1.000 / 1.000 / 1.000 | – | plan exact 0.989 / 0.982 / 0.982 |
| | melody time in the **new** key's scale | 0.971 / 0.974 / 0.965 | (old key's scale: 0.782 / 0.907 / 0.643) | chord roots in new key 0.923 / 0.931 / 0.924 |
| | mean melody pitch shift (semitones) | +0.55 [+0.13, +0.98] / +0.52 [+0.12, +0.93] / +0.76 [+0.34, +1.20] | requested +2 / +5 / −3 | |
| tempo | ×0.8 / ×1.25 | header 1.000 / 1.000; bar count kept 0.989 / 0.984 | – | duration ratio 1.253 / 0.801 (requested 1.250 / 0.800) |
| lyrics | whole song replaced | new lyrics sung 0.983 [0.979, 0.986] | chance 0.115 | old lyric "still sung" 0.116 = chance; plan exact 0.990 |
| | one section replaced | new lyrics in that section 0.982 [0.973, 0.989] | chance 0.087 | old section lyric 0.089 = chance |

MIDI-LLM (60 songs): bars+4 / label→bridge / tempo 1.000 (inputs); key +5: declared 1.000, melody in
new scale 0.975, mean pitch shift −0.59 [−1.72, +0.54]; lyrics_all: new lyrics 0.851, old 0.108 =
chance 0.106 (one refusal at G2P).

MuPT (224-225 songs, S1+S2; lyrics S1-S4): target section exact after bars −4 / +4 0.830 / 0.812,
change in the requested direction 0.926 / 0.905, but non-target sections exact only 0.738 / 0.787 and
the whole plan 0.27 / 0.32 (its unmodified-plan rate is 0.34); label → bridge 0.801; key +5 declared
1.000, melody in the new scale 0.967, mean pitch shift −0.09 [−0.49, +0.31]; tempo header 0.840 with
the bar count kept in only 0.342 (its early stops and over-generation make duration ratios
meaningless); lyrics_all new lyrics 0.496 vs chance 0.100, old lyric 0.102 = chance.

## Table 5b — does the output move more than reseeding? (cross − within, E3b, N = 224-225)

| intervention | melody (onset, pitch, dur) | contour | rhythm | chord changes | pitch-class JS | seed-paired d(A,B)/d(A,C) |
|---|---|---|---|---|---|---|
| bars −4 | +0.003 [+0.000, +0.005] | +0.000 | +0.005 | −0.008 | +0.007 | 0.916 |
| bars +4 | +0.002 | −0.002 | **+0.009 [+0.004, +0.014]** | −0.008 | +0.002 | 0.904 |
| label → bridge | +0.002 | +0.003 | +0.002 | +0.002 | +0.009 | 0.932 |
| **key +2** | +0.015 | +0.000 | −0.003 | **+0.190 [+0.173, +0.206]** | **+0.202 [+0.187, +0.216]** | 1.017 |
| **key +5** | +0.006 | +0.002 | −0.004 | **+0.072** | **+0.077** | 1.005 |
| **key −3** | +0.027 | −0.001 | +0.001 | **+0.262** | **+0.336 [+0.318, +0.353]** | 1.028 |
| tempo ×0.8 | +0.001 | +0.000 | +0.005 | −0.006 | +0.007 | 0.956 |
| tempo ×1.25 | +0.000 | −0.002 | −0.001 | −0.008 | +0.001 | 0.940 |
| **lyrics (whole song)** | +0.003 [+0.000, +0.005] | +0.006 [+0.002, +0.010] | +0.007 [+0.002, +0.011] | +0.011 [+0.003, +0.019] | +0.009 [+0.001, +0.017] | **0.997 [0.990, 1.004]** |
| lyrics (one section) | +0.002 | −0.002 | +0.004 | −0.002 | +0.004 | 0.958 |
| replay (same prompt, same seed) | −0.252 | −0.146 | −0.194 | −0.268 | −0.111 | – |

The reseed floor itself is large (mean melody distance between two samples of the same prompt 0.93;
pitch-class JS 0.20; chord-change distance 0.68), which is what "beyond reseeding" is measured against. MIDI-LLM (60 songs):
lyrics_all +0.002 [−0.003, +0.008] melody, seed-paired ratio 0.980 [0.968, 0.992]; key +5 is the only
intervention with a pitch-class effect (ratio 1.88). MuPT: key +5 moves chord changes +0.063 and
pitch-class JS +0.076 beyond reseeding; every other MuPT intervention, including lyrics_all
(rhythm −0.004, chords −0.006, pitch-class −0.003, all n.s.), is at the reseed floor.

**Local vs global edits** (`tables/interventions_effects_sections.*`): a −4-bar change moves the
*target* section's contour/rhythm/chords by +0.035-0.039 beyond reseeding while the sections before
and after stay at the reseed floor (|effect| ≤ 0.010); a label change moves only the target's
contour (+0.012).

## Table 5c — does the melody follow the lyric's Mandarin tones?

Direction agreement between consecutive sung syllables' pitch movement and the movement their
lexical tones imply (Chao pitch values, 3-3 sandhi; chance 0.5), own lyric vs the counterfactual
lyric at the same syllable slots:

| melody | own | other | own − other |
|---|---|---|---|
| pseudo-GT references (225) | 0.520 | 0.494 | **+0.026 [+0.013, +0.038]** |
| E3b, unmodified samples | 0.505 | 0.506 | −0.001 [−0.006, +0.004] |
| E3b, lyric-swapped samples | 0.503 | 0.504 | −0.001 [−0.006, +0.004] |
| MuPT, unmodified / swapped (225 songs) | 0.495 / 0.494 | 0.494 / 0.498 | +0.001 [−0.007, +0.009] / −0.004 [−0.013, +0.004] |

## Measured facts

1. **Structure-length and label interventions are obeyed essentially perfectly by Qwen-ABC**
   (0.991-1.000 for the target, 0.996-0.998 for the other sections). MuPT moves the target section
   in the requested direction (0.91-0.93) but hits it exactly only 0.80-0.83 of the time and
   disturbs the other sections (0.74-0.79 exact).
2. **Key interventions are obeyed as harmony and scale, not as transposition.** The declared key
   always changes, 97% of melody time moves into the new scale and 92-93% of chord roots, but the
   melody's mean pitch moves only +0.5-0.8 semitones for requests of +2, +5 or −3: the model
   re-composes in the new key at the same register instead of transposing (distance to the *transposed*
   original, 0.924-0.935, equals the reseed floor 0.925: it is not a transposed copy).
3. **Tempo** is obeyed as a header; bar count and plan are unchanged and the melody moves no more
   than reseeding (rhythm +0.005 at ×0.8).
4. **Lyric interventions are realized exactly as text** (new lyric recall 0.98, old lyric at chance),
   **but they change the melody no more than reseeding does**: whole-song effect ≤ 0.011 on every
   distance (vs 0.19-0.34 for a key change), seed-paired ratio 0.997 [0.990, 1.004]. MIDI-LLM
   (0.980) and MuPT (all effects n.s.) behave the same way: no system in this study composes a
   measurably different melody for different lyrics.
5. **The melody does not follow lexical tones** in E3b or MuPT (−0.001), although the pseudo-labelled
   training songs do, weakly (+0.026).

## Interpretation

Controls that are *written into the lead sheet's structure* — section lengths, labels, key
signature, tempo marking — are followed as instructions, and locally. Lyrics are followed as
**realization constraints**: the requested syllables are placed on the melody, but lyric content
does not measurably condition *what* melody is composed. That is the paper's most important
limitation, and it is consistent with the cramming results (syllables are fitted to a melody after
it is written). This does not establish that lyric conditioning of melody is unlearnable: the
corpus signal is weak (+0.026) and the pseudo-labels' note-syllable alignment is noisy.
