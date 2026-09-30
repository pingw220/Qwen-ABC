# Melody → chord (Table 4)

**Code:**
- `paper_eval/component_sota/chord_gen.py` (Qwen harmonizers);
- `accomontage.py` (AccoMontage2 adapter);
- `chord_metrics.py` (all metric definitions);
- `chord_eval.py` (every melody source × chord source).

**Tables:**
- `tables/melody_to_chord.*` (Table 4 and the factorial);
- `tables/melody_to_chord_paired.*`.

**Data:** `data/chords.parquet`.
**Figure:** `figures/fig5_melody_to_chord.*`.

## Systems

| chord source | how | checkpoint |
|---|---|---|
| **Qwen Melody→Chord** (`qwen`) | Qwen3.5-0.8B-Base fine-tuned on 10,243 train songs. The prompt holds meter, tempo, key, section plan and the melody (ABC-v2, no chords, no lyrics). The completion is one line per bar (`[r:k] beat:chord ...`), so **the melody is read-only by construction**. 2 epochs, 468 updates, E3b's recipe otherwise. | `experiments/component_sota/runs/chord_sft/final_model` |
| **Qwen Melody+Lyrics→Chord** (`qwen_lyr`) | same, with the lyric lines kept in the prompt; 590 updates | `.../chord_lyr_sft/final_model` |
| **AccoMontage2** (`am2`) | official `demo_SOME.py`, exactly as MIDI-SAG's ComposeFlow calls it (`--chord_style POP_STANDARD --chords_per_bar 1 --beat_subdivision 1 --key <key>`); its per-beat BTC chord output is mapped back onto the melody's bar grid | `MIDI-SAG/AccoMontage2` |
| pseudo-reference (`ref`) | the test song's own transcribed chords | – |
| diatonic baseline | per bar, the diatonic triad covering most melody duration; a cheap control | `chord_eval.diatonic` |

Decoding: T=1.0, top-p 0.95. Both harmonizers receive the same key. On the reference melody,
Qwen scores are the mean of 4 samples (S1–S4); every other combination uses 1 sample.

## Metrics

Chords are read on the beat grid.

**Reference similarity** (reference melody only, where the pseudo-reference chords belong to the same melody):
- root accuracy and major/minor accuracy per beat;
- chord chroma F1: 12-bin chord chroma per beat, micro-F1;
- progression similarity: LCS of the chord-change sequences divided by the longer length;
- cadence agreement: same root on the last beat of each reference section.

**Melody–harmony compatibility** (reference-free):
- chord-tone ratio: duration-weighted melody notes that are chord tones;
- strong-beat chord-tone: the same on beats 1 and 3;
- strong-beat dissonance: a non-chord tone a semitone from a chord tone, on a strong beat;
- chroma compatibility: per-bar cosine of the melody pitch-class histogram and the chord chroma.

**Plausibility and diversity:**
- roots in key;
- cadence I/V (the section-final chord is I or V; minor also allows III and v);
- chords/bar, distinct chords, chord entropy, time on tonic, root motion by 4th/5th;
- JS divergence of relative-root, quality and chord-bigram distributions vs the held-out corpus.

## MEASURED FACT (reference melodies, 225 songs; AccoMontage2 on 211 4/4 songs)

| | Qwen | Qwen+lyrics | AccoMontage2 | diatonic | reference |
|---|---|---|---|---|---|
| success | 1.000 | 1.000 | 0.938 (3/4 unsupported) | 1.000 | 1.000 |
| root acc. vs ref | **0.336** | 0.337 | 0.221 | 0.320 | 1 |
| maj/min acc. vs ref | **0.302** | 0.304 | 0.178 | 0.289 | 1 |
| chord chroma F1 vs ref | **0.563** | 0.570 | 0.471 | 0.561 | 1 |
| progression sim. vs ref | **0.468** | 0.461 | 0.407 | 0.360 | 1 |
| cadence agreement | **0.446** | 0.449 | 0.283 | 0.380 | 1 |
| chord-tone ratio | 0.628 | 0.631 | 0.538 | **0.778** | 0.677 |
| strong-beat chord-tone | 0.660 | 0.659 | 0.544 | **0.801** | 0.704 |
| strong-beat dissonance ↓ | 0.153 | 0.155 | 0.181 | **0.075** | 0.133 |
| chroma compatibility | 0.522 | 0.520 | 0.467 | **0.643** | 0.549 |
| roots in key | 0.919 | 0.918 | 0.999 | 1.000 | 0.920 |
| cadence I/V | 0.704 | 0.702 | 0.562 | 0.631 | 0.740 |
| chords/bar | 1.00 | 0.97 | 0.87 | 0.52 | 0.98 |
| chord entropy | 2.48 | 2.43 | 2.45 | 2.02 | 2.62 |
| JS chord bigram vs corpus ↓ | **0.052** | 0.052 | 0.325 | 0.367 | 0 |

**Paired Qwen − AccoMontage2** (N=211): every row below is resolved in Qwen's favour except roots in key.

| metric | difference [95% CI] |
|---|---|
| root acc. | +0.113 [+0.098, +0.129] |
| chord chroma F1 | +0.090 [+0.078, +0.102] |
| progression sim. | +0.060 [+0.046, +0.075] |
| cadence agreement | +0.160 [+0.131, +0.188] |
| strong-beat chord-tone | +0.115 [+0.096, +0.136] |
| chroma compatibility | +0.057 [+0.046, +0.068] |
| strong-beat dissonance | −0.028 [−0.041, −0.015] |
| cadence I/V | +0.139 [+0.103, +0.174] |
| roots in key | −0.079 [−0.096, −0.063] (AM2 is ~100% diatonic) |

**Paired Qwen − diatonic baseline:**

| metric | difference [95% CI] |
|---|---|
| chord chroma F1 vs ref | +0.002 [−0.007, +0.012] (**no difference**) |
| root acc. | +0.016 [+0.002, +0.029] |
| progression sim. | +0.109 [+0.094, +0.123] |
| cadence agreement | +0.065 [+0.036, +0.095] |
| strong-beat chord-tone | −0.142 [−0.155, −0.129] (**the baseline is "more compatible"**) |
| chords/bar | +0.48 |
| chord entropy | +0.46 |

**Lyrics in the harmonizer prompt:** no measurable effect. Chroma F1 +0.007 [+0.001, +0.013];
every compatibility difference is within ±0.003.

**Harmonizing generated melodies** (1 sample each; strong-beat chord-tone for Qwen vs AccoMontage2):

| melody source | Qwen | AccoMontage2 |
|---|---|---|
| E3b | 0.674 | 0.585 |
| Qwen Melody-Only | 0.656 | 0.563 |
| CSL official | 0.554 | 0.553 |
| CSL retrained | 0.564 | 0.549 |

On CSL-L2M melodies the two harmonizers are tied on compatibility. Qwen's chords on CSL melodies
are less often in key (0.87–0.88 vs 0.95 on Qwen melodies), and CSL's key is estimated, not given.

## INTERPRETATION

- **Qwen Melody→Chord beats AccoMontage2 on all three layers:** reference similarity,
  compatibility and corpus plausibility. The only exception is roots in key, where AccoMontage2's
  purely diatonic output wins trivially.
- **The reference-similarity layer is weak evidence on its own.** A per-bar diatonic triad matches
  Qwen's chord F1 against the pseudo-reference (0.561 vs 0.563). The pseudo-reference chords are
  themselves automatic transcriptions, and Qwen was trained on labels of the same kind, which gives
  it an in-distribution advantage over AccoMontage2.
- **Compatibility metrics reward simplicity.** The diatonic baseline "beats" the real reference
  chords on every compatibility metric. These metrics are a sanity floor, not a quality ranking.
- **What distinguishes Qwen from the heuristic** is the progression layer: progression similarity,
  cadences, harmonic rhythm, chord variety, and the corpus bigram distribution (JS 0.05 vs 0.37).
  Qwen writes progressions that look like the corpus's; the heuristic and AccoMontage2 do not.
- Whether that sounds better is untested by humans: listening-study block C is prepared, not run.
  In audio, the chord sources are indistinguishable on SongEval and Audiobox
  (`AUDIO_CONTROL_EVALUATION.md`).
