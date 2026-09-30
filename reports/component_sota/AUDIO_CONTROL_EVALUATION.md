# Backing audio and closed-loop control (Tables 7 and 8)

**Code:**
- `paper_eval/component_sota/audio.py` (`prep-backing`, `render-backing`);
- `backing_eval.py`;
- analyzers in `tools/` and `scripts/component_sota/audio_tools.sbatch`.

**Tables:**
- `tables/backing_chord_audio.*` (Table 7);
- `tables/closed_loop_control.*` (Table 8);
- `tables/backing_paired.*`;
- `tables/backing_chord_f1_matrix.*`.

**Data:** `data/backing.parquet`.
**Figure:** `figures/fig8_symbolic_vs_audio_control.*`.

## Rendering

- Every lead sheet goes through the project's renderer: FastSinger vocal, then MIDI-SAG's adapter
  `run_midi_llm_to_midi_sag.py --harmonizer leadsheet` with MuseControlLite, using the lead sheet's
  own chords, tempo and key.
- Each lead sheet is cut to its first ≤ 95.1 s (two MuseControlLite windows).
- Songs: the frozen 36-song subset.
- 288 renders: 36 × 3 chord sources + 36 × 5 control conditions. All rendered, all analysed.

## Detectors (identical for every condition)

| detector | measures |
|---|---|
| BTC-ISMIR19 (large vocabulary) | detected chords → **Chord F1** (MIDI-SAG's definition: 12-bin chroma per 100 ms frame, micro-F1 vs the requested chords) |
| Krumhansl-Schmuckler on CQT chroma | **key** (replaces the missing key-CNN) |
| BeatNet | beats → **Rhythm F1** (70 ms vs the score's beat grid) and BPM (median inter-beat interval) |
| SongEval | Coherence / Musicality / Memorability / Clarity / Naturalness |
| Audiobox-aesthetics | audio quality predictors |

## Table 7: same melody and vocal, chords from Qwen / AccoMontage2 / reference

| chord source | Chord F1 ↑ | Key acc. ↑ | Rhythm F1 ↑ | SongEval Musicality | SongEval Coherence | Audiobox CE |
|---|---|---|---|---|---|---|
| Qwen | 0.923 [0.909, 0.935] | 0.583 [0.417, 0.750] | 0.936 | 3.87 [3.80, 3.93] | 4.05 | 7.65 |
| AccoMontage2 | 0.921 [0.905, 0.935] | **0.778** [0.639, 0.917] | 0.930 | 3.84 | 4.02 | 7.63 |
| reference | 0.888 [0.858, 0.913] | 0.361 [0.222, 0.528] | 0.926 | 3.84 | 4.03 | 7.66 |

**Paired** (N=36):

| contrast | Chord F1 | key accuracy | SongEval Musicality |
|---|---|---|---|
| Qwen − AM2 | +0.002 [−0.011, +0.015] | **−0.194 [−0.361, −0.028]** | +0.032 [−0.015, +0.076] |
| Qwen − ref | +0.035 [+0.010, +0.063] | +0.222 [+0.083, +0.389] | – |
| AM2 − ref | +0.033 [+0.006, +0.064] | +0.417 [+0.222, +0.583] | – |

**Is Chord F1 informative?** Scoring each render against a *different* source's requested chords for
the same song gives 0.46–0.54. The matched source gives 0.89–0.92. The detector clearly separates
the chords that were actually requested (`tables/backing_chord_f1_matrix.md`).

## Table 8: closed loop (request → score → audio → detector)

Requests are derived from E3b's S1 draft: key +5 semitones, tempo ×1.25. Each request is implemented
two ways:

| request | generative | deterministic |
|---|---|---|
| key +5 | `key_gen`: E3b regenerated from the transposed-key prompt (paper-final `key_p5`) | `key_transpose`: the draft transposed note by note, chords respelled |
| tempo ×1.25 | `tempo_gen`: regenerated (`tempo_x1.25`) | `tempo_direct`: the draft with only `Q:` changed |

| condition | requested value in the score | detected in audio | detector-bias-free check | abs. BPM error |
|---|---|---|---|---|
| draft (no request) | – | key acc. 0.528 | – | 4.3 |
| key: regenerate | 1.000 | 0.583 [0.417, 0.750] | key moved +5 vs the draft's detected key: **0.389 [0.222, 0.556]** | – |
| key: transpose | 1.000 | 0.528 [0.361, 0.694] | **0.778 [0.639, 0.917]** | – |
| tempo: regenerate | 1.000 | 1.000 (octave-tolerant, ±4%) | strict ±4%: 0.694 [0.528, 0.833] | 26.0 |
| tempo: edit `Q:` | 1.000 | 1.000 | strict ±4%: 0.750 [0.611, 0.889] | 21.2 |
| chords (Table 7) | 1.000 | Chord F1 0.92 (matched) vs 0.46–0.54 (mismatched) | – | – |

Paired contrasts:
- key, transpose − regenerate, relative-shift success: **+0.389 [+0.222, +0.556]**;
- tempo, direct − regenerate, strict success: +0.057 [−0.057, +0.171] (n.s.).

## INTERPRETATION

- **Chords survive rendering** for every chord source: F1 0.89–0.92 against the requested chords,
  about 0.5 against other chords. On rendered audio, Qwen and AccoMontage2 chords are
  indistinguishable in Chord F1, rhythm and quality predictors.
- **Key control survives rendering only through the detector's ceiling.**
  - The chroma key detector recovers the draft's own key only 53% of the time. Absolute key accuracy
    after a request (0.53–0.58) is therefore uninformative.
  - The relative test shows that deterministic transposition moves the audio's key as requested far
    more reliably (78%) than regeneration (39%). Regeneration produces a different song, whose
    detected key varies independently.
  - For key control, symbolic transposition of an existing draft is the dependable path; generative
    key conditioning is correct in the score (100%) but not reliably audible as a shift.
- **Tempo control survives rendering in 100% of songs up to octave errors** (±4%, ×0.5/×1/×2; the
  test cannot confuse ×1.25 with the original tempo). It survives in 69–75% strictly; the rest are
  half- or double-time beat-tracker readings.
- **Key accuracy is also sensitive to the harmony itself.** The same melody is detected in the
  requested key 78% of the time with AccoMontage2's diatonic chords, 58% with Qwen's and 36% with
  the transcribed reference chords. Chroma-based key detection rewards diatonic backing.
