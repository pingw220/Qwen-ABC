# Musical fluency vs control adherence

Are the differences between model families about *musical fluency* or about *following the
request*? Same 225 songs, same symbolic statistics for references and every system; 4 samples per
song (Qwen/MuPT paired-seed, MIDI-LLM seeds 1000-1003, 199 songs it accepts). Tables
`tables/musicality_scalars.*`, `tables/musicality_js.*`, `tables/musicality_vs_control.*`; code
`paper_eval/musicality.py`. Song-level bootstrap.

## Per-song statistics (mean of per-song means, 95% CI)

| | reference | Qwen-ABC E3b | MuPT | MIDI-LLM | Qwen E0 |
|---|---|---|---|---|---|
| pitch range (semitones) | 22.0 [21.2, 22.9] | 17.5 [17.2, 17.8] | 17.6 [17.3, 17.9] | 24.7 [24.4, 25.0] | 17.9 |
| notes per bar | 3.47 | 3.24 | 3.87 | 3.43 | 3.38 |
| mean \|interval\| | 2.39 | 2.08 | 2.00 | 2.32 | 2.01 |
| repeated-pitch intervals | 0.260 | 0.321 | 0.326 | 0.268 | 0.338 |
| rhythm entropy (bits) | 4.99 | 4.78 | 4.58 | 4.97 | 4.82 |
| distinct bars | 0.872 | 0.780 | 0.769 | 0.938 | 0.801 |
| pitch 4-gram repetition | 0.455 | 0.573 | 0.557 | 0.314 | 0.554 |
| chords per bar | 0.968 | 0.955 | 0.970 | 1.078 | 0.955 |
| chord vocabulary per song | 11.8 | 8.8 | 8.7 | 10.6 | 9.3 |
| chord-tone agreement | 0.677 | 0.657 | 0.677 | 0.626 | 0.663 |
| syncopation proxy | 0.176 | 0.157 | 0.131 | 0.173 | 0.153 |
| melody in declared key | 0.958 | 0.971 | 0.966 | 0.970 | 0.980 |

## Distribution distance to the references (Jensen-Shannon, bits; lower = closer)

| | E3b | MuPT | MIDI-LLM | E0 | reference split-half floor |
|---|---|---|---|---|---|
| scale degree (pc rel. key) | 0.0022 | 0.0028 | 0.0059 | 0.0057 | 0.0033 |
| interval | 0.0057 | 0.0094 | **0.0013** | 0.0090 | 0.0010 |
| duration | 0.0052 | 0.0063 | 0.0128 | 0.0015 | 0.0016 |
| onset position | 0.0027 | 0.0017 | 0.0024 | 0.0009 | 0.0006 |
| chord root | 0.0020 | 0.0024 | 0.0105 | 0.0025 | 0.0035 |
| chord quality | 0.0035 | 0.0054 | 0.0114 | 0.0070 | 0.0028 |
| chord transition | 0.0144 | 0.0173 | 0.0486 | 0.0193 | 0.0259 |

(Plug-in JS on pooled histograms is biased upward under resampling; the table's bootstrap
intervals therefore sit above some point estimates. Compare against the split-half floor, which
has the same bias.)

## Side by side (single sample)

| | exact structure | lyric recall | strict validity | JS interval | JS scale degree |
|---|---|---|---|---|---|
| Qwen-ABC E3b | 0.982 | 0.970 | 0.777 | 0.0057 | 0.0022 |
| MuPT | 0.341 | 0.483 | 0.203 | 0.0094 | 0.0028 |
| MIDI-LLM | 0.884 † | 0.734 | 0.763 ‡ | 0.0013 | 0.0059 |

## Measured facts

1. **Qwen-ABC and MuPT are nearly indistinguishable musically** (pitch range 17.5 vs 17.6, pitch
   repetition 0.32 vs 0.33, chord density 0.96 vs 0.97, distinct bars 0.78 vs 0.77, chord-tone 0.66
   vs 0.68) **and separated by 0.64 in exact structure and 0.49 in lyric recall.** The
   family difference is control, not fluency.
2. **Both LM-on-ABC systems are narrower and more repetitive than the corpus**: pitch range −4.5
   semitones, repeated-pitch intervals +0.06, distinct bars −0.09, chord vocabulary −3 per song.
3. **MIDI-LLM's melodic statistics are closest to the corpus** (interval JS 0.0013, at the noise
   floor) but it overshoots range (24.7 vs 22.0) and under-repeats (distinct bars 0.94 vs 0.87;
   4-gram repetition 0.31 vs 0.46); its harmony is furthest (chord-transition JS 0.049).
4. **ESS did not change musical statistics**: E0 and E3b are within ~0.4 semitones of range and
   ~0.02 on repetition measures, while structure differs by 0.43.

## Interpretation

In this setting musical fluency, as measured symbolically, is roughly shared by every system that
was finetuned on the corpus, and the models differ mainly in whether they write what was asked.
"More corpus-like statistics" is not a quality verdict: no listening test was run for this
comparison (listening_study/ is prepared, results pending).
