# AccoMontage2 comparison

AccoMontage2 is the external melody→chord system. Its melody→chord step is a dynamic-programming
search over a chord-progression library (Chorderator); no trained network is involved in that step.
It is used here **unmodified**, through `demo_SOME.py`, with the exact arguments MIDI-SAG's
ComposeFlow uses:

```
--beat_subdivision 1 --chord_style POP_STANDARD --chords_per_bar 1 --key <key>
```

Adapter: `paper_eval/component_sota/accomontage.py`.

## Adapter facts

- **Input:** a one-track melody MIDI with a single tempo, 4/4 and the key signature. The key is the
  plan's key; for CSL-L2M melodies it is estimated by Krumhansl-Schmuckler.
- **Output:** a per-beat BTC-format chord file, in seconds. It is mapped back onto the melody's tick
  grid, and chord times map back exactly.
- **4/4 only:** the 14 test songs in 3/4 are recorded as `unsupported_meter`, which gives success
  0.938 on every melody source.
- **Irregular bars:** songs with a pickup or irregular bars are harmonized on AccoMontage2's own
  regular grid. Its phrase segmentation may straddle our irregular bars; this is a limitation of
  the input format, not a bug.
- **Cost:** CPU only. The median is 12.2 s per song (measured `seconds` field, S1 runs; the first
  songs of each shard include start-up and are slower).
- **Settings:** one chord per bar is ComposeFlow's setting. The reference averages 0.98 chord
  changes per bar; AccoMontage2 produces 0.86–0.92. A finer setting was not tried.

## Result summary

Full tables are in `MELODY_TO_CHORD.md` and `tables/melody_to_chord*.md`.

| layer | Qwen − AccoMontage2 (reference melody, paired, N=211) |
|---|---|
| reference similarity | root acc. +0.113 [+0.098, +0.129], chroma F1 +0.090 [+0.078, +0.102], cadence +0.160 |
| compatibility | strong-beat chord-tone +0.115 [+0.096, +0.136], chroma compat. +0.057, dissonance −0.028 |
| plausibility | cadence I/V +0.139; JS bigram vs corpus 0.052 vs 0.325; roots in key **−0.079** (AM2 wins) |
| rendered audio (36 songs) | Chord F1 +0.002 [−0.011, +0.015]; SongEval Musicality +0.03 (n.s.); **key accuracy −0.194 [−0.361, −0.028]** (AM2 wins) |

## Where AccoMontage2 wins

These are measured facts:

1. **Roots in key:** 0.999 vs 0.919.
2. **Key detected in the rendered backing:** 0.778 vs 0.583 on the same melody and vocal. Its
   strictly diatonic harmony makes the mix's key unambiguous to a chroma key detector.
3. **Time on tonic:** 0.376 vs 0.313. This is more stable, though not necessarily better.
4. **On CSL-L2M melodies it ties Qwen on compatibility:** strong-beat chord-tone 0.553 vs 0.554
   (official) and 0.549 vs 0.564 (retrained). Its roots stay in key, while Qwen's drop to 0.87.

## INTERPRETATION

- On our corpus, Qwen is the better harmonizer by every symbolic measure except diatonicity. Part of
  the reference-similarity margin is an in-distribution advantage: Qwen trained on the same kind of
  pseudo-labels.
- After rendering, the difference largely disappears. Both chord sources are realized equally
  faithfully (Chord F1 0.92), and quality predictors do not separate them. AccoMontage2 is the
  "safer" choice for key clarity.
- A human preference test (listening-study block C) is required before claiming Qwen harmonizes
  *better*. No human data exists.
