# Section-label semantics: does changing a label change the music?

**Code:** `paper_eval/component_sota/section_semantics.py`.
**Tables:** `tables/section_semantics_features.*` and `tables/section_semantics_classifier.*`.
**Classifier:** `data/section_classifier.json`.

**Inputs:** E3b paper-final samples.
- `orig` (S1–S4).
- `label_swap` (verse↔chorus, S1–S2).
- `label_bridge` (verse/chorus → bridge, S1–S2).

The target section, its bars and its lyrics are identical across conditions. The printed
label is the only control that changes.

## Method

1. **Features of the target section:**
   - register relative to the song mean and relative maximum pitch;
   - notes per bar and pitch range;
   - mean absolute interval and 16th-note share;
   - distinct-bar share, chords per bar and tonic share;
   - syllables per bar.

   We take intervened minus unmodified samples, paired over songs.

2. **Classifier:** numpy logistic regression, verse vs chorus, on the same features.
   - Trained on 38,065 **train-split** reference sections.
   - Tested on 1,750 held-out **test-split** reference sections: **AUC 0.847, accuracy 0.778**.
   - No song overlaps between training and test.
   - It is then applied to the generated target sections.

## MEASURED FACT

| intervention | N | Δ relative register (semitones) | Δ P(chorus) from classifier |
|---|---|---|---|
| verse → chorus | 106 | **+0.91 [+0.57, +1.25]** | **+0.093 [+0.057, +0.129]** (0.391 → 0.484) |
| chorus → verse | 118 | −0.54 [−0.82, −0.27] | −0.060 [−0.090, −0.030] (0.636 → 0.576) |
| verse → bridge | 106 | +0.55 [+0.18, +0.93] | – |
| chorus → bridge | 118 | −0.40 [−0.67, −0.12] | – |
| *reference: chorus − verse, same song* | 223 | +2.53 [+2.22, +2.84] | – |

Other features barely move. Density, range and rhythm CIs cross 0 for chorus↔verse. The
exception is mean interval size, which grows for verse → bridge: +0.13 [+0.03, +0.23].

In real songs, choruses also carry more chords per bar (+0.104) and less tonic (−0.041). The
generated relabelled sections do not reproduce this: chords/bar +0.028 [−0.023, +0.081].

## INTERPRETATION

- The label is **not** a cosmetic token. Relabelling moves the section in the musically correct
  direction:
  - it becomes higher when labelled chorus and lower when labelled verse;
  - an independent classifier's P(chorus) follows the label.
- The effect is about **one third of the real verse/chorus register gap** (0.9 vs 2.5
  semitones) and is carried almost entirely by register. The model has learned "choruses sit
  higher", not the fuller harmonic and rhythmic contrast.
- A bridge label is realized mainly as moderated register and wider intervals.
- This is evidence of *partial* semantic control. Claims should stay at that level. The
  audio-level check ("which sounds more chorus-like?") is listening-study block E and is
  pending; no human data exists.
