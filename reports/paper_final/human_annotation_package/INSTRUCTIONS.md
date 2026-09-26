# Verifying pseudo lead sheets: annotator instructions

**Status: no annotations have been collected. Every human-GT number in the paper is PENDING until
records land in `annotations/`.**

## Why

Every test "label" (section plan, lyric placement, melody, chords) was produced automatically by
SheetSage-Pro from the recording. When a model deviates from the requested plan we cannot tell,
without a human, whether the model or the label is wrong. You are checking the **labels**, not
the model.

## What you get (per song, in `materials/<song_id>/`; git-ignored because it contains lyrics)

| file | what |
|---|---|
| `pianoroll.png` | top: the pseudo label (sections shaded and named with bar counts, chords in brown, one syllable per note; red = several syllables crammed on one note, light blue = melisma). Bottom: the E3b model's selector@4 output for the same prompt, for context only |
| `pseudo_label.mid` | the pseudo label as MIDI (melody + chords) |
| `pseudo_label.abc`, `pseudo_plan.json` | the label as ABC and as the prompt's section plan with lyric lines |
| `prediction_e3b_*.abc` | model outputs (context only; do not correct them) |

You also need the **original recording** (look it up by song id in the SheetSage-Pro production
instance, `sheetsage_zh_production_v2_lvcr/<song_id>/`, read-only). Judge the labels against the
recording, not against the MIDI.

## Procedure (about 8-12 minutes per song)

1. Open `materials/annotate.html` in a browser (works offline). Enter your annotator id once.
2. Pick the next song (✓ marks finished ones; songs are in the order of `candidates.csv`).
3. For **each section row**:
   * *label ok?* Is it the right kind of section? Use the conventional pop reading: *verse* =
     changing lyrics over a recurring melody; *chorus* = the recurring hook with (mostly) the same
     lyrics; *prechorus* = a build between verse and chorus; *bridge* = a contrasting section that
     occurs once, usually after the second chorus; *intro/outro/instrumental* = no (lead) vocal.
     If no, choose the correct label.
   * *bars ok?* Count bars in the recording (downbeats). If no, enter the correct count.
   * *boundary*: is the section **start** at the right bar? If not, how many bars later (+) or
     earlier (−) should it start? A lyric line split across two sections usually means a wrong
     boundary.
   * *lyric-alignment errors*: number of syllables sung on the wrong note, crammed onto one note,
     or placed in the wrong section (count; an estimate is fine above 10).
   * *wrong lyric syllables*: syllables whose **text** is wrong or which are not sung.
   * *melody / chord errors*: 0 none · 1 a few notes/chords · 2 phrase-level problem (octave
     flips, rhythm off by a beat, wrong chord quality throughout) · 3 section unusable.
4. Whole song: key correct? tempo/beat grid correct (half/double time)? your confidence 1-5,
   free notes. If sections must be merged/split/added, describe it in *notes*; the analyst will
   fill `corrected_sections` (see `schema.json`).
5. **Export all (JSON)** at the end of each session and put the file in `annotations/`
   (`annotations_<annotator>.json`). The browser keeps a copy, but export is the record.

## Rules

* Do not look at model metrics; judge the recording.
* When unsure, say so (*unsure* / low confidence) rather than guess.
* At least 20 songs should be double-annotated (two annotators) for agreement; the first 10
  `anchor_clean_subset` and first 10 `priority` songs are the default overlap set.

## After annotation

```bash
python -m paper_eval.annotation_rescore --annotations reports/paper_final/human_annotation_package/annotations \
    --system e3b_sel4=experiments/bon_r3b_20260918_100234/test_sum/generations \
    --system e3b_single=/gscratch/ark/pingw220/qwen_abc_r2_offload/evals/bon_e3b_test_T1.0/generations
```

prints, per system, structure exact / per-section label and bars accuracy / lyric recall against
the pseudo labels and against the human-corrected labels on the annotated songs, with paired
song-level bootstrap CIs, and splits every model deviation from the pseudo plan into *label error*
(model agrees with the human) and *model error*.
