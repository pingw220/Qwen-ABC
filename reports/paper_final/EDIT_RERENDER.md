# Edit → re-render: the lead sheet as an editable interface

**Claim tested:** a generated symbolic lead sheet can be edited, by hand or by re-conditioning the
model on one changed control, and the edit carries through to rendered audio as requested. This is
a demonstration of *editability*, measured on the score and on rendered durations. It is not a claim
about audio quality, and in particular not a comparison with commercial audio generators. The
vocal is FastSinger and the backing is MuseControlLite; both add their own artefacts.

## Setup

* **Songs:** two held-out test songs chosen from reference features only, before any model output
  was seen. The criteria were a later-half, 8-bar, all-4/4 chorus (the section E3b's infill training
  covers), all-Chinese lyrics, 70–125 BPM and no pathology flag. `4VfHbfbzNPZmX5KI4iAOyT` is also in
  the 34-song human-checked clean subset. `6coZcwGP3Cmhyqzvp6sEJS` was the second candidate. The
  edited section is `paper_eval.interventions.pick_target_section`'s choice (the one the
  intervention study uses).
* **Model:** E3b (`qwen_abc_r2_offload/runs/e3b_201131/final_model`), T=1.0 / top-p 0.95, with the
  paired-seed sampler (`paper_eval/sampler.py`).
* **Source score:** four whole-song samples (S1–S4) of the unedited plan. One was chosen by the
  repo's inference-only `sum` selector: S1 for song 1 and S4 for song 2. All eight were strict-valid
  with the exact plan.
* **Edits** (each changes exactly one control):

| demo | local / direct edit | model re-generation |
|---|---|---|
| A structure: target chorus 8 → 12 bars | **section infill**: E3b rewrites only that section given the rest of the score (the masked-late-section prompt it was trained on, `qwen_abc/longrange.py`), so every other section's text is kept byte-identical | whole song under the edited plan, same seed |
| B key +5 semitones | exact symbolic transposition of the parsed score (notes +5, chord symbols respelled, K: updated) | whole song under the new `Key:`, same seed |
| C tempo ×1.25 | only the `Q:` field changes | whole song under the new `Tempo:`, same seed |
| D lyrics of one section | **section infill** with that section's lyrics replaced: an all-Chinese donor, the same syllable count, `replace_lyrics` / `pick_donor` | — |

  For the infill edits, 4 seeds were drawn. The demo sample is the first seed (S1→S4) that is
  strict-valid with the requested label and bar count. These are inference-time signals only, and
  all four seeds are reported.
* **Rendering:** the repo's unchanged path. `scripts/export_leadsheet_midi.py` wrote all 16 lead
  sheets (0 skipped; FastSinger note/symbol pairing checked). Then
  `scripts/slurm/midi_sag_render.sbatch` ran FastSinger followed by MIDI-SAG/MuseControlLite's own
  runner. Nothing in MIDI-SAG, FastSinger or MuseControlLite was modified.

## Results

"Other sections" compares every non-edited section of the edit with the source. "Text" is
byte-identity of the ABC blocks; "parsed notes" is identity of every note's onset, duration,
pitch, lyric and melisma relative to the section start. The audio ratio is mix.wav duration,
edit ÷ source.

| song | edit | method | requested control realized | what else changed | audio duration ratio |
|---|---|---|---|---|---|
| 4VfHbfbz | A structure 8→12 | local infill | 12 bars (4/4 seeds on target) | other sections: text 8/8, parsed notes 8/8; strict-valid True | 1.0537 (= score 194.2 / 184.3 s) |
| 4VfHbfbz | A structure 8→12 | whole-song regen | 12 bars, plan exact 1.0 | other sections identical 2/8; lyric recall 0.982 | 1.0537 |
| 4VfHbfbz | B key Eb major→Ab major | direct transposition | every note +5: True; mean pitch +5.00; melody in new key 1.0 | rhythm, lyrics, chord count unchanged; strict-valid True | 1.0000 |
| 4VfHbfbz | B key Eb major→Ab major | regeneration | declared key Ab major; mean pitch +3.29 st; melody in new/old key 1.0/0.97 (source 0.936/1.0) | new melody; strict-valid True; lyric recall 0.988 | 1.0000 |
| 4VfHbfbz | C tempo 97→121 BPM | Q: edit | tempo 121; notes identical True | score 184.33→147.77 s | 0.8017 (req 0.8; 97/121 = 0.8017) |
| 4VfHbfbz | C tempo 97→121 BPM | regeneration | tempo 121; bars 75/75; plan exact 1.0 | notes/bar 4.747 vs 4.293; lyric recall 0.890; strict-valid True | 0.8017 |
| 4VfHbfbz | D lyrics (section 8, 54 syllables) | local infill | new-lyric recall in section 1.0 (seeds: 1.0, 1.0, 1.0, 1.0); old-lyric recall 0.0741 (chance: source vs new 0.0741) | other sections: text 8/8, parsed notes 8/8; bars kept 8; strict-valid True (seed S3) | 1.0000 |
| 6coZcwGP | A structure 8→12 | local infill | 12 bars (4/4 seeds on target) | other sections: text 7/7, parsed notes 6/7; strict-valid True | 1.0552 (= score 189.3 / 179.4 s) |
| 6coZcwGP | A structure 8→12 | whole-song regen | 12 bars, plan exact 1.0 | other sections identical 3/7; lyric recall 1.000 | 1.0552 |
| 6coZcwGP | B key E major→A major | direct transposition | every note +5: True; mean pitch +5.00; melody in new key 1.0 | rhythm, lyrics, chord count unchanged; strict-valid True | 1.0000 |
| 6coZcwGP | B key E major→A major | regeneration | declared key A major; mean pitch -0.54 st; melody in new/old key 0.495/0.473 (source 0.964/1.0) | new melody; strict-valid False; lyric recall 1.000 | 1.0000 |
| 6coZcwGP | C tempo 97→121 BPM | Q: edit | tempo 121; notes identical True | score 179.38→143.8 s | 0.8017 (req 0.8; 97/121 = 0.8017) |
| 6coZcwGP | C tempo 97→121 BPM | regeneration | tempo 121; bars 74/74; plan exact 1.0 | notes/bar 2.0 vs 3.216; lyric recall 1.000; strict-valid True | 0.7992 |
| 6coZcwGP | D lyrics (section 7, 26 syllables) | local infill | new-lyric recall in section 1.0 (seeds: 1.0, 1.0, 1.0, 0.8077); old-lyric recall 0.0385 (chance: source vs new 0.0385) | other sections: text 7/7, parsed notes 6/7; bars kept 8; strict-valid True (seed S1) | 1.0000 |

### Reading it

* **Local edits work as an interface.** Section infill hit the requested 12 bars on 8/8 seeds
  (both songs) and kept every other section's text byte-identical. The lyric replacement sang the
  new section lyrics with recall 1.0 on 7/8 seeds (0.81 on the eighth). Old-lyric recall in that
  section fell to chance level: it equals the LCS overlap between the source and the new text, 0.07
  and 0.04. Parsed notes of the other sections are identical except in song 2, where the infilled
  chorus ends on a tie and a held note now carries one extra note into the following outro. The
  outro's text is unchanged; the tie crosses the edit boundary. Strict validity of the infilled
  section varies by seed: 3/4 and 4/4 for structure; only 1/4 for song 1's lyric edit, whose other
  three seeds are lyric-alignment-invalid despite recall 1.0.
* **Direct edits are exact by construction.** Transposition moved every note by exactly +5 with
  rhythm and lyrics untouched. The Q: edit changed only the tempo. The rendered audio follows: every
  mix's duration equals its score duration, and the tempo edit renders at exactly 97/121 = 0.8017 of
  the source length.
* **Whole-song re-generation follows the control but writes a new song.** Every regenerated score
  realizes its target: 12-bar chorus with the exact plan, the new tempo with the bar count kept, the
  new K: declared. Only the prefix before the first divergence survives: 2/8 and 3/7 sections
  identical for the structure edit, the same seed-coupling limit documented in AUDIT.md §9. For key,
  declaring the key is not the same as writing in it:
  * **song 1:** the melody moves up 3.3 semitones on average and sits fully in Ab major (1.00; 0.97
    in the old Eb major, which shares 6 of 7 pitch classes);
  * **song 2 (a failure):** the model declares A major but writes a melody that is in neither A
    major (0.50) nor the original E major (0.47), and the score is not strict-valid.

  For tempo, the regenerated melodies changed density in opposite directions (notes/bar 4.29 → 4.75
  and 3.22 → 2.00); with two songs no tempo→density claim is made.
* **Implication for the interface:** key and tempo are best edited *symbolically*, which is exact.
  Structure and lyric edits need the model, and section infill confines the change to the edited
  section, whereas whole-song re-generation rewrites the song.

## Artifacts

* `edit_rerender_demo/demo_<song>.json`, `summary.json`: requested edits, all per-seed
  verification, source selection, render paths and durations (no lyrics).
* `edit_rerender_demo/scores/<song>__{source,A_local,A_whole,B_direct,B_model,C_direct,C_model,D_local}.abc`
  and `materials/leadsheets*/`: lead-sheet MIDI, FastSinger inputs and audits. **git-ignored**,
  they contain lyrics.
* `materials/generations/`: raw generations for all seeds (git-ignored).
* Audio: `/gscratch/ark/pingw220/qwen_abc_r2_offload/paper_final_edit_rerender/<song>__<variant>/`
  (`mix.wav`, `vocal_fs.wav`, backing), 16/16 rendered, 0 failed.
* Figure 6: `figures/edit_rerender_summary.{pdf,png}` (song 1: requested vs measured per demo, plus
  before/after piano rolls around the edited section).
* Logs: `job_logs/pf-edit-gen-40604536.out`, `job_logs/pf-edit-render-40604716.out`,
  `job_logs/pf-edit-render-40604717.out`.

## Compute

| job | what | partition / GPU | wall |
|---|---|---|---|
| 40604536 | generation (8 source samples, 16 infills, 6 regenerations) | ckpt-all (ckpt-ark), 1× L40S | 4 min 53 s |
| 40604716 | render song 1 (8 lead sheets) | ckpt-all (ckpt-ark), 1× L40S | 25 min 8 s |
| 40604717 | render song 2 (8 lead sheets) | ckpt-all (ckpt-ark), 1× L40S | 59 min 41 s |

The total is about 1.5 GPU-h. The ark L40S/L40 allocations were full with the main paper-final
arrays, so these ran on the preemptible checkpoint queue with `--requeue`, which is the repo's own
convention in `scripts/slurm/eval.sbatch`. Both stages are resumable. An earlier submission
(40604530, gpu-l40s) was cancelled while still pending and never ran.

## Not done

* No listener has judged these renders; whether each edit is *audible* as intended is a question for
  `listening_study/`.
* No audio-side pitch verification (f0) of the key edit was run; the key is verified on the score.
* mp3 collection (`scripts/collect_midi_sag_audio.py`) was not run, since it expects the listening-set
  directory layout; the wavs are the artifacts.

## Reproduce

```bash
source scripts/env.sh
sbatch -p ckpt-all -A ckpt-ark --gpus=l40s:1 --requeue scripts/paper_final/edit_rerender_gen.sbatch
python -m paper_eval.edit_rerender build
python scripts/export_leadsheet_midi.py reports/paper_final/edit_rerender_demo/scores/*.abc \
    --outdir reports/paper_final/edit_rerender_demo/materials/leadsheets
# one lead-sheet dir per song (symlinks), then the unchanged renderer:
LEADSHEETS=$PWD/reports/paper_final/edit_rerender_demo/materials/leadsheets_<song> \
OUT=/gscratch/ark/pingw220/qwen_abc_r2_offload/paper_final_edit_rerender \
  sbatch -p ckpt-all -A ckpt-ark --gres=gpu:l40s:1 --requeue \
  --chdir=$PWD/reports/paper_final/edit_rerender_demo/materials scripts/slurm/midi_sag_render.sbatch
python -m paper_eval.edit_rerender audio
python -m paper_eval.edit_rerender figure
```
