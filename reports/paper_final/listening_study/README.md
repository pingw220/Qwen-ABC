# Control-oriented listening study (package only — **human results PENDING**)

No listener has taken this study. Nothing here is a human result; `analysis.py` refuses to run
until response files exist, and its tests use synthetic responses generated inside the test.

## What it tests

Whether the edits the model was *asked* to make are audible, and whether they cost anything
musically. Items pair E3b's output for an intervention with its output for the original control
at the same sampling seed (`experiments/paper_final/gen/qwen_e3b/{orig,<condition>}/<song>_S1.json`).

| condition | 2AFC question (A / B / none) | correct answer |
|---|---|---|
| `bars_p4` / `bars_m4` | which clip's MIDDLE section is longer? | edited / original clip |
| `key_p5` / `key_m3` | which clip is in the higher key? | edited / original clip |
| `lyrics_sec` | which clip sings the words on screen? | edited clip |

Then, with the original labelled: *did the edited clip carry out the requested edit?*
(yes/partly/no) and *did everything else stay the same?* (1-5); and blinded plausibility of
each clip (1-5). Items are picked by a hash of the song id among songs where both outputs parse,
**not** by whether the model complied, so `edit_happened` estimates compliance as heard; the
automatic compliance flag is in `items.csv` for comparison.

Catch items per list: identical pair (answer: none), a clip against its own symbolic +7-semitone
transposition, and against a copy with the middle section doubled.

Known ambiguity, kept on purpose: the model can write the requested key in a different octave
register (in the first build one `key_p5` item moved the declared key up a fourth while the
melody's mean pitch fell 5.7 semitones). `items.csv: realized_register_agrees_with_request` flags
these, and `analysis.py` reports key accuracy with and without them.

## Files

| file | committed | content |
|---|---|---|
| `items.csv` | yes | answer key: item id, condition, question, song id, hashed stimulus names, realized measurements (bars, keys, register shift, lyric recall) |
| `lists/list_<k>.csv` | yes | experimenter's run sheet per list: trial order (shuffled per list), clip A/B (edited-first on alternating items, flipped between lists), edit text. Participants see only audio + question text |
| `design.json` | yes | questions, conditions, eligible songs per condition, counts, seed |
| `response_schema.csv` | yes | one row per participant × trial × question |
| `analysis.py` | yes | accuracy vs chance with song-level bootstrap CIs, Fleiss' kappa, Krippendorff's alpha, catch exclusion |
| `INSTRUCTIONS.md` | yes | text read to participants |
| `materials/` | **no** (lyrics, audio) | `stimuli/<hash>.song.json` excerpts, `lyrics_shown.json`, `audio/` pilot renders |

## Rebuild (deterministic) and render

```bash
source scripts/env.sh
python -m paper_eval.listening_study build --per-condition 12 --lists 4   # after the E3b intervention runs finish
python -m paper_eval.listening_study render-cpu --pilot 16               # fluidsynth backing + FastSinger inputs
sbatch scripts/paper_final/listening_render.sbatch                        # FastSinger vocal + mix (GPU)
```

Stimuli are excerpts: the target section with one neighbour on each side (key items: the
target section alone), capped at 24 bars with the middle section kept whole. Audio = FastSinger
vocal (same singer and flags as `scripts/slurm/midi_sag_render.sbatch`) over a General-MIDI
fluidsynth backing (chords + melody doubling). A MuseControlLite full-band render is possible
with `midi_sag_render.sbatch` but costs minutes of GPU per clip and adds a generative
accompaniment that differs between A and B for reasons unrelated to the edit, so the pilot uses
the deterministic backing.

Pilot status: see `pilot.json` (items) and `materials/audio/render_log.tsv` (vocal ok or
instrumental fallback). Stimulus names are hashes of (condition, song), so a rebuild keeps the
names of items that stay selected and the render job skips finished files.

## Running the study

Recommended: 4 lists × ≥6 participants (Mandarin speakers for `lyrics_sec`), headphones,
~45 minutes per list. Collect responses as CSV per `response_schema.csv` into `responses/`, then

```bash
python reports/paper_final/listening_study/analysis.py --responses reports/paper_final/listening_study/responses --out results.json
```

## Build history

The first build (50 items) ran while the E3b intervention outputs were still being generated. It was
rebuilt on the complete outputs (66 items; only 4 items of the first build are still selected), and
the 16-item pilot was re-rendered for the final design (job 40618328, 32 stimuli, all vocals ok;
the first pilot's renders remain on disk but belong to the superseded design).
