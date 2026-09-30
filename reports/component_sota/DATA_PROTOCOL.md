# Component-round data protocol

Inherits `reports/paper_final/EVALUATION_PROTOCOL.md` unchanged: split 10,243 / 273 / 225
(`abc_v2_20260915_120927`), song-level paired bootstrap (10,000 resamples, 95% percentile CI),
failures counted as failures, T=1.0 / top-p 0.95 paired-seed sampling for every Qwen model.

## Training data of the new Qwen models (same songs, same split)

| model | examples | built by | check |
|---|---|---|---|
| Melody-Only (`mel`) | 10,243 whole songs + 5,089 infill (the **same** 50% of songs and the same targets as E3b's mixture) = 15,332, exactly E3b's count | `paper_eval/component_sota/build_datasets.py --task mel` | every completion re-parses to the identical melody (notes, lyrics, melisma); infill completions contain no chord symbol |
| Melody→Chord (`chord`) | 10,243 | `--task chord` | chord lines re-parse to exactly the song's own chords (onset, symbol) |
| Melody+Lyrics→Chord (`chord_lyr`) | 10,243 | `--task chord_lyr` | same |

Inputs of the melody-only model are E3b's inputs exactly (plan, lyrics, key, meter, tempo,
`section i/N`); only the task sentence and the absence of chords in the target differ. The
harmonizer prompt holds meter, tempo, key, section plan and the melody in ABC-v2 without chords
(lyric lines removed for `chord`; kept for `chord_lyr`); the completion is only chord lines, so the
melody is read-only by construction.

Token budget: melody-only mixture 40.5M tokens (24.0M supervised) vs E3b's 44.7M (26.9M);
harmonizer 27.8M (9.0M supervised), with lyrics 34.2M.

## CSL-L2M retraining data

`paper_eval/component_sota/csl_l2m.py build-data` converts the canonical songs to CSL-L2M's
REMI-aligned events exactly as its `mid2events.py` builds them (64 positions per 4/4 bar, Bar event
on bar change, ALIGN per character, SEQ per lyric line, `*` melisma), with our split: 9,681 train /
260 validation / 211 test songs (the 562 / 13 / 14 non-4/4 songs cannot be represented). Only notes
carrying a Han character (and their melisma continuations) are kept; a note carrying k syllables is
split into k equal parts; durations snap to the 109 values in CSL-L2M's melody dictionary; lyric
vocabulary rebuilt from our train split (3,867 characters). The upstream dataloader reads the files
unchanged (checked).

## Evaluation songs

* Symbolic: all 225 test songs. Systems that cannot process a song (CSL-L2M / AccoMontage2: 3/4)
  score it as a failure; their numbers are also given on the 211 4/4 songs where relevant.
* Audio: a frozen 36-song subset of the 211 4/4 songs (`audio/subset.json`), stratified by song
  length, lyric length, mode and pseudo-label difficulty, selected by hash before any audio existed.

## Lyric interventions

Replacement lyrics are paper-final's `lyrics_all` donors: another held-out song with all-Chinese
lyrics and the closest total syllable count, cut to each section's original syllable count. All
four melody systems receive the identical replacement lyric and the identical four seeds.

**Subset caveats (disclosed, not corrected).** The spec suggested stratifying also by section count,
key and model agreement. Section count is covered only through the song-length tercile, key only
through mode (not tonic), and E3b's plan agreement is *recorded* per song (`e3b_plan_exact`) but was
not a stratum. We did not re-draw the subset: it was frozen on 2026-09-27 04:04, before any audio
existed, and re-selecting after results would be selection on outcomes.

## Measurement decisions made during the round (all applied to every system alike)

* **Backing excerpts:** backings are rendered on the first ≤ 95.1 s of each song (two
  MuseControlLite windows) to bound cost; SVS renders are whole songs.
* **ASR with VAD:** Paraformer-zh is run with `fsmn-vad` segmentation. The first pass without VAD
  transcribed only a fragment of each multi-minute song (PER ≈ 0.91 for every system, including
  the pseudo-reference); those outputs were kept as `*.asr_novad.json` and not used.
* **SongEval file naming:** SongEval keys its results by file stem, and every backing file is
  `mix.wav`; wavs are linked under unique names and mapped back (`scripts/component_sota/audio_tools.sbatch`).
* **BTC under new NumPy:** BTC-ISMIR19 is run through `paper_eval/component_sota/tools/btc_run.py`,
  which restores the removed `np.float`/`np.int` aliases; BTC's code is unchanged.
* **DD, timebase-invariant (DD_tb):** CSL-L2M's corpus notates melodies about 1.5–2× denser per bar
  than ours (5–8 vs 3.5 notes per bar on its 8 shipped example songs), so plain DD partly measures a
  notation convention. DD_tb takes the best of ×0.5 / ×1 / ×2 duration rescalings; it is reported next to DD, for every system.
* **Structure metrics** are n/a for systems that receive no section plan (CSL-L2M): their sections
  are derived post hoc from where each lyric line lands and are not scored against the plan.
