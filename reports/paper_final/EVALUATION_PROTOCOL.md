# Paper evaluation protocol (frozen 2026-09-25)

Machine-readable form: `configs/protocol.json`. Frozen song lists: `configs/test_song_ids.txt`
(225 ids, sha256 of the sorted list `8b511dde4379080b…`) and `configs/clean_subset_song_ids.txt`
(34 ids). Code: `paper_eval/` (this round) on top of `qwen_abc/` (parser, metrics, prompts).
Nothing here changes a metric used by an earlier report; new metrics are additions.

## 1. Data and split

| | |
|---|---|
| corpus | SheetSage-Pro `sheetsage_zh_training_views_v2_lvcr`, 11,673 Mandarin songs, all pseudo-labels (automatic transcription) |
| split | inherited from MIDI-LLM `phoneme_leadsheet_section_v4` (10,247 / 591 / 581), then held-out songs removed if their lyrics are an exact or near duplicate of a train song (6-syllable shingle containment ≥ 0.3; containment is bimodal so any threshold 0.2–0.8 gives the same set), and 4 songs whose ABC round trip failed |
| **canonical split** | **10,243 train / 273 validation / 225 test** (`data/generated/abc_v2_20260915_120927/split_manifest.json`) |
| verified leakage | 0 exact lyric duplicates; max test-vs-train lyric containment 0.043, melody containment 0.194 (LONG_STRUCTURE_EXPERIMENTS §9) |
| evaluation specs | `songs_test.jsonl` of `abc_v2_20260915_120927`: every system is prompted with and scored against these specs (cleaned section boundaries). E0 (ABC-v1) receives the same specs in its own prompt format |
| clean subset | 34 songs that pass 12 rule filters and a manual reading (CLEAN_TEST_SUBSET.md); selection never used model output |

All 225 songs are evaluated for every system whenever technically possible. A failed or refused
generation stays in the denominator of every rate.

## 2. Normalization

* **Lyrics.** A syllable is one CJK character or one maximal run of non-space non-CJK characters
  (a latin word) — `qwen_abc.prompt.split_syllables`. Requested syllables are the spec's lyric lines
  flattened in order; sung syllables are the lyric tokens attached to notes in note order (a note
  carrying several syllables contributes all of them). No punctuation or case folding is needed:
  specs contain neither.
* **ABC.** Generated text is parsed by `qwen_abc.abc.parse_abc`. ABC-v2 section comments and
  `[r:k]` counters are ABC remarks and are ignored by the parser (they are scored separately by
  `counter_report`). Chords are normalized by `normalize_chords` (a chord lasts until the next).
* **MIDI / canonical song.** Every system is converted to the same canonical `Song`: integer ticks
  of a 16th note, per-bar beat counts, sections on bar lines, monophonic melody, chord segments.
  MIDI-LLM output is converted by `scripts/midi_llm_baseline.py:to_song` (round-2 converter,
  unchanged), with its section's bar markers as bar lines.
* **Sections.** Labels ∈ {intro, verse, prechorus, chorus, bridge, instrumental, outro, other}
  (the cleaned test set uses 7; no prechorus). MIDI-LLM, which knows no `other`/`prechorus`
  distinction in this checkpoint, receives `verse` for untrained labels (round-2 rule, recorded per
  song); its label accuracy is not reported as learned behaviour.

## 3. Metric definitions

| metric | definition |
|---|---|
| generation success | the system returned an output at all (MIDI-LLM G2P refusals and crashes are failures) |
| parse validity (`parse_ok`) | the parser produced a song |
| strict validity (`strict_ok`) | parsed with **zero** structural errors: bar duration ≠ declared meter, lyrics without/before music, bad header fields, mid-bar section/meter changes, invalid chord symbol, off-grid/zero durations, broken or dangling ties, unsupported tokens, missing final barline, empty bar/section, missing tempo, lyric overflow (more syllables than notes in a line), lyric bar overflow, orphan melisma. MIDI-LLM: its own validator (not comparable; reported separately) |
| bar count | number of bars written (bar-line segments with content, including irregular bars) |
| **exact structure** (`section_plan_exact`) | the sequence of (label, bars) over sections equals the requested sequence exactly |
| section-count accuracy | number of sections equals the request |
| section-label accuracy | label sequence equals the request; per-section rate = sections at the same index with the requested label |
| per-section bar accuracy | `section_bars_exact_frac`: sections at the same index with the requested label **and** bar count |
| absolute bar-count error | \|total bars − requested total\| |
| early termination | EOS emitted with fewer sections than requested |
| over-generation | more sections than requested |
| countdown consistency (ESS models) | `counter_self_consistent_frac`: `[r:k]` equals the bars actually left in the written section; `counter_plan_frac`: equals bars left under the requested plan |
| **lyric recall** | LCS(requested syllables, sung syllables) / #requested — in-order coverage |
| lyric precision | LCS / #sung |
| order accuracy | `lyric_exact` (sung sequence identical to request) and section-local recall (LCS per section index) |
| omission rate | 1 − lyric recall |
| duplication | sung syllables beyond the LCS: 1 − precision |
| **syllable cramming** | `cram_syllable_frac`: syllables sung on a note that carries more than one syllable, / all sung syllables (corpus reference 0.057) |
| unsung notes | `wordless_note_frac`: notes with no syllable and no melisma; melisma = `melisma_note_frac` |
| music statistics | `qwen_abc.metrics.song_metrics` / `long_structure_metrics`: pitch range, note count, notes per bar, duration distribution, 16th/8th off-beat and syncopation proxies, distinct-bar fraction, chords per bar, chord-tone agreement, melody/chord roots in the declared key, n-gram repetition, same-label motif similarity |

Rates count a failed parse as a failure; content metrics (e.g. pitch range) are averaged over
parsed outputs and always reported with their own N.

## 4. Decoding

* Canonical sampling: **T = 1.0, top-p 0.95, top-k off** (round-2 operating point, chosen on the
  64-song validation subset, never on test).
* New generations use the paired-seed Gumbel-max sampler (`paper_eval/sampler.py`): an exact
  sample from the same truncated distribution; seeds S1=101, S2=202, S3=303, S4=404. Token budget:
  prompt + completion ≤ 9,216 (the longest reference completion is 6,703 tokens); batch 16,
  left-padded. Legacy round-2/R3/R4 runs used torch multinomial sampling at the same T/top-p with
  prompt+completion ≤ 8,192 (10,240 for E1 T1.0); none of their outputs hit the limit.
* MIDI-LLM: its own CLI and sampler, mode A (recommended constraints:
  `--forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4 --require-section-syllables
  --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0`), seeds 1000-1003, context overflow
  retried once with `max-new-tokens = 10240 − condition − 8`.

## 5. Single, selector, oracle

* **single** — one sample. Reported as the mean over the available samples of each song (the
  expected single-sample score; all seeds are exchangeable), with the per-seed spread.
* **selector@4** — of 4 samples, the one maximizing the `sum` rule of `scripts/select_best_of_n.py`
  (weights in `configs/protocol.json`); uses only prompt + output (whitelisted signals). Note: its
  signals include `section_plan_exact` and `lyric_recall`, so for those two metrics selector@4 is
  equivalent to oracle@4 by construction.
* **oracle@4** — per metric, the best of the 4 samples for that metric (an analysis upper bound;
  uses the reference where the metric does). **oracle@8** where 8 exchangeable samples exist
  (E3b, MuPT: 4 legacy + 4 paired-seed samples).

## 6. Interventions (INTERVENTIONAL_CONTROLLABILITY.md)

For each song a target section is chosen deterministically from its id (a verse/chorus with lyrics,
≥ 6 bars and ≥ 4 regular bars, preferring inner sections). Each intervention changes **one**
control and nothing else; `controls_changed` verifies every task (the driver refuses any other):
bars −4/−2/+2/+4 (regular bars added/removed, lyrics unchanged); label → bridge, verse↔chorus; key
+2/+5/−3 semitones; tempo ×0.8/×1.25; lyrics of the whole song (`lyrics_all`) or the target section
(`lyrics_sec`) replaced by another held-out, all-Chinese song's syllables with identical per-section
syllable counts (closest total length). Seeds: S1, S2 (lyrics_all: S1-S4); `orig` has S1-S4.

**Estimands.** (i) Target-specific adherence (requested vs realized value) per sample. (ii) Output
change beyond sampling noise, per song: *cross* = mean distance between the song's `orig` samples
and its intervened samples over all seed pairs; *within* = mean distance between `orig` samples
of different seeds; **effect = cross − within** (> 0 means the intervention moves the output
further than reseeding does). Also reported: the seed-paired ratio d(A,B)/d(A,C). Distances:
normalized-LCS on (onset-in-bar, pitch, duration) melody tokens, pitch, interval (contour),
(onset, duration) rhythm, chord changes and chords per beat; plus non-saturating feature
distances (pitch-class and interval histogram JS, |Δ pitch mean|, |Δ notes per bar|).
`replay` (orig S1 re-batched) gives the numeric floor of seed pairing.

## 7. OOD (OOD_GENERALIZATION.md)

Regimes defined from the training distribution (`data/train_distribution.json`): in-distribution
= the song's own plan; compositional = chorus-first opening, verse-final ending; extrapolative =
20+-bar and 28+-bar sections, +3/+6 chorus repeats (section counts up to 23 vs corpus max 17),
tempo 60/240 BPM. Adherence is also regressed on plan surprisal (bits under the corpus model).

## 8. Statistics

Song is the unit. 95% percentile bootstrap over songs with 10,000 resamples (seed 0); paired
comparisons resample song-level differences over songs present in both arms. With several samples
per song, the song-level mean is taken first (cluster bootstrap). Notes, bars or tokens are never
treated as independent units for headline significance. Differences are called significant only
when the CI excludes 0; everything else is reported as not resolved.

## 9. Checkpoints

`configs/protocol.json: models` (paths) and `experiment_registry.json` (paths, partial sha256,
training config). Current best Qwen-ABC = E3b (`qwen_abc_r2_offload/runs/e3b_201131/final_model`).
