# Paper-final audit (2026-09-25)

Scope: everything under `/mmfs1/gscratch/scrubbed/pingw220/music_acc` that bears on the paper,
with Qwen-ABC as the primary repository. Read-only for every other repository. The machine-readable
companion is `experiment_registry.json` (45 entries, built by `python -m paper_eval.registry` from
the run and evaluation directories themselves).

## 0. Repository state at the start of this round

| | |
|---|---|
| repo | `/mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC`, remote `git@github.com:pingw220/Qwen-ABC.git` |
| branch / commit | `main` at `f8bcb96` ("R5: the per-section lyric assignment is not needed information"), 3 commits ahead of `origin/main` |
| user work in progress (left untouched, never staged) | staged rename `scripts/build_repaired_dataset.py → scripts/build_song_variant.py` plus an unstaged edit to it; untracked `configs/r6_pinyin_sft.yaml`, `qwen_abc/pinyin_lyrics.py`, `experiments/r6_pinyin_20260925_104634/` (an R6 pinyin run, job 40602954 on gpu-a40, running during this audit) |
| paper-final branch | `paper-final-eval-2026`, created from `f8bcb96`; the user's staged/unstaged/untracked work rides along uncommitted |
| MIDI-LLM repo | `MIDI-LLM-phoneme-lyric-v1` at `bea5597`, tracked tree clean (only untracked manifests/reports) |

## 1. What already exists

### Data (verified from manifests, not from memory)

| item | value | source |
|---|---|---|
| SheetSage-Pro corpus (D1, `sheetsage_zh_training_views_v2_lvcr`) | **11,673** songs | `abc_v1_*/build_report.json: corpus_songs` |
| inherited MIDI-LLM v4 split | 10,247 / 591 / 581 | same, `inherited_split_songs` |
| held-out songs removed as lyric duplicates of train | test 291 exact + 64 near; validation 274 + 38 (+5 dup of test, +1 near) | `exclusion_reasons` |
| **canonical de-duplicated split** | **10,243 train / 273 validation / 225 test** | `abc_v2_20260915_120927/split_manifest.json` |
| canonical evaluation dataset | `data/generated/abc_v2_20260915_120927` (ABC-v2 specs on cleaned section boundaries) | every round-2+ evaluation |
| human-checked clean test subset | **34 songs** (42 pass 12 rule filters, 8 dropped by manual reading); `reports/clean_test_song_ids.txt` | `reports/CLEAN_TEST_SUBSET.md` |
| human annotation beyond the 34-song selection | **none**: the 34 were checked for pseudo-label *quality*, not re-annotated | |

### Checkpoints

| model | path | exists |
|---|---|---|
| E0 (ABC-v1, no ESS) seeds 1234 / 2345 | `experiments/direct_sft{,_seed2345}_20260915_023500/final_model` | yes |
| E1c (cleaning only) | `experiments/e1c_v1clean_sft_20260915_120949` | **no** (deleted after the 09-15 quota incident; generations kept) |
| E1 (ESS) seeds 1234 / 2345 | `qwen_abc_r2_offload/e1_final_model`, `e1_seed2345_final_model` (symlinked) | yes |
| E1-long (ESS, 758 updates, no infill) | `qwen_abc_r2_offload/runs/e1long_225021/final_model` | yes |
| E2a / E2b (4× batch) | E2a deleted; E2b in offload | E2b only |
| E3 (infill @ 524K) | deleted | no |
| **E3b (ESS + late-section reconstruction; current best)** | `qwen_abc_r2_offload/runs/e3b_201131/final_model` | yes |
| MuPT-1.07B finetune (R4) | `experiments/r4_mupt_20260923_001821/final_model` | yes |
| R3-A / R3-C / R5 | `experiments/r3a_*`, `r3c_*`, `r5_free_*` | yes |
| MIDI-LLM one-stage v4 | `MIDI-LLM-phoneme-lyric-v1/runs/original-midi-llm-onestage-multitask-v1-v4/full/final` | yes |

### Evaluations (225 test songs unless noted; all numbers below re-read from `summary.json`)

| experiment | samples/song | decoding | status |
|---|---|---|---|
| E0, E0s2, E1c, E1, E1s2, E1-long, E2a, E2b, E3, E3b | 1 | T0.8 / p0.95 | complete |
| E0, E1, E1s2, E1-long, E2b, E3b | 1 | T1.0 / p0.95 | complete |
| E3b, MuPT, R3-A, R3-C, R5, E3b-nocram | 4 | T1.0 / p0.95 | complete (900 files each) |
| E3b, MuPT, R3-A, R3-C, R5 selector@4 (`sum` rule) | 1 chosen of 4 | — | complete (`bon_r3b_20260918_100234/*_sum`) |
| MIDI-LLM mode A / mode B | 1 (seed 1000) | its own sampler | complete: 199 generated, 26 refused at G2P |
| decoding sweeps (E0, E1) | 1 | 7 settings | 64 validation songs |
| round-1 conditional probes (lyric swap, tempo, key) | 1 | T0.8 and greedy | **20 songs, round-1 models only** |
| continuation / infill tasks (E1, E3b, …) | 1 | T0.8 | complete |

### Reports

`DATA_AUDIT`, `ABC_SCHEMA`, `ABC_V2_SCHEMA`, `ABC_V2_DATA_CLEANING`, `ABC_V2_DATASET_VALIDATION`,
`DATASET_VALIDATION`, `REPORT` (round 1), `LONG_CONTEXT_AUDIT`, `LONG_STRUCTURE_EXPERIMENTS` (round 2),
`CLEAN_TEST_SUBSET`, `BEST_OF_N_DECODING` (R3-B), `R3A_SYLLABLE_BUDGET`, `CRAMMING_ATTACKS` (R3-C),
`R4_MUSIC_PRETRAINED_BASE`, `R5_FREE_LYRIC_ASSIGNMENT`, `LEADSHEET_TO_AUDIO`, plus 26 tables in
`reports/tables/`. Rendering: 19 lead sheets rendered through FastSinger + MuseControlLite
(`experiments/midi_sag_2026092*`, audio on `/gscratch/ark`).

## 2. What is paper-ready

* **The ESS progression at T0.8, single sample, one training seed** (E0 0.529 → E1c 0.658 → E1 0.858
  → E3b 0.978 exact structure), with the matched-update control E1-long (0.809) and a second
  training seed for E0 and E1. Paired bootstrap CIs exist (`compare_r2.py`, 2,000 resamples).
* **Best-of-4 for E3b** with a validation-selected selector and an explicit inference-only whitelist.
* **The MuPT comparison (R4)** at matched data, recipe, decoding and selector.
* **Cramming negative results (R3-A/B/C)** and **R5 free assignment**, including the listening
  catch on R3-C.
* **The clean-subset contrast for E0/E1**, not for E3b.

## 3. What is incomplete

| gap | why it matters | action this round |
|---|---|---|
| single-sample vs selector@4 vs oracle@4 only for E3b/MuPT; MIDI-LLM has one sample | fair decoding comparison | new MIDI-LLM seeds 1001-1003 (4 samples/song); oracle@4 computed for all |
| ESS ablation at the canonical decoding (T1.0) has one sample per song | sampling noise ±0.05 on validity-type metrics | 4 new samples/song for E0, E1, E1-long, E3b, MuPT |
| **no interventional controllability** (structure, label, key, tempo, lyrics) on current models | "0.996 adherence" is reconstruction of the song's own plan, not proof of control | 13-condition intervention suite on E3b (224 songs × 2-4 seeds), 6-condition suite on MuPT, 5-condition subset (60 songs) on MIDI-LLM |
| the lyric→melody question was only probed on 20 songs with round-1 models | central scientific question | `lyrics_all` (4 seeds) and `lyrics_sec` on all 224 songs |
| no OOD / compositional study | control boundary | 7 OOD plans × E3b, E1, E0 (all songs) and 4 × MuPT |
| no memorization analysis | reviewer concern | nearest-neighbour search against all 10,243 training songs |
| annotation-noise contrast not computed for E3b / MuPT / MIDI-LLM | residual-error attribution | recomputed on the 34-song subset for every system |
| long-range (by song position) analysis only as "late sections exact" | ESS claim is about long range | 5-bucket position analysis for E0/E1/E1-long/E3b |
| no edit → re-render demonstration; no listening-study package | editability claim | built this round (see EDIT_RERENDER.md, listening_study/) |

## 4. What needs regeneration

Nothing existing is invalid. New generations are needed only for what never existed (above). Two
ablation checkpoints (E1c, E3) cannot be regenerated because their weights were deleted; their
existing T0.8 generations remain the only evidence for them and are used as such. **No retraining
is required**: every scientific question in this round is answerable with existing checkpoints.

## 5. Currently unfair comparisons

1. **MIDI-LLM structure is an input.** Section count, bars per section, tempo and the bar grid are
   a click track in its input and pinned by its decoder. Its "exact structure" (0.884 = 199/225,
   i.e. 1.000 on generated songs) is by construction, not learned plan adherence. Its mode B (free
   labels) collapses to 0.027. Structure is therefore *not* compared across Qwen and MIDI-LLM.
2. **MIDI-LLM validity is its own validator**, Qwen's is strict ABC; not the same test.
3. **MIDI-LLM coverage**: 26/225 songs are refused at its G2P front end; they count as failures in
   every rate (generation success denominator 225).
4. **The headline 0.996 is selector@4** while the round-2 table is single-sample T0.8. The selector's
   score includes `section_plan_exact` and `lyric_recall` — computed from prompt and output only, so
   legitimate at inference — but it means selector@4 structure is essentially **oracle@4 structure**.
   Single-sample numbers must be shown next to it.
5. **Qwen vs MuPT** differ in tokenizer (MuPT extended with corpus characters), architecture,
   pretraining corpus/objective, and update count (1,070 vs 758, in MuPT's favour). The comparison
   is controlled for data, recipe, decoding and selector only.
6. **E1 → E3b is confounded with update count** (506 vs 758); E1-long (758 updates, no infill) is
   the matched control and must be used for the objective claim.
7. The published ESS "progression" (0.529 → 0.658 → 0.858 → 0.978 → 0.996) mixes a T0.8 training
   ablation with an inference-time selector at T1.0. They are separated in `ESS_ABLATION.md`.

## 6. Claims already supported

* ESS raises exact structure far beyond seed noise (E0 seeds 0.53/0.61 vs E1 seeds 0.86/0.94) and
  removes early EOS (4-9% → 0%).
* The late-section-reconstruction objective raises structure at matched updates (E1-long 0.809 →
  E3b 0.978, T0.8).
* T1.0/top-p 0.95 recovers corpus-like pitch range and repetition without losing structure.
* Symbolic-music pretraining (MuPT) was not sufficient for plan/lyric adherence in this setting.
* Qwen-ABC sings far more of the requested lyrics than MIDI-LLM (0.97 vs 0.73 recall, single sample).
* Four cramming interventions failed; R3-C's success on metrics was a listening failure.

## 7. Claims currently unsupported

* "The model is controllable": only reconstruction of the song's own plan was measured.
* "Lyrics condition the melody" or "lyrics do not condition the melody": 20 round-1 songs only.
* Any OOD generalization claim.
* "The model does not memorize": no nearest-neighbour search was ever run.
* "Residual E3b error is label noise": the clean-subset contrast was computed for E0/E1 only.
* Any claim of musical *quality* beyond informal listening passes.
* "The lead sheet is an editable interface": renders exist, controlled before/after edits do not.

## 8. Proposed experiments that already exist and are not rerun

* E0/E1c/E1/E1-long/E2a/E2b/E3/E3b T0.8 single-sample evaluations (reused as the T0.8 ablation).
* E3b and MuPT 4-sample T1.0 generations and their selector@4 choices (reused for FAIR_DECODING;
  the new paired-seed samples replicate them independently).
* MIDI-LLM seed-1000 mode A/B (reused as its first sample).
* R3-A/R3-C/nocram/R5 (reused in LYRIC_FAILURES).
* Round-1 probes (cited as history, not as evidence for the current models).

## 9. Sampler note discovered during the smoke test

The paper-final generations use a Gumbel-max paired-seed sampler (`paper_eval/sampler.py`) so that
"same seed" means identical per-step noise regardless of batching. It is an exact sampler for
T=1.0/top-p 0.95 (unit-tested). The smoke test showed that re-running the *same* prompt and seed in
a different batch composition reproduces the first ~150-250 tokens and then diverges: a bf16-level
near-tie flips one token and the two continuations decouple (melody distance 0.76-0.79 vs 0.90-0.98
for a reseed, 3 songs). Seed pairing therefore does not survive a full song, and every
intervention effect in this round is estimated distributionally (cross-condition minus
within-condition distances over several seeds per condition), with the seed-paired ratio reported
alongside. The `replay` condition quantifies the pairing floor on all 225 songs.
