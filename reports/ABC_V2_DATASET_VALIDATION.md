# Dataset validation: `abc_v2_20260915_120927`

| | |
|---|---|
| Path | `data/generated/abc_v2_20260915_120927/` (git-ignored, 552 MB plus E3 task files) |
| Parent | `data/generated/abc_v1_20260915_012459`: its canonical songs are the only input, and the corpus is not re-read |
| Built by | `scripts/build_abc_v2_dataset.py`, Slurm job 40179446 (16 CPU workers, ckpt-all). Cleaning rule `section_boundary_snap_v2`; the earlier build `abc_v2_20260915_113908` (rule v1) is superseded, see ABC_V2_DATA_CLEANING.md §2 |
| Rebuild | job 40179447, written to `abc_v2_20260915_120927_rebuildcheck`, kept for provenance |
| Validated by | `scripts/validate_abc_v2_dataset.py`, job 40179448. It re-derives everything from the written v2 **and** v1 files. Result: `validation.json` → **V2_VALIDATION_OK, 0 hard failures** |
| Schema / cleaning | `reports/ABC_V2_SCHEMA.md`, `reports/ABC_V2_DATA_CLEANING.md` |

## 1. Files

| file | content |
|---|---|
| `songs_{split}.jsonl` | cleaned canonical song, spec, ABC-v2, ABC-v1 of the cleaned song, cleaning decisions, pathology flags, token counts |
| `sft_{split}.jsonl` | **E1/E2**: ABC-v2 prompt → ABC-v2 completion |
| `sft_v1clean_{split}.jsonl` | **E1c** (cleaning-only ablation): ABC-v1 prompt → ABC-v1 completion on the cleaned boundaries |
| `sft_longrange_{train,validation}.jsonl` | **E3**: all whole-song ABC-v2 examples plus masked late-section examples for a deterministic 50% of songs (`scripts/build_longrange_tasks.py`) |
| `infill_{split}.jsonl` | one masked-section example per eligible song (evaluation) |
| `cleaning_log.jsonl` | every changed song: old/new sections and every decision |
| `split_manifest.json`, `build_report.json`, `longrange_report.json`, `validation.json` | provenance, counts, sha256 |

## 2. Split (reused, not redrawn)

| | train | validation | test |
|---|---|---|---|
| songs | **10,243** | **273** | **225** |
| identical to the v1 `split_manifest.json` | yes | yes | yes |
| song-id overlap with other splits | 0 | 0 | 0 |

**Leakage re-verification.** This recomputes the checks from the v2 songs, with and without the build's document-frequency filter (shingles shared by >100 train songs):

| held-out vs train | exact lyric duplicates | lyric 6-gram containment max / p95 / ≥0.3 | melody 8-note containment max / p95 / ≥0.3 |
|---|---|---|---|
| validation (df≤100 filter) | 0 | 0.225 / 0.027 / 0 | 0.043 / 0.008 / 0 |
| validation (no filter) | 0 | 0.225 / 0.027 / 0 | 0.043 / 0.008 / 0 |
| test (df≤100 filter) | 0 | 0.043 / 0.022 / 0 | 0.194 / 0.008 / 0 |
| test (no filter) | 0 | 0.043 / 0.022 / 0 | 0.194 / 0.008 / 0 |

Validation vs test, lyric containment max: 0.020.

* **The filter hides nothing:** the numbers are identical with and without it.
* **Round 1 reproduces:** these are exactly the round-1 residue values (DATASET_VALIDATION.md §1).
* **No split-rule bug:** the canonical de-duplicated split therefore stands unchanged.

## 3. Per-example checks (all 10,741 examples; each row is 10,243 / 273 / 225 = pass)

| check | train | validation | test |
|---|---|---|---|
| ABC-v2 parses | 10,243 | 273 | 225 |
| ABC-v2 parses strictly (zero parser errors) | 10,243 | 273 | 225 |
| ABC-v2 → canonical equals the stored cleaned song (onsets, durations, pitches, lyrics, melisma, chords, bars, sections, meta) | 10,243 | 273 | 225 |
| ABC-v1 (cleaned) round trip | 10,243 | 273 | 225 |
| `strip_v2(ABC-v2)` == ABC-v1 (cleaned), byte-exact | 10,243 | 273 | 225 |
| v1 writer reproduces the ABC-v1 (cleaned) file | 10,243 | 273 | 225 |
| **notes and lyric-note alignment equal to ABC-v1** | 10,243 | 273 | 225 |
| **chords equal to ABC-v1 (chord retention)** | 10,243 | 273 | 225 |
| bar beat counts equal to ABC-v1 | 10,243 | 273 | 225 |
| meter / tempo / key equal to ABC-v1 | 10,243 | 273 | 225 |
| section label sequence equal to ABC-v1 (**section counts** unchanged) | 10,243 | 273 | 225 |
| sections tile the bars exactly (**bar counts** conserved) | 10,243 | 273 | 225 |
| spec rebuilds from the song | 10,243 | 273 | 225 |
| ABC-v2 prompt / ABC-v1 prompt rebuild byte-identically from the spec | 10,243 | 273 | 225 |
| prompt syllables == sung ABC syllables, in order | 10,243 | 273 | 225 |
| **counters: present, self-consistent, equal to the plan; headers equal to the plan; every section ends on `[r:1]`** | 10,243 | 273 | 225 |
| number of bar counters == number of bars | 10,243 | 273 | 225 |
| **MIDI round trip** (ABC-v2 → canonical → MIDI → melody read back identical) | 10,243 | 273 | 225 |

The builder runs the same checks independently at build time: `build_report.json` has `check_failures: {}` and `split_equals_v1: true`.

## 4. Determinism

Two independent builds (jobs 40179446 and 40179447, different nodes) produced **identical sha256 for all 11 output files**. `songs_train.jsonl` is byte-identical under `cmp`. The builder uses no randomness and processes songs in sorted id order. The E3 infill selection is a sha256 hash of the song id.

## 5. Token lengths (Qwen3.5 tokenizer; SFT = prompt + completion + EOS)

| split | ABC-v1 median / max | ABC-v1-clean median / max | **ABC-v2** median / p95 / p99 / max | v2 / v1 tokens | v2 > 4,096 | v2 > 8,192 |
|---|---|---|---|---|---|---|
| train | 2,202 / 5,833 | 2,199 / 5,832 | **2,858 / 4,062 / 4,629 / 6,703** | 1.290 | 467 | 0 |
| validation | 2,435 / 3,960 | 2,435 / 3,960 | 3,079 / 4,179 / 4,711 / 5,008 | 1.281 | 16 | 0 |
| test | 2,346 / 3,790 | 2,347 / 3,790 | 3,047 / 4,149 / 4,659 / 4,832 | 1.284 | 16 | 0 |

**Train totals:**

* ABC-v1: 23.10M tokens;
* ABC-v1-clean: 23.10M tokens;
* ABC-v2: **29.80M** tokens (25.47M supervised completion + EOS);
* E3 mixture: 29.80M + 14.93M infill (1.46M supervised).

The fast `tokenizers` counts used by the v2 builder equal the stored AutoTokenizer counts of every unchanged song (64/64 checked in the smoke build; identical prompts give identical counts).

## 6. Long-range task files (E3)

| | train | validation | test |
|---|---|---|---|
| songs with an eligible later-half lyric section (≥2 bars) | 10,230 | 270 | 224 |
| target label chorus / verse / other | 6,950 / 3,035 / 245 | 198 / 69 / 3 | 163 / 57 / 4 |
| infill examples mixed into E3 training | 5,089 | 136 | – |
| infill example tokens median / max | 2,893 / 6,733 | 3,106 / 5,039 | – |

For each infill example, the gapped song plus the target section reconstructs the ABC-v2 song exactly; `split_abc_sections` is asserted lossless on every song.
