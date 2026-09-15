# Dataset validation: `abc_v1_20260915_012459`

| | |
|---|---|
| Path | `Qwen-ABC/data/generated/abc_v1_20260915_012459/` (git-ignored, 468 MB) |
| Built by | `scripts/build_abc_dataset.py` at commit `0e9806f` plus working-tree changes (bar regularization, chord respelling), all committed afterwards; Slurm job 40170859, 316 s on 30 CPU workers |
| Validated by | `scripts/validate_abc_dataset.py`, which re-derives every number below from the written files → `validation.json`: **VALIDATION_OK**, 0 hard failures |
| Source | SheetSage-Pro `sheetsage_zh_training_views_v2_lvcr` (11,673 songs), split inherited from MIDI-LLM `phoneme_leadsheet_section_v4` |

## 1. Songs and examples

| | inherited (MIDI-LLM v4) | **final** | SFT examples | CPT examples |
|---|---|---|---|---|
| train | 10,247 | **10,243** | 10,243 | 10,243 |
| validation | 591 | **273** | 273 | 273 |
| test | 581 | **225** | 225 | 225 |
| total | 11,419 | **10,741** | 10,741 | 10,741 |

One example per song for each task: SFT is prompt → full-song ABC; CPT is the full-song ABC text alone. The CPT and SFT song sets are identical within each split (checked).

### Exclusions (932 rows in `exclusions.jsonl`, each with its reason)

| reason | split | songs |
|---|---|---|
| not in the inherited split (MIDI-LLM's tier-C / mostly-wordless gate) | – | 254 |
| lyrics exact duplicate of a **train** song | test / validation | **291 / 274** |
| lyrics near-duplicate of a train song (6-syllable containment ≥ 0.30) | test / validation | 64 / 38 |
| lyrics duplicate of a **test** song | validation | 5 exact + 1 near |
| melody near-duplicate (8-note containment ≥ 0.30) not already caught | – | 0 |
| ABC round-trip mismatch (all 4: song has no key estimate, so `K:C` parses back as C major) | train / test | 3 / 1 |
| no sung syllables | train | 1 |
| over the 8,192-token gate | – | 0 |

**The leakage filter is the largest effect in the build.** Of the inherited held-out songs, 61% of test and 54% of validation were duplicates of training songs: alternate uploads of the same recording under a new id. Their lyric containment is almost always ≥ 0.8 (344 of 580 test songs, 308 of 591 validation songs). The distribution is bimodal: songs are either near 1.0 or below 0.1, so the 0.30 threshold is not a knife-edge choice.

Scoring on the inherited test set would have measured memorization for more than half of the songs. The price is a small test set (225 songs), so per-metric confidence intervals are wide. See REPORT.md.

Residual similarity after filtering (validator, recomputed from the written files):

| held-out vs train | lyrics containment max / p95 | melody containment max / p95 |
|---|---|---|
| validation | 0.225 / 0.027 | 0.043 / 0.008 |
| test | 0.043 / 0.022 | 0.194 / 0.008 |

Exact-duplicate checks across splits: identical ABC text 0, identical prompt 0, identical lyric text 0. Song-id overlap between any two splits: 0.

## 2. Token lengths (Qwen3.5 tokenizer, SFT = prompt + ABC + EOS)

| split | min | p10 | median | p90 | p95 | p99 | max | prompt median | ABC median |
|---|---|---|---|---|---|---|---|---|---|
| train | 602 | 1,588 | 2,202 | 2,986 | 3,194 | 3,679 | 5,833 | 342 | 1,857 |
| validation | 632 | 1,626 | 2,435 | 3,099 | 3,386 | 3,763 | 3,960 | 370 | 2,043 |
| test | 730 | 1,720 | 2,346 | 3,144 | 3,322 | 3,649 | 3,790 | 362 | 1,981 |

Share exceeding a context limit (train): >2,048 tokens 61.3%, >4,096 tokens **0.19%** (19 songs), >6,144 tokens 0%. Every example fits the 8,192 training context untruncated.

Totals:

| | train | validation | test |
|---|---|---|---|
| SFT tokens | 23.10 M | 0.65 M | 0.54 M |
| CPT tokens | 19.51 M | 0.55 M | 0.46 M |

For comparison, MIDI-LLM one-stage v3 needs a median of 3,969 tokens per song (condition + target, 10,240 context). The ABC form is about 45% shorter while also carrying the lyric text itself.

## 3. Round trips

| check | result |
|---|---|
| ABC written for every converted song parses back to an identical canonical song (build) | 11,415 / 11,419 (4 key-less songs excluded) |
| MIDI melody track read back equals canonical onsets, durations, pitches (build) | **11,419 / 11,419** |
| Re-parse of every written example (validator): parse success / strict (zero parser errors) / canonical equal | **10,741 / 10,741 / 10,741** |
| Prompt rebuilt byte-identically from the stored spec | 10,741 / 10,741 |
| Syllables listed in the prompt == syllables sung in the ABC, in order | 10,741 / 10,741 |
| Chord symbols that fail the chord grammar | 0 of 876,828 |
| Malformed examples | 0 |
| Deterministic rebuild (independent second build, sha256 of all 11 output files) | **identical** (§7) |

## 4. Content coverage (all kept songs)

| measure | value |
|---|---|
| source syllables kept in lyrics (sung or `~`-joined) | **99.58%** (0.42% dropped: line had no aligned note at all) |
| syllables joined onto a neighbour note with `~` | 3.24% |
| notes that start a syllable / melisma continuation / wordless | 78.45% / 18.11% / **3.44%** |
| syllables carried by ≥ 2 notes (melismatic) | 16.2% |
| notes carrying ≥ 2 syllables (`~`) | 2.07% |
| notes per sung syllable | 1.233 (MIDI-LLM corpus reference: 1.182) |
| song time covered by a real chord (not N.C.) | 97.97% |
| melody notes with a chord sounding | 98.51% (MIDI-LLM handoff: 98%) |
| bars that differ from the nominal meter after regularization | **5.12%** (raw downbeats: 24.2%) |
| source notes dropped (same-onset duplicates, after the end) | 0.80% |
| notes truncated by the following note | 1.6% |
| melody onsets more than ¼ tick off the beat grid | **0** |
| sections by bar share | chorus 41.2%, verse 35.8%, instrumental 9.3%, intro 8.2%, outro 4.0%, bridge 1.2%, other 0.2% |
| language: CJK syllables / latin syllables | 99.35% / 0.65% |
| songs containing any latin (code-switched) syllable | 11.1% |
| meter | 4/4 ≈ 94.5%, 3/4 ≈ 5.5% (corpus manifest) |

Irregular bars went from 24.2% to 5.1%. About 20k runs had a spurious mid-bar downbeat (for example 2 + 2 beats) and were re-barred exactly. 22.8k runs were packed into whole bars plus a shorter final bar. Beat positions never move; only barline placement inside a run changes (ABC_SCHEMA.md §6).

## 5. Manual inspection (21 songs)

Songs were dumped with `scripts/show_dataset_examples.py`: 16 train, every 640th song, and 8 test, every 28th song. 13 train and 8 test were read in full against their prompts; the files are `data/generated/.../inspect_{train,test}.md`.

**Conversion correctness (no problems found):**

* Every `w:` line lines up with its bar's attack notes. The tie across a line break (`F8-` → `F3/`) correctly consumes no syllable.
* Chord symbols sit on the right beat. A chord change inside a sustained note splits it with a tie (`E- "Ebm"E`).
* Key signatures and accidentals are consistent. For example, `=G` appears in an A-major song whose melody is really in E minor.
* English words (`hey man`, `msn`, `oo _ oops`, `wow`) survive as single syllables.
* Repeated choruses produce visibly repeated ABC, which is correct. It is also what CPT will learn.

**Data-quality problems inherited from the pseudo-labels (not conversion bugs):**

1. **Section boundaries cut lyric lines.** Examples: `P:chorus | 19 bars` begins with the lone line `欢`; `P:instrumental | 28 bars` contains `回头`; `天` lands in an instrumental. All-In-One boundaries fall one syllable early, so a section label and its lyrics disagree at edges. This appeared in at least 7 of the 21 songs read.
2. **Double-time beat tracking.** Songs at 231 and 250 BPM have 4-beat "bars" that are really half-bars, giving 28–52 bar sections. Their tempo and bar counts are consistently doubled, so they are internally coherent but musically mislabeled.
3. **Aligner melisma noise.** A few songs attach 3–5 syllables with `~` to one note while the next notes all carry `_` (`又~是~一~个~秋 _ 季 _ _ _`). The note↔syllable DP matcher fails on fast or soft passages. It appeared in 2 of 21 songs.
4. **Key or tuning disagreement.** In one test song the declared key is `Ab major` while chords and melody alternate between A and Ab spellings. This is consistent with a recording about a quarter-tone off standard tuning, where both the key ensemble and GAME pitch rounding vacillate.
5. **Held-out song lyric overlap with train.** Already handled by the build filter (§1).

None of these is fixable inside the ABC conversion without new upstream analysis. They cap how clean the lyric alignment a model can learn will be, and they are reported as a data bottleneck in REPORT.md.

## 6. Conversion statistics (all 11,419 converted songs)

| counter | value |
|---|---|
| notes | 3,321,271 |
| bars | 1,038,747 |
| chords (after merging equal neighbours) | 929,377 |
| melisma notes recovered by the MIDI-LLM window rule | 214,249 |
| continuation-flagged notes promoted to attack (first note of their syllable) | 59,300 |
| repeated-attack links turned into melisma | 51,707 |
| non-contiguous repeats made wordless | 263 |
| unmatched syllables joined / dropped | 87,742 / 11,618 |
| section starts not on a downbeat (barline inserted) | 1,548 |
| chord qualities not in the mapping table | 0 |

## 7. Determinism

The same build command was run again on a different node into a separate directory (Slurm job 40171068). The sha256 of all 11 output files (`songs_*`, `sft_*`, `cpt_*`, `split_manifest.json`, `exclusions.jsonl`) is **identical** to the original build. The duplicate directory was then removed.
