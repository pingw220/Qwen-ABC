# Qwen-ABC POC: lyrics-aware ABC lead sheets with Qwen3.5-0.8B

## Verdict

* **Direct SFT is already sufficient** for syntax, lyric following and conditional control at this scale.
  * Qwen3.5-0.8B-Base fine-tuned on about 10k Mandarin pop songs writes whole-song ABC lead sheets that always parse (225/225 held-out songs).
  * 60–67% are strictly valid across two training seeds, and 99.5% of bars have the right length.
  * The model sings 93–96% of the requested lyrics in order, with ≥98% precision.
  * It follows key and tempo 100% of the time, and changes melody when the lyrics change.
  * No MIDI-specific vocabulary was needed.
* **CPT did not help; in this setup it hurt adherence to the prompt.** The CPT used ABC-only continued pretraining on the same training songs. The result replicates across two SFT training seeds.
  * It lowered teacher-forced loss slightly in both seeds: test 0.4809 / 0.4802 vs 0.4837 / 0.4833 nats/token.
  * But it made generations stop early (45 / 62 vs 9 / 13 songs ended before the requested structure).
  * It cut lyric recall from 0.956 / 0.931 to 0.885 / 0.832.
  * It cut exact section plans from 0.55 / 0.63 to 0.32 / 0.34. Every paired 95% CI excludes 0, and the gap exceeds seed-to-seed variation.
  * A token-matched direct-SFT control shows the difference is not about seeing the data more often.
  * Strict ABC validity does **not** differ reliably: 0.67 / 0.60 vs 0.60 / 0.60.
* **The ABC representation works, but structure control is its weak point.** Only 55% of songs reproduce every section's bar count. Adding bar or section counters to the target is the cheapest next fix.
* **Data quality is the main bottleneck for musical quality, not the model.**
  * The corpus is pseudo-labelled.
  * Section boundaries cut lyric lines in about a third of inspected songs.
  * 54–61% of the inherited held-out songs were duplicates of training songs and had to be removed.
* **Scaling to 2B/4B is plausible but not yet justified by these results.** Validation loss plateaus at about 0.50 by 1.8 epochs regardless of extra epochs, and melodies are narrower and more repetitive than the reference. That could be capacity or label noise; one 2B run would tell them apart (§20).

All numbers below are on the same 225 held-out test songs, one sample per song (T=0.8, top-p 0.95, seed 1000), unless stated otherwise.

---

## 1. What data was found

See `reports/DATA_AUDIT.md`.

* **The only corpus with everything needed** is SheetSage-Pro `sheetsage_zh_training_views_v2_lvcr`: 11,673 Mandarin pop songs with melody (GAME vocal), chords (ISMIR2019-LVCR), beats and downbeats (beat_this), sections (All-In-One + SongPrep), key (ensemble), tempo, meter, lyrics, and Phonsa note↔syllable alignment. All labels are automatic.
* **Other sources lack chords and sections, or are tiny:**
  * share_singing: 191 songs of human lyric MIDI;
  * GTSinger: 28.6k singing segments, 9 languages;
  * MIR-ST500: melody only;
  * JamendoLyrics: lyric timing only.
* **English:** there is no English lead-sheet corpus.

## 2. Selected dataset

* **Corpus:** D1 above, restricted to the song split MIDI-LLM already uses (`phoneme_leadsheet_section_v4`: 10,247 / 591 / 581 songs).
* **Leakage filter:** held-out songs whose lyrics or melody near-duplicate a training song were removed before any example was built.
* **Final split:** 10,243 train, 273 validation and **225 test** songs.
* **Output:** `data/generated/abc_v1_20260915_012459` (built by job 40170859; a second build was byte-identical).
* **Validation:** `reports/DATASET_VALIDATION.md`. Every example parses back to its canonical form, the MIDI melody round trip is 11,419/11,419, and prompt lyrics exactly equal the ABC lyrics for all 10,741 examples.

## 3. ABC representation

See `reports/ABC_SCHEMA.md`.

* **Base:** standard ABC with `L:1/8` on a sixteenth-note tick grid.
* **Sections and bars:** sections as `P:label`. Barlines follow the corpus downbeats after removing spurious ones (irregular bars 24% → 5%), and remaining irregular bars get inline `[M:k/4]`.
* **Chords:** quoted symbols on the note or rest where they start. A note that crosses a chord change or barline is split with a tie.
* **Lyrics:** a bar-synchronized `w:` line under every sung music line, one token per note attack:
  * a character or word starts a syllable;
  * `_` is a melisma;
  * `*` is a wordless note;
  * `~` puts several syllables on one note;
  * `|` re-synchronizes at the next bar.
* **Tied continuations** consume no lyric token.
* **Round trip:** exact for everything the canonical model holds. The lossy parts are listed in the schema, e.g. absolute seconds and tempo drift.
* **Length:** the median song is 1,857 ABC tokens plus a 342-token prompt, 45% shorter than MIDI-LLM's median 3,969-token stream. 99.8% fit in 4,096 tokens.

Example (test song `1F3t9l7U8CnKmzVyFbR1aG`, never seen in training). Prompt, abridged:

```text
Tempo: 71 BPM
Key: C major
Structure:
P:intro | 4 bars
P:verse | 6 bars
离别曲含着泪唱起
声声都是难舍
字
P:chorus | 6 bars
字都是意义
舍不得这里的每个人
```

Direct-SFT generation:

```abc
X:1
M:4/4
L:1/8
Q:1/4=71
K:C
P:intro
"C"z8 | "F"z8 | "C"z8 | z8 |
P:verse
G,4 G, G, A,3/ G,/ | "G"D,4 z2 G, A, | "F"A, C3 C/ D/ C A,/ A,3/ | "C"C6 z2 |
w: 离 别 _ _ _ | 曲 含 着 | _ 泪 唱 _ _ _ _ | 起
"Am"z3/ A,5/ A,3/ C3/ A, | "C"G,3 G, G,2 z E, |
w: 声 声 都 是 | 难 _ 舍 字
P:chorus
"F"D,3/ F,3/ F, F,2 G, A, | "G"G,3/ G,9/ z2 | "C"C C C C C D E2 | "Em"E13/ z3/ |
w: 字 都 是 意 _ _ | 义 _ | 舍 不 得 这 里 的 每~个 | 人
```

Eight songs with input, reference and three Qwen models are in `experiments/samples_20260915_023500/<song>/`. Each has `input.txt`, `reference.abc/.mid/.wav`, and a directory per model with `generated.abc/.mid/.wav` and `metrics.json`. The MIDI-LLM baseline directories have MIDI and audio only.

## 4. Qwen3.5-0.8B setup

* **Model:** `Qwen/Qwen3.5-0.8B-Base` (HF sha `dc7cdfe2`) loaded as `Qwen3_5ForCausalLM`, text tower only, 752M parameters.
  * 24 layers: 18 Gated-DeltaNet linear attention and 6 gated full attention; hidden size 1024; tied embeddings; 248k vocab; bf16 checkpoint.
  * I verified the id and architecture from the Hub config and the weight index before training.
  * **Deviation to note:** I used the *Base* checkpoint rather than the post-trained `Qwen/Qwen3.5-0.8B` (a multimodal chat model). Continued pretraining and plain-text SFT are cleaner on a base model. The two share architecture and tokenizer, so switching is a one-line config change.
* **No new tokens:** Qwen's tokenizer covers ABC and every CJK character in the data.
* **Full-parameter finetuning, not LoRA:**
  * 0.8B fits on one 48 GB GPU with fp32 master weights, bf16 autocast and gradient checkpointing: peak 16 GB at 16k tokens per micro-batch.
  * CPT through LoRA would have been a weaker test of CPT.
* **Attention:** SDPA for the full-attention layers (flash-attn not installed). Linear-attention layers use flash-linear-attention 0.5.2 Triton kernels; causal-conv1d uses the PyTorch fallback because the prebuilt wheel needs a newer glibc.
* **No packing:** packed documents would share the linear-attention recurrent state. Length-bucketed right padding instead. Checked with `scripts/check_padding_invariance.py`: right padding changes a sequence's loss by at most 0.14% relative (bf16 kernel noise), job 40174165.
* **Loss:** summed token NLL divided by supervised tokens per optimizer step, computed only at supervised positions in 512-token chunks (the 248k vocab would otherwise need about 16 GB of logits).
  * SFT supervises completion tokens and EOS.
  * CPT supervises every token.
* **Optimizer:** AdamW (β2 0.95, no weight decay), learning rate 5e-5 with 3% warmup and cosine decay to 10%, grad clip 1.0.
  * About 131k padded tokens per step: 8 × 16k-token micro-batches.
  * Seed 1234.
* **Code:** `qwen_abc/train.py`, configs in `configs/`, the resolved config of every run in `experiments/<run>/resolved_config.json`.

## 5. Direct SFT (Experiment A)

* **Run:** `experiments/direct_sft_20260915_023500`, config `configs/sft_direct.yaml`, 2 epochs = 382 steps on 10,243 songs (23.1M tokens per epoch, 19.5M supervised).
* **Sanity progression before the full run:**
  1. **Overfit check** (64 songs, 30 epochs, `experiments/sanity_overfit64_*`): eval loss 1.31 → 0.0001. Greedy generation reproduced 8/8 memorized songs exactly (strict-valid, lyric recall 1.0, melody distance 0).
  2. **1k-song pilot** (`experiments/pilot_direct_sft_1k_*`, 38 steps): validation loss 1.30 → 0.72. 24/24 validation generations parsed, but none was strict, bar plans never matched, and some songs degenerated. Harmless as a pilot, and it confirmed that data scale matters.
  3. **Full run:** validation loss 1.302 → 0.640 (50 steps) → 0.570 (100) → 0.524 (200) → 0.507 (300) → **0.5015** (382). No preemptions.

## 6. CPT

* **Run:** `experiments/abc_cpt_20260915_023500`, config `configs/cpt_abc.yaml`.
* **Data:** ABC text only, from the same 10,243 training songs (no other ABC corpus exists), with loss on every token. 1 epoch = 159 steps, 19.5M tokens.
* **CPT validation loss** (ABC-only): 1.628 → 0.922 → 0.849 → **0.824**.
* **Teacher-forced SFT test loss, same prompt format:**
  * base model: 1.276;
  * CPT checkpoint: 0.631, even though it never saw a prompt.
* **Where CPT helped the SFT-format loss:** bars 0.80 → 0.10, ties 2.64 → 0.91, chords 1.55 → 0.69.
* **Where it did not:** copying lyric text from the prompt (0.521 → 0.505). SFT brings that to 0.013.

## 7. CPT → SFT (Experiment B)

* **Run:** `experiments/cpt_then_sft_20260915_023500`.
* **Recipe:** identical to A (config, data, seed, 382 steps), initialized from the CPT final checkpoint.
* **Validation loss:** 0.652 at step 0 (A started at 1.302), then 0.572 (50) → 0.545 (100) → 0.512 (200) → 0.501 (300) → **0.4982** (382).

**Token-matched control (C):** `experiments/control_direct_sft_tokmatched_20260915_023500`.

* Direct SFT for 2.85 epochs (545 steps), which sees as many training tokens as CPT + SFT (19.5M + 2×23.1M ≈ 2.85 × 23.1M).
* Validation loss plateaus at 0.503 by step 350 and stays there (0.5058 at 400, 0.5034 at 545).
* It separates "CPT as a stage" from "more passes over the same data".

## 8. GPU resources

* **Hardware:** every training and generation job ran on 1× **NVIDIA L40S** (46 GB) in the `ckpt-all` checkpoint partition, account `ckpt-ark`, `--requeue`. The trainer writes atomic resume snapshots every 50 steps; none was needed.
* **Why L40S on ckpt, not the ark `gpu-l40s` allocation:** at submission the ark L40S pool had 1 GPU free while a lab member had pending jobs there, and ark `gpu-l40` had 2 free. Taking the lab's last L40S would have competed with them. `ckpt-all` had 5 idle L40S nodes and does not consume the ark allocation.
* **No interference:** the user's running job 40168248 (`midillm-onestage-prod`, gpu-l40) and interactive job were never touched.
* **Hardware faults:** node g3121 threw `CUDA error: uncorrectable ECC error` three times, killing one probe job, one generation shard and one loss job. All three were resubmitted with `--exclude=g3121`. Results are unaffected (failed jobs wrote nothing), but the node should be reported to Hyak.
* **Dataset build:** 316 s on 30 CPU cores (ckpt-all).
* **Throughput:** about 5.2–6.3k training tokens/s per L40S at 16k tokens per micro-batch; generation about 50 tokens/s at batch 1.

## 9. Training duration

| run | steps | wall-clock (1× L40S) |
|---|---|---|
| overfit-64 sanity | 150 | 0.22 h |
| 1k pilot | 38 | 0.24 h |
| A: direct SFT | 382 | 2.05 h |
| CPT | 159 | 0.90 h |
| B: CPT → SFT | 382 | 2.05 h (2.95 h including CPT) |
| C: token-matched control | 545 | 2.93 h |
| A / B second seed (2345) | 382 each | 2.05 h / 2.06 h |

Evaluation cost about 4 GPU-hours per model: 4 shards × about 45 min, plus probes.

## 10. Dataset sizes

| | songs | SFT examples | CPT examples |
|---|---|---|---|
| train | 10,243 | 10,243 | 10,243 |
| validation | 273 | 273 | 273 |
| test | 225 | 225 | 225 |

## 11. Token counts

| | train | validation | test |
|---|---|---|---|
| SFT tokens (prompt + ABC + EOS) | 23.10 M | 0.65 M | 0.54 M |
| CPT tokens | 19.51 M | 0.55 M | 0.46 M |
| median SFT example / p99 / max | 2,202 / 3,679 / 5,833 | 2,435 / 3,763 / 3,960 | 2,346 / 3,649 / 3,790 |

## 12. Validation losses

Teacher-forced, SFT format, nats/token.

| run | validation (all 273) | test (all 225) | test ppl |
|---|---|---|---|
| base model (no training) | 1.3000 | 1.2761 | 3.58 |
| CPT checkpoint (no SFT) | 0.6503 | 0.6313 | 1.88 |
| **A: direct SFT** | 0.5002 | 0.4837 | 1.622 |
| **B: CPT → SFT** | 0.4969 | **0.4809** | 1.617 |
| C: token-matched control | 0.5021 | 0.4848 | 1.624 |

By category (test loss, nats/token):

| run | pitch | duration | lyric text | lyric `_` `*` `~` | lyric `\|` | chord | bar | rest | tie | structure |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.976 | 0.866 | 0.013 | 0.487 | 0.075 | 0.598 | 0.095 | 0.640 | 0.782 | 0.003 |
| B | 0.969 | 0.870 | 0.014 | 0.462 | 0.074 | 0.594 | 0.090 | 0.623 | 0.808 | 0.005 |
| C | 0.983 | 0.859 | 0.014 | 0.465 | 0.076 | 0.603 | 0.091 | 0.616 | 0.821 | 0.003 |

After SFT, what is left is almost entirely musical content: pitch, duration, rhythm and chord choice. Structure and lyric text cost almost nothing under teacher forcing. B's 0.003-nat advantage is spread thinly, at most 0.025 nats in any category (lyric markers, rests, pitch), and ties are slightly worse.

## 13. Generation examples

See §3 and `experiments/samples_20260915_023500/README.md`. Qualitatively, the direct-SFT output:

* keeps the requested sections, key and tempo;
* sings the lyrics in order, one syllable per note with short melismas (`起 _ _ _ _` is unusual but occurs);
* repeats phrase-level material across choruses;
* uses simple diatonic harmony (C, F, G, Am, Em).

Compared with the reference, its melodies are flatter: narrower range, more repeated notes, fewer sixteenth off-beats.

## 14. ABC validity

| run | parse | strict valid | MIDI success | stopped with EOS | bars with correct duration |
|---|---|---|---|---|---|
| A | 225/225 | 151/225 (67.1%) | 225/225 | 220/225 | 99.5% |
| B | 225/225 | 134/225 (59.6%) | 225/225 | 223/225 | 99.5% |
| C | 225/225 | 159/225 (70.7%) | 225/225 | 224/225 | 99.2% |

*Strict* means the parser recorded zero errors of any kind.

* **Most common error: lyric alignment.**
  * `orphan_melisma` is a `_` where the previous note had no syllable.
  * `lyric_overflow` is more tokens than notes in a bar. It is concentrated in a few degenerate songs; A has one song with over 1,400 errors.
  * Bar-duration errors are rare: 137 bars over 225 songs for A.
* **B (seed 1234)** often writes several `w:` lines under one music line (`lyrics_without_music_line`: 327, against 1 for A, 8 for C and 26 for B seed 2345). The training data never does this.

## 15. Lyric alignment quality

| metric | reference | A | B | C |
|---|---|---|---|---|
| lyric recall: requested syllables sung, in order | 1.000 | **0.956** | 0.885 | 0.956 |
| lyric precision: sung syllables that were requested | 1.000 | 0.983 | 0.986 | 0.992 |
| lyrics exactly as requested | 1.000 | 0.271 | 0.209 | 0.284 |
| recall inside the requested section | 1.000 | 0.953 | 0.870 | 0.952 |
| no alignment error in the song | 1.000 | 0.724 | 0.680 | 0.791 |
| notes per syllable | 1.231 | 1.285 | 1.454 | 1.280 |
| melisma notes | 0.161 | 0.203 | 0.234 | 0.198 |
| wordless notes | 0.036 | 0.036 | 0.049 | 0.041 |
| notes carrying several syllables (`~`) | 0.023 | 0.035 | 0.037 | 0.040 |

* **Word coverage:** the model almost never invents or duplicates words (precision ≥ 0.98) and misses about 4%.
* **Cramming:** when it runs out of notes it crams the rest of a line onto one note with `~`, e.g. `是 在~谁~的~怀~里~学~会~做~梦` in test song 05Qm. That is 1.5× the reference rate, and it inflates recall slightly.
* **Melismas:** they are placed plausibly but slightly too often.

## 16. Musical sanity metrics

Per-song means on 225 songs. The reference column is the corpus ABC of the same songs.

| metric | reference | A | B | C |
|---|---|---|---|---|
| notes per bar | 3.47 | 3.45 | 3.46 | 3.45 |
| pitch range (semitones) | 22.0 | **14.0** | 13.5 | 14.7 |
| mean \|interval\| (semitones) | 2.39 | 1.69 | 1.70 | 1.73 |
| leaps > 7 semitones | 0.044 | 0.016 | 0.019 | 0.019 |
| mean note duration (beats) | 0.82 | 0.86 | 0.82 | 0.86 |
| sixteenth off-beat onsets | 0.193 | 0.114 | 0.098 | 0.109 |
| chords per bar | 0.98 | 0.86 | 0.83 | 0.82 |
| time covered by a chord | 0.974 | 0.962 | 0.963 | 0.955 |
| melody notes with a chord | 0.981 | 0.974 | 0.977 | 0.970 |
| chord-tone agreement (duration-weighted) | 0.677 | 0.754 | 0.732 | 0.716 |
| valid chord symbols | 1.000 | 1.000 | 1.000 | 1.000 |
| distinct non-empty bars | 0.872 | **0.455** | 0.458 | 0.528 |
| max identical consecutive bars | 1.04 | 2.42 | 2.41 | 1.73 |

Corpus-level Jensen-Shannon divergence, generated vs reference:

| run | pitch class | scale degree | interval | onset position | duration |
|---|---|---|---|---|---|
| A | 0.003 | 0.026 | 0.041 | 0.011 | 0.017 |
| B | 0.002 | 0.020 | 0.034 | 0.016 | 0.022 |
| C | 0.007 | 0.020 | 0.034 | 0.010 | 0.013 |

* **Distributionally**, all three models are close to the corpus.
* **Per song they are conservative:**
  * melodic range is two thirds of the reference, with small steps;
  * syncopation is about half the reference;
  * about half of all bars exactly repeat another bar in the same song, where the reference has 13%;
  * chord-tone agreement is *above* the pseudo-label reference.
* **Degeneration** (one bar repeated many times) is rare but present: max identical consecutive bars averages 2.4, driven by a few songs.

### Conditional sensitivity

**Sampled decoding (T=0.8), 20 songs.** Key and tempo are always written as requested, but melody-distance numbers are uninformative here: re-seeding alone changes the melody almost completely (distance 0.93 for A).

**Greedy decoding, 20 songs.** The output is deterministic, so any change is caused by the condition.

| run | swapped-in lyrics sung (recall) | original lyrics still sung | melody distance after lyric swap (0 = same) | pitch-contour distance | tritone key change: notes in new scale | still in old scale | before change: in new / old scale |
|---|---|---|---|---|---|---|---|
| A | 0.854 | 0.100 | 0.778 | 0.524 | 1.000 | 0.030 | 0.000 / 1.000 |
| B | 0.650 | 0.084 | 0.861 | 0.589 | 1.000 | 0.075 | 0.062 / 1.000 |
| C | 0.802 | 0.092 | 0.923 | 0.428 | 1.000 | 0.060 | 0.033 / 1.000 |

**Lyrics are not ignored.**

* Swapping in another song's lyrics under the same structure makes the model sing the new text (recall 0.85). The original text all but disappears (0.10, equal to the 0.107 in-order overlap between the two unrelated lyric sets).
* The melody is rewritten (78% of note tokens change).

**The key is obeyed completely:**

* a tritone shift moves every note into the new scale;
* a +25% tempo change is written into `Q:` 100% of the time;
* density per bar is barely affected (notes-per-bar ratio 1.08 for A).

## 17. Direct SFT vs CPT → SFT

Paired bootstrap over the 225 test songs; `*` means the 95% interval excludes 0. Full tables are in `reports/results_main_225.md`.

| metric | A | B | B − A [95% CI] | C − A [95% CI] |
|---|---|---|---|---|
| test loss | 0.4837 | 0.4809 | −0.0028 | +0.0011 |
| strict valid | 0.671 | 0.596 | −0.076 [−0.160, +0.013] | +0.036 [−0.044, +0.120] |
| lyric recall | 0.956 | 0.885 | **−0.071 [−0.099, −0.045] \*** | +0.000 [−0.019, +0.018] |
| section labels + bars exactly as requested | 0.551 | 0.324 | **−0.227 [−0.311, −0.147] \*** | +0.058 [−0.018, +0.133] |
| section label sequence as requested | 0.951 | 0.751 | **−0.200 [−0.262, −0.138] \*** | +0.004 [−0.031, +0.044] |
| total bar count as requested | 0.564 | 0.356 | **−0.209 [−0.293, −0.124] \*** | +0.049 [−0.027, +0.124] |
| bar lengths as requested (per bar) | 0.959 | 0.906 | **−0.053 [−0.078, −0.031] \*** | +0.010 [−0.006, +0.027] |
| melisma fraction | 0.203 | 0.234 | +0.031 \* | −0.006 |
| chord-tone agreement | 0.754 | 0.732 | −0.019 [−0.042, +0.003] | −0.036 \* |
| distinct bars | 0.455 | 0.458 | +0.003 | +0.073 \* |

**Mechanism: B stops early.**

* 45/225 of B's songs end with fewer sections than requested, against 9 for A and 7 for C.
* For 10% of B's songs the generated song is ≤ 72% of the requested length; the median is exactly the requested length for all three.
* B's greedy lyric-swap recall is also lower (0.65 vs 0.85; seed 1234).
* Under teacher forcing B is marginally better; free-running, it relies less on the prompt.

A plausible explanation, not proven: 1 epoch of *unconditional* whole-song modelling on the same songs teaches a strong length and ending prior ("songs end after a typical number of choruses"), and 2 SFT epochs do not fully override it. The same pattern — teacher-forced loss improving while free-running adherence gets worse — appeared earlier in the SheetSage2 lyric-alignment work. **Teacher-forced loss is not a reliable model-selection criterion for this task.**

**What the control rules out.** C sees as many tokens as B but without a separate unconditional stage. C matches A on every structure and lyric metric. Its validation loss plateaus at 0.503, so extra epochs neither help nor hurt adherence, and C writes slightly less repetitive music. The B deficit is therefore specific to the CPT stage, not to extra data exposure.

**Replicate: second SFT seed.** Both SFT arms were re-run with seed 2345. The seed changes the batch order; B reuses the same CPT checkpoint.

* Runs: `experiments/direct_sft_seed2345_20260915_023500` and `experiments/cpt_then_sft_seed2345_20260915_023500`, jobs 40173062 / 40173063.
* Full table: `reports/results_seed_replicate_225.md`.

| metric | A seed 1234 | A seed 2345 | B seed 1234 | B seed 2345 |
|---|---|---|---|---|
| test loss | 0.4837 | 0.4833 | 0.4809 | 0.4802 |
| strict valid | 0.671 | 0.596 | 0.596 | 0.604 |
| lyric recall | 0.956 | 0.931 | 0.885 | 0.832 |
| recall inside requested section | 0.953 | 0.909 | 0.870 | 0.786 |
| section labels + bars as requested | 0.551 | 0.627 | 0.324 | 0.342 |
| section label sequence as requested | 0.951 | 0.889 | 0.751 | 0.618 |
| bar lengths as requested (per bar) | 0.959 | 0.950 | 0.906 | 0.866 |
| songs ending with fewer sections than requested | 9 | 13 | 45 | 62 |
| chord-tone agreement | 0.754 | 0.723 | 0.732 | 0.715 |

The CPT deficit on lyric recall and structure appears in both seeds and is larger than the difference between two direct-SFT seeds.

The strict-validity gap does not replicate: A's seed 2345 run matches B. Seed variance on that metric alone (±0.07) is as large as the first-seed difference.

Two seeds is still a small sample for training variance. The early-termination mechanism, though, is consistent: 45 and 62 early endings for B against 9 and 13 for A.

## 18. Failure cases

1. **Early termination (mostly B).** Generation emits EOS after, say, 6 of 11 requested sections (test song 09RY5jQZ5kNoMSU5toQlfU), dropping the remaining lyrics.
2. **Bar-count drift.** Section labels are right (95% for A) but a section runs 1–2 bars long or short (≈ 40% of songs have at least one such section). The model has no explicit counter; it must count bars over hundreds of tokens.
3. **Lyric cramming.** Leftover syllables go onto one note via `~` (3.5% of notes vs 2.3% in the reference), usually at the end of a section that ran out of bars.
4. **Lyric/music line misalignment.** `_` after a wordless note, or more `w:` tokens than notes in a bar. Rare per song but catastrophic in a few: one A song has 1,425 orphan melismas when the music line degenerated into repeated bars.
5. **Conservative melodies.** Range 14 vs 22 semitones, half the syncopation, 2× more exact bar repetition than the reference.
6. **Inherited label noise** that the model learns faithfully:
   * sections that start one syllable early;
   * double-time songs (231–250 BPM) with 28–52-bar sections;
   * songs a quarter-tone off standard tuning with vacillating keys.

## 19. Recommendation for the next experiment

1. **Keep direct SFT as the recipe and drop same-data CPT.** CPT is only worth revisiting with *more* ABC than the SFT set (e.g. public ABC or MusicXML lead-sheet corpora converted to this schema) and with the SFT prompt format mixed into CPT so the ending prior stays conditional.
2. **Fix structure following in the representation, not the model.**
   * Emit a bar counter per music line (e.g. `%8` or `[I:bar 5]`) or repeat `P:verse (8 bars)` in the target.
   * Optionally emit per-section syllable counts in the prompt.
   * Re-measure section-plan exactness, currently 55%.
3. **Clean the lyric/section boundary noise upstream.** Snap section starts to the lyric line that begins after them, and gate songs with double-time tempo estimates. Both are cheap and affect about a third of songs.
4. **Select checkpoints on generation metrics** (lyric recall, section-plan exactness, strict validity) on the 273 validation songs, not on loss.
5. **Compare against MIDI-LLM on a larger overlapping set** with its constraint decoding switched off, so the comparison measures the models rather than the decoders (§21).

## 20. Is scaling to ~2B / 4B justified?

Not yet as the next step; worth one run after item 2 above. Evidence either way:

* **For capacity limits:**
  * validation loss plateaus at 0.50 and extra epochs do not move it (control run);
  * the residual loss is almost entirely musical content (pitch 0.98, duration 0.87 nats);
  * generations are more repetitive and narrower than the data.
* **Against:**
  * all conditioning behaviours (lyrics, key, tempo, section labels) are already learned at 0.8B;
  * the dominant errors are structural counting and inherited label noise, which a larger model would learn *more* faithfully rather than fix;
  * the data is pseudo-labelled, so part of the 0.50 floor is irreducible noise.

A 2B run costs about 2.5× A (≈ 5 GPU-hours on one L40S) and would separate the two explanations cheaply. Run it only after the representation fix, otherwise it re-measures the counting problem.

## 21. Comparison with MIDI-LLM (same test songs, where practical)

MIDI-LLM one-stage v4-cd:

* checkpoint `runs/original-midi-llm-onestage-multitask-v1-v4-cd/full/final`;
* its repository at `bea5597` with 9 uncommitted files (its current working tree, including the onset pile-up fix);
* its own CLI and recommended v4 decode constraints;
* spec task `lyrics_to_leadsheet`, i.e. no genre tags — the same condition as Qwen-ABC except per-bar irregular beats.

It ran on 30 test songs (every 7th). Its held-out split is the same v4 split, so it never trained on them either.

* 5/30 could not be generated: MIDI-LLM refuses code-switched English (4 songs) and the character 嗯 (1 song), both at input.
* Outputs were converted to the canonical model and scored with the identical metric code (`scripts/midi_llm_baseline.py`).
* Tables: `reports/results_vs_midi_llm_30.md`.

| metric (30 songs; unsupported songs count as failures for rates) | MIDI-LLM v4-cd | Qwen A | Qwen B |
|---|---|---|---|
| produced a lead sheet | 25/30 | 30/30 | 30/30 |
| passed its own validator / strict ABC | 2/30 (5 not generated; all 23 other failures are `melody_onset_pileup`) | 22/30 | 19/30 |
| lyric recall (generated songs) | 0.694 over 30 | 0.962 | 0.878 |
| lyric precision | 0.659 | 0.997 | 0.992 |
| section labels + bars as requested | **0.833** (enforced by its decoder) | 0.567 | 0.367 |
| notes per bar (reference ≈ 3.5) | 4.42 | 3.59 | 3.29 |
| sixteenth off-beat onsets (reference 0.19) | 0.40 | 0.12 | 0.11 |
| pitch range (reference 22) | 30.3 | 15.4 | 13.2 |
| chords per bar (reference 1.0) | 2.11 | 1.00 | 0.92 |
| chord-tone agreement (reference 0.68) | 0.52 | 0.75 | 0.74 |
| distinct bars (reference 0.87) | **0.99** | 0.45 | 0.49 |

Caveats:

* **Structure is not a fair comparison:** MIDI-LLM's decoder pins section labels, bar counts and lyric-line assignment, while Qwen-ABC is unconstrained.
* **Strict validity uses two different validators.**
* **Lyric metrics depend on converting MIDI-LLM's pointer stream,** where re-attacks of the same syllable count as extra syllables.
* **Musical balance:**
  * MIDI-LLM is busier than the data (2× chords, 2× syncopation, wider range) and never repeats a bar exactly;
  * Qwen-ABC is closer to the reference density and harmony but under-varied.
  * Neither is clearly better musically on these numbers; listening is needed.

---

## Reproduction

| step | command / location |
|---|---|
| environment | `README.md` |
| tests | `python -m pytest -q tests` (31 passed) |
| dataset | job 40170859; `data/generated/abc_v1_20260915_012459` |
| overfit sanity | `experiments/sanity_overfit64_20260915_015134` |
| pilot | `experiments/pilot_direct_sft_1k_20260915_020620` |
| A / CPT / B / C | jobs 40171255 / 40171256 / 40171257 / 40171258, `experiments/*_20260915_023500` |
| evaluation | `scripts/slurm/submit_eval.sh`, `scripts/eval_loss.py`, greedy probes via `generate_eval.py --greedy-probes-only`, tables via `scripts/compare_results.py` |
| MIDI-LLM baseline | `experiments/midi_llm_onestage_v4cd_baseline_20260915_024503` (`specs/`, `gen/`, `eval_test/`) |
| samples | `scripts/make_samples.py` → `experiments/samples_20260915_023500` |

Git keeps small logs, configs and summaries. Weights, generation dumps (which contain lyrics) and audio are git-ignored and stay on scrubbed storage.
