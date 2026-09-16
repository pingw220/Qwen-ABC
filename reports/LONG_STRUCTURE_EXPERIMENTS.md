# Long-structure and scaling experiments (round 2)

All numbers are on the same **225 de-duplicated held-out songs**, one sample per song, unless stated
otherwise. "T0.8" = temperature 0.8 / top-p 0.95 (round-1 default), "T1.0" = temperature 1.0 / top-p 0.95.
Paired bootstrap CIs over songs; `*` marks an interval excluding 0.

## Verdict

1. **Representation was the main structural problem, and it is now largely fixed.** Writing the
   structural state into the text — a section header plus a per-bar countdown (ABC-v2) — raises
   "every section has exactly the requested label and bar count" from **0.53 to 0.86** (0.94 on the
   second seed), and removes early termination entirely (4% → 0%). Both halves contribute: cleaning
   section boundaries alone gives 0.66, counters add the rest. Adding the long-range objective on top
   (E3b) reaches **0.98**.
2. **Musical conservatism was a decoding artifact, not capacity, not data.** The same checkpoints
   sampled at **T=1.0 / top-p 0.95** move from 13–14 to 18.4 semitones of pitch range, from 0.47 to
   0.80 distinct bars, and match the corpus on syncopation, off-beat onsets and chord-tone agreement —
   with lyric recall and structure unchanged. The same holds on a hand-checked clean subset, so it is
   not label noise.
3. **A 4× larger effective batch does not help, and at equal epochs it hurts.** What lyric alignment
   and validity need is optimizer *updates*: training E1 for 50% more updates lifts strict validity
   0.63 → 0.72 and alignment-clean 0.65 → 0.83 without touching structure.
4. **8K context was never the limit** (longest example 6,703 tokens), but a long-range *objective*
   does help: adding masked late-section reconstruction at E1's batch size (E3b) raises exact
   structure to **0.978** against 0.809 for a matched-update control without it, and gives the best
   model in this report. The same mixture at a 4× batch showed no benefit, which is why the batch
   ablation had to come first.
5. **Pseudo-label noise still accounts for a large share of the residual error**: on the clean
   subset, exact structure is 0.88–0.94 and strict validity rises by 0.1–0.2 for every model.
6. **Scaling to ~2B is not justified by the pre-registered criteria** (§15 of the plan) and was not
   run: the musical-narrowness symptom that would have justified it is explained by decoding.
7. **Versus MIDI-LLM:** Qwen-ABC is much stronger on lyrics (recall 0.97 vs 0.73) and coverage
   (225/225 vs 199/225), comparable on musical statistics once decoded at T=1.0, and its structural
   exactness is now within reach of a system whose structure is *given by construction*.

---

## 1. What was run

| ID | model | data / representation | context | tokens/update | training task | seeds |
|---|---|---|---|---|---|---|
| **E0** | Qwen3.5-0.8B-Base | ABC-v1 (round 1) | 8K | 131K | whole-song SFT | 1234, 2345 |
| **E1c** | same | ABC-v1 on **cleaned** section boundaries | 8K | 131K | whole-song SFT | 1234 |
| **E1** | same | **ABC-v2** (cleaned + section header + per-bar countdown) | 8K | 131K | whole-song SFT | 1234, 2345 |
| **E2a** | same | ABC-v2 | 8K | **524K** | whole-song SFT, same 2 epochs (128 updates) | 1234 |
| **E2b** | same | ABC-v2 | 8K | **524K** | whole-song SFT, same 506 updates as E1 (~8 epochs) | 1234 |
| **E3** | same | ABC-v2 + masked late-section reconstruction | 8K | 524K | long-range mixture, 2 epochs (190 updates) | 1234 |
| **E3b** | same | same mixture | 8K | 131K | long-range mixture at E1's batch (758 updates) — **best model** | 1234 |
| **E1-long** | same | ABC-v2 | 8K | 131K | whole-song SFT for 758 updates (3 epochs): the control for E3b | 1234 |
| E2 LR probe | same | ABC-v2 | 8K | 524K | 30 updates at LR 1e-4 vs 5e-5 | 1234 |
| E4 (~2B) | **not run** | – | – | – | – | – |

Data: `data/generated/abc_v2_20260915_120927` (ABC_V2_DATASET_VALIDATION.md); the split is the
round-1 de-duplicated split, re-verified (§9).

## 2. Was 8K context a limitation? No

`reports/LONG_CONTEXT_AUDIT.md` has the full audit. In short:

* Qwen3.5-0.8B-Base has `max_position_embeddings` 262,144; 18 of 24 layers are Gated-DeltaNet
  linear attention with no position limit.
* The longest training example is **5,833 tokens (ABC-v1) / 6,703 (ABC-v2)**; nothing was ever
  truncated (`encode()` drops over-long rows, and it dropped none).
* Raising `max_seq_len` 8192 → 16384 would produce a **bit-identical run**: `max_seq_len` only feeds
  the drop filter, while batching depends on `tokens_per_micro_batch`.
* One 32K sequence trains in 11.4 GB on an L40S, so the ceiling is not memory.

The round-1 failures (bar drift, early EOS, exact repetition) happen inside 2–4K tokens. They are
counting, termination and copying problems, not context problems.

## 3. E1: explicit structural state (the central result)

Paired against E0 on the same 225 songs, both at T0.8 (E0 evaluated on the same cleaned prompts):

| metric | reference | E0 | E1 | E1 − E0 [95% CI] |
|---|---|---|---|---|
| **exact full structure** (labels + bars per section) | 1.000 | 0.529 | **0.858** | **+0.329 [+0.258, +0.404] \*** |
| exact bars-per-section sequence | 1.000 | 0.529 | 0.871 | +0.342 [+0.267, +0.422] \* |
| sections with exact label+bars (per section) | 1.000 | 0.904 | 0.977 | +0.073 [+0.052, +0.096] \* |
| total bar count = request | 1.000 | 0.533 | 0.871 | +0.338 [+0.262, +0.418] \* |
| \|total bars − requested\| | 0 | 2.71 | 1.07 | −1.64 [−2.52, −0.78] \* |
| **early EOS** | – | 0.040 | **0.000** | −0.040 [−0.067, −0.018] \* |
| late sections exact (last third) | 1.000 | 0.903 | 0.981 | +0.078 [+0.050, +0.109] \* |
| lyric recall | 1.000 | 0.955 | 0.964 | +0.008 [−0.008, +0.024] |
| section-local lyric recall | 1.000 | 0.948 | 0.958 | +0.010 [−0.008, +0.029] |
| strict-valid ABC | 1.000 | 0.711 | 0.631 | −0.080 [−0.151, +0.004] |
| no lyric-alignment error (seed-dependent, see below) | 1.000 | 0.782 | 0.653 | −0.129 [−0.200, −0.058] \* |
| pitch range | 22.01 | 14.07 | 13.30 | −0.77 [−1.59, +0.06] |
| distinct bars | 0.872 | 0.462 | 0.467 | +0.005 [−0.019, +0.028] |

**The counters are used, not decorative.** In E1's generations 100% of bars carry a counter, 98.3%
agree with the bars actually written, 98.5% agree with the requested plan, 99.5% of section headers
match the plan and 99.8% of sections end on `[r:1]`.

**Cleaning alone (E1c) explains about a third of the gain:** exact structure 0.529 → **0.658**
(+0.129 [+0.058, +0.200] \*). Adding counters takes it to 0.858. Note that E0's two seeds differ by
0.08 on this metric, so the cleaning effect is only about 1.5 seed-widths; the counter effect is 4×
larger than seed noise.

**Is there a cost in lyric alignment? Not consistently — this is a seed effect.** With seed 1234, E1
has more `lyric_overflow` songs than E0 (64 vs 38 of 225) and lower alignment validity
(0.653 vs 0.782). With seed 2345 the ordering reverses (35 vs 38 songs; 0.796 vs 0.724). Across the
two seeds E0 is 0.782 / 0.724 and E1 is 0.653 / 0.796, so the round-1-style conclusion "counters cost
validity" is **not supported**; the metric simply has seed noise of about ±0.07.

**The cramming *mechanism* is real in individual songs**, though. When the requested bars cannot hold
the requested syllables, E0 can add bars (breaking structure) while E1 keeps the bar count and crams
the leftovers onto one note with `~`. In test song `2HVhqSDfh…` (7 syllables per bar) the final 4-bar
chorus receives ~30 syllables on a single note in both E1 samples, where the corpus sings them as
continuous 16ths. **Recommendation:** put per-section syllable counts (or a notes-per-syllable
budget) in the prompt so the model can choose a denser rhythm instead of cramming.

**Second seed.** E1 with seed 2345 reproduces and exceeds the effect: exact structure **0.938**,
early EOS 0, strict validity 0.733, lyric recall 0.966. The two E1 seeds end at validation loss
0.38004 / 0.38007. Full seed table in §10.

## 4. Decoding: the diversity bottleneck

Sweep on a fixed 64-song **validation** subset (never the test set), for both E0 and E1; the winner
was then verified on all 225 test songs. E1, validation subset:

| setting | strict valid | exact structure | lyric recall | pitch range | distinct bars | 16th off-beats | syncopation | chord-tone |
|---|---|---|---|---|---|---|---|---|
| reference | 1.000 | 1.000 | 1.000 | 22.1 | 0.893 | 0.204 | 0.160 | 0.697 |
| greedy | 0.656 | 0.844 | 0.859 | 5.8 | 0.144 | 0.046 | 0.012 | 0.873 |
| T0.7 | 0.594 | 0.844 | 0.965 | 12.0 | 0.418 | 0.114 | 0.131 | 0.800 |
| T0.8 (round-1 default) | 0.641 | 0.734 | 0.967 | 13.8 | 0.543 | 0.164 | 0.159 | 0.746 |
| T0.9 | 0.578 | 0.812 | 0.972 | 15.8 | 0.674 | 0.171 | 0.149 | 0.724 |
| **T1.0 / p0.95** | **0.641** | **0.844** | **0.974** | **17.6** | **0.839** | **0.222** | **0.170** | **0.705** |
| T1.0 / p0.98 | 0.453 | 0.906 | 0.970 | 20.0 | 0.927 | 0.223 | 0.193 | 0.656 |
| T1.1 / p0.98 | 0.328 | 0.797 | 0.972 | 23.7 | 0.979 | 0.249 | 0.200 | 0.620 |

Verified on the full test set (225 songs):

| metric | reference | E1 @ T0.8 | E1 @ T1.0 | E0 @ T1.0 |
|---|---|---|---|---|
| pitch range | 22.01 | 13.30 | **18.40** | 18.41 |
| mean abs interval | 2.39 | 1.70 | 2.06 | 1.98 |
| 16th off-beat onsets | 0.193 | 0.129 | 0.182 | 0.184 |
| syncopation proxy | 0.176 | 0.126 | 0.165 | 0.151 |
| distinct bars | 0.872 | 0.467 | **0.797** | 0.797 |
| longest identical-bar run | 1.04 | 2.96 | 1.20 | 1.14 |
| chorus-chorus motif similarity | 0.358 | 0.579 | 0.409 | 0.410 |
| chorus motif preservation | 0.278 | 0.640 | 0.360 | 0.355 |
| chord-tone agreement | 0.677 | 0.730 | 0.658 | 0.671 |
| melody in declared key | 0.958 | 0.994 | 0.974 | 0.981 |
| chord roots in key | 0.920 | 0.950 | 0.932 | 0.935 |
| lyric recall | 1.000 | 0.964 | 0.970 | 0.975 |
| exact structure | 1.000 | 0.858 | 0.836 | 0.569 |
| strict valid | 1.000 | 0.631 | 0.591 | 0.578 |
| syllables crammed | 0.057 | 0.137 | 0.153 | 0.137 |

**Answer: mostly decoding (B), not capacity (A).** Temperature 0.8 with top-p 0.95 is a mode-seeking
distortion of a distribution that is already close to the corpus: at T=1.0 the model's own
statistics land near the reference on every rhythm and repetition metric, and the *repetition
structure* becomes healthier — chorus-to-chorus similarity falls from 0.58 (over-repeating) to 0.41,
against a corpus value of 0.36.

What temperature does **not** buy: pitch range is still 18.4 vs 22.0, and strict validity falls
(0.63 → 0.59 on test, more steeply at top-p 0.98). Higher temperature also raises lyric cramming.
**Recommended operating point: T=1.0, top-p 0.95** — best diversity at no measured cost in
control. Individual songs can still go wrong: one inspected sample wrote out-of-key chord symbols
("G#", "Dm" in G major), although in aggregate T=1.0 output is *more* diatonic than the corpus.

## 5. Clean subset: how much is label noise?

Full report: `reports/CLEAN_TEST_SUBSET.md` (34 songs, rule-based filters + manual reading of every
candidate, selection independent of any model output).

| metric | E0 full / clean | E1 T0.8 full / clean | E1 T1.0 full / clean |
|---|---|---|---|
| exact full structure | 0.529 / 0.765 | 0.858 / 0.882 | 0.836 / 0.941 |
| strict valid | 0.711 / 0.824 | 0.631 / 0.824 | 0.591 / 0.735 |
| no lyric-alignment error | 0.782 / 0.912 | 0.653 / 0.853 | 0.689 / 0.824 |
| lyric recall | 0.955 / 0.984 | 0.964 / 0.977 | 0.970 / 0.980 |
| distinct bars (ref 0.872 / 0.837) | 0.462 / 0.432 | 0.467 / 0.425 | 0.797 / 0.734 |
| pitch range (ref 22.0 / 18.5) | 14.07 / 14.03 | 13.30 / 13.18 | 18.40 / 17.29 |

* **Structure and validity errors concentrate in noisy songs** — pseudo-label quality is a major
  bottleneck for those metrics (E0: 0.53 → 0.77 just by restricting to clean songs).
* **Musical conservatism does not improve on clean songs**, so it is not label noise (it is decoding).
* Caveat: clean songs are also shorter and simpler, and the subset's own reference has a narrower
  pitch range (18.5 vs 22.0), so musical metrics must be read against it.

## 6. E2: effective batch size

Both arms use ABC-v2 and 524,288 padded tokens per update (32 micro-batches of 16,384), i.e. 4× E1.

**Learning rate was probed first, not assumed.** Two 30-update probes of the real schedule
(`stop_after_steps`, so the chosen one could resume without wasted compute):

| LR | loss @1 / @10 / @20 / @30 | grad-norm @10 / @30 | validation loss @30 |
|---|---|---|---|
| **5e-5** (= E1) | 1.003 / 0.706 / 0.588 / 0.519 | 3.1 / 1.7 | **0.522** |
| 1e-4 (≈ sqrt-scaled) | 1.003 / **1.109** / 0.716 / 0.620 | **9.9** / 4.9 | 0.620 |

At 1e-4 the loss *rose* over the first 10 updates and the gradient norm spiked, so **5e-5 was kept**
for both arms. (No linear scaling was attempted: 2e-4 would have been worse still.)

**E2a — same data epochs (2), therefore 4× fewer updates (128):**

| metric | reference | E1 (506 updates) | E2a (128 updates) | E2a − E1 |
|---|---|---|---|---|
| validation loss (whole-song, ABC-v2) | – | **0.3801** | 0.4138 | +0.034 |
| exact full structure | 1.000 | 0.858 | **0.916** | +0.058 |
| lyric recall | 1.000 | **0.964** | 0.890 | −0.074 |
| section-local lyric recall | 1.000 | **0.958** | 0.878 | −0.080 |
| strict-valid ABC | 1.000 | **0.631** | 0.293 | −0.338 |
| no lyric-alignment error | 1.000 | **0.653** | 0.391 | −0.262 |
| distinct bars | 0.872 | **0.467** | 0.416 | −0.051 |
| content-token test loss | – | **0.580** | 0.628 | +0.048 |

**E2b — same 506 optimizer updates as E1, which at this batch is ~8 epochs.** Its validation loss
bottoms at **0.3904 at step 250 (≈4 epochs)** and then rises steadily to **0.4872** at step 506:
at matched updates the large batch overfits, and even its best point is worse than E1's 0.3801.

| step | 100 | 175 | **250** | 325 | 400 | 506 |
|---|---|---|---|---|---|---|
| validation loss | 0.4232 | 0.3955 | **0.3904** | 0.4102 | 0.4514 | 0.4872 |

Generation metrics (225 test songs, T0.8):

| metric | reference | E1 | E2a (2 ep, 128 upd) | E2b (8 ep, 506 upd) | E3b (best) |
|---|---|---|---|---|---|
| exact full structure | 1.000 | 0.858 | 0.916 | 0.884 | **0.978** |
| lyric recall | 1.000 | 0.964 | 0.890 | 0.964 | 0.967 |
| section-local lyric recall | 1.000 | 0.958 | 0.878 | 0.954 | 0.966 |
| strict-valid ABC | 1.000 | 0.631 | 0.293 | 0.507 | **0.844** |
| no lyric-alignment error | 1.000 | 0.653 | 0.391 | 0.662 | **0.911** |
| early EOS | – | 0.000 | 0.013 | 0.018 | 0.000 |
| pitch range | 22.01 | 13.30 | 13.52 | 17.71 | 13.81 |
| distinct bars | 0.872 | 0.467 | 0.416 | 0.679 | 0.473 |

**Answer: a larger effective batch does not help.**

* At equal epochs (E2a) it trades lyric alignment and validity for a little structure: with only 128
  updates the model still learns the (easy, highly regular) counter/plan pattern but has not learned
  lyric-to-note alignment, which needs many updates.
* At equal updates (E2b) it sees 4× the tokens, overfits by epoch ~4, and ends with a worse
  validation loss (0.487 vs 0.380) and worse validity (0.507 vs 0.631) than E1, with no structural
  gain (0.884 vs 0.858, inside noise).
* An unexpected side effect: **the overfitted E2b samples more diversely at T=0.8** (pitch range 17.7,
  distinct bars 0.679, close to what E1 reaches only at T=1.0). Memorizing the training songs appears
  to sharpen the per-token distribution less than the well-fit model does, so temperature truncation
  bites less. It is not a good trade — validity and lyric alignment both fall — but it is a reminder
  that "more diverse output" can come from a *worse* model.
* Nothing in the long-range or repetition metrics improved: chorus-chorus motif similarity stays at
  0.61 (E2a) against E1's 0.58 and a corpus 0.36.

So for this task **optimizer updates, not tokens per update, are the scarce resource**, and 131K
tokens/update remains the better setting.

## 7. E3: a long-range objective

`reports/LONG_CONTEXT_AUDIT.md` §5 explains why "16K context" is not the experiment: the longest
example is 6,703 tokens, so E3 keeps `max_seq_len` 8192 and changes the **task** instead.

**What was added.** Masked late-section reconstruction: the prompt holds the plan, all lyrics and the
whole song in ABC-v2 with one later-half lyric section replaced by `% [missing section]`; the target
is that section. Sections whose label occurred earlier are preferred, so the model must reuse motif
identity from a distant chorus. 5,089 such examples (a deterministic 50% of songs) are mixed into
training, adding 14.9M tokens and 1.46M supervised tokens per epoch.

**Late-section continuation was deliberately not used as training data:** under teacher forcing,
"continue the song from a correct prefix" is already exactly what whole-song SFT does. It is used as
an *evaluation* instead, for every model.

**E3 does the new task well** (224 held-out songs, T0.8):

| metric | value |
|---|---|
| parses / one section / correct label | 1.000 / 0.996 / 1.000 |
| **bars exactly as requested** | **0.996** |
| section lyric recall | 0.967 |
| counter self-consistency | 1.000 |
| motif similarity to the reference target section | 0.454 |
| motif similarity to the earlier same-label section | 0.392 |
| *reference* target vs earlier same-label section | 0.338 |

The generated section resembles the true section (0.454) more than the true section resembles its own
earlier repeat (0.338) — the model reconstructs by copying the earlier chorus more literally than the
artist did.

**Whole-song generation: it depends entirely on the batch size the mixture is trained with.**

| metric (225 test songs, T0.8) | E1 | E2a | E3 (mixture @524K) | **E3b (mixture @131K)** |
|---|---|---|---|---|
| exact full structure | 0.858 | 0.916 | 0.867 | **0.978** |
| late sections exact | 0.981 | 0.976 | 0.977 | 0.996 |
| early EOS | 0.000 | 0.013 | 0.009 | 0.000 |
| lyric recall | 0.964 | 0.890 | 0.916 | **0.967** |
| section-local lyric recall | 0.958 | 0.878 | 0.910 | **0.966** |
| strict-valid ABC | 0.631 | 0.293 | 0.502 | **0.844** |
| no lyric-alignment error | 0.653 | 0.391 | 0.569 | **0.911** |
| distinct bars | 0.467 | 0.416 | 0.431 | 0.473 |
| validation loss | 0.3801 | 0.4138 | 0.3977 | **0.3780** |

* **E3 (the mixture at 524K tokens/update) sits between E1 and E2a** — exactly what the batch size
  alone predicts, so the long-range objective did not rescue the large batch.
* **E3b (the same mixture at E1's 131K tokens/update) is the best configuration measured**, and by a
  wide margin on validity: 0.844 strict / 0.911 alignment-clean against E1's 0.631 / 0.653 and E1
  seed 2345's 0.733 / 0.796.
* **At T=1.0 E3b keeps that and gains the musical diversity:** exact structure 0.982, lyric recall
  0.976, pitch range 17.9, distinct bars 0.771, syncopation 0.159, chord-tone 0.672 (corpus 0.677),
  chorus motif preservation 0.388 (corpus 0.278), strict validity 0.729.
* **Its secondary tasks are essentially solved:** infilling gives bars exact 1.000 and section lyric
  recall 0.984; continuation from a true prefix gives an exact section plan on 100% of songs with
  0.92 strict validity.

**Is E3b better because of the infill objective, or because it simply trains longer?** E3b runs 758
updates and 1.5× the tokens of E1 (the infill examples are extra data). The control **E1-long** —
whole songs only, same 758 updates (3 epochs), same batch, same seed — separates the two:

| metric (225 test songs, T0.8) | E1 (506 upd) | **E1-long (758 upd, no infill)** | **E3b (758 upd, + infill)** |
|---|---|---|---|
| exact full structure | 0.858 | 0.809 | **0.978** |
| late sections exact | 0.981 | 0.968 | 0.996 |
| strict-valid ABC | 0.631 | 0.724 | **0.844** |
| no lyric-alignment error | 0.653 | 0.831 | **0.911** |
| lyric recall | 0.964 | **0.978** | 0.967 |
| section-local lyric recall | 0.958 | 0.974 | 0.966 |
| distinct bars | 0.467 | 0.509 | 0.473 |
| validation loss (best / final) | 0.3801 / 0.3801 | 0.3801 / 0.3857 | **0.3780 / 0.3780** |

**Both effects are real and they are different effects:**

* **Training longer (E1 → E1-long) buys validity and lyric alignment**, not structure: strict validity
  0.631 → 0.724, alignment-clean 0.653 → 0.831, lyric recall 0.964 → 0.978, while exact structure
  does *not* improve (0.858 → 0.809, within the ±0.05 sampling and ±0.08 seed noise of that metric).
* **The infill objective buys structure**: at matched updates, E3b's exact structure is 0.978 against
  E1-long's 0.809 (+0.17), far outside noise, with alignment-clean also higher (0.911 vs 0.831).
  Learning to write "a section with exactly these bars, fitting between two fixed neighbours"
  transfers to writing every section of a whole song.

**Corrected reading of E3.** The first E3 run (same mixture at 524K tokens/update) showed no benefit,
but that was the batch size, not the objective: at E1's batch the same mixture gives the best model in
this report. **E3b at T=1.0 is the recommended configuration**: exact structure 0.982, lyric recall
0.976, strict validity 0.729, pitch range 17.9, distinct bars 0.771, chord-tone agreement 0.672
(corpus 0.677).

## 8. Comparison with MIDI-LLM

Repository `MIDI-LLM-phoneme-lyric-v1` at commit `bea5597` (clean tree), checkpoint
`runs/original-midi-llm-onestage-multitask-v1-v4/full/final` with its own registry (v4, the best
one-stage checkpoint by generation metrics), its own CLI and environment, on the **same 225 test
songs and the same specs** as Qwen-ABC. Two modes:

* **A — recommended constraints:** `--forbid-no-lyric --min-chord-sec 0.25 --max-notes-per-syllable 4
  --require-section-syllables --max-wordless-notes-per-bar 1.5 --max-chords-per-bar 3.0`.
* **B — least constrained the CLI allows:** every optional constraint off, plus
  `--free-lyric-assignment --free-section-labels`.

**A caveat that dominates the structural comparison:** with `--from-spec`, MIDI-LLM turns the spec
into a *click grid that is part of the model input*, and its decoder pins section count, bars per
section and bar markers. **Its bar counts cannot be wrong by construction**, in either mode. A
"structure-free" mode does not exist for these checkpoints. Mode B only frees section *labels* and
lyric-line assignment.

### Coverage

| | MIDI-LLM A | MIDI-LLM B | Qwen-ABC (E0/E1) |
|---|---|---|---|
| songs producing a lead sheet | 199 / 225 | 199 / 225 | **225 / 225** |
| refused at input (G2P: code-switched English, rare characters) | 26 | 26 | 0 |
| passed its own validator | 175 | 193 | – |
| median seconds per song (1× L40S) | 89 | 97 | ~4 (batch 16) |

The 26 refusals are a hard limitation of its phoneme front end (`mandarin_g2p.py`); 18.2% of the test
songs contain latin syllables. Qwen-ABC reads characters directly and needs no G2P.

Also worth noting: **its recommended constraints lower its own validity** (175 valid vs 193 in the
unconstrained mode). The failures are almost all `melody_onset_pileup`, the known stall where the
model runs out of room before a section boundary while `--require-section-syllables` keeps the
section open.

### Quality, on all 225 songs (unsupported songs count as failures for rates)

| metric | reference | MIDI-LLM A | MIDI-LLM B | Qwen E1 @T0.8 | Qwen E1 @T1.0 |
|---|---|---|---|---|---|
| produced any output | 1.000 | 0.884 | 0.884 | **1.000** | **1.000** |
| passed a validator (own / strict ABC) | 1.000 | 0.778 | 0.858 | 0.631 | 0.591 |
| exact section plan | 1.000 | 0.884 † | 0.027 | 0.858 | 0.836 |
| **lyric recall** | 1.000 | 0.730 | 0.581 | 0.964 | **0.970** |
| lyric precision | 1.000 | 0.843 | 0.815 | 0.991 | 0.995 |
| section-local lyric recall | 1.000 | 0.721 | 0.313 | 0.958 | 0.969 |
| pitch range | 22.01 | 24.19 | 26.09 | 13.30 | 18.40 |
| mean abs interval | 2.39 | 2.35 | 2.28 | 1.70 | 2.06 |
| notes per bar | 3.47 | 3.38 | 4.02 | 3.56 | 3.38 |
| 16th off-beat onsets | 0.193 | 0.200 | 0.207 | 0.129 | 0.182 |
| syncopation | 0.176 | 0.171 | 0.168 | 0.126 | 0.165 |
| chords per bar | 0.982 | 1.078 | 0.940 | 0.823 | 0.928 |
| chord-tone agreement | 0.677 | 0.625 | 0.628 | 0.730 | 0.658 |
| distinct bars | 0.872 | 0.937 | 0.934 | 0.467 | 0.797 |
| **chorus motif preservation** | 0.278 | 0.110 | 0.102 | 0.640 | **0.360** |
| chorus-chorus motif similarity | 0.358 | 0.270 | 0.277 | 0.579 | 0.409 |

† = 199/225, i.e. **1.000 on every song it generated**, because the grid is an input.

**Reading it honestly:**

* **Structure:** not a fair comparison. MIDI-LLM cannot deviate; Qwen-ABC is free-running and reaches
  0.86 by writing its own state. The meaningful statement is that an unconstrained model now gets
  close to a constrained decoder, and that when MIDI-LLM is allowed to choose labels (mode B) its
  section plan collapses to 0.027 — i.e. the *model* does not know the plan; the *decoder* does.
* **Lyrics:** Qwen-ABC is clearly stronger (0.97 vs 0.73 recall, 0.995 vs 0.84 precision), and it is
  the only system that handles all 225 songs. Caveat: MIDI-LLM's lyric metrics come from converting
  its pointer stream, where re-attacks count as extra syllables.
* **Musicality:** MIDI-LLM's note statistics sit closest to the corpus, but it barely repeats: chorus
  motif preservation 0.11 against a corpus 0.28. Qwen at T0.8 over-repeats (0.64) and at T1.0 is the
  closest of all systems to the corpus level (0.36). Neither "more varied" nor "more coherent" is a
  quality verdict — **no listening test was run, so no musical-quality claim is made here.**
* **Validity uses two different validators** and is not directly comparable.

## 9. Split re-verification

The canonical de-duplicated split (10,243 / 273 / 225) was reused unchanged and re-checked from the
written ABC-v2 files, with and without the build's document-frequency filter:

| held-out vs train | exact lyric duplicates | lyric containment max / ≥0.3 | melody containment max / ≥0.3 |
|---|---|---|---|
| validation | 0 | 0.225 / 0 | 0.043 / 0 |
| test | 0 | 0.043 / 0 | 0.194 / 0 |

Song-id overlap between any two splits is 0; validation-vs-test lyric containment max is 0.020. No
bug was found and the split was not modified.

## 10. Seeds and statistics

**Method.** Every comparison is paired over the same test songs, one sample per song, with 2,000-sample
paired bootstrap CIs (`scripts/compare_r2.py`). Rates count an unparseable output as a failure;
content metrics skip it. Decoding was selected on a **validation** subset, never on test.

**Two seeds per central arm** (225 test songs, T0.8):

| metric | E0 s1234 | E0 s2345 | E1 s1234 | E1 s2345 |
|---|---|---|---|---|
| exact full structure | 0.529 | 0.613 | **0.858** | **0.938** |
| late sections exact | 0.903 | 0.877 | 0.981 | 0.995 |
| early EOS | 0.040 | 0.089 | **0.000** | **0.000** |
| lyric recall | 0.955 | 0.943 | 0.964 | 0.966 |
| section-local lyric recall | 0.948 | 0.928 | 0.958 | 0.964 |
| strict-valid ABC | 0.711 | 0.640 | 0.631 | 0.733 |
| no lyric-alignment error | 0.782 | 0.724 | 0.653 | 0.796 |
| songs with `lyric_overflow` | 38 | 38 | 64 | 35 |
| validation loss | 0.5002 (v1) | 0.5002 (v1) | 0.3801 | 0.3800 |

* **The structure result is far larger than seed noise**: the two E0 seeds span 0.53–0.61, the two E1
  seeds 0.86–0.94, and the gap is ≥0.24 in every pairing.
* **Early EOS is eliminated in both E1 seeds.**
* **Strict validity and lyric-alignment validity have ±0.07 seed noise** and the two arms overlap;
  no claim is made about them.
* Lyric recall differences (±0.01) are inside noise.

**Sampling noise, separately from training noise.** E0 was regenerated with the round-2 batched
pipeline and compared with the round-1 batch-1 sample of the same checkpoint: no metric differed
beyond its CI (largest: strict validity 0.671 → 0.627, CI [−0.116, +0.027]). Two samples of the same
checkpoint therefore differ by up to ±0.05 on validity-type metrics.

**What is *not* claimed:** single-seed differences of <0.1 on strict validity or alignment validity,
the E2a-vs-E1 structure difference (single seeds, and E2a is confounded by undertraining), and
anything about musical *quality* (no listening test).

## 11. Training budget

Generated by `scripts/budget_report.py` (full JSON in `reports/tables/budget.json`). All runs are
full-parameter finetunes of Qwen3.5-0.8B-Base (752M parameters, all trainable), `max_seq_len` 8192,
`tokens_per_micro_batch` 16,384, fp32 master weights + bf16 autocast + gradient checkpointing,
peak VRAM 16.0 GB per GPU.

| run | GPUs | micro-batches / update | tokens/update budget | LR | updates | epochs | raw tokens | supervised tokens | tokens/s | wall-clock h | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 (round 1) | 1× L40S | 8 | 131K | 5e-5 | 382 | 2.0 | 45.97M | 39.02M | ~6.2K | 2.05 | 2.05 |
| E0 seed 2345 | 1× L40S | 8 | 131K | 5e-5 | 382 | 2.0 | 45.96M | 39.02M | ~6.2K | 2.06 | 2.06 |
| E1c | 2× L40S | 8 | 131K | 5e-5 | 382 | 2.0 | 46.19M | 39.02M | 12.3K | 1.11 | 2.21 |
| E1 | 2× L40S | 8 | 131K | 5e-5 | 506 | 2.0 | 59.61M | 50.94M | 12.2K | 1.46 | 2.92 |
| E1 seed 2345 | 2× L40S | 8 | 131K | 5e-5 | 506 | 2.0 | 59.61M | 50.94M | 12.1K | 1.60 | 3.20 |
| E2a | 2× L40S | 32 | 524K | 5e-5 | 128 | 2.0 | 59.61M | 50.94M | 11.4K | 1.50 | 2.99 |
| E2 LR probe (1e-4) | 2× L40S | 32 | 524K | 1e-4 | 30 | 0.47 | 14.19M | 12.12M | 11.2K | 0.37 | 0.75 |
| E2b | 2× L40S/L40 | 32 | 524K | 5e-5 | 506 | 7.9 | 236.1M | 201.7M | 11.2K | 7.10 (incl. restarts) | 14.2 |
| E3 | 2× L40S | 32 | 524K | 5e-5 | 190 | 2.0 | 89.47M | 53.87M | 12.7K | 1.98 | 3.95 |
| E3b | 2× L40S | 8 | 131K | 5e-5 | 758 | 2.0 | 89.47M | 53.87M | 13.3K | 2.00 | 3.99 |
| E1-long (control) | 2× L40S/L40 | 8 | 131K | 5e-5 | 758 | 3.0 | 89.40M | 76.39M | 12.2K | 2.25 | 4.49 |

Notes:

* "epochs" for E3/E3b counts epochs over the *mixture* (whole songs + infill); the whole-song part is
  2 epochs in both.
* Round-1 runs (E0) were single-GPU; round-2 runs use 2 GPUs with the equivalence-checked data-parallel
  path, so GPU-hours are about 2× wall-clock.
* **Evaluation cost** is now small: batch-16 sampling covers 225 test songs in 20–35 minutes on one
  L40S (~0.5 GPU-h), against ~2.5 GPU-h at batch 1 in round 1. The decoding sweeps are 7 × 64 songs
  per model (~1 GPU-h each).
* **MIDI-LLM baseline:** 199 songs × 2 modes, median 89 s and 97 s per song, 7.8 + 7.7 GPU-h, plus a
  re-run of 28 songs whose first attempt exceeded its context.
* **Total round-2 GPU time:** roughly 50 GPU-h of training (E2b's restarts after the storage incident
  account for about 8 of them), ~15 GPU-h of Qwen evaluation and ~16 GPU-h of MIDI-LLM.

**Where the artifacts are.** Round-2 runs up to E3 live in `experiments/`; E2b, E3b, E1-long, their
evaluations and the seed-2 evaluations were written to
`/gscratch/ark/pingw220/qwen_abc_r2_offload/{runs,evals,logs}` after scrubbed hit its quota (§15), and
the E1 checkpoints were moved there too (symlinked back into `experiments/`).

## 12. Listening / qualitative

Eight test songs were exported with input, reference and every model's ABC and MIDI, for E0 (T0.8 and
T1.0), E1 (T0.8 and T1.0), E3 and MIDI-LLM mode A
(`experiments/listening_20260915_listening/`, git-ignored: it contains lyrics). Audio was rendered
with fluidsynth and then deleted to survive the storage incident (§14); re-render from the MIDI with
`scripts/make_samples.py --audio`. The songs
were chosen from **reference features only** (`scripts/select_listening_songs.py`): simple
verse/chorus, long multi-section, bridge, repeated chorus, high lyric density, melisma-heavy,
code-switching, plus one random song. No song was chosen by looking at model output.

Audio was rendered but **not listened to**, so nothing below is a claim about musical quality; these
are observations from the notation, chosen to include failures. One of the eight songs
(`5CbNIRIgmr…`, code-switched) MIDI-LLM refused outright.

1. **Structure is visibly followed (E1).** In every sample the section headers, counters and bar
   counts match the request, and sections end on `[r:1]`.
2. **Dense lyrics turn into cramming, not into extra bars.** `2HVhqSDfh…` asks for ~7 syllables per
   bar; the reference sings continuous 16ths, while both E1 samples keep the requested 4-bar chorus
   and put the last ~30 syllables on one note (`她~的~睫~毛…`). E0 instead drifted and lost content
   (55 written lines against the reference's 71).
3. **Low-temperature degeneration is real and visible.** In the 15-section code-switched song
   `28I269iDvk…`, E1 at T0.8 collapses into a two-note ostinato (`C3/ B,/` repeated for whole
   sections) and crams the lyrics onto it (lyric recall 0.67 for that song). At T1.0 the same
   checkpoint writes a normal melody for it (recall 0.99). This is the single-song face of the
   aggregate "distinct bars 0.47 → 0.80".
4. **Melisma-heavy references stay sparse.** In `5rc5x4FKWvr…` (0.78 syllables/bar, 61% melisma
   notes in the reference) the model writes long held notes and few attacks, which matches the
   reference's shape; this song is also a good example of a reference whose alignment is itself
   doubtful.
5. **At T1.0 an occasional song leaves the key** in its chord symbols ("G#", "Dm" in G major), even
   though the aggregate stays more diatonic than the corpus.

## 13. Ablation table

One sample per song on the 225 test songs, T=0.8 unless noted; "content loss" is teacher-forced test
loss over musical/lyric tokens only (counters, headers and structure lines excluded), so it is
comparable across ABC-v1 and ABC-v2.

| ID | Model | ABC | Context | Tokens/update | Training task | structure exact | lyric recall | early EOS | pitch range | distinct bars | strict valid | test loss | content loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.8B | ABC-v1 | 8K | 131K | Direct SFT 2ep | 0.529 | 0.955 | 0.040 | 14.071 | 0.462 | 0.711 | 0.4837 | 0.5952 |
| E1c | 0.8B | ABC-v1 cleaned | 8K | 131K | Direct SFT 2ep | 0.658 | 0.955 | 0.036 | 14.484 | 0.449 | 0.640 | 0.4828 | 0.5938 |
| E1 | 0.8B | ABC-v2 | 8K | 131K | Direct SFT 2ep | 0.858 | 0.964 | 0.000 | 13.298 | 0.467 | 0.631 | 0.3661 | 0.5800 |
| E1-long | 0.8B | ABC-v2 | 8K | 131K | Direct SFT 3ep | 0.809 | 0.978 | 0.009 | 14.467 | 0.509 | 0.724 | 0.3705 | 0.5874 |
| E2a | 0.8B | ABC-v2 | 8K | 524K | Direct SFT 2ep (128 upd) | 0.916 | 0.890 | 0.013 | 13.518 | 0.416 | 0.293 | 0.3989 | 0.6280 |
| E2b | 0.8B | ABC-v2 | 8K | 524K | Direct SFT 8ep (506 upd) | 0.884 | 0.964 | 0.018 | 17.707 | 0.679 | 0.507 | 0.4667 | 0.7302 |
| E3 | 0.8B | ABC-v2 | 8K | 524K | + infill 2ep | 0.867 | 0.916 | 0.009 | 13.249 | 0.431 | 0.502 | 0.3828 | 0.6040 |
| E3b | 0.8B | ABC-v2 | 8K | 131K | + infill 2ep (758 upd) | 0.978 | 0.967 | 0.000 | 13.813 | 0.473 | 0.844 | 0.3643 | 0.5775 |
| E3b@T1.0 | 0.8B | ABC-v2 | 8K | 131K | + infill, T=1.0 | 0.982 | 0.976 | 0.000 | 17.938 | 0.771 | 0.729 | 0.3643 | 0.5775 |

Reading down the table: **cleaning** (E0 → E1c) buys +0.13 structure; **counters** (E1c → E1) another
+0.20 and removes early EOS; **more updates** (E1 → E1-long) buys validity and lyric recall but no
structure; **the long-range objective** (E1-long → E3b) buys +0.17 structure and +0.12 validity; and
**decoding** (E3b → E3b@T1.0) buys the musical diversity (pitch range 13.8 → 17.9, distinct bars
0.47 → 0.77) at a cost of 0.12 strict validity. The 4× batch (E2a, E3) is worse everywhere except a
structure number that comes with unusable validity.

## 14. Answers to the 16 questions

1. **Was 8K context actually a limitation?** No. The longest example is 6,703 tokens; nothing was
   truncated; raising `max_seq_len` to 16K would give a bit-identical run (§2).
2. **Did explicit structural counters solve bar/section counting?** Largely. Exact section+bar plans
   go 0.53/0.61 (E0 seeds) → 0.86/0.94 (E1 seeds), per-section exactness 0.90 → 0.98, total bar count
   0.53 → 0.87, and early EOS disappears. Counters agree with the plan 98.5% of the time. The
   residual 6–14% is mostly songs where the requested bars cannot hold the lyrics (§3).
3. **Did upstream section-boundary cleaning help?** Yes, but modestly and with a caveat: cleaning
   alone moves exact structure 0.529 → 0.658 (+0.13 [+0.06, +0.20]), roughly 1.5× the E0 seed spread.
   It also shifts the corpus slightly off the 4/8-bar grid, and it is a lyric-centric convention
   rather than a proven correction (ABC_V2_DATA_CLEANING.md §4).
4. **Did 512K tokens/update help?** No. At equal epochs it costs lyric recall (0.964 → 0.890) and
   validity; at equal updates it overfits (validation loss bottoms at 0.390 by ~4 epochs, ends 0.489
   vs E1's 0.380). Updates, not tokens, are the scarce resource (§6).
5. **Did meaningful 16K long-context tasks help?** The 16K part is moot (§2), but the long-range
   *objective* helped a lot **at the right batch size**. Mixed into training at 131K tokens/update
   (E3b) it raises exact structure to 0.978 vs 0.809 for a matched-update control without it, and
   gives the best model in this report; at 524K tokens/update (E3) it showed nothing. The model also
   performs the new task itself nearly perfectly (bars exact 1.000, section lyric recall 0.984) (§7).
6. **Did decoding explain any of the musical conservatism?** Most of it. At T=1.0/top-p 0.95 the same
   checkpoint moves pitch range 13.3 → 18.4 (reference 22.0), distinct bars 0.47 → 0.80 (0.87),
   syncopation 0.126 → 0.165 (0.176) and chorus repetition from far-above-corpus to corpus level,
   with lyric recall and structure unchanged (§4).
7. **How much error disappears on a clean test subset?** A lot of the structure/validity error: E0's
   exact structure 0.53 → 0.77 and E1's 0.86 → 0.88–0.94; strict validity +0.1–0.2 for every model.
   Musical conservatism does not improve, so it is not label noise (§5).
8. **Is Qwen3.5-0.8B capacity-limited?** No evidence that it is, for this task. Its sampling
   distribution already matches the corpus on rhythm, repetition and harmony when decoded at T=1.0;
   the remaining errors are lyric-note alignment in dense songs, pseudo-label noise, and a pitch
   range still ~15% below the corpus. None of these is obviously a capacity symptom. This was not
   tested directly (see 9).
9. **Did ~2B improve musical diversity?** Not run. The pre-registered trigger for scaling — "melody
   remains too narrow after the representation fix" — was not met once decoding was corrected. The
   exact model id is verified for a future run: `Qwen/Qwen3.5-2B-Base` (sha `b1485b2f`), about 2.5×
   E1's cost (~7 GPU-hours on 2 L40S).
10. **Is 4B justified?** No. 2B was not justified; 4B even less so. Data and objective work first.
11. **How does the best Qwen compare with MIDI-LLM?** Qwen-ABC E1 at T=1.0 covers all 225 songs
    (MIDI-LLM: 199), has much better lyric recall (0.970 vs 0.730) and precision (0.995 vs 0.843),
    comparable rhythm statistics, a narrower pitch range (18.4 vs 24.2, corpus 22.0), and repetition
    closest to the corpus. MIDI-LLM's structure is exact by construction (§8).
12. **Which system has stronger lyric control?** Qwen-ABC, clearly, and it is the only one that
    accepts code-switched English and rare characters.
13. **Which system has stronger structural control?** MIDI-LLM, but only because the decoder and the
    click grid *impose* it; free its labels and its exact-plan rate collapses to 0.027. Among freely
    generating models, Qwen-ABC with counters is the stronger one.
14. **Which system has stronger musical diversity?** MIDI-LLM produces wider ranges and almost never
    repeats a bar; Qwen at T=1.0 matches the corpus more closely on repetition and chord-tone
    agreement. "More diverse" is not "better": MIDI-LLM repeats *less* than real songs do (chorus
    motif preservation 0.11 vs corpus 0.28). No listening test was run, so no quality claim is made.
15. **Which differences come from decoder constraints rather than the learned model?** All of
    MIDI-LLM's structural advantage, and part of its validity advantage (its recommended constraints
    actually *lower* its own validator pass rate: 175 vs 193 of 199). Its lyric coverage gap is a
    front-end (G2P) limitation, not a model one.
16. **Recommended next research direction (in order):**
    1. **Fix the lyric-density mismatch in the representation**: put per-section syllable counts or a
       notes-per-syllable budget in the prompt. This is the mechanism behind the remaining structure
       failures and the cramming, and it is cheap.
    2. **Attack pseudo-label noise**, which still explains a large share of residual error: better
       section boundaries, double-time detection, and an aligner pass that does not cram syllables.
    3. **Adopt T=1.0/top-p 0.95** as the evaluation and product decoding point, and report musical
       metrics only against the corpus band.
    4. **Then**, if a capacity question remains, run the single 2B control — after, not before, the
       above.
    5. **Keep the long-range infilling objective at 131K tokens/update** (E3b: 2 epochs over the
       whole-song + infill mixture, 758 updates): that gives structure 0.98, validity 0.84 and lyric
       recall 0.97 at T=0.8, and structure 0.98 with corpus-like musical statistics at T=1.0. It also
       provides section infilling as a usable feature.

## 15. Reproduction

| step | command / location |
|---|---|
| environment | `README.md`, `scripts/env.sh` |
| tests | `python -m pytest -q tests` (52 passed) |
| ABC-v2 dataset | `scripts/build_abc_v2_dataset.py --v1-dir data/generated/abc_v1_20260915_012459 --output-dir data/generated/abc_v2_<ts>` (jobs 40179446/40179447), validated by `scripts/validate_abc_v2_dataset.py` (job 40179448) |
| E3 task files | `scripts/build_longrange_tasks.py --v2-dir <v2> --infill-fraction 0.5` |
| training | `CONFIG=configs/<cfg>.yaml RUN_DIR=experiments/<name>_<ts> NPROC=2 sbatch --gpus=l40s:2 -c 16 --mem=128G scripts/slurm/train.sbatch` |
| evaluation bundle | `RUN=experiments/<run> FMT=v2 bash scripts/slurm/submit_run_evals.sh` |
| decoding sweep | `CKPT=... DATA=... FMT=v2 OUT=experiments/r2sweep_<name> bash scripts/slurm/submit_decoding_sweep.sh` |
| continuation / infill | `scripts/eval_continuation.py`, `scripts/eval_infill.py` |
| clean subset | `scripts/select_clean_subset.py` then manual reading; ids in `reports/clean_test_song_ids.txt` |
| comparisons | `scripts/compare_r2.py --run name=<eval dir> ... [--subset reports/clean_test_song_ids.txt]` |
| budget | `scripts/budget_report.py --run name=<run dir> ...` |
| MIDI-LLM | `scripts/midi_llm_baseline.py specs`, `scripts/slurm/midi_llm_r2_shard.sh` (MODE=A/B), `scripts/slurm/midi_llm_r2_retry_context.sh`, `scripts/midi_llm_baseline.py convert`, `scripts/midi_llm_r2_report.py` |
| samples | `scripts/make_samples.py --song-ids reports/listening_song_ids.json --audio` |

Weights, generations (they contain lyrics) and audio stay on scrubbed and are git-ignored.

### Infrastructure notes

* **Batched generation.** All round-2 sampling uses left-padded batch-16 decoding (9× faster). It is
  statistically equivalent to round-1 batch-1 decoding: a second E0 sample generated this way differs
  from the round-1 sample by no more than sampling noise on every metric (`scripts/check_infra.py`,
  `scripts/check_left_padding.py`, LONG_CONTEXT_AUDIT.md §6).
* **Data parallelism.** `torchrun --nproc_per_node N` reproduces the single-GPU optimizer step
  (rank r takes micro-batches r, r+N, …; gradients summed; loss normalized by the update's global
  supervised-token count). Checked: step-1 loss identical to 8 decimals on 1 vs 2 GPUs, final eval
  loss 2.10223 vs 2.10206 after 3 steps (bf16 noise), 1.8× throughput.
* **Storage incident.** On 2026-09-15 ~16:55 the scrubbed filesystem hit a quota limit and every
  running job died with `Disk quota exceeded`, including the user's unrelated `midillm-onestage-prod`
  (job 40168248, 17.5 h). Qwen-ABC round-2 runs held ~100 GB, mostly 9 GB `resume_state` snapshots and
  per-N-step checkpoints. Mitigation: intermediate checkpoints deleted, snapshots disabled for the
  re-runs (`resume_every: 0`), superseded datasets removed. The trainer's atomic snapshot survived the
  outage, so E2b resumed from step 330 rather than restarting.
