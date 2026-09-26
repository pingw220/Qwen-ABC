# Memorization / copying analysis

**Question.** Do the models retrieve or closely copy training songs?

**Answer (measured).** No generation copies an informative melodic phrase from a training song at a
length the held-out references do not also reach. Generations are, on average, slightly *more*
similar to their nearest training song than real held-out songs are: E3b by +0.017 interval excess,
MuPT by +0.037. MIDI-LLM is slightly less similar (−0.008). All 31 raw flags (23 on the
main samples, 8 on intervention samples) turned out, on inspection, to be repeated-pitch or two-note
trill stretches. Under the informative-run criterion 1 of 1,322 generations is flagged, and it is a
+2/−2 trill interrupted once (see below). Meanwhile one held-out *reference* shares a 30-interval,
8-distinct-interval phrase with a training song, more exact overlap than any generation.

Code: `paper_eval/memorization.py`, `scripts/paper_final/memorization.sbatch`,
`tests/test_memorization.py`. Numbers: `data/memorization_stats.json`, per-song
`data/memorization.parquet` (ids + numbers only; nearest-neighbour ids in
`data/memorization_nn_ids.parquet`), table `tables/memorization.{csv,md,tex}`, figure
`figures/memorization_nn.{pdf,png}`. Jobs 40604645 and 40604982 (ckpt-all, CPU, ~3 min each), plus
40604924 (inspection).

## Method

* **Corpus searched:** all 10,243 training songs (`abc_v2_20260915_120927/songs_train.jsonl`).
  **Queries:** the 225 held-out references; E3b, MuPT (first of the four T=1.0 samples from R3-B/R4)
  and MIDI-LLM (mode A, seed 1000, 199 generated songs); and, as "novel request" evidence, the E3b
  paper-final samples (S1) for `bars_p4`, `ood_long20` and `lyrics_all` (224 / 224 / 225 songs).
* **Representations:** melodic intervals (clamped ±12, transposition invariant); contour (interval
  sign); rhythm (onset-in-bar, duration); motif (interval, inter-onset); chords per change relative
  to the tonic (key invariant); ABC music tokens (counters, comments and lyrics stripped; not for
  MIDI-LLM, which has no ABC); section plan (label:bars); lyric syllables (references only, since
  generations copy the prompt's lyrics).
* **Similarity:** normalized LCS (LCS / max length) against **every** training song, which is exact and
  bit-parallel (`paper_eval/seqsim.py`). A top-50-by-n-gram-containment shortlist was tried first and
  rejected: on 20 random references it found the true LCS neighbour for only 2 / 7 / 6 of 20 queries
  (interval / rhythm / chord), with gaps up to 0.18 (`data/memorization_shortlist_validation.json`).
  Also recorded: n-gram containment (exact over all songs), the song's **background** (mean
  similarity to all training songs), **excess** = NN − background, the longest exact common run with
  the NN, and the **exact bar-copy rate** (share of bars with ≥3 notes whose key-relative
  (onset, duration, pitch − tonic) signature occurs anywhere in training; 381,992 distinct such
  training bars).
* **Why excess and not raw NN similarity.** Normalized LCS rewards simple, repetitive melodies against
  *any* song. E3b's raw interval NN similarity (0.569) exceeds the references' (0.547), but so does
  its background (+0.005 [+0.000, +0.009]). Excess separates "close to one particular song" from
  "generic".
* **Null / baseline:** the held-out references, i.e. real songs by other artists that were never
  trained on. Their overlap with training is what ordinary Mandarin-pop idiom produces. Flags use the
  references' 99th percentile.
* **Statistics:** song-level bootstrap, 10,000 resamples; paired differences vs the reference of the
  same song.

## Results (measured)

Mean [95% CI] over songs; N = songs with an output.

| | N | interval NN sim | interval excess | motif excess | rhythm excess | chord excess | ABC excess | bar-copy rate | interval vs own reference |
|---|---|---|---|---|---|---|---|---|---|
| Pseudo-GT (held-out reference) | 225 | 0.547 [0.541, 0.552] | 0.157 [0.153, 0.162] | 0.138 [0.133, 0.145] | 0.213 [0.203, 0.223] | 0.379 [0.366, 0.393] | 0.121 [0.116, 0.127] | 0.082 [0.070, 0.095] | – |
| Qwen-ABC (E3b) | 225 | 0.569 [0.564, 0.574] | 0.175 [0.170, 0.180] | 0.149 [0.143, 0.155] | 0.217 [0.208, 0.226] | 0.402 [0.389, 0.414] | 0.126 [0.122, 0.129] | 0.105 [0.090, 0.121] | 0.452 [0.446, 0.459] |
| MuPT | 225 | 0.569 [0.564, 0.574] | 0.194 [0.187, 0.203] | 0.175 [0.168, 0.182] | 0.256 [0.246, 0.267] | 0.419 [0.406, 0.432] | 0.132 [0.129, 0.135] | 0.101 [0.084, 0.120] | 0.419 [0.409, 0.428] |
| MIDI-LLM | 199 | 0.542 [0.539, 0.546] | 0.150 [0.146, 0.154] | 0.139 [0.134, 0.144] | 0.213 [0.204, 0.223] | 0.362 [0.351, 0.372] | – | 0.046 [0.036, 0.057] | 0.443 [0.437, 0.449] |

Paired difference vs the same song's reference (system − reference):

| | interval excess | motif excess | rhythm excess | chord excess | bar-copy rate | longest exact interval run (notes) |
|---|---|---|---|---|---|---|
| E3b | +0.017 [+0.012, +0.023] | +0.010 [+0.004, +0.017] | +0.004 [−0.007, +0.014] | +0.022 [+0.007, +0.038] | +0.023 [+0.007, +0.040] | +0.07 [−0.34, +0.43] |
| MuPT | +0.037 [+0.028, +0.046] | +0.036 [+0.029, +0.044] | +0.043 [+0.032, +0.055] | +0.040 [+0.023, +0.057] | +0.019 [+0.002, +0.037] | +0.84 [+0.27, +1.39] |
| MIDI-LLM | −0.008 [−0.013, −0.003] | −0.000 [−0.007, +0.006] | −0.001 [−0.012, +0.011] | −0.019 [−0.036, −0.002] | −0.040 [−0.054, −0.026] | −0.48 [−0.88, −0.14] |

**Exact runs.** The longest exact common interval run with the nearest training song averages 7.0
notes for references, 7.1 for E3b, 7.9 for MuPT and 6.6 for MIDI-LLM. With unisons removed
(repeated pitches cannot then pad a run) and at least 3 distinct intervals required, the maxima
are **30 (a reference)**, 13 (E3b), 15 (MuPT) and 12 (MIDI-LLM). Songs with such an informative run
of ≥10 notes: references 10/225, E3b 10/225, MuPT 16/225, MIDI-LLM 7/199.

**Copying the held-out reference.** Generations are not reconstructions of their own reference song:
interval-sequence LCS with their own reference is 0.452 (E3b), below their similarity to the nearest
*training* song. Verbatim bars shared with the own reference are ≈0 (E3b 0.000 [0.000, 0.001]). The
section plan is reproduced (plan self-similarity 0.998), because it is the prompt.

**Split sanity (lyrics).** The references' lyric 6-gram containment in their nearest training song is
0.006 [0.005, 0.007] (max ≤ 0.043 per the split report), so the de-duplicated split holds.

**Hubs.** Some training songs are nearest neighbours to many queries. One is the interval NN of 12
MIDI-LLM outputs and 5 MuPT outputs, but of at most 3 references (it is not in the references' top 5). This is the "long simple melody matches
everything" effect, the reason excess rather than raw similarity is reported.

### Flagged cases and what inspection found

Raw flag = interval **and** motif excess above the references' 99th percentile (0.247 / 0.265), or an
exact interval run longer than 99% of references' (14.3 notes). Raw flags: E3b 3/225, MuPT 19/225,
MIDI-LLM 1/199; intervention samples 4/224 (`bars_p4`), 1/224 (`ood_long20`), 3/225 (`lyrics_all`).

Every flagged case was inspected (job 40604924: the common run, its content, position and the
reference's own similarity to the same training song):

* **All 23 are degenerate stretches, not phrases.** In 19 of them the shared run is ≥82% unisons, i.e.
  a note repeated 8-28 times (a recitation on one pitch; for MuPT often where many syllables sit on a
  monotone). The other four are two-note trills or neighbour-note alternations (−2/+2, some interleaved with unisons). The most extreme case, a MuPT
  output with a 60-interval run after removing unisons, is a trill with 2 distinct intervals.
* Several MuPT flags share the same training neighbour (three test songs each with two different
  training songs). The shared content is again a long repeated-pitch stretch present in both.
  This is a stylistic degeneracy of MuPT, which also shows as its larger rhythm excess (+0.043),
  not retrieval of that song.
* **Informative-run flags** (unison-free run > references' 99th percentile of 16.3 notes, with ≥3
  distinct intervals): **0/225 E3b, 0/225 MuPT, 0/199 MIDI-LLM, 0/224 `bars_p4`, 0/224 `ood_long20`, 1/225 `lyrics_all`.** The one
  `lyrics_all` flag (song `28I269iDvk…`, 34-interval run with its nearest training song) was inspected: 31 of
  its 34 intervals are a +2/−2 two-note trill, interrupted once by −5, +3 — which is what lifts it past the
  "≥ 3 distinct intervals" rule. It is the same two-note ostinato round 2 documented for this song
  (LONG_STRUCTURE_EXPERIMENTS §12), not a copied phrase. The informative-run rule should additionally
  exclude runs dominated by a two-note alternation; with that refinement the count is 0 of all 1,322 generations examined.

### Novel requests (full intervention sets, seed S1)

| E3b samples | N | interval excess | plan similarity to nearest training plan | exact bar-copy rate | informative flags |
|---|---|---|---|---|---|
| own plan (`orig`) | 225 | 0.175 | 0.591 | 0.105 | 0 |
| one section +4 bars (`bars_p4`) | 224 | 0.172 | 0.565 | 0.108 | 0 |
| one section ≥ 20 bars (`ood_long20`) | 224 | 0.169 | 0.558 | 0.132 | 0 |
| all lyrics replaced (`lyrics_all`) | 225 | 0.171 | 0.591 | 0.112 | 1 (trill, see above) |

Under a changed plan the output follows the new plan (its plan is further from every training plan,
0.558-0.565 vs 0.591) while melodic excess stays at the unedited level (0.169-0.172 vs 0.175). The
bar-copy rate rises slightly for the long-section requests (0.132 vs 0.105), consistent with the
added bars being filled with common, simple bar patterns; no informative phrase is shared with a
training song.

## Interpretation (not measured facts)

* Nearest-neighbour similarity of generations is within a few hundredths of real held-out songs on
  every representation. The only exact overlaps above the reference level are repeated-pitch and
  trill stretches, which any system that recites syllables on one pitch will "share" with the corpus.
  **We find no evidence that Qwen-ABC, MuPT or MIDI-LLM retrieve training songs.**
* The small positive excess of E3b (+0.01 to +0.02) and the larger one of MuPT (+0.04) are what a model
  that is somewhat more idiomatic, more repetitive, or more conservative than individual artists would
  show. They are not specific to one training song: excess is measured against each song's own
  nearest neighbour, and no neighbour stands out by an informative run. MuPT's larger rhythm and motif
  excess agrees with its long monotone stretches.
* Chord progressions are the most shared dimension for every system *and* for the references (chord
  excess ≈0.36-0.42): pop harmony is formulaic, so high chord similarity alone is not evidence of
  copying.
* Structure edits (`bars_p4`, `ood_long20`) produce plans absent from training while keeping melodic
  novelty at the unedited level. This is consistent with composing to the request rather than
  reproducing a stored song.

## Caveats

* One sample per song per system (first of four). Sample-to-sample variation of these statistics was
  not measured.
* Symbolic similarity only. Audio-level or lyric-melody pairing similarity was not tested.
* Normalized LCS on long songs has a high floor (background ≈0.39 for intervals), which is why excess
  and exact runs carry the argument.
* The references are pseudo-labels (automatic transcription). Transcription errors lower every
  similarity roughly equally for references and generations trained on them.

## Reproduction

```bash
sbatch scripts/paper_final/memorization.sbatch              # build-index (cached), query all systems, report
VALIDATE=1 sbatch scripts/paper_final/memorization.sbatch   # + shortlist-vs-brute-force check
python -m pytest -q tests/test_memorization.py
```
