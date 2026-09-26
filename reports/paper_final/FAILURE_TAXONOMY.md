# Failure taxonomy

Automatic, multi-label tags on every single sample (4 per song × 225 songs per system). Table
`tables/failure_taxonomy.*`; per-sample tags `data/failure_tags.parquet`; tag co-occurrence for E3b
`data/failure_cooccurrence_e3b.csv`; representative examples (ids + metrics, no lyrics)
`data/failure_examples.json`; reference bands `data/reference_bands.json`; code
`paper_eval/failures.py`.

Thresholds come from the held-out references wherever possible, so a tag means "outside what real
songs in this corpus do": syllable cramming = above the reference 95th percentile; degenerate
melody = distinct bars or pitch range below the reference 1st percentile, or repeated-pitch
intervals above its 99th; extreme pitch range = above the reference 99th percentile; repetitive
loop = more identical consecutive bars than the reference 99th percentile (min. 4); lyric omission
= recall < 0.9; lyric duplication = precision < 0.95; lyric order failure = section-local recall
more than 0.1 below global recall; invalid syntax = any non-lyric parser error (MIDI-LLM: its own
validator); suspected annotation error = a control failure on a song that fails ≥ 3 of the 12
pseudo-label quality rules.

## Share of samples with each tag (song-level 95% CI)

| tag | Qwen-ABC E3b | MuPT | MIDI-LLM | Qwen E0 (no ESS) |
|---|---|---|---|---|
| no tag at all | **0.370** | 0.000 | 0.122 | 0.253 |
| front-end refusal | 0 | 0 | 0.116 | 0 |
| parse failure | 0 | 0 | 0 | 0 |
| invalid syntax | 0.127 [0.101, 0.153] | 0.203 | 0.121 ‡ | 0.168 |
| wrong section count | 0.006 | **0.546** | 0 † | 0.020 |
| wrong section label (count right) | 0.004 | 0.088 | 0 † | 0 |
| wrong bar count | 0.008 | 0.098 | 0 † | **0.428** |
| premature EOS | 0.001 | **0.308** | 0 | 0.012 |
| excess continuation | 0.001 | 0.238 | 0 | 0.008 |
| lyric omission (recall < 0.9) | 0.056 | **1.000** | **0.623** | 0.038 |
| lyric duplication | 0.014 | 0.998 | 0.367 | 0.023 |
| lyric order failure | 0.002 | 0.198 | 0 | 0.006 |
| syllable cramming (> corpus p95) | **0.454 [0.418, 0.492]** | 0.489 | 0 § | 0.362 |
| timing inconsistency | 0.097 | 0.104 | 0 | 0.092 |
| degenerate melody | 0.220 [0.188, 0.252] | 0.223 | 0.010 | 0.200 |
| extreme pitch range | 0 | 0 | 0.006 | 0 |
| repetitive loop | 0.010 | 0.017 | 0 | 0.004 |
| missing chords | 0.003 | 0.006 | 0 | 0.003 |
| suspected annotation error | 0.128 | 0.240 | 0.153 | 0.180 |

† structure is an input for MIDI-LLM. ‡ own validator. § its conversion never joins syllables.

**What "invalid syntax" is for E3b** (900 samples, 201 non-strict): bar-duration mismatch 82,
lyric overflow 76, orphan melisma 48, broken tie 36, lyrics without a music line 17, lyric-bar
overflow 15. For MuPT (717/900 non-strict) it is overwhelmingly lyric alignment: lyric overflow 618,
orphan melisma 274, lyric-bar overflow 141.

## Dominant failure modes per system

* **Qwen-ABC E3b**: syllable cramming (45% of samples beyond the corpus 95th percentile),
  melodic narrowness/repetition (22% outside the corpus 1-99% band), and local syntax/timing slips
  (13% / 10%). Structural failures are ~1%, and a third of samples carry no tag at all.
* **MuPT**: plan failures (wrong section count 55%, premature EOS 31%, excess continuation 24%) and
  lyric failures (every sample below 0.9 recall); every sample carries at least one tag.
* **MIDI-LLM**: lyric omission (62%), refusals (12%) and duplicated syllables from its pointer
  stream (37%); melodically the least degenerate.
* **Qwen E0 (no ESS)**: wrong bar counts (43%) — the failure ESS removes.

## Representative examples

`data/failure_examples.json` lists up to three (song, seed) examples per tag and system, with their
metrics; the generations themselves are under `experiments/paper_final/gen/<model>/orig/` (git-ignored,
they contain lyrics). E.g. E3b `05Yqjul6jLsld0Vh8K36KH` S1-S3: exact plan, recall 0.88-0.97, cramming
0.26-0.37 — the typical E3b failure: the requested plan is followed, the lyrics are all there, too
many of them share notes.
