# Joint vs cascaded lead-sheet generation (Table 5)

**Code:** `paper_eval/component_sota/joint_cascade.py`, which assembles melody-stage metrics from
`data/melody_samples_orig.parquet` and chord-stage metrics from `data/chords.parquet`.
**Tables:** `tables/joint_vs_cascade.*` and `tables/joint_vs_cascade_paired.*`.
**Figure:** `figures/fig6_joint_vs_cascade.*`.

A **system** is a (melody source, chord source) pair. Every stage uses seed S1 on the 225 test
songs. A system succeeds on a song only if both stages produce output.

| system | stages | description |
|---|---|---|
| **Qwen Joint (E3b)** | 1 | melody, lyrics alignment and chords in one pass |
| Cascade-Qwen | 2 | Qwen Melody-Only → Qwen Melody→Chord |
| hybrid | 2 | Qwen Melody-Only → AccoMontage2 |
| hybrid | 2 | CSL-L2M (official or retrained, chunked) → Qwen Melody→Chord |
| **Cascade-Specialized** | 2 | CSL-L2M (official or retrained, chunked) → AccoMontage2 |
| re-harmonized joint | 2 | E3b's melody → Qwen Melody→Chord, or → AccoMontage2 |

## MEASURED FACT

Values are song-level means [95% CI].

| system | success | PD | MD | lyric recall | exact structure | strong-beat chord-tone | chroma compat. | cadence I/V | s/song (mel + chords) |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen Joint (E3b)** | 1.000 | 0.473 | 1.57 | 0.972 | **0.991** | **0.688** [0.665, 0.709] | 0.521 | 0.676 | 5.4 + 0 (GPU) |
| Cascade-Qwen | 1.000 | **0.501** | **1.55** | **0.987** | 0.916 | 0.656 [0.637, 0.675] | **0.531** | **0.736** | 4.1 + 2.4 (GPU) |
| Qwen Mel → AccoMontage2 | 0.938 | 0.501 | 1.55 | 0.987 | 0.916 | 0.563 | 0.487 | 0.575 | 4.1 GPU + 12.2 CPU |
| CSL off. → Qwen Chord | 0.933 | 0.233 | 1.90 | 0.926 | n/a | 0.554 | 0.492 | 0.529 | 5.6 + 2.4 (GPU) |
| **Cascade-Specialized** (CSL off. → AM2) | 0.933 | 0.233 | 1.90 | 0.926 | n/a | 0.553 | 0.463 | 0.560 | 5.6 GPU + 12.2 CPU |
| CSL retr. → Qwen Chord | 0.938 | 0.300 | 1.67 | 0.931 | n/a | 0.564 | 0.476 | 0.541 | 7.3 + 2.4 (GPU) |
| CSL retr. → AM2 | 0.938 | 0.300 | 1.67 | 0.931 | n/a | 0.549 | 0.448 | 0.551 | 7.3 GPU + 12.2 CPU |
| E3b melody → Qwen Chord | 1.000 | 0.473 | 1.57 | 0.972 | 0.991 | 0.674 | 0.524 | 0.697 | 5.4 + 2.4 |

**Joint − Cascade-Qwen** (paired, N=225):

| metric | difference [95% CI] |
|---|---|
| exact structure | **+0.076 [+0.040, +0.116]** |
| strong-beat chord-tone | **+0.032 [+0.007, +0.057]** |
| PD | −0.028 [−0.057, +0.001] |
| DD | −0.019 [−0.036, −0.002] |
| lyric recall | **−0.015 [−0.024, −0.008]** |
| crammed syllables | **+0.049 [+0.036, +0.063]** (worse) |
| cadence I/V | **−0.060 [−0.104, −0.018]** |
| chroma compatibility | −0.010 [−0.023, +0.004] |

**Joint − Cascade-Specialized (CSL official → AM2):** Joint is better on every metric except roots
in key (−0.068, where AccoMontage2 is diatonic by construction):

| metric | difference [95% CI] |
|---|---|
| success | +0.067 [+0.036, +0.102] |
| PD | +0.239 [+0.202, +0.275] |
| MD | −0.325 |
| lyric recall | +0.046 |
| strong-beat chord-tone | +0.131 [+0.104, +0.158] |
| chroma compatibility | +0.059 |
| cadence I/V | +0.119 |

The one cost is crammed syllables: +0.173, because CSL-L2M places one character per note by construction.

**Joint − re-harmonizing its own melody with Qwen Melody→Chord:** no resolved difference on any
chord metric (strong-beat chord-tone +0.014 [−0.006, +0.035]). E3b's joint chords are as compatible
as a dedicated harmonizer's on the same melody.

## Cost

| | Joint | Cascade-Qwen | Cascade-Specialized |
|---|---|---|---|
| models | 1 × Qwen3.5-0.8B (752M) | 2 × Qwen3.5-0.8B (752M each) | CSL-L2M (92M) + AccoMontage2 (search, CPU) |
| training (this repo) | 4.0 GPU-h (2×L40S × 2.0 h, 758 updates, 89.5M tokens) | 4.0 (Melody-Only) + 2.3 (harmonizer) = 6.3 GPU-h | official: 0 here (pretrained elsewhere); retrained: 5.2 GPU-h (1×L40S) |
| inference, median s/song | 5.4 GPU (batch 16, amortized) | 4.1 + 2.4 = 6.5 GPU | 5.6 GPU + 12.2 CPU (per song, no batching) |
| failure modes | none (1.000) | none (1.000) | 3/4 songs unsupported by both stages; long songs need chunking |
| local section edit | yes, measured (`MELODY_INFILLING.md`) | Melody-Only was trained with infill examples but its infill was **not evaluated** | no |

Seconds are the median `seconds` field of the S1 outputs. They are not a controlled latency benchmark:
- Qwen numbers amortize batch-16 generation over the batch;
- CSL-L2M runs one song at a time, with up to 5 attempts;
- AccoMontage2 runs on CPU.

## INTERPRETATION

- **Joint generation is not free, but it is the best single system here.**
  - Against the matched Qwen cascade it trades structure and chord–melody agreement (+7.6 points
    exact structure, +3.2 points strong-beat chord-tone) for slightly less reference-like melodies,
    more cramming and fewer I/V cadences.
  - Neither dominates. Which is "best" depends on whether structural control or melody/lyric fidelity matters more.
- **Against the specialized cascade (CSL-L2M → AccoMontage2), the joint model wins by large margins
  on everything except diatonicity and cramming,** at similar per-song cost and with one model instead of two.
- The strongest hybrid, Qwen Melody-Only → Qwen Chord, is itself a Qwen system.
  - Every system with a CSL-L2M melody stage is worst on melody.
  - Every system with an AccoMontage2 chord stage is worst on compatibility and cadences.
- The claim "one general model replaces several specialized models" is supported on these automatic
  metrics at similar compute. It is **not** yet supported by listening data.
