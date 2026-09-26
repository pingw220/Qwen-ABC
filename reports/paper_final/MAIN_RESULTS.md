# Main results: Qwen-ABC vs MuPT vs MIDI-LLM

225 de-duplicated test songs, T=1.0 / top-p 0.95 (MIDI-LLM: its own sampler, recommended mode-A
constraints), 4 samples per song. Table 2 = `tables/model_comparison.*` (single sample and
selector@4, with paired differences); code `paper_eval/decoding.py`, `paper_eval/main_tables.py`.
Failures count as 0 in every rate.

## Table 2 (single sample = mean of 4; 95% song-level CI; paired differences)

| metric | Qwen-ABC E3b | MuPT | MIDI-LLM | Qwen − MuPT | Qwen − MIDI-LLM |
|---|---|---|---|---|---|
| generation success | 1.000 | 1.000 | 0.884 [0.840, 0.924] | 0 | +0.116 [+0.076, +0.160] |
| valid output (strict ABC / own validator ‡) | 0.777 [0.743, 0.809] | 0.203 [0.172, 0.236] | 0.763 [0.717, 0.807] ‡ | +0.573 [+0.536, +0.612] | (not comparable) |
| exact structure | **0.982 [0.973, 0.991]** | 0.341 [0.304, 0.379] | 0.884 † | **+0.641 [+0.603, +0.678]** | (by construction) |
| section count correct | 0.994 | 0.454 | 0.884 † | +0.540 | |
| label sequence correct | 0.990 | 0.367 | 0.884 † | +0.623 | |
| sections exact (per section) | 0.995 | 0.791 | 0.884 † | +0.204 | |
| \|total bars − request\| | 0.21 | 12.05 | 0 † | | |
| early termination | 0.001 | 0.308 [0.273, 0.342] | — | −0.307 | |
| countdown consistency | 0.999 | 0.996 | — | +0.004 | |
| **lyric recall** | **0.970 [0.963, 0.976]** | 0.483 [0.467, 0.498] | 0.734 [0.695, 0.769] | **+0.488 [+0.472, +0.504]** | **+0.236 [+0.202, +0.274]** |
| lyric precision | 0.992 | 0.612 | 0.949 | +0.379 | +0.044 [+0.037, +0.049] |
| lyrics exactly as requested | 0.290 | 0.000 | 0.000 | +0.290 | +0.290 |
| section-local lyric recall | 0.969 | 0.428 | 0.726 | +0.541 | +0.243 |
| crammed syllables | 0.173 | 0.175 | 0 § | −0.002 [−0.013, +0.009] | § |
| wordless notes | 0.035 | 0.101 | 0.035 | −0.065 | −0.002 |
| pitch range | 17.49 | 17.57 | 24.71 | −0.07 [−0.47, +0.32] | −7.32 |
| notes per bar | 3.24 | 3.87 | 3.43 | −0.63 | −0.25 |
| chord-tone agreement | 0.657 | 0.677 | 0.626 | −0.020 | +0.035 |
| melody in declared key | 0.971 | 0.966 | 0.970 | +0.005 (n.s.) | +0.002 (n.s.) |
| distinct bars | 0.780 | 0.769 | 0.938 | +0.010 (n.s.) | −0.163 |

† MIDI-LLM's section count, bars per section, tempo and bar grid are **inputs** (a click track) and
pinned by its decoder: exact on all 199 generated songs; 0.884 = 199/225. This is *externally
constrained structure*, not learned plan adherence (with free labels, mode B, it collapses to 0.027
in round 2). ‡ MIDI-LLM validity is its own validator. § MIDI-LLM's lead-sheet conversion never
attaches two syllables to one note, so cramming is not comparable.

With selector@4 (same table, second block): Qwen structure 1.000 / validity 0.969 / lyric recall
0.980; MuPT 0.680 / 0.422 / 0.538; MIDI-LLM 0.884† / 0.871‡ / 0.788.

## Measured facts

1. Qwen-ABC is the only system that generates a lead sheet for every song, follows the requested
   plan in a free-running decoder (0.982 single sample), and sings nearly all requested lyrics (0.970).
2. MuPT, trained on the same data with the same recipe and decoding, writes musically comparable
   material (pitch range, key, chord density within CIs of Qwen) but follows the plan in only 34% of
   samples, stops early in 31%, and sings under half of the lyrics.
3. MIDI-LLM refuses 26/225 songs at its G2P front end (code-switched English), sings 73% of the
   lyrics, and has the most corpus-like melodic interval statistics; its structure is exact because
   it is given.

## Interpretation (not causal)

Qwen and MuPT differ in tokenizer (MuPT's vocabulary was extended with the corpus's characters),
architecture, pretraining corpus and objective, and update count (1,070 vs 758, in MuPT's favour).
The safe statement is: **in our setting, symbolic-music pretraining was not sufficient to acquire
the instruction-like control this task requires**, while a general-purpose LM finetuned the same way
acquired it. Structure failures of MuPT are on ASCII plan tokens, so fresh Chinese embeddings alone
do not explain them (R4 report).
