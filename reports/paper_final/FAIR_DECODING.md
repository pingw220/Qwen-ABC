# Fair decoding: single sample vs selector@4 vs oracle@4

Model quality is reported separately from search. Tables `tables/decoding_comparison.*` (Table 3),
`tables/decoding_deltas.*`, `tables/decoding_cost.*`; per-song data `data/decoding_per_song.json`;
code `paper_eval/decoding.py`. 225 test songs; failures count as 0 in rates; song-level 95%
bootstrap (10,000).

* **single** — mean over the song's 4 samples (the expected score of one sample).
* **selector@4** — the repo's `sum` rule (`scripts/select_best_of_n.py`), chosen on 64 validation
  songs; uses only prompt + output. It scores `section_plan_exact` and `lyric_recall` directly, so
  for those two metrics **selector@4 ≡ oracle@4 by construction**.
* **oracle@k** — per metric, the best of k samples (analysis upper bound).

## Table 3 (headline metrics)

| system | mode | strict validity | exact structure | lyric recall | crammed syllables |
|---|---|---|---|---|---|
| **Qwen-ABC E3b** (paired-seed ×4) | single | **0.777 [0.743, 0.809]** | **0.982 [0.973, 0.991]** | **0.970 [0.963, 0.976]** | 0.173 |
| | selector@4 | 0.969 [0.942, 0.991] | 1.000 | 0.980 | 0.116 |
| | oracle@4 | 0.973 | 1.000 | 0.992 | 0.092 |
| Qwen-ABC E3b (legacy ×4, R3-B) | single | 0.791 [0.759, 0.822] | 0.980 [0.969, 0.990] | 0.972 | 0.162 |
| | selector@4 (**the published 0.996**) | 0.978 [0.956, 0.996] | 0.996 [0.987, 1.000] | 0.982 | 0.114 [0.103, 0.127] |
| | oracle@4 | 0.982 | 1.000 | 0.994 | 0.086 |
| | oracle@8 (both sets pooled) | 0.991 | 1.000 | 0.996 | 0.067 |
| **MuPT** (paired-seed ×4) | single | **0.203 [0.172, 0.236]** | **0.341 [0.304, 0.379]** | 0.483 [0.467, 0.498] | 0.175 |
| | selector@4 | 0.422 | 0.680 [0.618, 0.742] | 0.538 | 0.133 |
| | oracle@4 / oracle@8 | 0.538 / 0.764 | 0.729 / 0.862 | 0.583 / 0.618 | 0.081 / 0.054 |
| MuPT (legacy ×4, R4) | single / selector@4 | 0.232 / 0.467 | 0.359 / **0.689** (published) | 0.490 / 0.539 | 0.173 / 0.146 |
| **MIDI-LLM** (mode A ×4) | single | 0.763 [0.717, 0.807] ‡ | 0.884 † | 0.734 [0.695, 0.769] | 0 § |
| | selector@4 / oracle@4 | 0.871 / 0.871 ‡ | 0.884 † | 0.788 / 0.800 | 0 § |
| E1-long (ESS) | single / selector@4 / oracle@4 | 0.700 / 0.938 / 0.938 | 0.853 / 0.978 / 0.996 | | |
| E1 (ESS) | single / selector@4 / oracle@4 | 0.574 / 0.920 / 0.924 | 0.844 / 0.969 / 1.000 | | |
| E0 (no ESS) | single / selector@4 / oracle@4 | 0.609 / 0.947 / 0.947 | 0.552 / 0.764 / 0.889 | | |

† MIDI-LLM structure is an input (click grid), exact on every generated song; 0.884 = 199/225 songs
generated. ‡ its own validator. § its converter never places two syllables on one note.

## Measured facts

1. **The published 0.996 is selector@4; the single-sample number is 0.980-0.982.** Selection adds
   only +0.016 structure for E3b because the model is already at ceiling. What selection buys for E3b is
   **validity** (0.78-0.79 → 0.97-0.98) and cramming (0.16-0.17 → 0.114-0.116; oracle@4 0.086-0.092, corpus 0.057).
2. **Two independent 4-sample sets agree** (legacy torch sampling vs the paired-seed sampler):
   E3b single structure 0.980 vs 0.982, validity 0.791 vs 0.777; MuPT 0.359 vs 0.341.
3. **Selection matters much more for weaker models**: E1 +0.125, E0 +0.212, MuPT +0.339 structure.
   MuPT's published 0.689 is a selector@4 figure; its single-sample structure is 0.34 and its
   single-sample validity 0.20.
4. **Selector ≈ oracle** for validity (E3b 0.969 vs 0.973; 0.978 vs 0.982): the inference-time rule
   is within 0.4-0.5 points of what 4 samples allow. oracle@8 raises E3b validity to 0.991.
5. **Cost**: E3b 4.7-6.5 GPU-s per sample on an L40S at batch 16 (1.2-1.6 GPU-h for 4 samples of all
   225 songs); MuPT 19-21 GPU-s per sample (4.8-5.4 GPU-h, mostly L40); MIDI-LLM 87 s median per
   song process (one song per process; 24.5 GPU-h for its 957 paper-final tasks).

## Interpretation

For the paper, report E3b single-sample numbers as the model result (structure 0.982, validity
0.777, lyric recall 0.970) and selector@4 as an inference-time procedure (structure 0.996-1.000,
validity 0.969-0.978). The structural claim does not depend on selection; the validity claim does.
