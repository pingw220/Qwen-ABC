# Paper result summary (paper-final round, 2026-09-25)

Controllable full-song lead-sheet generation from pseudo-labelled music. All numbers are on the
frozen protocol (`EVALUATION_PROTOCOL.md`): 225 de-duplicated held-out Mandarin songs, T=1.0 /
top-p 0.95, song-level 95% bootstrap CIs (10,000 resamples), failures counted as failures.
"Single" = mean over 4 samples per song. **MF** = measured fact, **INT** = interpretation.

New this round: 19,706 Qwen/MuPT generations (E3b 8,966; E0 2,455; E1 2,455; E1-long 900;
MuPT 4,930), 957 MIDI-LLM runs, 16 edit→re-render renders, 62 listening-pilot renders; 75.9
GPU-h (45.4 L40S incl. 19.2 on checkpoint L40S, 30.5 L40; `COMPUTE_LEDGER.csv`).

---

### 1. Can pseudo-labelled data train reliable full-song lead-sheet generators?
**MF.** Qwen-ABC E3b, trained only on SheetSage-Pro pseudo-labels, produces a lead sheet for 225/225
songs; single-sample exact structure **0.982 [0.973, 0.991]**, lyric recall **0.970 [0.963, 0.976]**,
strict ABC validity **0.777 [0.743, 0.809]** (N = 225 songs × 4). On the 34 human-checked clean
songs its structure is 1.000 and validity 0.868 [0.794, 0.934]. — `MAIN_RESULTS.md`, `tables/model_comparison.*`
**INT.** Yes for structure and lyric text; validity needs selection (Q4); lyric placement (cramming)
remains below corpus quality (Q16).

### 2. Final Qwen / MIDI-LLM / MuPT results (single sample)
| | Qwen-ABC E3b | MuPT | MIDI-LLM |
|---|---|---|---|
| generation success | 1.000 | 1.000 | 0.884 [0.840, 0.924] (26 G2P refusals) |
| exact structure | 0.982 [0.973, 0.991] | 0.341 [0.304, 0.379] | 0.884 † (input, by construction) |
| lyric recall | 0.970 [0.963, 0.976] | 0.483 [0.467, 0.498] | 0.734 [0.695, 0.769] |
| valid output | 0.777 (strict ABC) | 0.203 (strict ABC) | 0.763 (own validator) |
| early termination | 0.001 | 0.308 | – |

Paired: Qwen − MuPT structure **+0.641 [+0.603, +0.678]**, lyric recall +0.488 [+0.472, +0.504];
Qwen − MIDI-LLM lyric recall **+0.236 [+0.202, +0.274]**. — `MAIN_RESULTS.md`

### 3. Single-sample decoding
**MF.** See Q2. Two independent 4-sample sets (legacy torch sampling vs the new paired-seed sampler)
agree: E3b structure 0.980 vs 0.982, validity 0.791 vs 0.777; MuPT structure 0.359 vs 0.341. —
`FAIR_DECODING.md`

### 4. How much of Qwen's best number comes from selector@4?
**MF.** The published 0.996 is selector@4 (legacy set; 1.000 on the new set). Single-sample
structure is 0.980-0.982, so selection adds **+0.016**. Selection mainly buys validity: 0.777-0.791 →
**0.969-0.978**, and cramming 0.162-0.173 → 0.114-0.116. Caveat: the selector scores
`section_plan_exact` and `lyric_recall` directly (legitimate at inference), so for those metrics
selector@4 = oracle@4. MuPT's published 0.689 is also selector@4 (single 0.341-0.359). —
`FAIR_DECODING.md`, `tables/decoding_comparison.*`

### 5. Oracle@4
**MF.** E3b oracle@4: structure 1.000, validity 0.973-0.982, lyric recall 0.992-0.994, cramming
0.086-0.092; oracle@8 validity 0.991. The selector is within 0.4-0.5 points of oracle@4 on validity.
MuPT oracle@4/@8 structure 0.729 / 0.862.

### 6. Does Explicit Structural State improve long-range reliability?
**MF.** At T1.0 (4 samples/song): E0 (no ESS) 0.552 → E1 (cleaning + ESS) 0.844: **+0.292
[+0.247, +0.338]**; late-section failure −0.063 [−0.079, −0.048]; premature EOS eliminated in every
song position. + more updates alone (E1-long − E1): +0.009 [−0.022, +0.040] (n.s.); + late-section
reconstruction at matched updates (E3b − E1-long): **+0.129 [+0.101, +0.157]**. On extrapolative
plans E0 falls to 0.276 while E3b holds 0.956 (Q13). — `ESS_ABLATION.md`, `tables/ess_ablation_T10.*`
**INT.** Yes; the effect is a training/representation effect, separate from selection.

### 7. Where in the song does the pre-ESS model fail?
**MF.** E0's structural failures are *spread across the song* (6-8% of sections already in the first
fifth, peak 11-12% at 60-80%; first-to-last change +0.035 [−0.002, +0.074], n.s.). What grows toward
the end is premature EOS (0 → 2.9% of last-fifth sections at T0.8), wrong labels (+0.036 [+0.015,
+0.062]) and lyric omission (+0.115 [+0.077, +0.157]). E3b's section failure is flat at 0.3-0.6% in
every fifth; its lyric omission and cramming still rise late (0.032 → 0.054; 0.160 → 0.230). —
`LONG_RANGE_ANALYSIS.md`, `figures/fig3_*`

### 8. Does the current model obey STRUCTURE interventions?
**MF.** One section ±2/±4 bars (224 songs × 2 seeds): target exact **0.991-1.000**; non-target
sections exact 0.997-0.998 (unmodified baseline 0.996); whole plan 0.978-0.980. The change is local:
the target section's contour/rhythm/chords move +0.035-0.039 beyond reseeding (−4 bars) while the
sections before and after stay at the reseed floor. MuPT: 0.81-0.83 target, 0.74-0.79 elsewhere.
— `INTERVENTIONAL_CONTROLLABILITY.md` Table 5a
**INT.** Yes; this is controllability, not only reconstruction.

### 9. SECTION-LABEL interventions?
**MF.** verse/chorus → bridge: label 0.998 [0.993, 1.000], duration kept 0.996; verse ↔ chorus:
1.000 / 0.998. Musical content of a relabelled section changes little (target contour +0.012 beyond
reseeding). **INT.** The label is obeyed; no claim that a "bridge" sounds like a bridge.

### 10. KEY interventions?
**MF.** +2 / +5 / −3 semitones: declared key 1.000; melody time in the new key's scale 0.971 / 0.974
/ 0.965; chord roots in the new key 0.92-0.93; pitch-class distribution moves far beyond reseeding
(JS +0.20 / +0.08 / +0.34). But mean melody pitch shifts only **+0.5 to +0.8 semitones** and the
output is no closer to the transposed original than a reseed is (0.924-0.935 vs 0.925). —
Table 5a/5b. **INT.** Key is obeyed harmonically and tonally by re-composing in the new key at the
same register, not by transposing; a direct symbolic transposition is the exact edit (Q17).

### 11. TEMPO interventions?
**MF.** ×0.8 / ×1.25: tempo header 1.000 / 1.000; bar count kept 0.989 / 0.984; plan exact
0.984 / 0.973; melody/rhythm change at the reseed floor (rhythm +0.005 [+0.000, +0.011] at ×0.8).
MIDI-LLM takes tempo as input. **INT.** Tempo is followed as a marking, not as a compositional
condition — which is the desired behaviour for a lead sheet.

### 12. Does changing lyrics affect melody beyond reseeding noise?
**MF. No, not measurably.** Whole-song lyric replacement (225 songs × 4 seeds): new lyrics sung
0.983 [0.979, 0.986], old lyric at chance (0.116 vs 0.115); melody change beyond reseeding ≤ 0.011
on every distance (melody +0.003 [+0.000, +0.005], chord changes +0.011) vs +0.19-0.34 for a key
change; seed-paired ratio d(A,B)/d(A,C) **0.997 [0.990, 1.004]**. MIDI-LLM (exact seed pairing):
0.980 [0.968, 0.992]; MuPT: all effects n.s. Mandarin tone-melody agreement: references follow
their own lyric's tones slightly (+0.026 [+0.013, +0.038]); E3b −0.001 [−0.006, +0.004], MuPT
+0.001 / −0.004 (n.s.). — `INTERVENTIONAL_CONTROLLABILITY.md` Table 5b/5c
**INT.** Lyrics are followed as realization constraints, not as a compositional condition; the
historical "lyric swap ≈ reseed" diagnostic holds for the current best checkpoint.

### 13. How does control degrade ID → compositional → extrapolative OOD?
**MF.** Exact plan (seed S1): E3b **0.991 → 0.984 → 0.956**; E1 0.853 → 0.900 → 0.806; E0 0.542 →
0.565 → **0.276**; MuPT 0.338 → 0.296 → 0.231. E3b's only resolved drop is +6 chorus repeats
(−0.121 [−0.165, −0.080]; section counts up to 23 vs corpus max 17: 0.76 exact above 17 vs 0.93
below), with no section-count errors. — `OOD_GENERALIZATION.md`, `figures/fig5_*`
**INT.** Compositional recombination is easy; ESS makes extrapolation robust up to section counts
beyond the training range.

### 14. Are generated songs unusually similar to training songs?
**MF.** Exact search against all 10,243 training songs: nearest-neighbour interval excess E3b
+0.017 [+0.012, +0.023] above held-out references (MuPT +0.037, MIDI-LLM −0.008); informative-run
flags 1 of 1,322 generations, and that one is a two-note trill; a held-out *reference* shares a longer
real phrase with training (30 intervals) than any generation. Structure-edited samples follow novel
plans with unchanged melodic excess. — `MEMORIZATION_ANALYSIS.md`
**INT.** No evidence of retrieval or copying.

### 15. What fraction of apparent error is likely pseudo-label error?
**MF.** E3b single-sample structure errors: 1.8% overall, **0** on the 34 clean songs and 0 on the 42
songs that pass every quality rule (dose-response ρ = +0.14 [+0.02, +0.26]). Upper bound on the
label-attributable share of strict-validity errors: 0.41 [0.06, 0.70]; lyric recall 0.24 [−0.04,
0.46]. For MuPT/E0/E1 most error is model error (MuPT still 0.58 failure on rule-clean songs). No
human re-annotation exists; a 100-song package is prepared. — `ANNOTATION_NOISE.md`
**INT.** E3b's residual structure error is plausibly label noise; its validity and cramming errors
are only partly so. The clean subset is also simpler, so these are upper bounds.

### 16. Dominant model failure modes
**MF.** E3b single samples: syllable cramming beyond the corpus 95th percentile 45%, melody outside
the corpus 1-99% band on repetition/range 22%, local syntax errors 13% (bar-duration mismatch,
lyric overflow, orphan melisma, broken ties), structure ~1%; 37% of samples carry no tag. MuPT: wrong
section count 55%, premature EOS 31%, lyric omission 100%. MIDI-LLM: lyric omission 62%, refusals
12%. — `FAILURE_TAXONOMY.md`, `LYRIC_FAILURES.md`

### 17. Does the edit → re-render pipeline work?
**MF.** 2 held-out songs × 4 edits, 16/16 renders through FastSinger + MuseControlLite, no upstream
code changed. Local edits: chorus 8 → 12 bars by section infill (12 bars on 8/8 seeds, all other
sections byte-identical), one section's lyrics by infill (new-lyric recall 1.0 on 7/8 seeds), key +5
by direct transposition (every note +5), tempo ×1.25 by the `Q:` field (render length 0.8017 of the
source). Regeneration variants: whole-song regeneration rewrites most other sections; key
regeneration failed on one song (declared A major, melody in neither key). — `EDIT_RERENDER.md`
**INT.** The symbolic lead sheet works as an editable interface; the reliable edits are local
(infill) or direct symbolic ones. No audio-quality claim.

### 18. Fully paper-ready results
Q1-Q16 and the corresponding tables/figures: complete N, CIs, frozen protocol, committed code.
Q17 is a demonstration (N = 2 songs).

### 19. Results that require human evaluation
Musical quality/plausibility of any system; whether edits are perceived as intended
(`listening_study/`, 66 items, 4 counterbalanced lists, 16-item audio pilot rendered — **responses
PENDING**); human re-annotation to separate label from model error (`human_annotation_package/`,
100 songs — **PENDING**). No human results were fabricated.

### 20. Strongest defensible claims
1. A general-purpose 0.8B LM finetuned on automatically transcribed lead sheets writes complete,
   parseable Mandarin lead sheets that follow a requested section plan in 98% of single samples and
   sing 97% of the requested lyrics, on held-out, de-duplicated songs.
2. Writing the structural state into the sequence (ESS) plus a late-section reconstruction objective
   is what makes long-range structure reliable (+0.29 and +0.13 exact structure, both far outside
   seed noise), including on extrapolative plans (0.956 vs 0.276 without ESS).
3. The model is **interventionally** controllable on structure length, section labels, key and tempo:
   single-control changes are realized essentially exactly and locally.
4. Lyrics are realized, not composed for: replacing the lyrics changes the melody no more than
   reseeding and the melody ignores lexical tones — true of all three systems here.
5. In our setting, symbolic-music pretraining (MuPT) was not sufficient to acquire this
   instruction-like control: musically comparable output, far lower plan and lyric adherence.
6. The models do not copy training songs; residual structural error of the best model coincides with
   pseudo-label problems.
Selector@4 is an inference-time add-on (+0.016 structure, +0.19 validity) and should be reported as
such, next to single-sample numbers.
