# Melody section infilling vs whole-song regeneration

**Model:** E3b as trained (`/gscratch/ark/pingw220/qwen_abc_r2_offload/runs/e3b_201131/final_model`); **no infill model was trained**,
because E3b's training mixture already contains section infill.

**Prompt:** the infill prompt is byte-identical to E3b's training prompts. We verified this on
224/224 songs by rebuilding the recon_late prompts and comparing them with the training JSONL.

**Decoding:** T=1.0, top-p 0.95, paired seeds S1 and S2.

**Code:** `paper_eval/component_sota/infill.py` generates the tasks; `infill_eval.py` scores them.
**Tables:** `tables/infill.{csv,md,tex}` and `tables/infill_vs_regen.{csv,md,tex}`.
**Per-sample data:** `data/infill.parquet`.
**Raw outputs:** `experiments/component_sota/infill/qwen_e3b/<kind>/<song>_<seed>.json` (2,038 generations).

## Tasks (225 held-out songs)

| kind | what is asked | N songs |
|---|---|---|
| recon_late | Reconstruct a held-out **later-half** reference section; this is the position E3b was trained on. | 224 |
| recon_early | Reconstruct a **first-half** reference section; E3b never trained on these positions. | 129 |
| edit_resample | Rewrite one section of E3b's own S1 draft, with the same plan and lyrics. | 222 |
| edit_lyrics | Rewrite one section of the draft with **new lyrics** (paper-final `lyrics_sec` intervention). | 222 |
| edit_extend | Rewrite one section of the draft **4 bars longer** (paper-final `bars_p4` intervention). | 222 |

**Comparator:** whole-song regeneration for the same request. This is the paper-final E3b
generation under the same intervention, from the same prompt, with the whole song resampled:
`orig` S2/S3 for resample, `lyrics_sec` S1/S2, `bars_p4` S1/S2.

## MEASURED FACT

Untouched sections are those other than the edited one. Song-level means, 95% bootstrap CI, N=222 songs.

| edit | untouched sections byte-identical, infill | same, regeneration | untouched notes identical, infill | same, regeneration | untouched chords identical, infill | same, regeneration |
|---|---|---|---|---|---|---|
| resample | **1.000** [1.000, 1.000] | 0.001 [0.000, 0.003] | 0.977 [0.970, 0.984] | 0.163 [0.147, 0.179] | 0.984 | 0.005 |
| new lyrics | **0.999** [0.997, 1.000] | 0.029 [0.024, 0.035] | 0.976 [0.969, 0.983] | 0.168 [0.153, 0.183] | 0.984 | 0.035 |
| +4 bars | **0.997** [0.993, 1.000] | 0.050 [0.044, 0.056] | 0.974 [0.965, 0.981] | 0.178 [0.162, 0.193] | 0.980 | 0.059 |

Melody change in the untouched sections (1−LCS) is 0.007–0.009 for infill and 0.90–0.96 for
regeneration. The paired difference is −0.95 [−0.953, −0.943] for resample.

**Target success.** This is where infill pays, and the effects are small:

| edit | target label+bars, infill | same, regeneration | paired diff | target lyric recall, infill | same, regeneration | paired diff |
|---|---|---|---|---|---|---|
| resample | 0.989 | 0.993 | −0.005 [−0.018, +0.007] | 0.962 | 0.975 | −0.013 [−0.031, +0.003] |
| new lyrics | 0.984 | 0.993 | −0.009 [−0.025, +0.005] | **0.927** | **0.982** | **−0.054 [−0.083, −0.028]** |
| +4 bars | 0.971 | 1.000 | **−0.029 [−0.047, −0.014]** | 0.963 | 0.975 | −0.012 [−0.025, +0.001] |

**Boundaries.** The absolute pitch jump into the target is 4.2–4.5 semitones for infill and
4.0–4.4 for regeneration. Reference songs have 4.52 at their own section boundaries. The rest
before the target is 6.2–7.9 beats for infill, 5.2–6.3 for regeneration and 5.22 for the
reference. The infill targets are in key 0.957–0.968 of the time, the same as regeneration
(0.962–0.969).

**Reconstruction.**
- recon_late reproduces label and bars 1.000 of the time, with lyric recall 0.985.
- recon_early (untrained positions) reproduces label and bars 0.965 [0.930, 0.992], with lyric recall 0.933.
- Interval similarity to the held-out reference section is 0.624 late vs 0.569 early.
- Strict validity of the assembled song is 0.821 late vs 0.771 early.

**Why "notes identical" is 0.97 while "bytes identical" is 1.00.** Notes are compared as
(onset, duration, pitch, melisma) within the section. A tie that crosses a section boundary
changes the last or first note of the neighbouring section without changing its text. In
2,850 untouched sections of `edit_resample`, all 61 sections with identical text but
different notes (2.1%) are **adjacent to the edited section**; none is further away.

## INTERPRETATION

- For local edits, infill gives the locality guarantee that regeneration cannot. Regeneration
  rewrites about 84% of untouched notes and about 95% of untouched chords, even though every
  other control is unchanged.
- The price is a small, measurable loss on the target:
  - the new lyric is realized less completely (−5.4 points of recall);
  - extending by 4 bars fails more often (−2.9 points).
  - Both are consistent with a conditional task that sees fixed context on both sides.
- Infill transfers to first-half positions that were never trained, at about 3.5 points lower
  structural success. It is a genuine capability, not memorized positions.
- **Framing recommendation:** an editability demonstration with a strong, cleanly measured
  locality result. It should not be framed as a new method: there is no infill-specific
  training beyond the E3b mixture.
- Boundary smoothness is measured only by pitch-jump and rest-gap proxies, which are in the
  reference range. Whether seams *sound* natural is listening-study block D, still pending.
