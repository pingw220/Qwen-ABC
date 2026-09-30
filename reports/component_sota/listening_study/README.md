# Component-round listening study (package only)

> **NO HUMAN RESULTS HAVE BEEN COLLECTED.** This directory holds a ready-to-run, blinded study
> design. Every number in this round's reports comes from automatic metrics. Nothing here may be
> cited as a listener preference until responses exist and `analyze` has been run on them.

**Code:** `paper_eval/component_sota/listening_study.py` (`build`, `prep-render`, `render`, `stage`, `analyze`).
**Tests:** `tests/test_component_sota_listening.py` checks the following:
- participant files contain no system names;
- A/B sides are balanced;
- the analysis runs on simulated data (written only to a scratch path, labelled SIMULATED).

## Blocks and questions

For each 2AFC question the choices are A, B or no difference. Block D is rated 1–5.

| block | comparison | questions | test items |
|---|---|---|---|
| A: Lyrics→Melody | pairs of melody sources on the same song, both sung by the same SVS (the renderer is chosen by hash, not quality). Pairs: Melody-Only vs CSL-official, Melody-Only vs E3b, E3b vs CSL-official, CSL-official vs CSL-retrained, Melody-Only vs reference | fits the lyrics better? · more coherent? · more natural phrasing? · stronger phrase/section structure? | 30 (6 per pair) |
| B: SVS robustness | FastSinger vs SoulX-Singer on the same melody, for 5 melody sources; a piano render of the target melody is played first | clearer singing? · more natural? · follows the target melody better? | 20 (4 per source) |
| C: Harmonization | backing mixes with the same melody and vocal; chords from Qwen / AccoMontage2 / reference | accompaniment fits the melody better? · more harmonically coherent? · more natural? | 18 (6 per pair) |
| D: Infilling locality | the ORIGINAL E3b draft vs an EDITED version, where one section is rewritten by section infill or by whole-song regeneration (which method is blinded). The clip spans the section before, the target and the section after | does the edited section connect naturally? (1–5) · did unrelated sections stay consistent? (1–5) | 24 |
| E: Section semantics | original vs relabelled sample (verse→chorus / →bridge); same lyrics and bars | which version of the marked section sounds more chorus-like / bridge-like? | 12 |
| QC | 2 identical pairs (expect "no difference") and 2 pairs where one clip is degraded (expect the clean one) | as the source block | 4 |

## Status

- 108 items: 104 test and 4 QC.
- **102 of 104 test items have audio; 2 are `pending_audio`.** One SoulX render fails on a character
  missing from its phoneme set; one FastSinger render fails on a CSL-L2M phrase exceeding its maximum
  input length. They stay in the design; items are never re-drawn by availability.
- Songs are drawn from the frozen 36-song audio subset (`../audio/subset.json`) by a salted hash.
- Four counterbalanced lists (`lists/list_{0..3}.csv`): within every (block, pair) stratum each item
  appears with system X on side A in two lists and on side B in the other two. Trial order is
  shuffled per list (seed 20260927); blocks stay contiguous.
- Stimuli are 30 s excerpts, or the section window for D/E, with 50 ms fades. They are staged under
  `/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio/listening_study/stimuli/`, with
  opaque clip IDs. **No audio is in git.**
- D/E vocals were rendered with FastSinger from the E3b draft, the infilled, regenerated and
  relabelled songs (`listening_render/`, 54 renders).

## Files

| file | participant-facing? |
|---|---|
| `lists/list_k.csv`, `questions.json` | yes (no system names, no song IDs) |
| `response_schema.json`, `response_template.csv` | yes (for the collection tool) |
| `design.json` | no (counts only) |
| `KEY_DO_NOT_SHARE.json` | **no**: item → systems, songs, A/B assignment per list |
| `STAGING_DO_NOT_SHARE.csv` | **no**: clip ID → source audio path |

## Analysis plan

Preregistered in code: `listening_study.py analyze --responses R.csv`.

1. **Rater exclusion:** exclude raters with more than 1 QC error.
2. **Per block and question:** P(prefer system 1), with "no difference" counted as 0.5. The 95%
   CIs come from a song-cluster bootstrap and, separately, a rater-cluster bootstrap (10,000
   resamples each); both are reported.
3. **Block D:** mean ratings with the same bootstrap; infill vs regenerate paired by song.
4. **Inter-rater reliability:** Krippendorff's α (nominal) and Fleiss' κ per block.
5. **Power note:** with 12 raters × 6 items per pair (72 judgements per pair), a true preference of
   65% is detected with only about 73% power at α = 0.05 (two-sided normal approximation, ignoring
   clustering). Clustering by song lowers this further. We recommend 16 or more raters, which gives
   96+ judgements per pair and about 85% power before clustering.
