# Component-SOTA + decomposition evaluation (2026-09)

This round tests whether Qwen-ABC's components are competitive with specialized systems. Each part
of the lead-sheet pipeline is evaluated on its own, then the parts are recombined.

The parts are:
- lyrics→melody;
- melody section infilling;
- melody→chord;
- joint vs cascaded generation;
- two singing-voice renderers;
- backing audio;
- closed-loop audio control;
- section-label semantics.

All symbolic results use the 225 held-out test songs of the canonical split (10,243 / 273 / 225).
Audio results use a frozen, stratified 36-song subset.

**Start with [`FINAL_COMPONENT_SOTA_REPORT.md`](FINAL_COMPONENT_SOTA_REPORT.md)**, which answers
the 23 questions of the brief. For how the paper should use these results, see
[`PAPER_INTEGRATION_RECOMMENDATION.md`](PAPER_INTEGRATION_RECOMMENDATION.md).

## Systems

| name in tables | what it is | checkpoint / code |
|---|---|---|
| Qwen Full (E3b) → melody | the paper model (joint melody + chords + lyrics); chords stripped for melody-only comparisons | `/gscratch/ark/pingw220/qwen_abc_r2_offload/runs/e3b_201131/final_model` |
| Qwen Melody-Only | Qwen3.5-0.8B-Base trained on E3b's mixture without chords; same recipe and 758 updates (equal-update control) | `experiments/component_sota/runs/mel_sft/final_model` (`configs/mel_sft.yaml`) |
| Qwen Melody→Chord | Qwen3.5-0.8B-Base harmonizer: melody read-only in the prompt, only chord lines generated; 2 epochs (468 updates) | `experiments/component_sota/runs/chord_sft/final_model` |
| Qwen Melody+Lyrics→Chord | same, with the lyric lines kept in the prompt; 590 updates | `experiments/component_sota/runs/chord_lyr_sft/final_model` |
| CSL-L2M official (EXTERNAL PRETRAINED SYSTEM COMPARISON) | the authors' released lyrics-only checkpoint, trained on **their** corpus and run through their unmodified code; whole-song ("as released") and section-chunked (≤ 24 lines per call) | `/gscratch/ark/pingw220/third_party/csl_l2m_ckpt/pretrained_CSLL2M_onlyLyrics.pt` |
| CSL-L2M retrained | the same architecture and trainer, retrained on **our** training split (9,681 4/4 songs; about 29k steps) | `experiments/component_sota/runs/csl_l2m_retrain/params/step_29000-RC_0.581.pt` |
| AccoMontage2 | official `demo_SOME.py` (POP_STANDARD, 1 chord/bar, given key); melody→chord output only | `MIDI-SAG/AccoMontage2` |
| diatonic baseline | per bar, the diatonic triad that covers the most melody duration | `chord_eval.diatonic` |
| SVS-A FastSinger / SVS-B SoulX-Singer | the project's SVS, and MIDI-SAG's SoulX-Singer with its shipped zh prompt | see `SVS_CROSS_RENDER.md` |
| backing renderer | MIDI-SAG adapter + MuseControlLite (`--harmonizer leadsheet`) | see `AUDIO_CONTROL_EVALUATION.md` |

Decoding for every Qwen model: T=1.0, top-p 0.95, paired Gumbel seeds S1–S4
(`paper_eval/sampler.py`). CSL-L2M is seeded per (song, seed). AccoMontage2 is deterministic.

## Reports

| file | content |
|---|---|
| `AUDIT.md` | starting state, external systems found, CSL-L2M issues, what had to be trained |
| `DATA_PROTOCOL.md` | datasets, split, audio subset, measurement decisions made during the round |
| `LYRICS_TO_MELODY.md`, `CSL_L2M_COMPARISON.md` | Table 1: Qwen vs CSL-L2M (official and retrained) |
| `LYRIC_CONDITIONING.md` | Table 2: lyric swap vs reseed, tone analysis |
| `MELODY_INFILLING.md` | Table 3: section infill vs whole-song regeneration |
| `MELODY_TO_CHORD.md`, `ACCOMONTAGE2_COMPARISON.md` | Table 4: harmonizers |
| `JOINT_VS_CASCADE.md` | Table 5: joint vs cascaded systems and cost |
| `SVS_CROSS_RENDER.md` | Table 6: melody source × SVS |
| `AUDIO_CONTROL_EVALUATION.md` | Tables 7–8: backing by chord source; closed-loop key/tempo/chords |
| `SECTION_SEMANTICS.md` | label intervention vs musical change; held-out classifier |
| `FAILURE_ANALYSIS.md` | every failure category, with counts |
| `COMPUTE_LEDGER.csv`, `experiment_registry.json` | every job; every model, dataset and artifact |
| `listening_study/` | blinded A/B package for blocks A–E. **No human results have been collected.** |
| `tables/` | CSV + Markdown + LaTeX for every table |
| `figures/` | PDF + PNG for Figures 1–8 |
| `data/` | per-sample / per-song parquet behind every table |
| `configs/`, `jobs/` | configs as run; Slurm logs and `SUBMISSIONS.txt` |
| `audio/subset.json` | the frozen audio subset |

Raw generations, renders and checkpoints are **not** in git: they contain lyrics, audio or
weights. They live under `experiments/component_sota/` (git-ignored) and
`/gscratch/ark/pingw220/qwen_abc_r2_offload/component_sota_audio/`.

## Reproduction

```bash
source scripts/component_sota/env.sh                      # worktree on PYTHONPATH, PAPER_EVAL_OUT_ROOT set
# data + training
python -m paper_eval.component_sota.build_datasets --task mel|chord|chord_lyr
CONFIG=configs/component_sota/mel_sft.yaml RUN_DIR=experiments/component_sota/runs/mel_sft sbatch scripts/component_sota/train.sbatch   # likewise chord_sft, chord_lyr_sft
python -m paper_eval.component_sota.csl_l2m build-data && sbatch scripts/component_sota/csl_train.sbatch
# generation
python -m paper_eval.generation --model qwen_mel --suite bon                           # + --suite interventions --conditions lyrics_all
$CSL_PY -m paper_eval.component_sota.csl_l2m infer --which official|retrain [--ckpt ...] [--chunked]
python -m paper_eval.component_sota.chord_gen --harmonizer chord|chord_lyr --melodies ref:S1,...
python -m paper_eval.component_sota.accomontage --melodies ref:S1,...
python -m paper_eval.component_sota.infill generate --seeds S1,S2
# symbolic evaluation
python -m paper_eval.component_sota.melody_eval [--condition lyrics_all]
python -m paper_eval.component_sota.lyric_conditioning
python -m paper_eval.component_sota.infill_eval
python -m paper_eval.component_sota.chord_eval && python -m paper_eval.component_sota.joint_cascade
python -m paper_eval.component_sota.section_semantics
# audio
python -m paper_eval.component_sota.audio subset|prep-svs|render-svs|prep-backing|render-backing
LIST=... MODE=vocal|mix sbatch scripts/component_sota/audio_tools.sbatch   # + scripts/component_sota/btc_only.sbatch
python -m paper_eval.component_sota.svs_eval evaluate && python -m paper_eval.component_sota.backing_eval evaluate
# package, figures, ledger
python -m paper_eval.component_sota.listening_study build && ... prep-render && ... render && ... stage
python -m paper_eval.component_sota.figures
python -m paper_eval.ledger --report-dir reports/component_sota --submissions jobs/SUBMISSIONS.txt   # path relative to --report-dir
pytest tests/test_component_sota.py tests/test_component_sota_listening.py
```
