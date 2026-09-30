# Component-SOTA round: audit (2026-09-27)

## Starting state

| | |
|---|---|
| main checkout | `Qwen-ABC` on `paper-final-eval-2026` at `b7edc01` (pushed); the user's WIP (staged rename `scripts/build_repaired_dataset.py → build_song_variant.py`, its unstaged edit, untracked R6 pinyin files) left untouched |
| this round | separate **git worktree** `music_acc/Qwen-ABC-component`, branch `component-sota-eval-2026` created from `b7edc01`; git-ignored assets (`data/generated`, `.venv`, `.hf_cache`, `experiments/paper_final`, `models`) symlinked from the main checkout and excluded in `info/exclude` |
| code path | `scripts/component_sota/env.sh` reuses the main venv but puts the worktree first on `PYTHONPATH` (the main `scripts/env.sh` hard-codes the main checkout) |

## Paper-final facts this round inherits (verified from the artifacts)

| item | value | source |
|---|---|---|
| split | 10,243 / 273 / 225 (de-duplicated) | `data/generated/abc_v2_20260915_120927/split_manifest.json`, `reports/paper_final/configs/test_song_ids.txt` |
| E3b checkpoint | `/gscratch/ark/pingw220/qwen_abc_r2_offload/runs/e3b_201131/final_model` | `experiment_registry.json` |
| base model / tokenizer | `Qwen/Qwen3.5-0.8B-Base` (752M params, 248k vocab) | `resolved_config.json` |
| representation | ABC-v2 = ABC + ESS (`% section i/N | B bars`, per-bar `[r:k]`); prompt: meter, tempo, key, plan (label, bars, irregular beats, section i/N), lyric lines per section | `qwen_abc/abc_v2.py` |
| training | whole-song + 50%-songs late-section infill mixture (15,332 examples), max_seq_len 8192, 131,072 tokens/update (2 GPUs × grad-accum), AdamW LR 5e-5, warmup 3%, cosine to 0.1×, 2 epochs = **758 updates**, seed 1234 | `configs/e3b_v2_longrange_smallbatch_sft.yaml` |
| decoding | T=1.0, top-p 0.95, top-k off; paired-seed sampler S1-S4 | `reports/paper_final/EVALUATION_PROTOCOL.md` |
| infill ability | trained only on **later-half** lyric sections (`choose_infill_target`); prompt = plan + lyrics + ABC with `% [missing section]` | `qwen_abc/longrange.py` |
| existing generations reused | E3b `orig` S1-S4, `lyrics_all` S1-S4, `lyrics_sec`, `bars_p4`, `key_p5`, `tempo_x1.25`, `label_swap`, `label_bridge` (paper-final) | `experiments/paper_final/gen/qwen_e3b/` |

## External systems found (all local, used unmodified)

| system | location / commit | status |
|---|---|---|
| **CSL-L2M** (AAAI-25) | `MIDI-SAG/lyrics2melody_new` (MIDI-SAG `cc282feb`, clean) | code present, **checkpoints absent**; downloaded the authors' released `pretrained_CSLL2M.pt`, `pretrained_CSLL2M_onlyLyrics.pt`, `pretrained_VQVAE.pt` (Google Drive links in its README) to `/gscratch/ark/pingw220/third_party/csl_l2m_ckpt/` |
| **AccoMontage2** | `MIDI-SAG/AccoMontage2` (`demo_SOME.py`, chorderator database present) | runs on CPU, 4/4 only, outputs per-beat BTC chord labels |
| **SVS-A: FastSinger** | `music_acc/fastsinger` (`suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag@400`, spk 6) | the project's SVS |
| **SVS-B: SoulX-Singer** | `MIDI-SAG/SoulX-Singer` + `MIDI-SAG_checkpoints/SoulX-Singer/model.pt` (704M params), shipped zh prompt + precomputed prompt metadata | executable; used by MIDI-SAG's ComposeFlow |
| backing renderer | MIDI-SAG `tools/midi_llm_adapter/run_midi_llm_to_midi_sag.py` + MuseControlLite | the project's renderer (`reports/LEADSHEET_TO_AUDIO.md`) |
| audio analyzers | BTC-ISMIR19 (large voca), BeatNet (MIDI-SAG's Rhythm F1), RMVPE (`MIDI-SAG_checkpoints/rmvpe_model.pt`), SongEval (`/gscratch/ark/pingw220/andysu/SongEval`), Audiobox-aesthetics | present |
| Mandarin ASR (PER) | none installed with weights; `funasr` present in `envs/midi-sag` | downloaded Paraformer-zh (`iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404`, 954 MB) to `/gscratch/ark/pingw220/third_party/modelscope` |
| key-CNN (MIDI-SAG key accuracy) | environment does not exist locally | replaced by Krumhansl-Schmuckler on CQT chroma, identical for every condition |

## CSL-L2M issues found before running (documented, handled in our adapter)

1. The released lyrics-only checkpoint has **no POS/tone embedding weights**, but the shipped
   `CSLL2M_withOnlyLyrics.yaml` sets `f_pos/f_tone: True` and loads with `strict=False`: running it
   as shipped adds randomly initialized POS/tone embeddings to the conditioning without error. We
   use f_pos=f_tone=False (loading then reports 0 missing keys, 4 unused leftover tables).
2. No random seeding upstream; the adapter seeds torch/numpy/random per (song, seed).
3. 4/4 only; Han characters only (Latin words crash it); fixed tempo 90 and 8 silent bars in its
   MIDI writer; section labels ignored by the lyrics-only model; decoder trained on ≤ 2048 events
   (~43 lines) while inference has no cap; a rejection sampler that gives up after 128 failed
   positions (5 retries).
4. The full-control model (`pretrained_CSLL2M.pt`) needs 12 statistical musical attributes that
   `generate.py` copies from a *retrieved training song* plus a hard-coded key and emotion, and
   indexes POS/tone per line instead of per character (a bug); it is therefore not run as a
   lyrics->melody system here. The lyrics-only model is the fair external comparison.

## What had to be trained

| model | why | status |
|---|---|---|
| Qwen Melody-Only | did not exist | trained (E3b recipe, chord-free mixture, 758 updates) |
| Qwen Melody→Chord (+ lyrics variant) | did not exist | trained (E3b recipe, 2 epochs) |
| CSL-L2M retrained on our split | controlled architecture comparison | trained with the unmodified upstream trainer |
| infill model | E3b already does section infill | **not trained**: E3b evaluated directly, including first-half (untrained) positions |

## Addendum (2026-09-29): issues found during the round

Measurement defects found and fixed while running the round are listed in
`FAILURE_ANALYSIS.md` §6 and `DATA_PROTOCOL.md`: ASR without VAD, SongEval stem collisions, BTC
under the new NumPy/torch, the RMVPE return shape, CSL structure scoring, and a table overwrite.
No system output was changed by any fix; each would have corrupted a metric. No upstream repository
(MIDI-SAG, CSL-L2M, AccoMontage2, FastSinger, SoulX-Singer, BTC, SongEval, MuseControlLite) was
modified. Compatibility shims live in `paper_eval/component_sota/tools/` (e.g. `btc_run.py`).

Job logs: `jobs/` holds every Slurm log under 1 MB, plus `SUBMISSIONS.txt`. The three larger logs
(CSL-L2M retraining and two whole-song inference shards, mostly progress bars) stay on disk,
untracked, in the same directory.
