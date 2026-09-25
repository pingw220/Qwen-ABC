# Qwen-ABC

Proof of concept: can a general-purpose LLM (Qwen3.5-0.8B) learn **lyrics-aware lead-sheet generation** in human-readable **ABC notation**, without a MIDI-specific vocabulary?

Two recipes, identical except for one stage:

| | recipe |
|---|---|
| **A. Direct SFT** | Qwen3.5-0.8B-Base → SFT (lyrics + structure → ABC) |
| **B. CPT + SFT** | Qwen3.5-0.8B-Base → ABC continued pretraining → the same SFT |

Both use the same base model, SFT data, SFT hyper-parameters, test songs and evaluation.

Round 2 asks what limits long-form structure, diversity, coherence and lyric alignment: representation
(ABC-v2 counters), data cleaning, effective batch size, a long-range objective, decoding, and model scale.

Reports:

* `reports/DATA_AUDIT.md`: what data exists and why D1 was chosen
* `reports/ABC_SCHEMA.md`: the ABC-v1 format (lyrics, melisma, sections, round trip)
* `reports/DATASET_VALIDATION.md`: ABC-v1 dataset sanity checks
* `reports/REPORT.md`: round-1 results and verdict
* `reports/LONG_CONTEXT_AUDIT.md`: Qwen context, what the trainer really does, token lengths
* `reports/ABC_V2_SCHEMA.md`: ABC-v2 (section header + per-bar countdown) and the variants measured
* `reports/ABC_V2_DATA_CLEANING.md`: section-boundary cleaning, decisions and before/after examples
* `reports/ABC_V2_DATASET_VALIDATION.md`: ABC-v2 dataset checks, determinism, leakage re-verification
* `reports/CLEAN_TEST_SUBSET.md`: the 34-song clean subset, and model error vs label noise
* `reports/LONG_STRUCTURE_EXPERIMENTS.md`: **round-2 results and verdict**
* `reports/LEADSHEET_TO_AUDIO.md`: ABC → FastSinger → MuseControlLite, and the note↔syllable pairing it needs
* `reports/BEST_OF_N_DECODING.md`: best-of-n sampling (R3-B) — strict validity 0.81 → 0.98 with no training
* `reports/R3A_SYLLABLE_BUDGET.md`: R3-A — a syllable budget in the prompt does **not** fix cramming (negative result)
* `reports/CRAMMING_ATTACKS.md`: **three attacks on cramming**, none adopted — and what listening caught that the metrics did not
* `reports/R4_MUSIC_PRETRAINED_BASE.md`: **MuPT vs Qwen** — symbolic-music pretraining transfers musical competence, not instruction-following
* `reports/R5_FREE_LYRIC_ASSIGNMENT.md`: **R5** — the per-section lyric assignment is not needed; the model can allocate but cannot budget

## Layout

```
qwen_abc/        library: canonical model, corpus adapter, ABC writer/parser, MIDI, prompt, splits, train, metrics
                 round 2: abc_v2.py (counters), cleaning.py (section boundaries), longrange.py (E3 tasks),
                 leadsheet_midi.py + fastsinger.py (inputs for the FastSinger / MuseControlLite renderers)
scripts/         build / validate / inspect dataset, train, generate+evaluate, samples
                 round 2: build_abc_v2_dataset.py, validate_abc_v2_dataset.py, build_longrange_tasks.py,
                 select_clean_subset.py, eval_continuation.py, eval_infill.py, compare_r2.py,
                 budget_report.py, check_infra.py, check_left_padding.py, midi_llm_r2_report.py
scripts/slurm/   Slurm launchers (L40S and L40 variants)
configs/         training configs (sft_direct, cpt_abc, sft_after_cpt)
tests/           pytest: round trip, lyric alignment, parser errors, splits/leakage, determinism
data/generated/  built datasets (git-ignored)
experiments/     one timestamped directory per run (logs and configs tracked, weights ignored)
```

## Environment

```bash
cd /mmfs1/gscratch/scrubbed/pingw220/music_acc/Qwen-ABC
/usr/bin/python3.12 -m venv .venv && .venv/bin/pip install uv
.venv/bin/uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/uv pip install --python .venv/bin/python "transformers>=5.2" accelerate datasets safetensors mido pytest pyyaml flash-linear-attention
source scripts/env.sh   # PATH, PYTHONPATH, HF cache on scrubbed
```

## Reproduce

```bash
source scripts/env.sh
python -m pytest -q tests

# 1. dataset (CPU node, ~15 min); the split is fixed before examples are written
TS=$(date +%Y%m%d_%H%M%S)
OUT=data/generated/abc_v1_$TS sbatch scripts/slurm/build_dataset_cpu.sbatch
python scripts/show_dataset_examples.py --data-dir data/generated/abc_v1_$TS -n 20 --output /tmp/examples.md

# 2. Experiment A
CONFIG=configs/sft_direct.yaml RUN_DIR=experiments/direct_sft_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS" sbatch scripts/slurm/run_direct_sft_l40s.sbatch

# 3. Experiment B
CONFIG=configs/cpt_abc.yaml RUN_DIR=experiments/abc_cpt_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS" sbatch scripts/slurm/run_cpt_l40s.sbatch
CONFIG=configs/sft_after_cpt.yaml RUN_DIR=experiments/cpt_then_sft_$(date +%Y%m%d_%H%M%S) \
  OVERRIDES="data_dir=data/generated/abc_v1_$TS model_name_or_path=experiments/abc_cpt_<ts>/final_model" \
  sbatch scripts/slurm/run_cpt_sft_l40s.sbatch

# 4. evaluation (same test songs, same seeds, same decoding for both)
python scripts/generate_eval.py --checkpoint experiments/<run>/final_model \
  --data-dir data/generated/abc_v1_$TS --output-dir experiments/<run>/eval_test --num-songs 100 --probes
```

The exact commands, job ids and dataset path used for the reported results are listed in `reports/REPORT.md`.

To reproduce the full evaluation of one checkpoint (4 generation shards + probes + aggregation, with dependencies):

```bash
CKPT=experiments/<run>/final_model OUT=experiments/<run>/eval_test SHARDS=4 NUM_SONGS=0 bash scripts/slurm/submit_eval.sh
python scripts/eval_loss.py --checkpoint experiments/<run>/final_model --data-dir data/generated/abc_v1_20260915_012459 \
  --split test --output experiments/<run>/loss_test.json
python scripts/compare_results.py --run A=experiments/direct_sft_<ts> --run B=experiments/cpt_then_sft_<ts> > table.md
```

## Looking at / listening to lead sheets

```bash
# adds <name>.clean.abc (ABC-v1 text), <name>.musicxml and engraved <name>.pN.svg next to every .abc
python scripts/build_viewable_samples.py experiments/<sample dir>
# audio (needs fluidsynth; see scripts/make_samples.py for the soundfont path)
python scripts/make_samples.py --data-dir <dataset> --song-ids reports/listening_song_ids.json --audio --output-dir experiments/<out> --eval name=<eval dir>
```

MuseScore opens the `.musicxml` (melody, lyrics, chord symbols, section marks) and the `.mid` directly;
plain `.abc` needs an ABC plugin or a tool such as EasyABC / abcjs. Requires `music21` and `verovio`.

For a sung, full-band recording the lead sheet goes through FastSinger and MuseControlLite, the same
two models MIDI-LLM's lead sheets go through — see `reports/LEADSHEET_TO_AUDIO.md`:

```bash
python scripts/export_leadsheet_midi.py --sample-dir experiments/<samples> --outdir experiments/<out>/leadsheets
LEADSHEETS=$PWD/experiments/<out>/leadsheets OUT=<audio dir on /gscratch/ark> \
    sbatch --chdir=$PWD/experiments/<out> scripts/slurm/midi_sag_render.sbatch
python scripts/collect_midi_sag_audio.py --render-dir <audio dir> --sample-dir experiments/<samples>
```

## Results

**Round 1** (`reports/REPORT.md`): direct SFT on ABC-v1 works; same-data CPT hurts prompt adherence.

**Round 2** (`reports/LONG_STRUCTURE_EXPERIMENTS.md`), 225 de-duplicated held-out songs, one sample
per song:

| | E0 (ABC-v1) | E1 (ABC-v2) | **E3b (ABC-v2 + infill)** | E3b @ T=1.0 | reference |
|---|---|---|---|---|---|
| exact section+bar plan | 0.53 | 0.86 | **0.98** | 0.98 | 1.00 |
| early EOS | 4.0% | 0% | 0% | 0% | 0% |
| lyric recall | 0.955 | 0.964 | 0.967 | 0.976 | 1.00 |
| strict-valid ABC | 0.71 | 0.63 | **0.84** | 0.73 | 1.00 |
| pitch range (semitones) | 14.1 | 13.3 | 13.8 | **17.9** | 22.0 |
| distinct bars | 0.46 | 0.47 | 0.47 | **0.77** | 0.87 |

**R3-B** (`reports/BEST_OF_N_DECODING.md`): four samples at T=1.0 with an inference-time selector
(no training, 2.1 GPU-h) — strict-valid **0.978**, exact structure **0.996**, lyric recall **0.982**,
crammed syllables 0.163 → **0.114**, with pitch range and interval distribution unchanged.

**R3-C** (`reports/CRAMMING_ATTACKS.md`): four attempts on syllable cramming, **none adopted**.
Repairing the targets takes crammed syllables to 0.043 (corpus 0.057) with validity, structure and
lyric recall unchanged — and a listening pass then found near-monotone melodies (repeated-pitch
intervals 0.330 → 0.356, worse on 99 of 225 songs) and thinned chords (songs with coverage < 0.95:
18 → 30). Splitting a crammed note into equal same-pitch pieces teaches the model to write runs of
repeated notes. Banning the join token at decode time drives cramming to zero and validity to 0.342,
because in ABC the notes are committed before the lyric line is written. Cramming is decided when
the melody is written, and it is in the data because the note↔lyric matcher put it there.

* **Representation fixes structure:** cleaned section boundaries + a per-bar countdown (`[r:k]`) take
  exact structure 0.53 → 0.86; a long-range infilling objective at the same batch size takes it to 0.98.
* **Melodic conservatism was decoding, not capacity:** temperature 1.0 / top-p 0.95 recovers corpus-like
  range, syncopation and repetition with no loss of lyric or structure control.
* **A 4× effective batch does not help;** optimizer updates are the scarce resource.
* **8K context was never a limit** (longest example 6,703 tokens).
* **A music-pretrained base does not help** (`reports/R4_MUSIC_PRETRAINED_BASE.md`): MuPT-1.07B, pretrained
  on 10B tokens of ABC, ties Qwen on everything it writes — parse rate, bar durations, internal countdown
  consistency, pitch range, interval distribution, chord density — and loses on everything it was asked for:
  exact structure 0.689 vs 0.996, lyric recall 0.539 vs 0.982, despite 41% more optimizer steps.
* **Scaling to ~2B was not justified** by the pre-registered criteria and was not run.
