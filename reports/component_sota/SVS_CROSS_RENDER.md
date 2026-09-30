# Two-SVS cross-render: melody source × singing-voice renderer (Table 6)

**Code:**
- `paper_eval/component_sota/audio.py` (`prep-svs`, `render-svs`);
- analyzers in `tools/`;
- `svs_eval.py`.

**Tables:** `tables/svs_cross_render.*` and `tables/svs_effects.*` (renderer effect per source,
interactions, melody main effects, rank agreement).
**Data:** `data/svs.parquet`.
**Figure:** `figures/fig7_svs_cross_render.*`.

## Design

- **Songs:** the frozen 36-song audio subset (`audio/subset.json`, 4/4, stratified, frozen before any audio).
- **Melody sources (S1):** pseudo-reference, Qwen Full (E3b), Qwen Melody-Only, CSL-L2M official
  chunked, CSL-L2M retrained chunked.
- **Renderers:**
  - **SVS-A FastSinger:** `suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag@400`, speaker 6, the project's SVS.
  - **SVS-B SoulX-Singer:** MIDI-SAG's checkpoint (704M), `--control score`, shipped zh prompt.
- **Identical symbolic input to both.** One sung-note list per (source, song) is built by
  `qwen_abc.fastsinger.sung_notes`: wordless notes are dropped, crammed notes divided, melisma held.
  - FastSinger gets its two input files.
  - SoulX gets a MIDI with one lyric event per note, with melisma written as MIDI-SAG's adapter does.
  - The list itself (`target.json`) is the reference for all metrics.
  - Whole songs are rendered, not excerpts.
- **Metrics:**

| metric | definition |
|---|---|
| PER / CER | Paraformer-zh ASR **with fsmn-vad segmentation**, vs the target syllables. Phonemes are toneless pinyin initials + finals. |
| note pitch accuracy | RMVPE median F0 over the middle 60% of each note, within ±50 cents of the target; also octave-folded |
| octave-error rate | fraction of notes off by 1200 ± 100 cents |
| mean absolute cents | octave-folded |
| voicing F1 | voiced frames vs target note spans, 10 ms frames |
| phrase-onset hit rate | a voiced onset within 100 ms of each phrase start |
| duration ratio | rendered length / score length |
| Audiobox PQ / CE | Audiobox-aesthetics production quality and content enjoyment |

## MEASURED FACT (36 songs; SoulX fails on 1 song for every source, FastSinger on 1 CSL-retrained song)

| melody source | SVS | PER ↓ | note pitch acc. ↑ | abs. cents ↓ | voicing F1 ↑ | phrase onsets ≤ 100 ms ↑ | Audiobox CE ↑ |
|---|---|---|---|---|---|---|---|
| pseudo-reference | FastSinger | 0.051 [0.040, 0.061] | 0.929 | 17.7 | 0.964 | 0.920 | 6.39 |
| | SoulX | 0.065 [0.049, 0.083] | 0.923 | 24.4 | 0.930 | 0.734 | 6.14 |
| Qwen Full (E3b) | FastSinger | 0.074 [0.060, 0.089] | 0.898 | 22.5 | 0.961 | 0.897 | 6.17 |
| | SoulX | 0.085 [0.070, 0.103] | 0.929 | 22.8 | 0.925 | 0.700 | 5.85 |
| Qwen Melody-Only | FastSinger | 0.064 [0.055, 0.073] | 0.927 | 18.2 | 0.964 | 0.918 | 6.37 |
| | SoulX | 0.091 [0.072, 0.112] | 0.926 | 23.4 | 0.930 | 0.728 | 6.03 |
| CSL-L2M official | FastSinger | 0.067 [0.052, 0.083] | 0.872 | 38.2 | 0.948 | 0.839 | 6.70 |
| | SoulX | 0.053 [0.038, 0.070] | 0.864 | 37.0 | 0.910 | 0.728 | 6.36 |
| CSL-L2M retrained | FastSinger | **0.034** [0.026, 0.043] | **0.971** | **11.2** | 0.961 | 0.920 | **6.96** |
| | SoulX | **0.027** [0.020, 0.034] | 0.939 | 20.4 | 0.926 | 0.778 | 6.63 |

Octave errors are ≤ 0.007 everywhere, and duration ratios are 1.000–1.010.

**Renderer main effect (SoulX − FastSinger), consistent over all 5 sources:**
- voicing F1: −0.034 to −0.039, every CI excludes 0;
- phrase onsets within 100 ms: −0.11 to −0.20, every CI excludes 0;
- Audiobox CE is lower for SoulX in every source.

On PER and pitch the renderer effect depends on the melody source. That is the interaction below.

**Interactions** (difference of paired renderer effects):

| metric | contrast | estimate [95% CI] |
|---|---|---|
| PER | Melody-Only vs CSL-retrained | **+0.038 [+0.021, +0.055]** |
| PER | Melody-Only vs CSL-official | +0.042 [+0.017, +0.067] |
| PER | E3b vs CSL-retrained | +0.024 [+0.006, +0.043] |
| PER | Melody-Only vs reference | +0.012 [−0.005, +0.030] (n.s.) |
| note pitch accuracy | E3b vs CSL-retrained | +0.060 [+0.037, +0.086] |

SoulX is relatively *worse* at pronouncing Qwen melodies than CSL melodies.

**Melody main effects** (mean of both renderers):

| metric | contrast | estimate [95% CI] |
|---|---|---|
| PER | Qwen Melody-Only − CSL-retrained | **+0.048 [+0.037, +0.060]** (Qwen melodies less intelligible) |
| PER | Melody-Only − reference | +0.020 [+0.008, +0.033] |
| note pitch accuracy | Melody-Only − CSL-retrained | −0.028 [−0.048, −0.005] |
| note pitch accuracy | Melody-Only − CSL-official | +0.061 [+0.023, +0.102] |

**Rank agreement** of the 5 melody sources across renderers (Spearman, 5 points):
- CER and voicing: 0.90;
- pitch and onsets: 0.60;
- PER: 0.50.

**Failures (renderer × input):**
- SoulX fails on song `7v46DZBqB4xmaTHRygC8gy` for every source: the lyric character 揹 is missing
  from its phoneme set (`KeyError: 'zh_揹'`).
- FastSinger fails on one CSL-retrained melody: "Input length exceeds the maximum length", i.e. a
  CSL phrase with no line break long enough for FastSinger.

## INTERPRETATION

- **Singability favours CSL-L2M-retrained melodies, not Qwen's.** Its one-character-per-note
  melodies with no cramming (Qwen crams 12.6–17.3% of syllables vs 5.7% in the reference) are:
  - recognized best by ASR (PER 0.027–0.034);
  - sung most accurately in pitch.

  This is the clearest case in the round where a specialized system wins, and it is reported as such.
  It is consistent with cramming being a known Qwen weakness: sections with crammed syllables
  are hard for any SVS to articulate.
- **Official CSL-L2M melodies are sung *less* accurately in pitch** (0.86–0.87; 37–38 cents).
  Plausible causes are its dense, wide-range lines (6.5 notes/bar, range 19.7). Its PER is middling.
- **FastSinger is the more faithful renderer for timing:** voicing +0.035, phrase onsets +0.11 to +0.20.
  SoulX's relative PER penalty is larger on Qwen melodies; this interaction is significant.
- **Melody rankings are only partly stable across renderers.** Rank correlations of 0.5–0.9 over 5
  sources are weak evidence either way. Any claim about "singable melodies" should be made with
  both renderers and should report the interaction.
- Audio-quality predictors (Audiobox CE) rank the CSL melodies highest. They measure the rendered
  voice, not melodic quality, and have no human validation here: listening-study block B is pending.
