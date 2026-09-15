# Data audit for the Qwen-ABC POC

Audited 2026-09-15, read-only; nothing outside `Qwen-ABC/` was modified. Base: `/mmfs1/gscratch/scrubbed/pingw220/music_acc` (abbreviated `$MA`). Counts come from counting files or manifest rows unless marked "sample" (a seeded random sample of 150–200 songs).

## 1. Relevant repositories

| repo | what it is | relevance |
|---|---|---|
| `$MA/sheetsage-pro` | SheetSage-Pro: audio → lead sheet pipeline. Separation, beat_this beats/downbeats, GAME vocal notes, ISMIR2019-LVCR chords (v2), All-In-One + SongPrep sections, Phonsa lyric/phoneme alignment, note↔lyric DP matcher, key ensemble | **produces the corpus used here** |
| `$MA/MIDI-LLM-phoneme-lyric-v1` (branch `experiment/original-midillm-multitask-section-onestage-v1`) | lyrics-aware lead-sheet generation on MIDI-LLM (Llama-3.2-1B + AMT MIDI vocabulary); section and one-stage variants, v4 data = LVCR chords + real key | **the system this POC is compared against**; its song split is inherited |
| `$MA/MIDI-LLM` | the original MIDI-LLM code | background only |
| `$MA/sheetsage2-lyrics-poc`, `$MA/SheetSage2` | lyrics-conditioned audio transcription experiments on GTSinger | not lead-sheet generation |
| `$MA/sheetsage-pro-release`, `sheetsagepp_frozen_*` | release copy / frozen snapshots of the pipeline | duplicates |
| `$MA/phonsa`, `songprep`, `all-in-one`, `BTC-ISMIR19`, `GAME`, `RMVPE`, `whisperX`, `VocalRender`, `fastsinger`, `MIDI-SAG` | upstream components or singing-voice synthesis | not data |
| `$MA/Qwen-ABC` | this repo (was an empty `git init`) | |

## 2. Datasets found

| # | path | format | songs / examples | languages | role |
|---|---|---|---|---|---|
| **D1** | `$MA/sheetsage-pro/dataset/processed/sheetsage_zh_training_views_v2_lvcr` | `manifest.jsonl` + `phonemes/<song_id>/{leadsheet.json, leadsheet.mid, provenance.json}` (`phoneme_leadsheet_v1`) | **11,673** songs (tier A 9,453 / B 2,202 / C 18; 1 excluded upstream for validation failure) | zh (all rows), about 0.9% latin syllables (code-switching) | **selected** |
| D2 | `.../processed/sheetsage_zh_production_v2_lvcr/<song_id>/` | full pipeline instance (audio, stems, analysis, intermediate, exports) | 11,683 dirs | zh | source of D1 |
| D3 | `$MA/MIDI-LLM-phoneme-lyric-v1/manifests/phoneme_leadsheet_section_v4` | section windows built from D1, with `{split}_songs.jsonl` | 104,562 / 6,014 / 5,958 sections; **10,247 / 591 / 581 songs** | zh | split source, comparison |
| D4 | `.../manifests/one_stage_full_song_v3` | whole-song AMT token streams from D3 | 9,801 / 301 / 270 songs | zh | MIDI-LLM one-stage (v4 generation) |
| D5 | `$MA/sheetsage-pro/dataset/{qq_gt,spotify}_{aligned_lyrics,phoneme,prompt,audio}` | txt / TextGrid (model predicted) / arrow | 11,460 + 1,585 | zh, some en | raw inputs to D2: provider lyrics, captions |
| D6 | `$MA/sheetsage-pro/for_pingw220/share_singing` (+ `test_data/share`, 1,000 MIDIs) | human MIDI, one lyric event per note | 191 (1,000) | zh | only human lyric↔note GT; no chords |
| D7 | `$MA/data/gtsinger_hf`, `sheetsage-pro/dataset/gtsinger_git` | json/TextGrid/musicxml/wav | 28,628 segments, 9 languages (zh 7,139, en 4,827) | multi | singing segments, no chords or sections |
| D8 | `sheetsage-pro/dataset/mir_st500*` | onset/offset/pitch json | 500 | zh pop | melody only, no lyrics |
| D9 | `$MA/jamendolyrics` | txt/csv word timing | 79 (en 20) | en de es fr | no notes |
| D10 | `$MA/eval_data/sheetsage_pro_lbd`, `sheetsage2-lyrics-poc/manifests*` | eval predictions / GTSinger windows | — | — | transcription evaluation |
| D11 | `$MA/infer_out`, `midi_export_v2`, `samples`, `midi_export`, `midi_results` | MIDI-LLM generations | 7 authored specs | zh | `midi_export`, `midi_results` and most of `samples` are **stale** (pre onset-pile-up fix); `midi_export_v2` is the valid set |

There is **no English lead-sheet corpus** (melody + chords + lyrics). English exists only as GTSinger singing segments (no chords, no sections), 580 qq/spotify English songs with no melody extraction (`en_phoneme_blocked`), and Jamendo lyric timing. The POC is therefore Mandarin, with English only as in-song code-switching.

## 3. D1 fields (per song `leadsheet.json`, about 4 MB each, mostly alignment provenance)

| field | content |
|---|---|
| `global` | `tempo_bpm`, `meter` (4/4: 11,027, 3/4: 646), `key` (e.g. "B major", populated 100% in v2_lvcr), `key_agreement` (high 57% / medium 43%, sample), `language`; `title/artist/genre` are null |
| `beats[]` | `time_sec`, `is_downbeat` (beat_this); median 400 beats per song |
| `downbeats[]` | `time_sec` |
| `sections[]` | `label` (All-In-One + SongPrep: chorus, verse, intro, inst, outro, bridge, silence, end), start/end quantized to downbeats |
| `melody_notes[]` | `pitch_midi`, `onset_sec/offset_sec`, `onset_grid_time_sec/offset_grid_time_sec` (quantized to `beat_subdivision_4`), `lyric`, `lyric_unit_id`, `is_lyric_continuation`, cents, confidence, source `game_vocal_csv` |
| `recognized_chords[]` | `symbol` (`B:maj`), `root`, `quality` (maj/min/min7/7/maj7/N/sus4/dim/aug), `bass` degree, `start_beat/end_beat`, `source_model` = ISMIR2019-LVCR (the top-level `chord_default_source: click_quantized_btc` string is stale) |
| `lines[]`, `words[]`, `syllables[]` | normalized simplified-Chinese text, time spans, line ids; `words[].mandarin.pinyin` (toneless), initial/final |
| `phonemes[]` | Phonsa phonemes with times, lexical/nonlexical class |
| `note_lyric_alignment.items[]` | note ↔ syllable DP matches (`is_lyric_continuation` for melisma) |
| `note_phoneme_alignment.records[]` | note ↔ phoneme relations (attack / sustained nucleus / continuation) |
| `leadsheet.mid` | 3-track MIDI export |

### Presence checklist (D1)

| lyrics | Chinese chars | pinyin | phonemes | syllable boundaries | lyric-note alignment | melody notes | chords | tempo | meter | beats | downbeats | sections | key | MIDI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | yes | yes, toneless | yes | yes (times) | yes (pseudo-label, DP) | yes (vocal) | yes (LVCR) | yes (global) | yes (global) | yes | yes | yes | yes | yes |

**Everything in D1 is automatic transcription, not ground truth.** Known quality facts from the MIDI-LLM work:

* median `note_lyric_coverage` is 0.977;
* 10.6% of notes are unlinked in the raw data, and 4.1% after the melisma recovery used below;
* chord-tone agreement between melody and chords is about 0.63 in the corpus itself (a ceiling of recognition against a separated vocal).

Per-song medians (from `manifest.jsonl`):

| notes | syllables | chords | sections | downbeats | lyric lines | melismas | tempo |
|---|---|---|---|---|---|---|---|
| 280 | 221 | 81 | 10 | 91 | 27 | 31 | 111 BPM |

Tempo ranges from 37.5 to 333 BPM (p10 75, p90 143).

## 4. Existing split logic

* **SheetSage-Pro / MIDI-LLM v1:** 90/5/5 on `sha256('phoneme_leadsheet_v1:' + musical_semantic_hash)`, with song version groups kept together.
* **MIDI-LLM v4 section manifests (D3)** record the resulting assignment in `{train,validation,test}_songs.jsonl`: 10,247 / 591 / 581 songs, `cross_split_group_intersection: 0`. 236 mostly-wordless songs (>20% wordless melody notes) and tier C are excluded.
* **One-stage (D4)** removes further songs, notably `cross_split_identity_collision` (545 songs).

## 5. Leakage risks (important)

1. **Exact-duplicate lyrics across splits.** Every `song_group_id` in D3 contains exactly one song (10,247 / 591 / 581 groups for as many songs), so version grouping protects nothing. MIDI-LLM's one-stage builder found **282 of 581 test** and **263 of 591 validation** songs whose phoneme-level lyrics exactly equal a train song's (re-uploads and alternate versions under new ids). D3 itself does not remove them. **Mitigation here:** remove held-out songs whose lyrics or melody near-duplicate a train song (6-syllable / 8-note shingle containment ≥ 0.30; exact lyric hash) before any example is built (`qwen_abc/splits.py`, numbers in DATASET_VALIDATION.md).
2. **Splits redrawn between MIDI-LLM data generations.** v2 and v3 test sets are ~90% inside each other's train split, because the hash key includes chords and LVCR changed them. This POC takes one generation (v4/LVCR) only. Never compare against a MIDI-LLM checkpoint trained on a different generation's train split using this test set.
3. **External eval sets.** 8 share_singing/eval191 songs appear (by lyrics) in qq/spotify, i.e. in D1. This is irrelevant here because share_singing is not used.
4. **Same song, different representations.** CPT and SFT examples are generated from the already-split song list, so a song's ABC can never be in CPT-train while its lyrics are in SFT-test. The validator enforces identical CPT/SFT song sets per split.

## 6. Timing structure measured (sample of 200 songs)

| measurement | result |
|---|---|
| melody onset position within the beat (×4) | 0: 47.3%, 2: 33.5%, 3: 11.0%, 1: 8.2% — **all on the sixteenth grid** |
| note duration (sixteenths) | 2: 44%, 4: 16%, 1: 15%, 3: 11%, 0: 0.7% (zero length) |
| overlapping consecutive notes | 1.5% |
| notes before the first beat / after the last beat | 0 / 0 |
| chord starts exactly on a beat time | 98.2% |
| beats per detected bar | varies; the conversion rate is in DATASET_VALIDATION.md |
| section starts on a downbeat | 89% (others at time 0.0 before the first beat, or at the audio end) |
| notes per syllable unit | 1: 86.3%, 2: 12.6%, 3: 1.1%, 4: <0.1% |
| notes with no lyric unit (raw) / continuation flag | 10.3% / 11.7% |
| octave of melody | MIDI 48–71 dominant (sounding vocal pitch; no +12 offset, unlike SheetSage2 output) |

## 7. How MIDI-LLM represents the same data (for comparability)

MIDI-LLM (one-stage v4-cd, the current best) sees a whole song in one sequence of up to 10,240 tokens.

* **Condition:**
  * a JSON text prompt (`task`, `num_sections`, genre/mood tags from `qq_gt_prompt`, `key`, `tempo_bpm`, `meter`, `language`);
  * the lyrics as **phoneme tokens** grouped by `struct:line_start` / `struct:syllable_start` (the model never sees a character);
  * the **given click track** (beats/downbeats and section plan, anticipated AMT controls; never generated);
  * optional countdown tokens.
* **Target:** section plan plus AMT MIDI events (onset, duration, pitch), each note preceded by an alignment pointer `align:syllable:NNN` / `align:continuation:NNN` / `align:no_lyric`. Chords are a separate instrument track.
* **Tasks:** 4 families sampled 30:30:15:15 — prompt+lyrics→lead sheet, lyrics→lead sheet, melody+lyrics→chords, chords+lyrics→melody. There is also scheduled sampling, and decode-time constraints (key, chord density, notes-per-syllable caps).
* **Data preparation:** melisma recovery `_augment_alignment_with_melisma`, and a whole-song gate at 20% wordless melody.

What Qwen-ABC keeps comparable:

* same corpus (D1, v2_lvcr);
* same song split (D3 lists) plus stricter leakage removal;
* same melisma recovery rule;
* same section label canonicalization;
* same "structure is given" condition: tempo, meter, per-section bar counts and irregular bar lengths are in the prompt, as the click track is for MIDI-LLM;
* whole-song generation (the one-stage granularity).

What differs deliberately:

* characters instead of phonemes;
* one task (lyrics+structure → lead sheet) instead of four;
* no genre/mood prompt tags (the POC keeps the condition minimal);
* no decode-time constraints;
* no scheduled sampling;
* human-readable ABC instead of a MIDI vocabulary.

## 8. Recommendation

Use **D1** for the POC, restricted to the D3 song split, with near-duplicate held-out songs removed. It is the only corpus that has all of the following at scale:

* melody;
* chords;
* sections;
* beats/downbeats;
* key;
* tempo;
* note-level lyric alignment.

It is also exactly the data the MIDI-LLM baseline uses. GTSinger and share_singing lack chords and sections and are too small for lead-sheet SFT. They remain candidates for a later human-GT lyric-alignment evaluation.

## 9. Representative raw examples (D1, song `000tQ09FGX0glTUrvCLmQZ`)

```json
"global": {"key": "B major", "key_agreement": "high", "language": "zh", "meter": "4/4", "tempo_bpm": 125.0}
"melody_notes[0]": {"note_id": "n00000", "pitch_midi": 59, "onset_sec": 17.521958, "offset_sec": 17.782448,
                    "onset_grid_time_sec": 17.52, "offset_grid_time_sec": 17.77, "lyric": "月",
                    "lyric_unit_id": "s00000", "is_lyric_continuation": false, "quantization_grid": "beat_subdivision_4"}
"recognized_chords[1]": {"symbol": "B:maj", "root": "B", "quality": "maj", "bass": null, "start_sec": 1.1,
                         "end_sec": 12.92, "start_beat": 0, "end_beat": 24, "source_model": "ISMIR2019-LVCR"}
"sections": [["intro",0.0,17.04],["verse",17.04,44.12],["chorus",44.12,59.72],["chorus",59.72,75.24],
             ["verse",75.24,96.62],["chorus",96.62,133.54],["verse",133.54,157.0],["chorus",157.0,184.12],
             ["outro",184.12,199.32]]
"lines[0]": {"line_id": "l0000", "text": "月台火车在开动祝你一路顺风", "start_sec": 17.44, "end_sec": 25.12}
"syllables[1]": {"syllable_id": "s00001", "text": "台", "start_sec": 17.76, "end_sec": 17.98, "line_id": "l0000"}
"words[1].mandarin": {"character": "台", "pinyin": "tai", "initial": "t", "final": "ai", "tone": null}
"note_lyric_alignment.items[0]": {"note_id": "n00000", "lyric_unit_id": "s00000", "lyric_text": "月",
                                  "match_type": "dp_match", "confidence": 0.8014, "is_lyric_continuation": false}
"beats[0:2]": [{"time_sec": 1.1, "is_downbeat": true}, {"time_sec": 1.58, "is_downbeat": false}]
```

MIDI-LLM v4 section manifest row (D3, `00PAUS0IOugDtuYfx1MvEr:sec000`), abridged:

```json
{"task": "phoneme_text_to_leadsheet_section", "section_label": "verse", "split": "test",
 "text_prompt": "Musical description: pop, rock, male_vocal, rnb, guitar, drums, bass, happy, romantic, relaxing, 120.0 bpm, A# major.",
 "metadata": {"key": "C# major", "tempo_bpm": 166.6667, "meter": "4/4", "quality_tier": "A"},
 "melody_notes_local[0]": {"note_id": "n00000", "onset_sec": 12.004, "offset_sec": 12.3651, "pitch": 68, "syllable_id": "s00000", "continuation": false},
 "chord_events_local[0]": {"onset_sec": 0.34, "offset_sec": 14.92, "symbol": "A:min"},
 "phoneme_units[0]": {"syllable_id": "s00000", "symbol": "uo", "role": "whole"},
 "excluded_hanzi_fields": ["text", "..."]}
```

The caption tag ("A# major") disagrees with the key estimate ("C# major"). This is one reason the POC uses `global.key` and no caption.
