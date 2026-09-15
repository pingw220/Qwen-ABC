# Lyrics-aware ABC lead-sheet schema (`qwen_abc_leadsheet_v1`)

Implementation: `qwen_abc/abc.py` (writer and parser), `qwen_abc/canonical.py` (internal model), `qwen_abc/source.py` (corpus adapter).
Tests: `tests/test_abc_roundtrip.py`, `tests/test_splits_and_source.py`.

The format is standard ABC 2.1 plus four conventions: `P:` for section labels, inline `[M:k/4]` for irregular bars, bar-synchronized `w:` lines, and one rule for tied notes in lyric alignment. Every choice below follows from what the corpus actually contains (see DATA_AUDIT.md §6):

* melody onsets lie exactly on a quarter-beat grid (0 of ~58k sampled notes are off it);
* chords change only on beats;
* downbeats come from beat_this, and a minority of bars are not the nominal length (rate in DATASET_VALIDATION.md);
* lyrics are one Mandarin character (or one English word) per syllable, linked to notes.

## 1. Time model

| quantity | unit |
|---|---|
| tick | 1/4 beat (sixteenth note; the beat is always the quarter note: meters are 4/4 and 3/4) |
| ABC `L:` | `1/8`, so 1 ABC length unit = 2 ticks |
| bar | its own integer beat count, taken from the corpus downbeats |
| song start | the first detected beat; audio before it is dropped |
| tempo | one global quarter-note BPM (`Q:1/4=N`, the corpus `tempo_bpm`, rounded) |

Duration text (`d` = ticks): even `d` gives `d/2` units, written `""` for 1, else the digits (`B2` = quarter note). Odd `d` is written `d/`, i.e. d/2 in ABC shorthand (`B/` = sixteenth, `B3/` = dotted eighth, `B15/` = 7.5 eighths). No tuplets, no broken rhythm (`>`), no grace notes.

## 2. Formal grammar (what the writer emits)

```
file        := header body
header      := "X:1\n" "M:" N "/4\n" "L:1/8\n" "Q:1/4=" BPM "\n" "K:" KEY "\n"
KEY         := TONIC ["m"]            TONIC in {C Db D Eb E F F# G Ab A Bb B} (major)
                                      {C C# D Eb E F F# G G# A Bb B} (minor)
body        := section+
section     := "P:" LABEL "\n" (musicline lyricline?)+
LABEL       := intro | verse | prechorus | chorus | bridge | instrumental | outro | other
musicline   := bar{1,4} "\n"               (at most 4 bars per line; a section always starts a line)
bar         := [ "[M:" K "/4] " ] element (" " element)* " |"
element     := [ '"' CHORD '"' ] ( note | rest )
note        := [ACC] PITCH DUR ["-"]      ACC in {^ _ =}; "-" = tie into the next element
rest        := "z" DUR
PITCH       := letter C..B / c..b with "," / "'" octave marks (C = MIDI 60, c = 72)
CHORD       := ROOT [SUFFIX] ["/" BASS] | "N.C."
SUFFIX      := m | 7 | maj7 | m7 | dim | aug | sus4 | sus2 | dim7 | m7b5 | 6 | m6 | 9 | maj9 | m9 | mM7 | 7sus4 | 5
lyricline   := "w: " wtoken (" " wtoken)*
wtoken      := SYLLABLE ("~" SYLLABLE)* | "_" | "*" | "|"
```

### Element boundaries

A bar is written as a left-to-right sweep. A new element starts at every note onset, every note offset, every chord onset and every bar line. So a note is split, and tied with `-`, when a barline or a chord change falls inside it. A rest is split the same way. The chord symbol attaches to the element that starts at the chord's onset and lasts until the next chord symbol.

### Accidentals

Pitches are spelled from the key signature: diatonic pitches get the key's spelling, others a natural if one exists, else a sharp (sharp or C keys) or a flat (flat keys). An explicit accidental is written whenever *either* ABC propagation convention (same-octave or all-octaves, to the end of the bar) would otherwise read a different pitch. The writer's output is therefore unambiguous under both. The parser applies same-octave propagation.

Chord roots use the key's accidental flavour: sharps in sharp keys, flats in flat keys, and C# Eb F# Ab Bb in C major / A minor.

## 3. Lyrics and alignment (the important part)

A `w:` line belongs to the music line directly above it. Its tokens are consumed left to right by that line's **attack notes**. An attack note is a note head that is not the tied continuation of the previous head. Rests never consume a token.

| token | meaning |
|---|---|
| `我` / `hey` | this note starts a new syllable (a CJK character or one latin word) |
| `隆~隆` | this note starts several syllables at once (used for aligner-unmatched syllables, §5) |
| `_` | melisma: this note continues the syllable of the immediately preceding note |
| `*` | this note has no lyric (humming, intro melody, aligner gap) |
| `\|` | skip to the first note after the next bar line |

Writer rules:

* each bar's tokens are listed in order;
* trailing `*` tokens of a bar are omitted;
* bars are separated by `|`;
* trailing empty bars are omitted;
* a music line with no sung notes has no `w:` line.

Because every bar re-synchronizes on `|`, a miscount by a model only damages one bar.

Standard ABC tokens that are parsed but never written: `-` (syllable break inside a word), `\` (escape).

Semantics in the canonical model:

* **one syllable → one note**: the note carries `lyric=(s,)`;
* **one syllable → many notes (melisma)**: the first note carries `lyric=(s,)`; each later note has `melisma=True` and no text. MIDI export writes the text once, on the attack. This avoids MIDI-LLM's old 累累累 spelling bug.
* **tied note**: one canonical note. Its continuation heads consume no `w:` token.
* **rest**: not a note; never aligned.
* **note without lyric**: `lyric=None, melisma=False` (`*`).
* **several syllables on one note**: `lyric=(s1, s2)` (`~`).
* **code switching**: latin words are single syllable tokens mixed freely with CJK characters (`w: hey boy 坦 白`).

`_` is valid only directly after a note that has a syllable or is itself a melisma; otherwise the parser records `orphan_melisma`.

## 4. SFT prompt (the condition)

The prompt mirrors MIDI-LLM, where tempo and bar structure (the click track) are always given and never generated. Lyrics are grouped by section and by the corpus' lyric line:

```text
Task: write a lead sheet in ABC notation (melody, chord symbols, aligned lyrics).
Language: zh
Meter: 4/4
Tempo: 125 BPM
Key: B major
Structure:
P:intro | 8 bars
P:verse | 13 bars | beats 4x6 2 4x6        <- only when a section has irregular bars
月台火车在开动祝你一路顺风
甜酸苦辣在无言中尽在无言中
P:chorus | 8 bars
...

ABC:
```

The completion is the ABC file followed by EOS. Validation (`scripts/validate_abc_dataset.py`) checks two things on every example: the syllables listed in the prompt are exactly the ABC's sung syllables, in order, and the prompt rebuilds byte-identically from the stored spec.

## 5. Example

This is a real training song (000tQ09FGX0glTUrvCLmQZ, abridged):

```abc
X:1
M:4/4
L:1/8
Q:1/4=125
K:B
P:intro
"B"z8 | z8 | z8 | z8 |
z8 | z8 | "Abm"z4 "F#"z4 | "B"z8 |
P:verse
z2 B, B,3/ B,2 B,3/ | B,2 C B,3/ C3/ D z | z F, F, G F2 G2 | F8- |
w: 月 台 火 车 | 在 _ _ 开 动 | 祝 你 一 路 顺 | 风
F3/ z2 D F2 F D/ | "Ebm"F3 F D C B, z | "E"z2 C D3/ B,5/ G, | "F#"F,15/ z/ |
w: 甜 酸 苦 _ | 辣 在 无 言 中 | 尽 在 无 言 | 中
```

How to read the verse:

* `在 _ _` sings 在 over three notes (a melisma).
* `F8-` in bar 4 is tied into `F3/` on the next line. 风 is sung once, so the continuation consumes no token and the next line's first token `甜` goes on the following note, D.
* The final bar `"F#"F,15/ z/` holds 中 for 7.5 eighths.

## 6. Conversion from the corpus (`source.py`)

1. **Beat grid.** Every time is mapped to a fractional beat index by interpolating between the song's own `beats`, then rounded to the nearest tick. Melody uses `onset_grid_time_sec` / `offset_grid_time_sec`, the corpus' quantized times. Onset residuals over a quarter tick are counted (0 on the real songs checked).
2. **Bars** run between consecutive `is_downbeat` beats; a pickup before the first downbeat becomes a short first bar.
3. **Sections** snap to the nearest beat, and a barline is inserted there if needed. Labels are normalized with MIDI-LLM's alias table (`inst` → `instrumental`, `end` → `outro`, `silence` → `other`). The first section absorbs any lead-in.
4. **Melody** becomes monophonic:
   * a later note at the same onset is dropped;
   * an overlap truncates the earlier note;
   * a zero-length note becomes one tick.
5. **Lyric links** come from `note_lyric_alignment`, plus MIDI-LLM's melisma recovery: an unlinked note whose onset lies inside a syllable's time window continues that syllable. Walking the notes in order:
   * the first note of a syllable is its attack;
   * immediately following notes of the same syllable are `_`;
   * a syllable re-linked after a wordless gap becomes wordless (`noncontiguous_repeat_as_wordless`);
   * a continuation flag on a syllable's first note is promoted to an attack.
6. **Unmatched syllables** (in the lyrics but on no note) are joined with `~` onto the previous attack of the same lyric line, else onto the next one. They are dropped only if their line has no sung syllable at all. The lyric *text* therefore survives even where the aligner found no note.
7. **Chords**: LVCR/Harte `root:quality` plus a bass degree maps to ABC symbols. `N` maps to `N.C.`. Consecutive equal chords merge, and chords last until the next chord.
8. **Key**: `global.key` (key-ensemble verdict), normalized to the spelling with fewer accidentals (`C# major` → `Db major`, `Ab minor` → `G# minor`).

## 7. Round-trip assumptions and guarantees

Checked on every song at build time, and again by the validator from the written files:

* `parse_abc(song_to_abc(song))` equals `song` on meter, tempo, key, bar beat counts, sections, every note's onset/duration/pitch/lyric/melisma flag, and chords. Chords are compared after normalization, since duration is implied by the next onset.
* Writing the canonical song to MIDI and reading the melody track back gives identical onsets, durations and pitches (at 480 PPQ, 120 MIDI ticks per canonical tick).
* The writer emits zero parser errors; every bar's content equals its declared `[M:]`.

## 8. Known lossy conversions (corpus → ABC)

| lost | why it is acceptable for this POC |
|---|---|
| absolute seconds, tempo drift between beats | the model works metrically; the MIDI plays at constant tempo |
| audio before the first beat | no melody is ever there (0 notes in the sample) |
| pitch cents, velocity, note confidence | lead-sheet level |
| same-onset duplicate notes, overlaps | the vocal line is monophonic; counted per song in `stats` |
| sub-sixteenth timing | the corpus is already quantized to this grid |
| chord durations shorter than the gap to the next chord | chords are segmental in the corpus (no gaps) |
| exact chord spelling (C# vs Db) and unrecognized qualities | pitch classes are preserved; unknown qualities are counted |
| the lyric **line** a syllable belongs to | kept in the prompt, not in the ABC |
| phonemes, pinyin, tones, syllable timing | not needed for character-level lyric conditioning |
| order of aligner-unmatched syllables relative to notes | they ride on a neighbour note via `~` |

## 9. Parser behaviour on imperfect (generated) ABC

The parser never raises on generated text; it records `errors` and still returns a song when at least one bar was read.

* **Errors (make an output non-strict):** `bar_duration_mismatch`, `lyric_overflow`, `lyric_bar_overflow`, `orphan_melisma`, `broken_tie`, `dangling_tie`, `invalid_chord_symbol`, `off_grid_duration`, `unsupported_token`, `missing_final_barline`, `section_mid_bar`, `empty_section`, `missing_tempo`, `lyrics_without_music_line`.
* **Tolerated (warnings):** `|]`, `||`, `:|`, other header fields, annotations (`"^text"`), a missing key (read as C), no leading `P:` (an `other` section is inserted).
* **Short bars:** a bar shorter than declared keeps its written length and is padded up to a whole beat, so the metrical position of later bars stays well defined.
