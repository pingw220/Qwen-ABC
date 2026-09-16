# From an ABC lead sheet to a full-band recording

Qwen-ABC writes lead sheets, not audio. To hear one, it has to go through the
same two models MIDI-LLM's lead sheets go through: **FastSinger** sings the
melody, and **MuseControlLite** (Stable Audio Open 1.0, driven by MIDI-SAG's
inference script) generates the backing around that vocal.

The path is:

```
generated.abc
  └─ scripts/export_leadsheet_midi.py          (this repo)
       ├─ <name>.mid       lead sheet: conductor markers, melody+lyrics, chords, click
       ├─ <name>.fs.mid    melody FastSinger sings
       └─ <name>.fs.txt    one symbol per note of that melody
  └─ scripts/slurm/midi_sag_render.sbatch
       ├─ FastSinger                     -> vocal_fs.wav
       └─ MIDI-SAG run_midi_llm_to_midi_sag.py --vocal-audio vocal_fs.wav
            ├─ chord condition from the lead sheet's own chord track
            ├─ rhythm condition from the click track
            ├─ structure condition + 47.55 s windows from the section markers
            └─ mix.wav, backing.wav
  └─ scripts/collect_midi_sag_audio.py         -> full_band.mp3 next to the score
```

Nothing in MIDI-SAG or FastSinger is modified. Both are called exactly as their
own scripts call them; this repo only writes their inputs.

## Why an ABC-specific adapter rather than reusing MIDI-LLM's

MIDI-SAG's adapter (`MIDI-SAG/tools/midi_llm_adapter/`) exists because MIDI-LLM
lead sheets need repair before a renderer can use them: their click track
disagrees with their own tempo header by 1–19 BPM, their section boundaries pile
notes onto a single tick, and their melody is not reliably monophonic. Feeding
it ABC would mean reconstructing a grid we already have exactly.

Writing the renderer's inputs from the canonical `Song` instead gives, by
construction:

* **one timeline** — the click is the score's own bar/beat grid, so click,
  notes, chords and tempo map cannot disagree;
* **exact section starts** — sections begin on barlines in ABC, so there is no
  stitching pile-up to detect or tolerate;
* **a monophonic melody** — ABC is a single voice, so nothing has to be dropped
  to make the vocal singable.

MIDI-SAG's own validator agrees: all 19 exported lead sheets pass with **0
errors** (the warnings are irregular bars, which are real in this corpus).

## The two files FastSinger needs

`midi_lyric_to_jlines` pairs the melody MIDI and the lyric text **by position
only**: symbol *k* sings note *k*. Too many symbols and the tail is sliced off;
too few and `#` is padded at the end, so one missing symbol shifts every later
word onto the wrong note. A previous FastSinger run on MIDI-LLM lead sheets
misplaced up to 220 of 409 symbols this way.

`qwen_abc/fastsinger.py` builds both files from one note list, so the two can
not disagree, and `scripts/export_leadsheet_midi.py` refuses to write a lead
sheet whose counts do not match. Four rules, each counted in the audit JSON:

| case | what happens | why |
|---|---|---|
| wordless note (an instrumental line) | not sung, absent from both files | it is not the vocal; if it stayed, the next word would be sung on it |
| melisma note (`~` in ABC) | one `#` | holds the previous syllable, which is what `#` means |
| note carrying several syllables | the note is divided between them | the corpus joins a syllable it could not place onto its neighbour; dropping all but the first cost 18% of the lyrics on one song |
| non-Chinese syllable | sung as a held vowel | FastSinger's Mandarin lexicon has no pronunciation for it |

Over the 19 exported lead sheets: 6 861 melody notes → **6 786 sung symbols**,
202 notes divided between syllables, 388 wordless notes not sung, and **37
syllables (0.5%) dropped** for want of room. FastSinger then reports "Total
number of notes: N, Total number of words: N" for every song — no slicing, no
padding.

## Sections and the generation window

MuseControlLite generates in 47.554 s windows anchored on section starts, and
its structure condition is an embedding over eight tags. `plan_segments`
converts our labels (`intro verse prechorus chorus bridge instrumental outro
other`) to those tags — only `prechorus → verse` and `instrumental → inst` are
not the identity — and splits any section wider than one window into equal parts
at barlines, keeping the section's own tag. On these songs no split was needed:
the widest segment is 26.9 s.

## Running it

```bash
source scripts/env.sh
python scripts/export_leadsheet_midi.py --sample-dir experiments/listen_now_<ts> \
    --outdir experiments/midi_sag_<ts>/leadsheets

LEADSHEETS=$PWD/experiments/midi_sag_<ts>/leadsheets \
OUT=/gscratch/ark/pingw220/qwen_abc_r2_offload/midi_sag_<ts> \
    sbatch --chdir=$PWD/experiments/midi_sag_<ts> scripts/slurm/midi_sag_render.sbatch

python scripts/collect_midi_sag_audio.py \
    --render-dir /gscratch/ark/pingw220/qwen_abc_r2_offload/midi_sag_<ts> \
    --sample-dir experiments/listen_now_<ts>
```

About 3.5 min of L40 time per song (20 s of singing, the rest diffusion) and
~200 MB of wav, which is why the audio lives on `/gscratch/ark` and only the
mp3 comes back next to the score. Both stages skip songs that are already
rendered, so the job can be requeued.

## What a listening comparison can and cannot say

Every model in a sample directory goes through the *same* singer, the *same*
harmonization source and the *same* backing model, so a difference between two
mixes comes from the lead sheet. It does not follow that the better-sounding
mix is the better lead sheet: the vocal is synthetic, the backing is generated
per 47 s window, and both add their own artefacts. Treat these as auditions of
the notation, not as evidence of musical quality — that needs the listening
study, not this pipeline.
