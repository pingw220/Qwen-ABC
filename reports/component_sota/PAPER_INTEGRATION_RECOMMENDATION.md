# Paper integration recommendation

This is based only on the measured results in `FINAL_COMPONENT_SOTA_REPORT.md`. No human study has
been run.

## A. Strongest defensible thesis

> A single 0.8B general-purpose language model, fine-tuned on text lead sheets, is competitive with
> or better than specialized systems at each symbolic stage of lead-sheet generation:
> - lyrics→melody, above CSL-L2M even when CSL-L2M is retrained on the same data;
> - melody→chord, above AccoMontage2.
>
> Generating the whole lead sheet jointly costs little melody fidelity. It buys structural adherence
> and chord–melody agreement, and enables local section editing that whole-song regeneration cannot
> provide.

State it with the qualifiers: *on automatic, reference-based metrics against transcribed
pseudo-references*, *Mandarin pop*, and *pending human evaluation*.

## B. Main contributions (3–5)

1. **Lead sheets as text for a general LLM, with explicit section structure.** Joint generation
   reaches 0.991 exact structure (seed S1; 0.982 over 4 samples) at the same cost as a
   melody-only model.
2. **A controlled decomposition study.**
   - Melody-Only vs Full at equal updates: the multitask cost is quantified (Q3–4).
   - CSL-L2M official vs retrained on the same split separates data from architecture (Q1–2).
3. **A harmonizer baseline result:** Qwen Melody→Chord > AccoMontage2 on reference similarity,
   compatibility and corpus plausibility. Report it with the diatonic-heuristic control.
4. **Local editing by section infill:** 99.7–100% of untouched sections are byte-identical, vs
   0.1–5% for regeneration. The small, quantified cost falls on the target section.
5. **A lyric-conditioning finding** (a negative result, reported as such): across three
   lyrics→melody systems, replacing the lyrics moves the melody barely more than reseeding
   (ratios 1.000–1.005), and no system follows Mandarin tones.

## C. Main-paper tables and figures

| paper item | source |
|---|---|
| Table: lyrics→melody | Table 1 (`tables/lyrics_to_melody.md`) with CSL official (whole and chunked), CSL retrained (chunked), Melody-Only, Full, reference; columns PD, DD, MD, lyric recall, cramming, structure, success |
| Table: joint vs cascade | Table 5 (5 main rows: Joint, Cascade-Qwen, Qwen Mel→AM2, CSL→Qwen Chord, CSL→AM2) |
| Table or figure: infill | Table 3 or Figure 4 (locality vs target success) |
| Figure | Figure 1 (decomposition), Figure 6 (joint vs cascade) |
| Short table | Table 4 (harmonizers), including the diatonic control row |

## D. Appendix

- Table 2 and Figure 3 (lyric conditioning, tone test).
- Table 6 and Figure 7 (SVS cross-render and the interaction, including the finding that
  CSL-retrained melodies are the most singable).
- Tables 7–8 and Figure 8 (backing audio; closed-loop control).
- Section semantics.
- The CSL-L2M adapter and failure-by-length analysis.
- DD_tb and the notation-timebase issue.
- Measurement fixes.
- The listening-study protocol (as a protocol, not as results).

## E. Should lyrics→melody become a major contribution?

**Yes, as a component result, not as the headline.**
- The margin over CSL-L2M is large and robust to retraining.
- But the only external baseline is one system, the metrics are reference-based, and CSL-L2M's
  melodies turned out *more singable* (lower PER, higher pitch accuracy under both SVS).
- Present it as "competitive/superior on published metrics", together with the singability caveat.

## F. Should melody→chord become a major contribution?

**A secondary contribution.**
- Qwen clearly beats AccoMontage2 symbolically.
- However, frame-level similarity to pseudo-labels does not separate Qwen from a diatonic heuristic,
  part of the margin is in-distribution, and rendered audio shows no difference.
- The defensible claim is about progression and cadence plausibility and melody compatibility,
  not correctness.

## G. Should joint-vs-cascade be central?

**Yes. It is the most paper-relevant new result.**
- It shows what joint generation costs and buys. Joint vs matched cascade is a structure-vs-fidelity
  trade, not a free lunch.
- The specialized cascade loses on almost everything at similar per-song cost.
- It also supports the "one general model instead of several specialized ones" framing, with the
  qualifier: on automatic metrics.

## H. How to frame infilling

**As an editability demonstration with a strong quantitative result**, placed in the main paper and
kept short. It is not a core method: no infill-specific model was trained beyond E3b's mixture.
Report the target-side cost:
- new-lyric recall −5.4 points;
- +4-bar extension −2.9 points.

## I. How to discuss YuE / large audio models

- Do **not** compete on absolute audio quality. This round measured no end-to-end audio model, and
  our audio numbers depend on the renderer (Q14–16).
- Position the work as the *symbolic, controllable* layer: explicit structure, editable sections,
  deterministic transposition.
  - Key control is reliably audible only via symbolic transposition (78% vs 39%).
  - Tempo survives rendering (100% up to octave).
  - Chords survive rendering (F1 0.92).
- This complements audio models; it does not replace them.

## J. Claims we should NOT make

- "Better than all lyrics-to-song / lyrics-to-melody systems": one external baseline only.
- "Better audio quality than YuE" (or any audio model): not measured.
- "Qwen melodies are more singable": false. CSL-retrained melodies have lower PER under both renderers.
- "Lyrics causally determine the melody" / "the model sets lyrics to music": lyric swap ≈ reseed.
- "The model follows Mandarin tones": no effect detected.
- "Section labels carry section semantics": only a partial register effect, about 1/3 of the real gap.
- "Qwen's chords are correct / pseudo-reference chords are the unique right harmony": the reference
  is a transcription, and a diatonic heuristic matches Qwen on chord F1.
- "Qwen harmonization sounds better than AccoMontage2": audio metrics show no difference, and
  humans are pending.
- "Music pretraining hurts": not tested here.
- "Joint generation is free": it is a measured trade.
- Any listener-preference claim: the listening study is packaged, not run.
