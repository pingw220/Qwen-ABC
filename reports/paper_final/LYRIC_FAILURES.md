# Lyric realization and cramming

Tables `tables/lyric_failures.*` (Table 8), `tables/lyric_pressure.*`; correlations
`data/lyric_correlations.json`; position profile in `LONG_RANGE_ANALYSIS.md`; code
`paper_eval/failures.py`. 225 test songs; E3b = 4 paired-seed samples per song unless noted.

**Definitions.** *Crammed syllables*: share of sung syllables that sit on a note carrying more than
one syllable (the ABC `~` join). *Lyric recall*: in-order LCS coverage of the requested syllables.
*Wordless notes*: neither a syllable nor a melisma. *Melisma notes*: continue the previous syllable.

## Current model vs corpus and every earlier attempt (single samples, mean of 4)

| system | crammed syllables | lyric recall | notes >1 syllable | wordless notes | melisma notes | strict validity |
|---|---|---|---|---|---|---|
| pseudo-GT reference | 0.057 [0.047, 0.070] | 1.000 | 0.023 | 0.036 | 0.161 | 1.000 |
| **E3b** (paired-seed) | **0.173 [0.164, 0.183]** | 0.970 | 0.057 | 0.035 | 0.185 | 0.777 |
| E3b (legacy R3-B samples) | 0.162 [0.153, 0.172] | 0.972 | 0.054 | 0.033 | 0.191 | 0.791 |
| E3b selector@4 | 0.114-0.116 | 0.980-0.982 | | | | 0.969-0.978 |
| R3-A: syllable budget in the prompt | 0.160 | 0.978 | 0.051 | 0.044 | 0.204 | 0.724 |
| R3-C: repaired targets | 0.063 | 0.973 | 0.014 | 0.053 | 0.191 | 0.633 |
| decode-time ban of `~` | **0.000** | 0.904 | 0.000 | 0.036 | 0.187 | **0.369** |
| R5: free lyric assignment | 0.162 | 0.900 | 0.047 | 0.061 | 0.250 | 0.786 |

(The R3-B/R3-C/ban/R5 rows re-score the existing 4-sample generations with the same code; they
reproduce CRAMMING_ATTACKS.md and R5_FREE_LYRIC_ASSIGNMENT.md, whose selector@4 numbers differ.)

## Where cramming happens

**Requested lyric density** (syllables per bar in the requested section):

| syllables/bar | sections (E3b samples) | E3b cramming | reference cramming | E3b section lyric recall |
|---|---|---|---|---|
| < 2 | 1,459 | 0.285 [0.245, 0.325] | 0.142 [0.105, 0.182] | 0.929 |
| 2-4 | 3,413 | 0.165 [0.152, 0.179] | 0.056 [0.043, 0.072] | 0.969 |
| 4-6 | 2,052 | 0.191 [0.177, 0.204] | 0.051 [0.039, 0.065] | 0.969 |
| 6-8 | 514 | 0.205 [0.179, 0.232] | 0.049 [0.032, 0.070] | 0.953 |
| ≥ 8 | 177 | 0.298 [0.209, 0.405] | 0.256 [0.143, 0.391] | 0.882 |

**Song position** (E3b, T1.0): cramming rises from 0.160 in the first fifth to 0.230 [0.214, 0.246]
in the last; lyric omission from 0.032 to 0.054.

**Song length and label quality** (Spearman over 225 songs): total syllables vs cramming +0.15,
vs lyric recall −0.25; reference (label) cramming vs generated cramming +0.17.

## Measured facts

1. The current model crams **3× the corpus rate** (0.173 vs 0.057) while singing 97% of the lyrics
   in order; selection reduces it to 0.114-0.116; the 4-sample oracle is 0.086-0.092.
2. Cramming is U-shaped in requested density: it is highest where the request is *sparse* (<2
   syllables/bar: 0.29, 2× the corpus there) and where it is *very dense* (≥8: 0.30, near the corpus's
   own 0.26). In between it is flat at 0.17-0.21 vs the corpus's 0.05.
3. Cramming and omission both rise toward the end of the song.
4. Every attack that removed cramming broke something else: the decode-time ban costs half of
   validity (0.37); repaired targets pass the metrics but listening found near-monotone melodies
   (CRAMMING_ATTACKS.md); a syllable budget in the prompt changes nothing.
5. Old-lyric leakage after a lyric swap is at chance (INTERVENTIONAL_CONTROLLABILITY.md): lyrics that
   are sung are the requested ones.

## Interpretation

Lyrics are realized faithfully as *text* (recall 0.97, zero leakage) but their *placement* on notes
is weak: the model does not budget syllables against the melody it writes, most visibly in sparse
sections and late in the song. (Why sparse sections cram most is not established here; that bucket
includes the 1-2-syllable fragment sections that R5 traced to boundary-split lyric lines.) This agrees
with the interventional result that lyric content hardly shapes the melody, and with the
tone-melody test (the melody ignores lexical tones the corpus weakly follows). The negative results
are kept on record rather than optimized away.
