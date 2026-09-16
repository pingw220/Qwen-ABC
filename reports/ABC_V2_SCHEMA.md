# ABC-v2 schema (`qwen_abc_leadsheet_v2`): explicit long-range structural state

Implementation: `qwen_abc/abc_v2.py` (writer, prompt, stripper, counter checker), with hooks in `qwen_abc/abc.py:song_to_abc`.
Tests: `tests/test_abc_v2.py`. Data: `data/generated/abc_v2_20260915_120927`.

ABC-v1 (`reports/ABC_SCHEMA.md`) is unchanged and still used. ABC-v2 is ABC-v1 **plus two annotations**, applied to songs whose section boundaries went through the conservative cleaning pass (`reports/ABC_V2_DATA_CLEANING.md`).

## 1. Why

In round 1 the model almost always wrote the right section *labels* (95%), but only 55–63% of songs had every section's *bar count* right.

* Per section, 90% of sections had the right length. A song has about 10 sections, so a small per-section error compounds.
* The information needed was in the prompt (`P:verse | 8 bars`).
* But the model had to count barlines over hundreds of tokens, with no visible state.

ABC-v2 replaces that implicit counting with state written into the text itself.

## 2. The two additions

```abc
X:1
M:4/4
L:1/8
Q:1/4=97
K:B
P:intro
% section 1/9 | 6 bars
[r:6] "G#m"z6 "D#m"z2 | [r:5] z2 "C#m"z6 | [r:4] "F#"z6 "G#m"z2 | [r:3] z2 "D#m"z6 |
[r:2] "C#m"z8 | [r:1] "F#"z8 |
P:verse
% section 2/9 | 17 bars
[r:17] "G#m"z C/ D/ D2 D2 "D#m"D D | [r:16] z4 "C#m"D2 C2 | [r:15] D C2 D- "F#"D/ z7/ | [r:14] z2 "G#m"z2 D2 D2 |
w: 明 信 片 的 风 景 | 去 过 | 都 叹 气 | 看 天
...
[r:1] "C#m"G, A,/ B,/- "E"B,3/ z/ "F#"z4 |
w: 打 喷 嚏
P:chorus
% section 3/9 | 12 bars
```

1. **Section header.** Directly after every `P:label` line comes a comment line `% section i/N | B bars`: the section's index, the number of sections in the song, and the section's bar count.
2. **Per-bar countdown.** Every bar starts with the remark field `[r:k]`, where k is the number of bars left in the section, counting the current bar.
   * The first bar carries `[r:B]` (copied from the header) and the last carries `[r:1]`.
   * A section ends exactly when the model has written `[r:1] … |`.
   * An irregular bar keeps its `[M:k/4]` right after the counter: `[r:3] [M:2/4] …`.

The prompt adds `| section i/N` to every structure line, so the header is copied, never counted:

```text
Task: write a lead sheet in ABC notation (melody, chord symbols, aligned lyrics, bar countdown).
...
Structure:
P:intro | 17 bars | beats 2 4x16 | section 1/7
<lyric lines>
P:verse | 23 bars | section 2/7
```

Nothing else changes. The following are all identical to v1:

* the header fields;
* note, rest, chord, tie and accidental syntax;
* up to 4 bars per music line;
* the `w:` alignment rules;
* the `|` resynchronization in lyrics.

## 3. Grammar delta (relative to ABC_SCHEMA.md §2)

```
section     := "P:" LABEL "\n" header "\n" (musicline lyricline?)+
header      := "% section " I "/" N " | " B " bars"          I in 1..N, B = bars in this section
bar         := "[r:" K "] " [ "[M:" K "/4] " ] element (" " element)* " |"    K = B - (bar index in section)
```

## 4. Design choices and alternatives measured

All variants use the same 225 test songs and the Qwen3.5 tokenizer. Mean tokens per ABC file, ABC-v1 = 2,036:

| variant | example | tokens / song | overhead | round trip |
|---|---|---|---|---|
| ABC-v1 | – | 2,036 | 0% | 225/225 |
| header only | `% section 2/9 \| 17 bars` | 2,155 | +5.8% | 225/225 |
| **header + per-bar countdown (chosen)** | `[r:17]` | **2,634** | **+29.4%** | 225/225 |
| header + per-bar count-up with total | `[r:5/17]` | 2,866 | +40.8% | 225/225 |
| header + verbose inline field | `[I:bar 5/17]` | 3,056 | +50.1% | 225/225 |
| header + line-level range | `[r:5-8/17]` at line start | 2,403 | +18.0% | 225/225 |
| header + line-level countdown | `[r:14]` at line start | 2,283 | +12.1% | 225/225 |

Tokenization facts behind the choice:

* Qwen splits every digit into its own token.
* `[r:5]` costs 4 tokens (`[r`, `:`, `5`, `]`) at line start and 5 after a barline.
* `[I:bar 5/8]` costs 8 tokens.
* `% section 2/7 | 8 bars` costs about 10 tokens.

Why the per-bar countdown:

* **Tied to each bar:** the state sits at the start of the bar it describes, so the model never counts barlines. It only decrements a number it wrote a few tokens earlier.
* **Local end signal:** "the section ends after `[r:1]`" is a purely local decision. With count-up (`k/N`) the model must compare two numbers; that costs 40% more tokens and gives no extra information.
* **Line-level variants are cheaper** (+12–18%) but leave up to 4 bars of implicit counting per line. They were not chosen because round 1 already showed that implicit counting is what fails. Train-set cost of the choice: 23.1M → 29.8M SFT tokens per epoch (+29%, `validation.json`). Every example still fits in 8K (max 6,703).
* **Deterministic:** one output per song, and byte-identical rebuilds.
* **Standard ABC:**
  * `%` starts a comment, and `[r:…]` is the ABC 2.1 inline *remark* field, so standard ABC tools ignore both.
  * `parse_abc` already skips both, so a v2 file parses to exactly the same canonical song and MIDI as its v1 form.
  * `strip_v2()` removes them and returns the exact v1 text (checked on every example).
* **Easy to parse:** `counter_report()` checks the state with two regular expressions.

## 5. Evaluation of the state itself (`counter_report`)

For generated ABC-v2 text, per song:

| metric | meaning |
|---|---|
| `counter_present_frac` | bars that carry `[r:k]` |
| `counter_self_consistent_frac` | k equals the bars actually remaining in the section as written, i.e. the model's state matches what it did |
| `counter_plan_frac` | k equals the bars remaining under the requested plan, for the same section index |
| `header_plan_frac` | section headers equal to the plan (index, count, bars) |
| `sections_ending_on_counter_1` | sections whose last bar says `[r:1]` |

All five are exactly 1.0 on every one of the 10,741 reference examples (`validation.json`).

These separate two failure modes:

* **Wrong state:** the counter drifts from the plan, so `counter_plan_frac` is low.
* **Ignored state:** the counter is right, but the model ends the section anyway, so self-consistency is low.

## 6. Limits

* The countdown enforces nothing: it is text the model may get wrong. Unlike MIDI-LLM, there is no decoder constraint.
* The overhead is paid on every bar, including instrumental bars of rests.
* Counters describe the requested plan. When the plan itself is noisy (section boundaries from All-In-One), the model learns to count to a noisy target. Hence the cleaning pass and the E1c ablation, which is cleaning without counters.
