# R5: the model does not need to be told which section sings which lyrics

Every prompt in this project has handed the model the lyric lines already sorted
into sections. That sorting comes from the corpus labels, and the labels cut
lyric lines mid-phrase often enough to be audible — 31% of test songs have a
section holding one or two syllables, which is why the openings sound almost
wordless.

R5 removes the assignment: the prompt gives the section plan (labels, bars,
`section i/N`) and then the whole lyric as one block, with the
boundary-split fragments **rejoined** (`qwen_abc/abc_free.py`). Rejoining alone
takes fragment lines from 6.6% to 1.0% and songs containing one from 29.2% to
9.0% — the defect leaves the prompt entirely. Completions are byte-identical to
E3b's, so this is one variable.

## The information is not needed

225 held-out songs, T=1.0 / top-p 0.95, best-of-4, paired bootstrap CIs:

| metric | E3b (assignment given) | R5 (model decides) | difference |
|---|---|---|---|
| strict-valid ABC | 0.978 | 0.978 | +0.000 [−0.022, +0.027] |
| exact structure | 0.996 | **1.000** | +0.004 [+0.000, +0.013] |
| no lyric-alignment error | 0.982 | 0.987 | +0.004 [−0.018, +0.027] |
| early EOS | 0.000 | 0.000 | — |
| lyric recall | 0.982 | 0.959 | **−0.023** |
| section-local lyric recall | 0.981 | **0.566** | **−0.415** |
| syllables sung crammed | 0.114 | 0.124 | +0.010 [−0.003, +0.022] |

Validity is identical, structure is perfect, and 96% of the lyrics still get sung
in order. What changes is *where* the words land: the model places them somewhere
other than the corpus did on nearly half the material.

That number cannot say who is right, because the corpus placement is the thing
under suspicion. Only listening settles it, and a listening pass over four songs
did.

## What listening found

| song | what the model did with the freedom |
|---|---|
| `1tKSEmbS7vmv7cFc24sjlI` | **worked**: left intro, instrumental and outro wordless on its own, redistributed the rest, kept every syllable (recall 1.000, local 0.953) and cut cramming 0.141 → 0.103. Cost: repeated bars 0.326 → 0.562, and a listener called the melody repetitive |
| `7qDiugtLVz4njwHDGCfdB5` | close to the corpus placement (local 0.917) |
| `5CbNIRIgmrh1W9xNnhtSz1` | **allocated unevenly**: cramming 0.103 → 0.394 in some sections while wordless notes went 0.011 → 0.090 in others. A listener heard exactly this: "melody with no lyrics in places" |
| `1Fo23uW11ZN4JCFCeB0ngS` | **front-loaded**: 46 syllables into a section the corpus gives 4, then ran out — recall 0.626, late-song recall 0.985 → 0.731. A listener heard it "stop before the song ended" |

So the model *can* allocate — it works out by itself that an instrumental section
has no words, which is the one judgement the labels get right by construction —
but it does not **budget**. It spends syllables early and runs short, or piles
them into one section and leaves another with a melody and nothing to sing.

Freeing the prompt also barely moved the defect it was aimed at: sections holding
one or two syllables went 3.5% → 2.9% (corpus 3.8%), songs affected 29.8% →
26.7%. The reason is structural — the *completions* still carry the corpus
assignment, so the target still teaches the fragmentation that the prompt no
longer contains.

## What it says to do next

The failure has a name and an obvious counterpart. R3-A put a syllable budget in
the prompt and it did nothing, because the assignment was already given and the
budget had no job. Here the assignment is free and budgeting is precisely the
missing skill. **Free assignment plus a per-section syllable budget** is the
experiment this result asks for, and it is one dataset variant and one ~5 GPU-h
run.

## An aside worth recording: some songs are mislabelled, not badly generated

`7qDiugtLVz4njwHDGCfdB5` is a school anniversary anthem. Its labels open with
`bridge` and put `intro` second — an ordering that cannot be right — and its
closing section, "就在今天为您唱 / 祝您生日快乐", is labelled `chorus` rather than
an outro. The fragment `吉水` appears as a standalone line five times, split off
from `吉水二中我们的梦…`. A listener judged the middle of this song good and the
opening and ending wrong, which is exactly where the labels are wrong.

Nothing a model does can fix that, and no metric computed against those labels
will report it. It is the same conclusion the cramming work reached from another
direction: the ceiling here is the annotation.

## Reproduction

```bash
python scripts/build_prompt_variant.py --prompt free \
    --v2-dir data/generated/abc_v2_20260915_120927 --output-dir data/generated/abc_free_<ts>
CONFIG=configs/r5_free_lyrics_sft.yaml RUN_DIR=experiments/r5_free_<ts> \
    sbatch --partition=gpu-a40 --account=ark --gpus=2 -c 8 --mem=120G scripts/slurm/train.sbatch
CKPT=<run>/final_model DATA=data/generated/abc_free_<ts> OUT=experiments/evals_r5/... FMT=free \
    ARGS="--split test --seeds 4 --temperature 1.0 --top-p 0.95" sbatch scripts/slurm/eval.sbatch
```

Run `experiments/r5_free_20260924_100314` (job 40565949, 762 updates against
E3b's 758, 6.1 GPU-h), evaluation job 40587702 (~3 GPU-h), dataset
`data/generated/abc_free_20260924_095912`. Samples and audio:
`experiments/listen_now_20260924`, packaged as
`experiments/leadsheets_free_lyrics_20260924.tar.gz`.
