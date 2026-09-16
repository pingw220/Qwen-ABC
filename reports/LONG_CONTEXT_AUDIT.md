# Long-context audit: Qwen3.5-0.8B-Base and the Qwen-ABC training setup

Audited 2026-09-15. Sources:

* model config: `.hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/config.json`;
* modeling code: Transformers 5.17.0, `models/qwen3_5/modeling_qwen3_5.py` and `masking_utils.py`;
* trainer: `qwen_abc/train.py`;
* token counts: `data/generated/abc_v1_20260915_012459/sft_*.jsonl` (stored AutoTokenizer counts) and `data/generated/abc_v2_20260915_120927/validation.json`;
* GPU measurements: `experiments/infra_checks_20260915/check_infra.json` and `left_padding_check.json` (1× L40S).

## Verdict

* **8K context was not a limitation.** The longest SFT example in any split is 5,833 tokens for ABC-v1 and 6,703 for ABC-v2. No example was ever truncated or dropped.
* **The long-range failures happen well inside the context.** Bar-count drift, early EOS and repetition occur in songs of 2–4K tokens.
* **Raising `max_seq_len` to 16K changes nothing.** The run would be bit-identical in batching and optimizer steps (§3).
* **What a real long-context experiment needs** is a different task, not a larger window; see §5 and experiment E3.

## 1. Model context and positions

| property | value |
|---|---|
| `max_position_embeddings` | **262,144** |
| layers | 24: 18 Gated-DeltaNet linear attention + 6 gated full attention (every 4th layer) |
| full attention | 8 query heads / 2 KV heads, head dim 256 |
| RoPE | multimodal RoPE (`mrope_interleaved`, sections [11, 11, 10]); text positions are identical in all 3 axes |
| RoPE theta | 10,000,000 |
| rotated dims | `partial_rotary_factor` 0.25, so 64 of 256 head dims rotate |
| RoPE scaling | `rope_type: default`, no YaRN or linear scaling configured |
| linear attention | no positional encoding: causal conv (kernel 4) plus gated delta-rule recurrence, so no position limit |

**Context scaling.** Transformers supports RoPE scaling through `rope_parameters`, but none is configured and none is needed at 262K native.

**Practical limit in this stack.** Measured with fp32 master weights, bf16 autocast, gradient checkpointing and chunked CE, one sequence of random tokens per step:

| sequence length | peak allocated | fwd+bwd time |
|---|---|---|
| 8,192 | 6.8 GB | 14.9 s (first call; includes Triton autotuning) |
| 16,384 | 8.0 GB | 3.3 s |
| 32,768 | 11.4 GB | 82 s |

Memory is not the limit: 32K fits in 11 GB of a 46 GB L40S.

The 32K step was about 25× slower per token than 16K. This was not profiled; likely causes are kernel re-tuning for a new shape or an SDPA fallback kernel, since flash-attn is not installed. For long training, throughput would need re-measuring.

**Generation cap.** `qwen_abc/generate.py` caps prompt + generation at `max_total` tokens:

* round 1: 8,192 total and 7,680 new tokens;
* round 2: `--max-total` is a flag, and ABC-v2 evaluations use 10,240.

## 2. What the trainer does (verified in `qwen_abc/train.py`)

| question | answer |
|---|---|
| truncation? | **No.** `encode()` *drops* an example longer than `max_seq_len` and records the count. Every run so far has `train_dropped_over_max_len: 0`. |
| one song = one training document? | **Yes.** SFT row = prompt + ABC + EOS. The loss is on the ABC and EOS only; the prompt label is `-100`. |
| packing | **Disabled.** Each row is its own sequence. |
| padding | Right padding with `attention_mask`; labels `-100` on pads. |
| batching | Length-bucketed. Examples are sorted by length with a seeded random tie-break, then filled greedily while `max_len_in_batch × rows ≤ tokens_per_micro_batch` (16,384 *padded* tokens). Batch order is reshuffled every epoch with seed `seed + epoch`. |
| gradient accumulation | `accum = round(tokens_per_update / (tokens_per_micro_batch × world_size))`; round 1 used 8 micro-batches on 1 GPU. |
| definition of `tokens_per_update` | A **padded-token budget**: an update has `accum × world` micro-batches of ≤16,384 padded tokens each (≤131,072). E0 averaged **121K real tokens and 102K supervised tokens** per update (23.10M / 191 and 19.51M / 191). |
| updates per epoch | `ceil(micro_batches / (accum × world))`. E0: ceil(1,528 / 8) = **191 per epoch, 382 total**. The same 2 epochs of ABC-v2 give 250 per epoch (the examples are 29% longer). |
| loss normalization | Summed token NLL over all micro-batches of the update, divided by the update's **supervised** token count (`labels[1:] != -100`, counted before the step). Padding never enters the numerator or the denominator, and it is not a mean of per-micro-batch means. |
| recurrent-state leakage | **None possible.** There is no packing, and right pads follow the real tokens. Transformers also zeroes padded positions before every Gated-DeltaNet mixer during prefill. Round 1 checked that right padding changes a sequence's loss by ≤0.14% (bf16 noise; `scripts/check_padding_invariance.py`). |
| is right padding safe? | Yes (row above). |
| does `max_seq_len` 8192 → 16384 alone change anything? | **No.** `max_seq_len` is read only by the drop filter in `encode()`, which drops nothing at 8,192. Batch construction depends only on `tokens_per_micro_batch`, so the batches, update count, LR schedule and data order are identical. |

**Round-2 additions.** They change no optimizer step:

* **Data-parallel option (torchrun).** Rank r runs micro-batches r, r+N, … of the same update. Gradients are summed, and the loss uses the update's global supervised count, so N GPUs reproduce the 1-GPU step. The equivalence check is in §6.
* **Other options:** `save_every` (intermediate HF checkpoints for generation-based selection) and `stop_after_steps` (a short probe that keeps the full run's LR schedule).
* **Accounting:** logs now also record supervised tokens seen, tokens/s and peak reserved memory.

## 3. Token lengths (Qwen3.5 tokenizer)

### ABC-v1 (round 1, `abc_v1_20260915_012459`)

| split | part | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| train (10,243) | prompt | 342 | 456 | 491 | 563 | 912 |
| | ABC + EOS | 1,858 | 2,542 | 2,723 | 3,130 | 4,921 |
| | total SFT example | 2,202 | 2,986 | 3,194 | 3,679 | 5,833 |
| validation (273) | prompt | 370 | 482 | 535 | 611 | 711 |
| | ABC + EOS | 2,044 | 2,637 | 2,857 | 3,226 | 3,402 |
| | total | 2,435 | 3,099 | 3,386 | 3,763 | 3,960 |
| test (225) | prompt | 362 | 462 | 509 | 564 | 646 |
| | ABC + EOS | 1,982 | 2,710 | 2,826 | 3,057 | 3,222 |
| | total | 2,346 | 3,144 | 3,322 | 3,649 | 3,790 |

| total SFT example longer than | train | validation | test |
|---|---|---|---|
| 4,096 | 19 (0.19%) | 0 | 0 |
| 8,192 | 0 | 0 | 0 |
| 16,384 | 0 | 0 | 0 |
| 32,768 | 0 | 0 | 0 |
| truncated or dropped | 0 | 0 | 0 |

### ABC-v2 (round 2, `abc_v2_20260915_120927`: cleaned boundaries + counters)

| split | part | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| train | prompt | 415 | 547 | 584 | 658 | 1,033 |
| | ABC + EOS | 2,441 | 3,260 | 3,516 | 4,025 | 5,670 |
| | total | 2,858 | 3,794 | 4,062 | 4,629 | 6,703 |
| validation | total | 3,079 | 3,846 | 4,179 | 4,711 | 5,008 |
| test | total | 3,047 | 4,006 | 4,149 | 4,659 | 4,832 |

| ABC-v2 total longer than | train | validation | test |
|---|---|---|---|
| 4,096 | 467 (4.6%) | 16 | 16 |
| 8,192 / 16,384 / 32,768 | 0 / 0 / 0 | 0 | 0 |

ABC-v2 is 1.29× the tokens of ABC-v1. The E3 masked-section examples (§5) have a median of 2,893 and a maximum of 6,733 tokens. **Nothing in this project needs more than 8K.**

## 4. Why "16K" would not be a long-context experiment

A song is 2–7K tokens. Doubling the window adds only padding headroom that no example uses (§2, last row). The long-range problems in round 1 are:

* **bar-count drift inside a section:** a counting problem over a few hundred tokens;
* **early EOS after 6 of 11 sections:** a termination prior;
* **exact repetition of bars:** a local copying preference.

None of these is caused by information falling outside the window. The first two are addressed by representation (ABC-v2 counters, E1), the third by decoding (sweep) and data (clean subset).

## 5. What round 2 does instead (E3)

**Late-section continuation from a true prefix is not new supervision.** Under teacher forcing, whole-song SFT already trains exactly that. It is therefore used as an *evaluation* for every model, free-running from a correct prefix.

**Masked late-section reconstruction (infilling) is the long-range objective that whole-song SFT cannot provide:**

* The prompt holds the plan, all lyrics and the whole song, with one later-half lyric section replaced by a gap.
* Sections whose label already occurred earlier are preferred, so the model must reuse motif identity from a distant chorus.
* The model sees context on *both* sides.

Built by `scripts/build_longrange_tasks.py`:

* 10,230 of 10,243 train songs are eligible; a deterministic 50% (5,089) are mixed into E3 training;
* the infill examples add 14.9M tokens (1.46M supervised) per epoch;
* the longest example is 6,733 tokens.

E3 therefore stays at `max_seq_len` 8,192. Setting 16,384 would be a no-op, and it is not claimed as a 16K run.

## 6. Other infrastructure checks from this audit

* **Batched generation.**
  * **Throughput:** left-padded batch-16 sampling runs 241.6 tok/s against 26.9 tok/s at batch 1 (9×), peaking at 2.9 GB.
  * **Greedy agreement:** batched and batch-1 greedy outputs agree for 59–155 tokens and then diverge, except in 1 of 6 songs, which stays identical for all 400 tokens.
  * **Padding vs noise:** teacher-forced scoring (`scripts/check_left_padding.py`) shows the difference is bf16 batch-shape kernel noise, not padding. The *unpadded* longest row of a batch differs from its batch-1 score as much as padded rows do (argmax agreement 99.3%, max |Δ log p| of the argmax token 0.13), and left-padded rows match right-padded rows (98.6–99.4% vs 98.6–99.6%). Prefill zeroes padded inputs to the linear-attention mixers.
  * **Consequence:** all round-2 evaluations sample at batch 16. E0 is regenerated in the same pipeline, so comparisons never mix protocols. The round-1 batch-1 samples stay as an independent second sample of E0.
* **Data-parallel equivalence:** see `experiments/infra_checks_20260915/ddp_equiv_world{1,2}/train_log.jsonl` and LONG_STRUCTURE_EXPERIMENTS.md §Budget.
