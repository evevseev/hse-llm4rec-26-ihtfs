# Session 2: frozen-pool RRF and Qwen reranker

This implementation uses the session1 candidate protocol and prompt. It retains
the starter's `popularity`, `cbf`, `latest_buy`, and `latest_buy_or_sell` outputs,
adds the exact 20/20/10 deduplicated candidate pool and an RRF Top-10, and sends
the same 50 candidates to Qwen. Invalid or incomplete answers are recorded and
use that user's RRF Top-10 without retrying.

The data adapter and baseline/evaluator functions are imported from the pinned
course starter under `data/upstream/`. Session 2 supplies the frozen-pool,
prompt, validation, and PyTorch process-block runner. Local data, weights, and
response logs are not copied into the repository.

## Prepare and verify the May development pass

From the repository root:

```bash
data/venv/bin/python session2/experiment.py prepare \
  --split dev_may \
  --data-dir data/FAR-Trans \
  --output-dir session2/outputs/dev_may

data/venv/bin/python session2/experiment.py check \
  --output-dir session2/outputs/dev_may
```

The development cohort is selected deterministically at the May cutoff and
excludes the June 100-user cohort without consulting the May labels. `check`
prints one user's pool, confirms RRF and Qwen have identical candidate
membership, and sends an intentionally malformed response through the adapter
to verify the RRF fallback.

## Run and resume inference

The first measured segment is exactly ten calls from the May development pass:

```bash
data/venv/bin/python session2/experiment.py run \
  --output-dir session2/outputs/dev_may \
  --model-dir data/models/qwen2.5-1.5b-instruct \
  --device auto --max-new-calls 10

data/venv/bin/python session2/experiment.py probe-batch \
  --output-dir session2/outputs/dev_may \
  --model-dir data/models/qwen2.5-1.5b-instruct \
  --device auto --batch-size 2
```

`auto` chooses CUDA, then Apple MPS; CPU inference is deliberately disabled for
this GPU timing run. The local snapshot is checked against the pinned revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; generation is greedy FP16 with 96
new tokens and an 8192-token input budget. Each response attempt is appended to
`responses.jsonl`, and generated text is cached by prompt and model settings.
Malformed responses are completed fallback cases, not retried calls.

`timing.json` records model-load time, device/dtype, per-call generation
durations, completed calls, and remaining calls. The May output directory now
contains the completed 100-user pass. June A and B use separate runners and
output directories.

## First measured May segment

The first ten development calls were run and saved on Apple MPS with FP16
(`torch 2.11.0`, `transformers 4.57.6`; CUDA was unavailable). Pool/prompt
preparation took 5.12 s, model loading took 12.08 s, and the ten generations
took 123.56 s total (12.36 s mean, 12.21 s median). A direct 300-call
extrapolation is 61.8 minutes for generation alone, before setup; this exceeds
the plan's one-hour cap and 45-minute target on this machine.

All ten raw completions were wrapped in Markdown code fences despite the
frozen “JSON only” instruction. The strict response contract therefore counted
all ten as `invalid_json` and used the corresponding RRF Top-10.

Batch-2 was then compared on those same two saved requests. Both raw responses
and both resulting Top-10 lists matched the singleton records exactly; batch
generation took 25.41 s versus 27.49 s for the two singleton generations. The
batch path was used for the remaining May users. Of the full 100-user May pass, 30
responses were generated but rejected by the unchanged strict JSON contract;
the other 70 requests hit MPS out-of-memory errors and used RRF fallback. No
failed response was retried. Consequently the May `qwen` and `rrf` metrics are
identical and do not represent valid Qwen ranking coverage. The complete
response/fallback log, batch comparison and final metrics are in
`outputs/dev_may/`.

## June Run A: MLX parity gate and result

June A uses a separate `outputs/june_a/` directory. The MLX conversion is
separate from the source snapshot; its saved tensors are FP16. The MLX probe
was faster on two saved May prompts (9.41 s versus 27.49 s), but the raw answers
did not match PyTorch and one resulting Top-10 differed, so the parity gate
selected PyTorch. June A then ran singleton requests in blocks of five users,
each block in a fresh process. It completed all 100 users without an MPS memory
error.

All 100 June PyTorch outputs were Markdown-fenced rather than raw JSON. The
unchanged strict validator recorded `invalid_json` and used RRF for every user.
Accordingly the June A `qwen` and `rrf` metrics are identical (all-user nDCG@10
0.035536; 11 users have nonempty truth). The MLX parity record, generation
outputs, responses, block progress and metrics are saved under `outputs/june_a/`.
The MLX probe was a one-time backend check. A new clone can reproduce June A
directly with PyTorch and does not need MLX:

```bash
data/venv/bin/python session2/june_a.py prepare \
  --data-dir data/FAR-Trans --output-dir session2/outputs/june_a
data/venv/bin/python session2/june_a.py run \
  --output-dir session2/outputs/june_a --backend torch --block-size 5
```

## June Run B: candidate-order sensitivity

June B reuses the June A backend (PyTorch) and the same five-user fresh-process
blocks. Before inference, an automated check confirmed for all 100 users that
the candidate IDs, aliases, source ranks, RRF Top-10, baselines, features and
history were unchanged; only the candidate-row order was permuted. The row
order changed for every user.

The B run completed without a memory error. Raw model responses matched A for
3/100 users and differed for 97/100. The final Top-10 order matched for 100/100
users with full set overlap: both runs' 100 responses were rejected as
`invalid_json` because they were Markdown-fenced, so strict fallback produced
the same RRF Top-10 in A and B. The per-user comparison is in
`outputs/june_b/comparison_with_a.json`.

```bash
data/venv/bin/python session2/june_b.py prepare \
  --data-dir data/FAR-Trans \
  --a-dir session2/outputs/june_a \
  --output-dir session2/outputs/june_b
data/venv/bin/python session2/june_b.py run \
  --a-dir session2/outputs/june_a \
  --output-dir session2/outputs/june_b --block-size 5
data/venv/bin/python session2/june_b.py compare \
  --a-dir session2/outputs/june_a --b-dir session2/outputs/june_b
```

## Separate post-hoc fence-removal analysis

`posthoc_fence_repair.py` reads the saved A/B outputs only. It removes exactly
one outer Markdown JSON code fence, then applies the same strict JSON schema,
ten-unique-alias, in-pool checks, and RRF fallback. It writes to
`outputs/posthoc_fence_repair/`; the official A/B responses, predictions and
metrics are not edited.

This sensitivity analysis accepted 86/100 A responses and 93/100 B responses;
the rest still failed unchanged validation. Recomputed `qwen_fence_repaired`
nDCG@10 is 0.026642 for A and 0.018238 for B, versus RRF 0.035536. With repaired
responses, A/B Top-10s match for 3 users and differ for 97 (mean set overlap
2.89/10). The original strict results remain unchanged: every fenced response
falls back to RRF there.

Example user `870E814DB274CFA95600` has two relevant assets. After fence removal,
A puts one in its Top-10 and gets nDCG@10 0.613147; B puts both at ranks 1 and 3
and gets 0.919721. Before this post-hoc pass, both used the same RRF fallback
and scored 0.806574. No model was run again.

```bash
data/venv/bin/python session2/posthoc_fence_repair.py \
  --a-dir session2/outputs/june_a \
  --b-dir session2/outputs/june_b \
  --output-dir session2/outputs/posthoc_fence_repair
```

## Tests

```bash
data/venv/bin/python -m unittest discover -s session2 -p 'test_*.py' -v
```
