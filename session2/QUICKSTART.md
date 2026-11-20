# Session 2 quickstart

Commands below reproduce June A/B from a fresh clone. The run uses PyTorch
FP16 in fresh five-user processes; MLX is not needed. The course starter,
FAR-Trans data, model weights, and raw responses stay in ignored local
directories.

## 1. Get the course starter and create the runtime

From the repository root, with `uv` and Python 3.13 available:

```bash
mkdir -p data/upstream
git clone --filter=blob:none --sparse \
  https://github.com/dryjins/RecSys-LLMs.git data/upstream/RecSys-LLMs
git -C data/upstream/RecSys-LLMs checkout 51337032503c56e3a28ef646a24aa016782e1bd3
git -C data/upstream/RecSys-LLMs sparse-checkout set project/far-trans

uv venv data/venv --python 3.13
uv pip install --python data/venv/bin/python \
  -r session2/requirements.txt
```

## 2. Add local data and the pinned model

Download and extract the official FAR-Trans archive so these files are directly
inside `data/FAR-Trans/`: `transactions.csv`, `asset_information.csv`,
`customer_information.csv`, and `close_prices.csv`.

```bash
data/venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir data/models/qwen2.5-1.5b-instruct
```

## 3. Prepare and run June A, then June B

```bash
data/venv/bin/python session2/june_a.py prepare \
  --data-dir data/FAR-Trans --output-dir session2/outputs/june_a
data/venv/bin/python session2/june_a.py run \
  --output-dir session2/outputs/june_a --backend torch --block-size 5

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

`run` resumes from saved response records. Each worker process handles at most
five users; an inference memory error stops the parent run with completed
responses retained. June B requires completed A and checks that only candidate
row order changed. Optional fence-removal sensitivity analysis, kept separate
from the primary results:

```bash
data/venv/bin/python session2/posthoc_fence_repair.py \
  --a-dir session2/outputs/june_a \
  --b-dir session2/outputs/june_b \
  --output-dir session2/outputs/posthoc_fence_repair
```

The checked aggregate outputs from the existing run are in
`results/session2_summary.json`. Per-user outputs and raw model responses are
local only under `session2/outputs/`.
