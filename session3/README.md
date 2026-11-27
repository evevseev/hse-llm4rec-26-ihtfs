# Session 3 JSON-constrained reranking

Prepare and complete June Run A using the
[Session 2 quickstart](../session2/QUICKSTART.md). From the repository root,
install the constrained-decoding dependency in that environment:

```bash
uv pip install --python data/venv/bin/python -r session3/requirements.txt
```

Run two June A users as a smoke test, then resume the remaining users in fresh
five-user processes:

```bash
data/venv/bin/python session3/json_constraint.py run --max-new-users 2
data/venv/bin/python session3/json_constraint.py run
```

After generation is complete, validate and compare both branches:

```bash
data/venv/bin/python session3/json_constraint.py compare
```

The default inputs are `session2/outputs/june_a`; the constrained responses,
aggregate, and per-user comparison details are written under
`session3/outputs/june_a_json_constrained/`.

The [aggregate results](../results/session3_summary.json) and the
[independent check](../results/session3_independent_checks.json) are tracked.
Recompute the latter from the local FAR-Trans CSVs and response files:

```bash
data/venv/bin/python session3/eval/verify_june.py \
  --guided-responses session3/outputs/june_a_json_constrained/responses.jsonl
```
