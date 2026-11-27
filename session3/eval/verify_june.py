#!/usr/bin/env python3
# Generated with an AI assistant for an independent result check.
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from math import log2
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "session2" / "eval"))
from verify_metrics import ground_truth

BASELINE_NAMES = ("popularity", "cbf", "latest_buy", "latest_buy_or_sell")
FENCE = re.compile(r"```json\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)
DISCOUNTS = [1 / log2(rank + 1) for rank in range(1, 11)]


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse(text, pool, fallback, repair=False):
    if text is None:
        return fallback, "generation_error", False
    fence = FENCE.fullmatch(text) if repair else None
    try:
        obj = json.loads(fence[1] if fence else text)
    except json.JSONDecodeError:
        return fallback, "invalid_json", bool(fence)
    if not isinstance(obj, dict) or set(obj) != {"ranked_aliases"}:
        return fallback, "invalid_schema", bool(fence)
    aliases = obj["ranked_aliases"]
    if not isinstance(aliases, list) or len(aliases) != 10:
        return fallback, "invalid_count", bool(fence)
    if any(not isinstance(alias, str) for alias in aliases):
        return fallback, "invalid_alias_type", bool(fence)
    if len(set(aliases)) != 10:
        return fallback, "duplicate_alias", bool(fence)
    if any(alias not in pool for alias in aliases):
        return fallback, "out_of_pool_alias", bool(fence)
    return [pool[alias] for alias in aliases], None, bool(fence)


def score(ranking, truth):
    hits = [rank for rank, asset in enumerate(ranking, 1) if asset in truth]
    gain = sum(DISCOUNTS[rank - 1] for rank in hits)
    ideal = sum(DISCOUNTS[:len(truth)])
    return {"hit_ranks": hits, "dcg": gain, "idcg": ideal,
            "ndcg": gain / ideal if ideal else 0.0}


def verify(data, reference, guided):
    protocol = json.loads((reference / "protocol.json").read_text())
    cutoff, end = protocol["label_window"]
    eligible, ordered, ground = ground_truth(data, cutoff, end)
    cohort = ordered[:100]
    inputs = json_lines(reference / "frozen_inputs.jsonl")
    labels = json_lines(reference / "labels.jsonl")
    free = json_lines(reference / "responses.jsonl")
    constrained = json_lines(guided)
    for tag, records in (("inputs", inputs), ("labels", labels), ("free", free), ("guided", constrained)):
        matches = [record["user_id"] for record in records]
        if matches != cohort or matches != protocol["users"]:
            raise ValueError(f"{tag} does not match the independent cohort")

    methods = {}
    errors = {key: Counter() for key in ("free", "free_unwrapped", "guided", "guided_unwrapped")}
    fences = Counter()
    candidate_recalls = []
    token_count_matches = 0
    paired = Counter()
    cases = []
    failure_examples = {}
    for index, (row, label, left, right) in enumerate(zip(inputs, labels, free, constrained), 1):
        uid = row["user_id"]
        truth = ground[uid]
        pool = row["alias_to_id"]
        pool_assets = set(pool.values())
        if pool_assets != set(row["pool_ids"]) or len(pool) != 50 or not pool_assets <= eligible:
            raise ValueError(f"Invalid fixed candidate pool at cohort position {index}")
        prefix, candidates = row["prompt"].rsplit("\nFIXED_CANDIDATES_JSON:\n", 1)
        order = [candidate["candidate_alias"] for candidate in json.loads(candidates)]
        if sorted(pool) != order:
            raise ValueError(f"Wrong June A order at cohort position {index}")
        history = json.loads(prefix.rsplit("\nPAST_HISTORY_JSON:\n", 1)[1])
        if any(event["days_since_last_buy"] <= 0 for event in history):
            raise ValueError(f"Non-past history at cohort position {index}")
        prompt_hash = hash_bytes(row["prompt"].encode())
        if any(item["prompt_sha256"] != prompt_hash for item in (row, left, right)):
            raise ValueError(f"Prompt mismatch at cohort position {index}")
        if any(item["protocol_hash"] != protocol["protocol_hash"] for item in (row, left, right)):
            raise ValueError(f"Protocol mismatch at cohort position {index}")
        if truth != set(label["relevant_ids"]):
            raise ValueError(f"Labels differ from raw transactions at cohort position {index}")
        if truth:
            candidate_recalls.append(len(truth & pool_assets) / len(truth))
        token_count_matches += right.get("input_tokens") == left.get("input_tokens")

        rankings = {name: row["baseline_rankings"][name] for name in BASELINE_NAMES}
        rankings.update(rrf=row["rrf_top10"], input_order=[pool[alias] for alias in order[:10]])
        reasons = {}
        for name, record, repair in (
            ("free", left, False), ("free_unwrapped", left, True),
            ("guided", right, False), ("guided_unwrapped", right, True),
        ):
            ranking, reason, fenced = parse(record["raw_response"], pool, row["rrf_top10"], repair)
            if not repair and (
                record["ranked_ids"] != ranking or record["used_rrf_fallback"] != bool(reason)
            ):
                raise ValueError(f"{name} saved response disagrees with independent parser at {index}")
            errors[name][reason or "accepted"] += 1
            fences[name] += fenced
            rankings[name] = ranking
            reasons[name] = reason
        scores = {}
        for method, ranking in rankings.items():
            if len(ranking) != 10 or len(set(ranking)) != 10 or not set(ranking) <= eligible:
                raise ValueError(f"Invalid ranking for {method} at cohort position {index}")
            scores[method] = score(ranking, truth)
            methods.setdefault(method, []).append((bool(truth), scores[method]["ndcg"]))
        paired["guided_differs_from_rrf_order"] += rankings["guided"] != rankings["rrf"]
        paired["guided_differs_from_unwrapped_order"] += rankings["guided"] != rankings["free_unwrapped"]
        guided_failure = reasons["guided"]
        if guided_failure and guided_failure not in failure_examples:
            example = {
                "cohort_position": index, "output_tokens": right.get("output_tokens"),
                "used_rrf": rankings["guided"] == rankings["rrf"],
            }
            if guided_failure in {"invalid_count", "duplicate_alias", "out_of_pool_alias"}:
                reported = json.loads(right["raw_response"])["ranked_aliases"]
                example["reported_alias_count"] = len(reported)
                example["out_of_pool_alias_count"] = sum(
                    isinstance(alias, str) and alias not in pool for alias in reported
                )
            failure_examples[guided_failure] = example
        if truth:
            guided_score = scores["guided"]["ndcg"]
            rrf_score = scores["rrf"]["ndcg"]
            comparison = "above" if guided_score > rrf_score else "below" if guided_score < rrf_score else "equal"
            paired[f"with_buys_guided_{comparison}_rrf"] += 1
            cases.append({
                "cohort_position": index, "relevant_count": len(truth),
                "relevant_in_pool": len(truth & pool_assets), "guided_failure": reasons["guided"],
                "same_guided_and_rrf_top10": rankings["guided"] == rankings["rrf"],
                "scores": {name: scores[name] for name in ("rrf", "free_unwrapped", "guided")},
            })

    return {
        "cohort_size": len(cohort), "nonempty_users": len(candidate_recalls),
        "eligible_assets": len(eligible), "cutoff": cutoff, "label_end_exclusive": end,
        "reference_protocol_sha256": protocol["protocol_hash"],
        "ordered_input_file_sha256": hash_bytes((reference / "frozen_inputs.jsonl").read_bytes()),
        "labels_file_sha256": hash_bytes((reference / "labels.jsonl").read_bytes()),
        "unconstrained_responses_file_sha256": hash_bytes((reference / "responses.jsonl").read_bytes()),
        "constrained_responses_file_sha256": hash_bytes(guided.read_bytes()),
        "per_prompt_hashes_checked": len(inputs), "model_input_token_count_matches": token_count_matches,
        "scores": {
            method: {"all": mean(s for _, s in values),
                     "with_buys": mean(s for has_buys, s in values if has_buys)}
            for method, values in methods.items()
        },
        "validation": errors, "fences_removed": fences,
        "paired": paired,
        "guided_failure_examples": failure_examples,
        "mean_candidate_recall_with_buys": mean(candidate_recalls),
        "users_with_buys_cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute June scores from the data and raw responses.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "FAR-Trans")
    parser.add_argument("--reference", type=Path, default=ROOT / "session2" / "outputs" / "june_a")
    parser.add_argument("--guided-responses", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir, args.reference, args.guided_responses), indent=2))


if __name__ == "__main__":
    main()
