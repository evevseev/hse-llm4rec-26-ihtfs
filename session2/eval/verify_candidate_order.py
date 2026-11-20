#!/usr/bin/env python3
# Generated with an AI assistant to verify the controlled June comparison.
import argparse
import json
from pathlib import Path
from statistics import mean


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_inputs(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    records = jsonl(folder / "frozen_inputs.jsonl")
    if [r["user_id"] for r in records] != protocol["users"]:
        raise ValueError("Frozen input order differs from the protocol")
    return protocol, records


def prompt_parts(prompt):
    prefix, candidates = prompt.rsplit("\nFIXED_CANDIDATES_JSON:\n", 1)
    return prefix, json.loads(candidates)


def check_comparison(order_a, order_b):
    a, inputs_a = read_inputs(order_a)
    b, inputs_b = read_inputs(order_b)
    protocol_keys = (
        "cutoff", "label_window", "users", "candidate_pool_size", "candidate_quotas",
        "rrf_k", "alias_seed", "model_id", "model_revision", "prompt_sha256",
        "max_new_tokens", "do_sample", "num_beams", "context_limit", "history_limit",
        "source_data_sha256", "source_code_sha256",
    )
    for key in protocol_keys:
        if a[key] != b[key]:
            raise ValueError(f"June protocols differ in {key}")
    labels = jsonl(order_a / "labels.jsonl")
    if labels != jsonl(order_b / "labels.jsonl"):
        raise ValueError("June truth differs across candidate orders")

    changed = 0
    coverage = []
    truth = {row["user_id"]: set(row["relevant_ids"]) for row in labels}
    for left, right in zip(inputs_a, inputs_b):
        for field in ("pool_ids", "id_to_alias", "source_ranks", "rrf_top10", "baseline_rankings"):
            if left[field] != right[field]:
                raise ValueError(f"Expected fixed field {field} to match")
        prefix_a, rows_a = prompt_parts(left["prompt"])
        prefix_b, rows_b = prompt_parts(right["prompt"])
        alias_a = {row["candidate_alias"]: row for row in rows_a}
        alias_b = {row["candidate_alias"]: row for row in rows_b}
        if prefix_a != prefix_b or alias_a != alias_b or len(alias_a) != len(rows_a):
            raise ValueError("Candidate order changed the history or candidate attributes")
        changed += rows_a != rows_b
        relevant = truth[left["user_id"]]
        if relevant:
            coverage.append(len(relevant & set(left["pool_ids"])) / len(relevant))
        if set(left["alias_to_id"].values()) != set(left["pool_ids"]):
            raise ValueError("The alias map differs from the fixed pool")

    result = {
        "compared_users": len(inputs_a),
        "identical_protocol_fields": list(protocol_keys),
        "candidate_order_changed_for_users": changed,
        "mean_candidate_recall_nonempty": mean(coverage) if coverage else None,
    }
    response_paths = [folder / "responses.jsonl" for folder in (order_a, order_b)]
    if all(path.exists() for path in response_paths):
        left_responses, right_responses = (jsonl(path) for path in response_paths)
        overlap = []
        raw_differences = 0
        ranking_differences = 0
        for left, right, row in zip(left_responses, right_responses, inputs_a):
            if left["user_id"] != row["user_id"] or right["user_id"] != row["user_id"]:
                raise ValueError("Response order differs from the fixed user order")
            for response in (left, right):
                ranking = response["ranked_ids"]
                if len(set(ranking)) != 10 or not set(ranking) <= set(row["pool_ids"]):
                    raise ValueError("The result is outside the fixed pool")
                if response["used_rrf_fallback"] and ranking != row["rrf_top10"]:
                    raise ValueError("Fallback differs from RRF")
            overlap.append(len(set(left["ranked_ids"]) & set(right["ranked_ids"])))
            raw_differences += left["raw_response"] != right["raw_response"]
            ranking_differences += left["ranked_ids"] != right["ranked_ids"]
        result["users_with_both_responses"] = len(overlap)
        result["users_with_different_raw_responses"] = raw_differences
        result["users_with_different_final_rankings"] = ranking_differences
        result["mean_common_top_10_assets"] = mean(overlap) if overlap else None
    return result


def main():
    parser = argparse.ArgumentParser(description="Check that the June test varies only candidate row order.")
    parser.add_argument("order_a", type=Path)
    parser.add_argument("order_b", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_comparison(args.order_a, args.order_b), indent=2))


if __name__ == "__main__":
    main()
