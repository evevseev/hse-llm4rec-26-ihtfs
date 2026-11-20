#!/usr/bin/env python3
# Generated with an AI assistant to cross-check the saved-response diagnostic.
import argparse
import json
import re
from collections import Counter
from math import fsum, log2
from pathlib import Path


FENCE = re.compile(r"^```(?:json)?[ \t]*\r?\n(.*?)\r?\n```$", re.IGNORECASE | re.DOTALL)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score(ranking, truth):
    weights = [1 / log2(i + 2) for i in range(10)]
    if not truth:
        return 0.0
    gain = fsum(w for item, w in zip(ranking, weights) if item in truth)
    return gain / fsum(weights[:len(truth)])


def check(directory):
    inputs = read(directory / "frozen_inputs.jsonl")
    output = read(directory / "responses.jsonl")
    if [row["user_id"] for row in output] != [row["user_id"] for row in inputs]:
        raise ValueError("Stored responses do not cover the full cohort in order")
    truth = {row["user_id"]: set(row["relevant_ids"])
             for row in read(directory / "labels.jsonl")}
    counts = Counter()
    scores = []
    example = None
    for i, (source, response) in enumerate(zip(inputs, output), 1):
        raw = response["raw_response"]
        match = FENCE.fullmatch(raw.strip())
        counts["markdown_wrappers_removed"] += bool(match)
        labels = truth[source["user_id"]]
        failure = None
        try:
            payload = json.loads(match.group(1) if match else raw)
            if not isinstance(payload, dict) or set(payload) != {"ranked_aliases"}:
                raise ValueError("invalid_schema")
            aliases = payload["ranked_aliases"]
            if not isinstance(aliases, list) or len(aliases) != 10:
                raise ValueError("invalid_count")
            if any(not isinstance(alias, str) for alias in aliases):
                raise ValueError("invalid_alias_type")
            if len(set(aliases)) != 10:
                raise ValueError("duplicate_alias")
            if not set(aliases) <= source["alias_to_id"].keys():
                raise ValueError("out_of_pool_alias")
            ranking = [source["alias_to_id"][alias] for alias in aliases]
        except json.JSONDecodeError:
            failure = "invalid_json"
        except ValueError as error:
            failure = str(error)
        if failure:
            counts[failure] += 1
            ranking = source["rrf_top10"]
        else:
            counts["accepted"] += 1
        ndcg = score(ranking, labels)
        scores.append((ndcg, bool(labels)))
        old = response["ranked_ids"]
        if example is None and labels and not failure and old != ranking:
            example = {
                "cohort_position_1_based": i,
                "strict_hit_ranks": [j for j, item in enumerate(old, 1) if item in labels],
                "after_unwrapping_hit_ranks": [j for j, item in enumerate(ranking, 1) if item in labels],
                "relevant_count": len(labels),
                "strict_ndcg_at_10": score(old, labels),
                "after_unwrapping_ndcg_at_10": ndcg,
            }
    nonempty = [value for value, has_truth in scores if has_truth]
    return {
        "counts": dict(counts),
        "ndcg_at_10": fsum(value for value, _ in scores) / len(scores),
        "ndcg_at_10_nonempty": fsum(nonempty) / len(nonempty),
        "first_changed_case_with_nonempty_truth": example,
    }


def main():
    parser = argparse.ArgumentParser(description="Independently recompute the diagnostic for fenced JSON.")
    parser.add_argument("output_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({folder.name: check(folder) for folder in args.output_dirs}, indent=2))


if __name__ == "__main__":
    main()
