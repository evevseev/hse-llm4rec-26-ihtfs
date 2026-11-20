#!/usr/bin/env python3
# Generated with an AI assistant for an independent result check.
import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from hashlib import sha256
from math import fsum, log2
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def ground_truth(data, cutoff, end):
    known_assets = {
        row["ISIN"] for row in rows(data / "asset_information.csv")
        if row["timestamp"][:10] < cutoff
    }
    priced_assets = {
        row["ISIN"] for row in rows(data / "close_prices.csv")
        if row["timestamp"][:10] < cutoff and float(row["closePrice"]) > 0
    }
    eligible = known_assets & priced_assets
    known_users = {
        row["customerID"] for row in rows(data / "customer_information.csv")
        if row["timestamp"][:10] < cutoff
    }
    truth = defaultdict(set)
    warm = set()
    for row in rows(data / "transactions.csv"):
        user, asset, day = row["customerID"], row["ISIN"], row["timestamp"][:10]
        if (user not in known_users or asset not in eligible
                or row["transactionType"] != "Buy" or int(row["transactionID"]) < 0):
            continue
        if day < cutoff:
            warm.add(user)
        elif day < end:
            truth[user].add(asset)
    ordered_users = sorted(warm, key=lambda user: (sha256(f"42:{user}".encode()).hexdigest(), user))
    return eligible, ordered_users, truth


def verify(data, predictions, cutoff, horizon, num_users):
    end = (date.fromisoformat(cutoff) + timedelta(days=horizon)).isoformat()
    eligible, users, truth = ground_truth(data, cutoff, end)
    selected = users[:num_users]
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows(predictions):
        grouped[row["method"]][row["user_id"]].append((int(row["rank"]), row["ISIN"]))

    discounts = [1 / log2(rank + 1) for rank in range(1, 11)]
    methods = {}
    example = None
    for method, rankings in sorted(grouped.items()):
        if set(rankings) != set(selected):
            raise ValueError(f"{method}: different users from the independent warm cohort")
        scores = []
        for user in selected:
            ranked = sorted(rankings[user])
            assets = {asset for _, asset in ranked}
            if [rank for rank, _ in ranked] != list(range(1, 11)) or len(assets) != 10 or not assets <= eligible:
                raise ValueError(f"{method}: invalid recommendation list")
            relevant = truth[user]
            hits = [rank for rank, asset in ranked if asset in relevant]
            gain = fsum(discounts[rank - 1] for rank in hits)
            ideal = fsum(discounts[:len(relevant)])
            score = gain / ideal if ideal else 0.0
            scores.append((score, len(relevant)))
            if method == "latest_buy" and hits == [2, 3, 4] and len(relevant) == 6 and example is None:
                example = {
                    "method": method, "cohort_position_1_based": selected.index(user) + 1,
                    "relevant_count": len(relevant), "hit_ranks": hits,
                    "dcg_at_10": gain, "idcg_at_10": ideal, "ndcg_at_10": score,
                }
        nonempty = [score for score, count in scores if count]
        methods[method] = {
            "users": len(scores), "nonempty_truth_users": len(nonempty),
            "ndcg_at_10": fsum(score for score, _ in scores) / len(scores),
            "ndcg_at_10_nonempty": fsum(nonempty) / len(nonempty) if nonempty else None,
        }

    return {
        "cutoff": cutoff, "end_exclusive": end, "eligible_assets": len(eligible),
        "methods": methods, "verification_case": example,
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute the June teaching metrics from raw CSVs and saved predictions.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cutoff", default="2021-06-01")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--users", type=int, default=100)
    args = parser.parse_args()
    result = verify(args.data_dir, args.predictions, args.cutoff, args.horizon_days, args.users)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
