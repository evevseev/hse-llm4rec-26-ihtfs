#!/usr/bin/env python3
# Generated with an AI assistant to check the published ranker's comparison protocol.
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Check the provided ranker's CBF-order control.")
    parser.add_argument("--upstream", type=Path, default=ROOT / "data" / "upstream")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "FAR-Trans")
    parser.add_argument("--june-a", type=Path, default=ROOT / "session2" / "outputs" / "june_a")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "far-trans-llm"))
    from data_adapter import load

    freeze = json.loads((args.june_a / "protocol.json").read_text())
    starter, _, _, truth, contexts, _, _, _, manifest = load(SimpleNamespace(
        starter_dir=args.upstream / "far-trans", data_dir=args.data_dir,
        cutoff=freeze["cutoff"], horizon_days=30, max_users=100, candidate_count=50,
    ))
    if (manifest["users"] != freeze["users"]
            or manifest["data_sha256"] != freeze["source_data_sha256"]
            or manifest["end_exclusive"] != freeze["label_end_exclusive_from_starter"]):
        raise ValueError("Course ranker inputs differ from the June A evaluation protocol")

    rows = []
    for user in manifest["users"]:
        context = contexts[user]
        candidates = [row["id"] for row in context["candidate_assets"]]
        relevant = truth.get(user, set()) & set(context["eligible_asset_ids"])
        rows.append({"relevant_count": len(relevant),
                     "ndcg_at_10": starter.ndcg10(candidates[:10], relevant)})
    print(json.dumps({
        "llm_ranker_result_source": "https://raw.githubusercontent.com/dryjins/RecSys-LLMs/799e54c0493cf7e2b014152548ae5c18ee24a8f5/project/far-trans-llm/HW2_RESULTS.md",
        "published_llmrank_with_fallback": {
            "ndcg_at_10": 0.0201, "ndcg_at_10_nonempty": 0.1827, "fallbacks": 22,
        },
        "published_llmrank_inference_rerun": False,
        "local_cbf_order_first10": starter.aggregate(rows),
        "same_june_a_users_data_hashes_cutoff_and_end": True,
    }, indent=2))


if __name__ == "__main__":
    main()
