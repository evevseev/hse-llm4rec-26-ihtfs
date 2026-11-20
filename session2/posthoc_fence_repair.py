#!/usr/bin/env python3
"""Post-hoc fence-only JSON parse sensitivity analysis for June A/B outputs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import experiment


ROOT = experiment.ROOT
OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs'
DEFAULT_A = OUTPUT_ROOT / 'june_a'
DEFAULT_B = OUTPUT_ROOT / 'june_b'
DEFAULT_OUT = OUTPUT_ROOT / 'posthoc_fence_repair'

FENCED_JSON = re.compile(r'```json\r?\n(.*?)\r?\n```', flags=re.IGNORECASE | re.DOTALL)


def strip_markdown_fence(raw: str) -> tuple[str, bool]:
    """Remove exactly one complete outer ```json fence; do not repair other text."""
    match = FENCED_JSON.fullmatch(raw)
    if match is None:
        return raw, False
    return match.group(1), True


def load_split(directory: Path) -> tuple[dict, dict[str, dict], dict[str, dict], dict[str, set[str]]]:
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('split') not in ('main_a', 'main_b'):
        raise ValueError(f'{directory} is not a June A/B run')
    inputs = {row['user_id']: row for row in experiment.read_jsonl(directory / 'frozen_inputs.jsonl')}
    responses = {row['user_id']: row for row in experiment.read_jsonl(directory / 'responses.jsonl')}
    labels = {row['user_id']: set(row['relevant_ids'])
              for row in experiment.read_jsonl(directory / 'labels.jsonl')}
    if set(inputs) != set(responses) or set(inputs) != set(labels):
        raise ValueError(f'{directory}: expected a complete response and label record for every user')
    if list(inputs) != protocol['users']:
        raise ValueError(f'{directory}: input order disagrees with the frozen protocol')
    return protocol, inputs, responses, labels


def evaluate_split(directory: Path) -> dict:
    protocol, inputs, responses, labels = load_split(directory)
    methods = {name: [] for name in
               ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'rrf', 'qwen_fence_repaired')}
    per_user = []
    audit = []
    for uid in protocol['users']:
        context = inputs[uid]
        saved = responses[uid]
        inner, removed = strip_markdown_fence(saved.get('raw_response') or '')
        ranked, failure = experiment.parse_alias_response(
            inner, context['alias_to_id'], context['rrf_top10'])
        relevant = labels[uid]
        original_ranked = saved['ranked_ids']
        original_ndcg = experiment.starter.ndcg10(original_ranked, relevant)
        posthoc_ndcg = experiment.starter.ndcg10(ranked, relevant)
        for method in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'):
            methods[method].append({
                'user_id': uid, 'relevant_count': len(relevant),
                'ndcg_at_10': experiment.starter.ndcg10(
                    context['baseline_rankings'][method], relevant),
            })
        methods['rrf'].append({
            'user_id': uid, 'relevant_count': len(relevant),
            'ndcg_at_10': experiment.starter.ndcg10(context['rrf_top10'], relevant),
        })
        methods['qwen_fence_repaired'].append({
            'user_id': uid, 'relevant_count': len(relevant), 'ndcg_at_10': posthoc_ndcg,
        })
        per_user.append({
            'user_id': uid,
            'wrapper_removed': removed,
            'old_failure_reason': saved.get('failure_reason'),
            'posthoc_failure_reason': failure,
            'relevant_count': len(relevant),
            'old_top10': original_ranked,
            'posthoc_top10': ranked,
            'relevant_ids': sorted(relevant),
            'old_ndcg_at_10': original_ndcg,
            'posthoc_ndcg_at_10': posthoc_ndcg,
            'top10_changed': ranked != original_ranked,
        })
        audit.append({
            'user_id': uid, 'wrapper_removed': removed,
            'response_before': saved.get('raw_response'),
            'response_after_fence_removal': inner,
            'posthoc_failure_reason': failure,
            'posthoc_top10': ranked,
        })
    aggregate = {name: experiment.starter.aggregate(rows) for name, rows in methods.items()}
    failures = Counter(row['posthoc_failure_reason'] or 'valid' for row in per_user)
    return {
        'split': protocol['split'], 'protocol_hash': protocol['protocol_hash'],
        'users': len(per_user), 'metrics': aggregate,
        'wrapper_removed_count': sum(row['wrapper_removed'] for row in per_user),
        'posthoc_valid_count': failures.get('valid', 0),
        'posthoc_fallback_count': len(per_user) - failures.get('valid', 0),
        'posthoc_failure_reasons': dict(failures),
        'per_user': per_user, 'audit': audit,
    }


def build_report(a_dir: Path = DEFAULT_A, b_dir: Path = DEFAULT_B,
                 output_dir: Path = DEFAULT_OUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_a, result_b = evaluate_split(a_dir), evaluate_split(b_dir)
    a_by_user = {row['user_id']: row for row in result_a['per_user']}
    b_by_user = {row['user_id']: row for row in result_b['per_user']}
    if set(a_by_user) != set(b_by_user):
        raise ValueError('June A/B cohorts differ')
    comparison = []
    for uid in a_by_user:
        a, b = a_by_user[uid], b_by_user[uid]
        comparison.append({
            'user_id': uid, 'posthoc_top10_equal': a['posthoc_top10'] == b['posthoc_top10'],
            'posthoc_set_overlap': len(set(a['posthoc_top10']) & set(b['posthoc_top10'])),
            'a_posthoc_failure_reason': a['posthoc_failure_reason'],
            'b_posthoc_failure_reason': b['posthoc_failure_reason'],
        })
    all_split_rows = [(result_a, row) for row in result_a['per_user']] + [
        (result_b, row) for row in result_b['per_user']]
    changed_metric_examples = [(split_result, row) for split_result, row in all_split_rows
                               if row['wrapper_removed'] and row['posthoc_failure_reason'] is None
                               and row['old_ndcg_at_10'] != row['posthoc_ndcg_at_10']]
    changed_examples = [(split_result, row) for split_result, row in all_split_rows
                        if row['wrapper_removed'] and row['posthoc_failure_reason'] is None
                        and row['top10_changed']]
    chosen_pair = (changed_metric_examples[0] if changed_metric_examples else
                   changed_examples[0] if changed_examples else next(
                       ((split_result, row) for split_result, row in all_split_rows
                        if row['wrapper_removed']), None))
    example = None
    if chosen_pair:
        split_result, chosen = chosen_pair
        same_user_rows = {}
        for run in (result_a, result_b):
            run_row = next(row for row in run['per_user'] if row['user_id'] == chosen['user_id'])
            audit_row = next(row for row in run['audit'] if row['user_id'] == chosen['user_id'])
            same_user_rows[run['split']] = {
                'raw_before': audit_row['response_before'],
                'after_removing_only_outer_fence': audit_row['response_after_fence_removal'],
                'failure_after_unchanged_validation': run_row['posthoc_failure_reason'],
                'old_top10': run_row['old_top10'],
                'posthoc_top10': run_row['posthoc_top10'],
                'old_ndcg_at_10': run_row['old_ndcg_at_10'],
                'posthoc_ndcg_at_10': run_row['posthoc_ndcg_at_10'],
            }
        example = {
            'user_id': chosen['user_id'], 'relevant_ids': chosen['relevant_ids'],
            'runs': same_user_rows,
            'explanation': (
                'The original strict json.loads saw the Markdown fence and fell back to RRF. '
                'This post-hoc pass removes only that outer fence, then applies the same '
                'schema, uniqueness, pool-membership, and RRF-fallback checks.'
            ),
        }
    report = {
        'analysis': 'post-hoc only: strip one outer Markdown JSON fence; no model calls',
        'strict_validation_unchanged': True,
        'main_a_artifacts_modified': False, 'main_b_artifacts_modified': False,
        'a_protocol_hash': result_a['protocol_hash'], 'b_protocol_hash': result_b['protocol_hash'],
        'a': {key: value for key, value in result_a.items() if key not in ('per_user', 'audit')},
        'b': {key: value for key, value in result_b.items() if key not in ('per_user', 'audit')},
        'a_b_posthoc_comparison': {
            'users': len(comparison),
            'top10_exactly_equal': sum(row['posthoc_top10_equal'] for row in comparison),
            'top10_different': sum(not row['posthoc_top10_equal'] for row in comparison),
            'mean_set_overlap': sum(row['posthoc_set_overlap'] for row in comparison) / len(comparison),
        },
        'example': example,
    }
    payloads = {
        'summary.json': report,
        'posthoc_metrics_a.json': result_a['metrics'],
        'posthoc_metrics_b.json': result_b['metrics'],
        'posthoc_per_user_a.json': result_a['per_user'],
        'posthoc_per_user_b.json': result_b['per_user'],
        'posthoc_audit_a.jsonl': result_a['audit'],
        'posthoc_audit_b.jsonl': result_b['audit'],
        'posthoc_a_b_comparison.json': comparison,
    }
    for name, value in payloads.items():
        path = output_dir / name
        if name.endswith('.jsonl'):
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                     for row in value), encoding='utf-8')
        else:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps({
        'output_dir': str(output_dir),
        'a': report['a'], 'b': report['b'],
        'a_b_posthoc_comparison': report['a_b_posthoc_comparison'],
        'example': example,
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-dir', type=Path, default=DEFAULT_A)
    parser.add_argument('--b-dir', type=Path, default=DEFAULT_B)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_report(args.a_dir, args.b_dir, args.output_dir)


if __name__ == '__main__':
    main()
