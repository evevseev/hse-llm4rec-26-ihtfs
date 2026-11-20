#!/usr/bin/env python3
"""June Run B: fixed-pool row-order perturbation using the June A backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import experiment
import june_a


ROOT = experiment.ROOT
JUNE_A_DIR = Path(__file__).resolve().parent / 'outputs' / 'june_a'
JUNE_B_DIR = Path(__file__).resolve().parent / 'outputs' / 'june_b'
TORCH_MODEL = experiment.MODEL_DEFAULT
MLX_MODEL = june_a.MLX_MODEL


def _prompt_candidates(row: dict) -> list[dict]:
    marker = 'FIXED_CANDIDATES_JSON:\n'
    return json.loads(row['prompt'].split(marker, 1)[1])


def _prompt_history(row: dict) -> list[dict]:
    marker = 'PAST_HISTORY_JSON:\n'
    suffix = row['prompt'].split(marker, 1)[1].split('\n\nFIXED_CANDIDATES_JSON:', 1)[0]
    return json.loads(suffix)


def _read_inputs(directory: Path) -> dict[str, dict]:
    return {row['user_id']: row for row in experiment.read_jsonl(directory / 'frozen_inputs.jsonl')}


def prepare_b(output_dir: Path = JUNE_B_DIR, data_dir: Path = ROOT / 'data' / 'FAR-Trans',
              a_dir: Path = JUNE_A_DIR) -> dict:
    a_state = json.loads((a_dir / 'run_state.json').read_text(encoding='utf-8'))
    a_protocol = json.loads((a_dir / 'protocol.json').read_text(encoding='utf-8'))
    if a_state.get('status') != 'completed' or a_protocol.get('split') != 'main_a':
        raise ValueError('June A must be completed before preparing June B')
    experiment.build_run(output_dir, data_dir, 'main_b', 100)
    a_rows, b_rows = _read_inputs(a_dir), _read_inputs(output_dir)
    a_users = json.loads((a_dir / 'protocol.json').read_text(encoding='utf-8'))['users']
    b_users = json.loads((output_dir / 'protocol.json').read_text(encoding='utf-8'))['users']
    if a_users != b_users or set(a_rows) != set(b_rows):
        raise AssertionError('June A/B user cohorts or ordering differ')
    per_user = []
    for uid in a_users:
        left, right = a_rows[uid], b_rows[uid]
        candidate_a = _prompt_candidates(left)
        candidate_b = _prompt_candidates(right)
        order_a = [row['candidate_alias'] for row in candidate_a]
        order_b = [row['candidate_alias'] for row in candidate_b]
        candidates_same = sorted(candidate_a, key=lambda row: row['candidate_alias']) == \
            sorted(candidate_b, key=lambda row: row['candidate_alias'])
        history_same = _prompt_history(left) == _prompt_history(right)
        same = {
            'same_pool_ids': left['pool_ids'] == right['pool_ids'],
            'same_candidate_membership': set(left['pool_ids']) == set(right['pool_ids']),
            'same_id_alias_map': left['id_to_alias'] == right['id_to_alias'],
            'same_source_ranks': left['source_ranks'] == right['source_ranks'],
            'same_rrf_top10': left['rrf_top10'] == right['rrf_top10'],
            'same_baselines': left['baseline_rankings'] == right['baseline_rankings'],
            'same_candidate_rows_ignoring_order': candidates_same,
            'same_history_rows': history_same,
            'candidate_order_changed': order_a != order_b,
            'candidate_order_a_sha256': hashlib.sha256('\n'.join(order_a).encode()).hexdigest(),
            'candidate_order_b_sha256': hashlib.sha256('\n'.join(order_b).encode()).hexdigest(),
        }
        if not all(same[key] for key in ('same_pool_ids', 'same_candidate_membership',
                                         'same_id_alias_map', 'same_source_ranks',
                                         'same_rrf_top10', 'same_baselines',
                                         'same_candidate_rows_ignoring_order',
                                         'same_history_rows',
                                         'candidate_order_changed')):
            raise AssertionError(f'{uid}: June A/B frozen inputs are not an order-only change: {same}')
        per_user.append({'user_id': uid, **same})
    check = {
        'status': 'passed', 'split_a': 'main_a', 'split_b': 'main_b',
        'users': len(a_users), 'all_candidate_memberships_identical': True,
        'all_source_ranks_and_rrf_unchanged': True,
        'all_candidate_row_orders_changed': all(row['candidate_order_changed'] for row in per_user),
        'prompt_sha256_a': json.loads((a_dir / 'protocol.json').read_text(encoding='utf-8'))['prompt_sha256'],
        'prompt_sha256_b': json.loads((output_dir / 'protocol.json').read_text(encoding='utf-8'))['prompt_sha256'],
        'per_user': per_user,
    }
    (output_dir / 'candidate_pool_check.json').write_text(
        json.dumps(check, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: check[key] for key in check if key != 'per_user'}, indent=2))
    return check


def run_b(output_dir: Path = JUNE_B_DIR, a_dir: Path = JUNE_A_DIR,
          block_size: int = 5) -> dict:
    check_path = output_dir / 'candidate_pool_check.json'
    if not check_path.exists() or json.loads(check_path.read_text(encoding='utf-8'))['status'] != 'passed':
        raise ValueError('Prepare June B and pass the frozen candidate-pool check first')
    a_state = json.loads((a_dir / 'run_state.json').read_text(encoding='utf-8'))
    if a_state.get('status') != 'completed':
        raise ValueError('June A must be completed before June B')
    backend = a_state['backend']
    if backend not in ('torch', 'mlx'):
        raise ValueError(f'Unexpected June A backend: {backend}')
    if block_size < 1 or block_size > 10:
        raise ValueError('Use fresh short blocks of 1-10 users')

    frozen = experiment.read_jsonl(output_dir / 'frozen_inputs.jsonl')
    initial_count = len(experiment.read_jsonl(output_dir / 'responses.jsonl'))
    worker = Path(__file__).resolve().parent / 'june_a.py'
    model_dir = MLX_MODEL if backend == 'mlx' else TORCH_MODEL
    while True:
        done = {row['user_id'] for row in experiment.read_jsonl(output_dir / 'responses.jsonl')}
        pending = [row for row in frozen if row['user_id'] not in done]
        if not pending:
            experiment.finalize(output_dir)
            state = {
                'status': 'completed', 'split': 'main_b', 'backend': backend,
                'completed_responses': len(done), 'pending_responses': 0,
                'new_responses_this_invocation': len(done) - initial_count,
                'block_size': block_size, 'matched_june_a_backend': True,
                'june_b_started': True,
            }
            (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                       encoding='utf-8')
            return state
        command = [sys.executable, str(worker), 'worker', '--output-dir', str(output_dir),
                   '--backend', backend, '--split', 'main_b', '--model-dir', str(model_dir),
                   '--max-users', str(min(block_size, len(pending)))]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            done = {row['user_id'] for row in experiment.read_jsonl(output_dir / 'responses.jsonl')}
            state = {
                'status': 'stopped_after_worker_error', 'split': 'main_b', 'backend': backend,
                'completed_responses': len(done), 'pending_responses': len(frozen) - len(done),
                'new_responses_this_invocation': len(done) - initial_count,
                'block_size': block_size, 'worker_exit_code': result.returncode,
                'matched_june_a_backend': True, 'june_b_started': True,
            }
            (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                       encoding='utf-8')
            print(json.dumps(state, indent=2))
            return state


def compare_runs(a_dir: Path = JUNE_A_DIR, b_dir: Path = JUNE_B_DIR) -> dict:
    a_protocol = json.loads((a_dir / 'protocol.json').read_text(encoding='utf-8'))
    b_protocol = json.loads((b_dir / 'protocol.json').read_text(encoding='utf-8'))
    if a_protocol['split'] != 'main_a' or b_protocol['split'] != 'main_b':
        raise ValueError('Expected June A and June B outputs')
    a_inputs, b_inputs = _read_inputs(a_dir), _read_inputs(b_dir)
    a_responses = {row['user_id']: row for row in experiment.read_jsonl(a_dir / 'responses.jsonl')}
    b_responses = {row['user_id']: row for row in experiment.read_jsonl(b_dir / 'responses.jsonl')}
    users = a_protocol['users']
    if set(a_responses) != set(users) or set(b_responses) != set(users):
        raise ValueError('Both June runs must contain one response record for every user')
    details = []
    for uid in users:
        left, right = a_inputs[uid], b_inputs[uid]
        if set(left['pool_ids']) != set(right['pool_ids']):
            raise AssertionError(f'{uid}: candidate pool membership differs')
        response_a, response_b = a_responses[uid], b_responses[uid]
        overlap = len(set(response_a['ranked_ids']) & set(response_b['ranked_ids']))
        details.append({
            'user_id': uid,
            'same_candidate_membership': set(left['pool_ids']) == set(right['pool_ids']),
            'raw_response_equal': response_a.get('raw_response') == response_b.get('raw_response'),
            'top10_order_equal': response_a['ranked_ids'] == response_b['ranked_ids'],
            'top10_set_overlap': overlap,
            'a_failure_reason': response_a.get('failure_reason'),
            'b_failure_reason': response_b.get('failure_reason'),
        })
    raw_equal = sum(row['raw_response_equal'] for row in details)
    top10_equal = sum(row['top10_order_equal'] for row in details)
    report = {
        'status': 'completed', 'a_protocol_hash': a_protocol['protocol_hash'],
        'b_protocol_hash': b_protocol['protocol_hash'], 'users': len(users),
        'candidate_membership_equal_for_all': all(row['same_candidate_membership'] for row in details),
        'raw_responses_equal': raw_equal,
        'raw_responses_differ': len(users) - raw_equal,
        'top10_exact_order_equal': top10_equal,
        'top10_exact_order_differ': len(users) - top10_equal,
        'mean_top10_set_overlap': sum(row['top10_set_overlap'] for row in details) / len(users),
        'a_fallback_count': sum(row['a_failure_reason'] is not None for row in details),
        'b_fallback_count': sum(row['b_failure_reason'] is not None for row in details),
        'per_user': details,
    }
    (b_dir / 'comparison_with_a.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_user'}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze June B and assert it differs only in candidate row order')
    prepare.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'FAR-Trans')
    prepare.add_argument('--output-dir', type=Path, default=JUNE_B_DIR)
    prepare.add_argument('--a-dir', type=Path, default=JUNE_A_DIR)
    run = commands.add_parser('run', help='Run June B using the June A backend in fresh blocks')
    run.add_argument('--output-dir', type=Path, default=JUNE_B_DIR)
    run.add_argument('--a-dir', type=Path, default=JUNE_A_DIR)
    run.add_argument('--block-size', type=int, default=5)
    compare = commands.add_parser('compare', help='Compare raw responses and final Top-10 with June A')
    compare.add_argument('--a-dir', type=Path, default=JUNE_A_DIR)
    compare.add_argument('--b-dir', type=Path, default=JUNE_B_DIR)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_b(args.output_dir, args.data_dir, args.a_dir)
    elif args.command == 'run':
        run_b(args.output_dir, args.a_dir, args.block_size)
    else:
        compare_runs(args.a_dir, args.b_dir)


if __name__ == '__main__':
    main()
