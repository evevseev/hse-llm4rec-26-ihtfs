#!/usr/bin/env python3
# Generated with an AI assistant.
"""Run and compare the JSON-constrained June A reranking branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SESSION3 = Path(__file__).resolve().parent
SESSION2 = SESSION3.parent / 'session2'
sys.path.insert(0, str(SESSION2))

import experiment
import posthoc_fence_repair


DEFAULT_SOURCE = SESSION2 / 'outputs' / 'june_a'
DEFAULT_OUTPUT = SESSION3 / 'outputs' / 'june_a_json_constrained'
BLOCK_SIZE = 5

JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'ranked_aliases': {
            'type': 'array',
            'items': {'type': 'string'},
        },
    },
    'required': ['ranked_aliases'],
    'additionalProperties': False,
}


def load_source(source_dir: Path) -> tuple[dict, list[dict]]:
    protocol = json.loads((source_dir / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('split') != 'main_a':
        raise ValueError('The JSON-constrained branch requires the frozen June A inputs')
    rows = experiment.read_jsonl(source_dir / 'frozen_inputs.jsonl')
    if [row['user_id'] for row in rows] != protocol['users']:
        raise ValueError('Frozen-input order does not match the June A protocol')
    if any(row.get('protocol_hash') != protocol['protocol_hash'] for row in rows):
        raise ValueError('Frozen inputs contain a different protocol hash')
    for row in rows:
        actual_hash = hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest()
        if actual_hash != row['prompt_sha256']:
            raise ValueError(f"{row['user_id']}: frozen prompt text/hash mismatch")
    return protocol, rows


def ensure_output(source_dir: Path, output_dir: Path) -> tuple[dict, list[dict]]:
    protocol, rows = load_source(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'run_manifest.json'
    manifest = {
        'study': 'June A JSON-constrained generation branch',
        'source_split': 'main_a',
        'source_protocol_hash': protocol['protocol_hash'],
        'source_frozen_inputs_sha256': experiment.sha256_file(source_dir / 'frozen_inputs.jsonl'),
        'model': experiment.MODEL_DEFAULT.name,
        'model_revision': experiment.MODEL_REVISION,
        'backend': 'pytorch',
        'device': 'mps',
        'dtype': 'torch.float16',
        'do_sample': False,
        'num_beams': 1,
        'max_new_tokens': experiment.MAX_NEW_TOKENS,
        'context_limit': experiment.CONTEXT_LIMIT,
        'constraint': 'lm-format-enforcer==0.11.3',
        'json_schema': JSON_SCHEMA,
        'pool_size': experiment.POOL_SIZE,
        'block_size': BLOCK_SIZE,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise ValueError('Existing output directory belongs to different frozen inputs/settings')
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
    responses = experiment.read_jsonl(output_dir / 'responses.jsonl')
    seen = set()
    allowed_users = set(protocol['users'])
    for response in responses:
        uid = response.get('user_id')
        if uid not in allowed_users or uid in seen:
            raise ValueError('Constrained responses contain an unknown or duplicate user')
        if response.get('protocol_hash') != protocol['protocol_hash']:
            raise ValueError('Constrained responses belong to a different source protocol')
        seen.add(uid)
    return protocol, rows


def make_prefix_allowed_tokens_fn(tokenizer):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer, JsonSchemaParser(JSON_SCHEMA))


def worker(source_dir: Path, output_dir: Path, model_dir: Path,
           max_users: int) -> int:
    protocol, rows = load_source(source_dir)
    response_path = output_dir / 'responses.jsonl'
    completed = {row['user_id'] for row in experiment.read_jsonl(response_path)}
    pending = [row for row in rows if row['user_id'] not in completed][:max_users]
    if not pending:
        return 0

    load_started = time.perf_counter()
    generator = experiment.Session1Generator(model_dir, 'mps')
    model_load_seconds = time.perf_counter() - load_started
    constraint_started = time.perf_counter()
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(generator.tokenizer)
    constraint_setup_seconds = time.perf_counter() - constraint_started

    generation_seconds = 0.0
    generated_users = 0
    block_started = time.perf_counter()
    for row in pending:
        uid = row['user_id']
        try:
            result = generator.generate(
                row['prompt'], experiment.MAX_NEW_TOKENS,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
            raw = result['text']
            ranked, failure = experiment.parse_alias_response(
                raw, row['alias_to_id'], row['rrf_top10'])
            if result['input_tokens'] + experiment.MAX_NEW_TOKENS > experiment.CONTEXT_LIMIT:
                ranked, failure = list(row['rrf_top10']), 'oversized_context'
            seconds = float(result['seconds'])
            generation_seconds += seconds
            response = {
                'user_id': uid,
                'protocol_hash': protocol['protocol_hash'],
                'prompt_sha256': row['prompt_sha256'],
                'raw_response': raw,
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'generation_seconds': seconds,
                'backend': 'torch',
                'ranked_ids': ranked,
                'used_rrf_fallback': failure is not None,
                'failure_reason': failure,
            }
        except Exception as error:
            failure_reason = ('oversized_context' if isinstance(error, ValueError) and
                              str(error) == 'oversized_context' else
                              f'generation_error:{type(error).__name__}:{error}')
            response = {
                'user_id': uid,
                'protocol_hash': protocol['protocol_hash'],
                'prompt_sha256': row['prompt_sha256'],
                'raw_response': None,
                'input_tokens': None,
                'output_tokens': None,
                'generation_seconds': None,
                'backend': 'torch',
                'ranked_ids': list(row['rrf_top10']),
                'used_rrf_fallback': True,
                'failure_reason': failure_reason,
            }
        experiment.append_jsonl(response_path, response)
        generated_users += 1

    experiment.append_jsonl(output_dir / 'block_timings.jsonl', {
        'user_ids': [row['user_id'] for row in pending],
        'model_load_seconds': model_load_seconds,
        'constraint_setup_seconds': constraint_setup_seconds,
        'generation_seconds_total': generation_seconds,
        'worker_wall_seconds': time.perf_counter() - block_started,
        'generated_users': generated_users,
        'device': generator.device,
        'dtype': 'torch.float16',
    })
    return 0


def run(source_dir: Path, output_dir: Path, model_dir: Path,
        max_new_users: int = 0) -> dict:
    protocol, rows = ensure_output(source_dir, output_dir)
    if max_new_users < 0:
        raise ValueError('--max-new-users must be >= 0 (0 means all pending users)')
    response_path = output_dir / 'responses.jsonl'
    initial = len(experiment.read_jsonl(response_path))
    target_new = max_new_users or len(rows)
    worker_path = Path(__file__).resolve()

    while True:
        completed = {row['user_id'] for row in experiment.read_jsonl(response_path)}
        pending = [row for row in rows if row['user_id'] not in completed]
        if not pending or len(completed) - initial >= target_new:
            break
        budget = target_new - (len(completed) - initial)
        block_count = min(BLOCK_SIZE, len(pending), budget)
        command = [
            sys.executable, str(worker_path), 'worker',
            '--source-dir', str(source_dir), '--output-dir', str(output_dir),
            '--model-dir', str(model_dir), '--max-users', str(block_count),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            completed_count = len(experiment.read_jsonl(response_path))
            state = {
                'status': 'stopped_after_worker_error',
                'completed_responses': completed_count,
                'pending_responses': len(rows) - completed_count,
                'worker_exit_code': result.returncode,
                'new_responses_this_invocation': completed_count - initial,
            }
            (output_dir / 'run_state.json').write_text(
                json.dumps(state, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(state, indent=2), file=sys.stderr)
            return state

    responses = experiment.read_jsonl(response_path)
    completed = {row['user_id'] for row in responses}
    timings = experiment.read_jsonl(output_dir / 'block_timings.jsonl')
    state = {
        'status': 'completed' if len(completed) == len(rows) else 'partial',
        'source_protocol_hash': protocol['protocol_hash'],
        'completed_responses': len(completed),
        'pending_responses': len(rows) - len(completed),
        'new_responses_this_invocation': len(completed) - initial,
        'block_size': BLOCK_SIZE,
        'model_load_seconds_total': sum(row['model_load_seconds'] for row in timings),
        'generation_seconds_total': sum(
            float(row['generation_seconds'] or 0.0) for row in responses),
        'june_a_responses_modified': False,
    }
    (output_dir / 'timing.json').write_text(
        json.dumps({
            'model_load_seconds_total': state['model_load_seconds_total'],
            'generation_seconds_total': state['generation_seconds_total'],
            'blocks': timings,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'run_state.json').write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def _model_call_failure_reason(response: dict) -> str | None:
    reason = str(response.get('failure_reason') or '')
    if reason.startswith(('generation_error:', 'model_call_error:', 'inference_error:')):
        return ':'.join(reason.split(':', 2)[:2])
    return None


def _response_audit(row: dict, response: dict) -> tuple[dict, list[str], list[str]]:
    raw = response.get('raw_response')
    model_failure = _model_call_failure_reason(response)
    raw_present = isinstance(raw, str)
    syntax_valid = False
    if raw_present:
        try:
            json.loads(raw)
            syntax_valid = True
        except json.JSONDecodeError:
            pass

    strict_ranking, strict_failure = experiment.parse_alias_response(
        raw, row['alias_to_id'], row['rrf_top10'])
    stripped_text, wrapper_removed = posthoc_fence_repair.strip_markdown_fence(
        raw if raw_present else '')
    stripped_syntax_valid = False
    if raw_present:
        try:
            json.loads(stripped_text)
            stripped_syntax_valid = True
        except json.JSONDecodeError:
            pass
    stripped_ranking, stripped_failure = experiment.parse_alias_response(
        stripped_text, row['alias_to_id'], row['rrf_top10'])

    saved_ranking = response.get('ranked_ids')
    saved_decision_match = saved_ranking is not None and strict_ranking == saved_ranking

    token_counts = {
        'input_tokens': response.get('input_tokens'),
        'output_tokens': response.get('output_tokens'),
    }
    audit = {
        'raw_response_present': raw_present,
        'json_syntax_valid': syntax_valid,
        'validator_accepted_top10': strict_failure is None and raw_present,
        'validator_failure_reason': strict_failure if raw_present else None,
        'model_call_failure_reason': model_failure,
        'wrapper_removed': wrapper_removed,
        'fence_stripped_json_syntax_valid': stripped_syntax_valid,
        'fence_stripped_validator_accepted_top10': stripped_failure is None and raw_present,
        'fence_stripped_failure_reason': stripped_failure if raw_present else None,
        'saved_failure_reason': response.get('failure_reason'),
        'saved_used_rrf_fallback': response.get('used_rrf_fallback'),
        'validated_decision_matches_saved': saved_decision_match,
        'token_counts': token_counts,
        'generation_seconds': response.get('generation_seconds'),
    }
    return audit, strict_ranking, stripped_ranking


def _index_complete_responses(response_rows: list[dict], users: list[str], branch: str) -> dict:
    expected = set(users)
    indexed = {}
    duplicates = []
    extras = []
    for row in response_rows:
        user_id = row.get('user_id')
        if user_id in indexed:
            duplicates.append(user_id)
        else:
            indexed[user_id] = row
        if user_id not in expected:
            extras.append(user_id)
    missing = expected - set(indexed)
    if duplicates or extras or missing:
        raise ValueError(
            f'{branch} responses are not a complete one-row-per-user cohort: '
            f'missing={len(missing)}, extra={len(extras)}, duplicate={len(duplicates)}')
    return indexed


def _manifest_summary(directory: Path, protocol: dict, fallback: str) -> dict:
    preferred = directory / 'run_manifest.json'
    manifest_path = preferred if preferred.exists() else directory / fallback
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    try:
        relative_path = manifest_path.resolve().relative_to(experiment.ROOT.resolve())
        display_path = str(relative_path)
    except ValueError:
        display_path = manifest_path.name
    return {
        'file': display_path,
        'sha256': experiment.sha256_file(manifest_path),
        'split': manifest.get('split', manifest.get('source_split')),
        'protocol_hash': manifest.get('protocol_hash',
                                     manifest.get('source_protocol_hash', protocol['protocol_hash'])),
        'model': manifest.get('model_id', manifest.get('model')),
        'model_revision': manifest.get('model_revision', manifest.get('source_model_revision')),
        'backend': manifest.get('primary_backend', manifest.get('backend')),
        'device': manifest.get('device'),
        'dtype': manifest.get('runtime_dtype', manifest.get('dtype')),
        'constraint': manifest.get('constraint'),
    }


def _timing_summary(directory: Path, responses: list[dict], branch: str) -> dict:
    timing_path = directory / 'timing.json'
    timing = json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else {}
    state_path = directory / 'run_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    block_rows = experiment.read_jsonl(directory / 'block_timings.jsonl')
    generation_values = [float(row['generation_seconds']) for row in responses
                         if isinstance(row.get('generation_seconds'), (int, float))]
    input_values = [int(row['input_tokens']) for row in responses
                    if isinstance(row.get('input_tokens'), (int, float))]
    output_values = [int(row['output_tokens']) for row in responses
                     if isinstance(row.get('output_tokens'), (int, float))]

    def total_from_blocks(key: str) -> float | None:
        values = [float(row[key]) for row in block_rows if isinstance(row.get(key), (int, float))]
        return sum(values) if values else None

    model_load = total_from_blocks('model_load_seconds')
    if model_load is None and isinstance(timing.get('model_load_seconds_total'), (int, float)):
        model_load = float(timing['model_load_seconds_total'])
    mask_setup = total_from_blocks('constraint_setup_seconds')
    if mask_setup is None and isinstance(timing.get('constraint_setup_seconds_total'), (int, float)):
        mask_setup = float(timing['constraint_setup_seconds_total'])
    return {
        'generation_seconds_total': sum(generation_values),
        'generation_seconds_measured_calls': len(generation_values),
        'model_load_seconds_total': model_load,
        'model_load_seconds_last_process_snapshot': state.get(
            'model_load_seconds_this_process'),
        'mask_preparation_seconds_total': mask_setup,
        'input_tokens_total': sum(input_values),
        'input_tokens_mean': sum(input_values) / len(input_values) if input_values else None,
        'input_token_count': len(input_values),
        'output_tokens_total': sum(output_values),
        'output_tokens_mean': sum(output_values) / len(output_values) if output_values else None,
        'output_token_count': len(output_values),
        'timing_note': (
            ('June A artifacts do not record cumulative model-load time; '
             'the unconstrained branch has no mask-preparation stage')
            if branch == 'june_a' and model_load is None else None),
    }


def _audit_summary(per_user: list[dict], branch: str) -> dict:
    audits = [row['validation'][branch] for row in per_user]
    parser_fallbacks = Counter(row['validator_failure_reason'] for row in audits
                               if row['raw_response_present'] and
                               row['validator_failure_reason'])
    stripped_fallbacks = Counter(row['fence_stripped_failure_reason'] for row in audits
                                 if row['raw_response_present'] and
                                 row['fence_stripped_failure_reason'])
    model_failures = Counter(row['model_call_failure_reason'] for row in audits
                             if row['model_call_failure_reason'])
    other_fallbacks = Counter(
        row['saved_failure_reason'] or 'missing_raw_response'
        for row in audits
        if not row['raw_response_present'] and not row['model_call_failure_reason'])
    return {
        'responses': len(audits),
        'raw_responses_present': sum(row['raw_response_present'] for row in audits),
        'raw_responses_missing': sum(not row['raw_response_present'] for row in audits),
        'json_syntax_valid': sum(row['json_syntax_valid'] for row in audits),
        'json_syntax_invalid': sum(
            row['raw_response_present'] and not row['json_syntax_valid'] for row in audits),
        'validator_accepted_top10': sum(row['validator_accepted_top10'] for row in audits),
        'validator_fallbacks_by_reason': dict(parser_fallbacks),
        'wrappers_removed': sum(row['wrapper_removed'] for row in audits),
        'fence_stripped_json_syntax_valid': sum(
            row['fence_stripped_json_syntax_valid'] for row in audits),
        'fence_stripped_validator_accepted_top10': sum(
            row['fence_stripped_validator_accepted_top10'] for row in audits),
        'fence_stripped_fallbacks_by_reason': dict(stripped_fallbacks),
        'model_call_failures': sum(bool(row['model_call_failure_reason']) for row in audits),
        'model_call_failure_reasons': dict(model_failures),
        'non_model_missing_output_fallbacks_by_reason': dict(other_fallbacks),
        'saved_rrf_fallback_decisions': sum(
            row['saved_used_rrf_fallback'] is True for row in audits),
        'saved_decisions_checked': len(audits),
        'validated_decisions_match_saved': sum(
            row['validated_decision_matches_saved'] for row in audits),
    }


def compare(source_dir: Path, constrained_dir: Path, output_path: Path) -> dict:
    protocol, rows = load_source(source_dir)
    original_rows = experiment.read_jsonl(source_dir / 'responses.jsonl')
    constrained_rows = experiment.read_jsonl(constrained_dir / 'responses.jsonl')
    users = protocol['users']
    if len(set(users)) != len(users):
        raise ValueError('June A protocol contains repeated users')
    if not users:
        raise ValueError('June A protocol has no users to compare')
    original = _index_complete_responses(original_rows, users, 'June A')
    constrained = _index_complete_responses(constrained_rows, users, 'Constrained')
    label_rows = experiment.read_jsonl(source_dir / 'labels.jsonl')
    labels = {row['user_id']: set(row['relevant_ids']) for row in label_rows}
    if len(labels) != len(label_rows) or set(labels) != set(users):
        raise ValueError('June A labels do not cover the frozen cohort')

    per_user = []
    for frozen in rows:
        uid = frozen['user_id']
        source_response = original[uid]
        constrained_response = constrained[uid]
        actual_prompt_hash = hashlib.sha256(frozen['prompt'].encode('utf-8')).hexdigest()
        source_hash = source_response.get('prompt_sha256')
        constrained_hash = constrained_response.get('prompt_sha256')
        checks = {
            'frozen_input_hash_matches_text': frozen['prompt_sha256'] == actual_prompt_hash,
            'june_a_response_hash_matches_text': source_hash == actual_prompt_hash,
            'constrained_response_hash_matches_text': constrained_hash == actual_prompt_hash,
            'response_hashes_equal': source_hash == constrained_hash,
        }
        if not all(checks.values()):
            raise ValueError(f'{uid}: prompt text/hash mismatch: {checks}')
        for response in (source_response, constrained_response):
            if response.get('protocol_hash') != protocol['protocol_hash']:
                raise ValueError(f'{uid}: response protocol hash mismatch')

        june_a, june_a_top10, june_a_fence_stripped_top10 = _response_audit(
            frozen, source_response)
        constrained_audit, constrained_top10, _ = _response_audit(
            frozen, constrained_response)
        for branch, audit in (('June A', june_a), ('Constrained', constrained_audit)):
            if not audit['validated_decision_matches_saved']:
                raise ValueError(f'{uid}: {branch} revalidated ranking differs from saved top-10')

        candidate_text = frozen['prompt'].split('FIXED_CANDIDATES_JSON:\n', 1)[1]
        candidates_in_prompt = json.loads(candidate_text)
        aliases = [candidate['candidate_alias'] for candidate in candidates_in_prompt[:10]]
        prompt_order = [frozen['alias_to_id'][alias] for alias in aliases]
        experiment.starter.validate_ranking(prompt_order, frozen['pool_ids'])
        rankings = {
            'popularity': frozen['baseline_rankings']['popularity'],
            'cbf': frozen['baseline_rankings']['cbf'],
            'latest_buy': frozen['baseline_rankings']['latest_buy'],
            'latest_buy_or_sell': frozen['baseline_rankings']['latest_buy_or_sell'],
            'rrf': frozen['rrf_top10'],
            'june_a': june_a_top10,
            'june_a_fence_stripped': june_a_fence_stripped_top10,
            'june_a_json_constrained': constrained_top10,
            'prompt_order_top10': prompt_order,
        }
        per_user.append({
            'user_id': uid,
            'relevant_count': len(labels[uid]),
            'prompt_sha256': {**checks, 'full_prompt_sha256': actual_prompt_hash},
            'validation': {
                'june_a': june_a,
                'june_a_json_constrained': constrained_audit,
            },
            'rankings': rankings,
            'ndcg_at_10': {
                name: experiment.starter.ndcg10(ranking, labels[uid])
                for name, ranking in rankings.items()
            },
        })

    methods = list(per_user[0]['ndcg_at_10'])
    metrics = {
        method: experiment.starter.aggregate([{
            'user_id': row['user_id'],
            'relevant_count': row['relevant_count'],
            'ndcg_at_10': row['ndcg_at_10'][method],
        } for row in per_user])
        for method in methods
    }

    source_manifest = _manifest_summary(source_dir, protocol, 'protocol.json')
    constrained_manifest = _manifest_summary(constrained_dir, protocol, 'run_manifest.json')
    if source_manifest['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('June A manifest belongs to a different protocol')
    branch_timings = {
        'june_a': _timing_summary(source_dir, original_rows, 'june_a'),
        'june_a_json_constrained': _timing_summary(
            constrained_dir, constrained_rows, 'june_a_json_constrained'),
    }
    if constrained_manifest['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('Constrained run manifest belongs to a different June A protocol')
    report = {
        'source_manifests': {
            'june_a': source_manifest,
            'june_a_json_constrained': constrained_manifest,
        },
        'source_protocol_hash': protocol['protocol_hash'],
        'prompt_template_sha256': protocol['prompt_sha256'],
        'users': len(users),
        'response_coverage': {
            'june_a': len(original_rows),
            'june_a_json_constrained': len(constrained_rows),
        },
        'prompt_hash_verification': {
            'all_full_prompt_hashes_match': True,
            'all_branch_prompt_hashes_equal': True,
            'verified_users': len(per_user),
        },
        'validation': {
            'june_a': _audit_summary(per_user, 'june_a'),
            'june_a_json_constrained': _audit_summary(per_user, 'june_a_json_constrained'),
        },
        'timings_and_tokens': branch_timings,
        'metrics': metrics,
        'paired_top10': {
            'june_a_equals_fence_stripped': sum(
                row['rankings']['june_a'] == row['rankings']['june_a_fence_stripped']
                for row in per_user),
            'fence_stripped_equals_constrained': sum(
                row['rankings']['june_a_fence_stripped'] ==
                row['rankings']['june_a_json_constrained'] for row in per_user),
            'mean_fence_stripped_constrained_set_overlap': sum(
                len(set(row['rankings']['june_a_fence_stripped']) &
                    set(row['rankings']['june_a_json_constrained'])) for row in per_user
            ) / len(per_user),
            'prompt_order_equals_rrf': sum(
                row['rankings']['prompt_order_top10'] == row['rankings']['rrf']
                for row in per_user),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    details_path = constrained_dir / 'comparison_details.json'
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(per_user, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run', help='Generate the constrained branch and resume safely')
    run_parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    run_parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument('--model-dir', type=Path, default=experiment.MODEL_DEFAULT)
    run_parser.add_argument('--max-new-users', type=int, default=0,
                            help='Limit this invocation; 0 processes all pending users')
    compare_parser = commands.add_parser('compare', help='Compare June A raw, fence-stripped and constrained')
    compare_parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    compare_parser.add_argument('--constrained-dir', type=Path, default=DEFAULT_OUTPUT)
    compare_parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'comparison.json')
    worker_parser = commands.add_parser('worker', help=argparse.SUPPRESS)
    worker_parser.add_argument('--source-dir', type=Path, required=True)
    worker_parser.add_argument('--output-dir', type=Path, required=True)
    worker_parser.add_argument('--model-dir', type=Path, required=True)
    worker_parser.add_argument('--max-users', type=int, required=True)
    args = parser.parse_args()

    if args.command == 'run':
        state = run(args.source_dir, args.output_dir, args.model_dir, args.max_new_users)
        if state['status'] == 'stopped_after_worker_error':
            raise SystemExit(1)
    elif args.command == 'compare':
        compare(args.source_dir, args.constrained_dir, args.output)
    else:
        raise SystemExit(worker(args.source_dir, args.output_dir, args.model_dir,
                                args.max_users))


if __name__ == '__main__':
    main()
