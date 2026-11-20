#!/usr/bin/env python3
"""MLX parity probe and resumable, block-isolated June Run A."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import experiment


ROOT = experiment.ROOT
SESSION2 = Path(__file__).resolve().parent
MAY_DIR = SESSION2 / 'outputs' / 'dev_may'
JUNE_DIR = SESSION2 / 'outputs' / 'june_a'
TORCH_MODEL = experiment.MODEL_DEFAULT
MLX_MODEL = ROOT / 'data' / 'models' / 'qwen2.5-1.5b-instruct-mlx-fp16'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def model_tensor_dtype(model_dir: Path) -> list[str]:
    from safetensors import safe_open

    path = model_dir / 'model.safetensors'
    with safe_open(str(path), framework='numpy') as weights:
        return sorted({weights.get_slice(key).get_dtype() for key in weights.keys()})


class MlxGenerator:
    def __init__(self, model_dir: Path):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model_dir = model_dir
        self.model, self.tokenizer, self.config = load(str(model_dir), return_config=True)
        self.weight_dtypes = model_tensor_dtype(model_dir)
        if self.weight_dtypes != ['F16']:
            raise RuntimeError(f'MLX weights are not entirely FP16: {self.weight_dtypes}')
        self.metadata = {
            'backend': 'mlx-lm', 'mlx_model_path': str(model_dir),
            'source_model_revision': experiment.MODEL_REVISION,
            'mlx_version': _package_version('mlx'),
            'mlx_lm_version': _package_version('mlx-lm'),
            'weight_tensor_dtypes': self.weight_dtypes,
            'do_sample': False, 'sampler': 'argmax',
            'max_new_tokens': experiment.MAX_NEW_TOKENS,
        }

    def generate(self, prompt: str) -> dict:
        from mlx_lm import generate

        chat_prompt = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        input_tokens = len(self.tokenizer.encode(chat_prompt))
        if input_tokens + experiment.MAX_NEW_TOKENS > experiment.CONTEXT_LIMIT:
            raise ValueError('oversized_context')
        self.mx.synchronize()
        started = time.perf_counter()
        raw = generate(self.model, self.tokenizer, chat_prompt, max_tokens=experiment.MAX_NEW_TOKENS,
                       verbose=False)
        self.mx.synchronize()
        return {
            'text': raw, 'input_tokens': input_tokens,
            'output_tokens': len(self.tokenizer.encode(raw, add_special_tokens=False)),
            'seconds': time.perf_counter() - started,
        }


def _package_version(package: str) -> str:
    from importlib.metadata import version
    return version(package)


def probe_mlx(output_dir: Path, may_dir: Path = MAY_DIR, model_dir: Path = MLX_MODEL) -> dict:
    """Compare two saved singleton prompts with MLX without touching the May artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    may_manifest = load_json(may_dir / 'protocol.json')
    if may_manifest['split'] != 'dev_may':
        raise ValueError('Reference examples must come from the frozen May development run')
    frozen = {row['user_id']: row for row in experiment.read_jsonl(may_dir / 'frozen_inputs.jsonl')}
    py_responses = {row['user_id']: row for row in experiment.read_jsonl(may_dir / 'responses.jsonl')}
    users = [uid for uid in may_manifest['users'] if uid in py_responses][:2]
    if len(users) != 2:
        raise ValueError('Need two saved May PyTorch responses for parity comparison')
    started = time.perf_counter()
    try:
        generator = MlxGenerator(model_dir)
        load_seconds = time.perf_counter() - started
        comparisons = []
        for uid in users:
            source = frozen[uid]
            single = py_responses[uid]
            one_started = time.perf_counter()
            mlx_result = generator.generate(source['prompt'])
            ranked, failure = experiment.parse_alias_response(
                mlx_result['text'], source['alias_to_id'], source['rrf_top10'])
            comparisons.append({
                'user_id': uid,
                'pytorch_raw_response': single['raw_response'],
                'mlx_raw_response': mlx_result['text'],
                'raw_response_match': mlx_result['text'] == single['raw_response'],
                'pytorch_top10': single['ranked_ids'], 'mlx_top10': ranked,
                'top10_match': ranked == single['ranked_ids'],
                'mlx_failure_reason': failure,
                'pytorch_generation_seconds': single['generation_seconds'],
                'mlx_generation_seconds': mlx_result['seconds'],
                'mlx_input_tokens': mlx_result['input_tokens'],
                'mlx_output_tokens': mlx_result['output_tokens'],
                'measured_wall_seconds': time.perf_counter() - one_started,
            })
        torch_seconds = sum(row['pytorch_generation_seconds'] for row in comparisons)
        mlx_seconds = sum(row['mlx_generation_seconds'] for row in comparisons)
        result = {
            'status': 'completed', 'may_protocol_hash': may_manifest['protocol_hash'],
            'mlx': generator.metadata, 'model_load_seconds': load_seconds,
            'pytorch_generation_seconds_total': torch_seconds,
            'mlx_generation_seconds_total': mlx_seconds,
            'faster_than_pytorch': mlx_seconds < torch_seconds,
            'all_raw_responses_match': all(row['raw_response_match'] for row in comparisons),
            'all_top10_match': all(row['top10_match'] for row in comparisons),
            'use_mlx_for_june_a': (mlx_seconds < torch_seconds and
                                   all(row['raw_response_match'] for row in comparisons) and
                                   all(row['top10_match'] for row in comparisons)),
            'comparisons': comparisons,
        }
    except Exception as error:
        result = {
            'status': 'unavailable_or_failed', 'may_protocol_hash': may_manifest['protocol_hash'],
            'use_mlx_for_june_a': False,
            'error': f'{type(error).__name__}: {error}',
            'memory_error': _is_memory_error(error),
        }
    path = output_dir / 'mlx_probe.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in result if key != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return result


def _is_memory_error(error: BaseException) -> bool:
    text = f'{type(error).__name__}: {error}'.casefold()
    return any(token in text for token in ('out of memory', 'memoryerror', 'allocation failed',
                                            'failed to allocate', 'mps backend out of memory'))


def _append_failure(path: Path, user_id: str, backend: str, error: BaseException,
                    protocol_hash: str) -> None:
    experiment.append_jsonl(path, {
        'user_id': user_id, 'backend': backend, 'protocol_hash': protocol_hash,
        'timestamp_unix': time.time(), 'memory_error': _is_memory_error(error),
        'error': f'{type(error).__name__}: {error}',
    })


def run_worker(output_dir: Path, backend_name: str, model_dir: Path, max_users: int,
               split: str = 'main_a') -> int:
    protocol = load_json(output_dir / 'protocol.json')
    if split not in ('main_a', 'main_b') or protocol['split'] != split:
        raise ValueError(f'Runner split mismatch: requested {split}, prepared {protocol["split"]}')
    frozen_rows = experiment.read_jsonl(output_dir / 'frozen_inputs.jsonl')
    response_path = output_dir / 'responses.jsonl'
    completed_rows = experiment.read_jsonl(response_path)
    completed = {row['user_id'] for row in completed_rows}
    if any(row.get('protocol_hash') != protocol['protocol_hash'] for row in completed_rows):
        raise ValueError('June responses contain a different protocol hash')
    pending = [row for row in frozen_rows if row['user_id'] not in completed]
    block = pending[:max_users]
    if not block:
        return 0

    load_started = time.perf_counter()
    try:
        if backend_name == 'mlx':
            generator = MlxGenerator(model_dir)
        elif backend_name == 'torch':
            generator = experiment.Session1Generator(model_dir, 'auto')
        else:
            raise ValueError(f'Unsupported backend {backend_name}')
    except Exception as error:
        _append_failure(output_dir / 'block_failures.jsonl', block[0]['user_id'], backend_name,
                        error, protocol['protocol_hash'])
        return 75 if _is_memory_error(error) else 1
    load_seconds = time.perf_counter() - load_started
    block_started = time.perf_counter()
    generated_count = 0
    for row in block:
        uid = row['user_id']
        try:
            result = generator.generate(row['prompt']) if backend_name == 'mlx' else \
                generator.generate(row['prompt'], experiment.MAX_NEW_TOKENS)
        except Exception as error:
            if isinstance(error, ValueError) and str(error) == 'oversized_context':
                experiment.append_jsonl(response_path, {
                    'user_id': uid, 'protocol_hash': protocol['protocol_hash'],
                    'prompt_sha256': row['prompt_sha256'], 'raw_response': None,
                    'generation_seconds': None, 'backend': backend_name,
                    'ranked_ids': list(row['rrf_top10']), 'used_rrf_fallback': True,
                    'failure_reason': 'oversized_context',
                })
                completed.add(uid)
                generated_count += 1
                continue
            _append_failure(output_dir / 'block_failures.jsonl', uid, backend_name,
                            error, protocol['protocol_hash'])
            state = {
                'status': 'stopped_on_generation_error', 'split': split, 'backend': backend_name,
                'failed_user_id': uid, 'completed_responses': len(completed),
                'pending_responses': len(frozen_rows) - len(completed),
                'model_load_seconds_this_process': load_seconds,
                'block_elapsed_seconds': time.perf_counter() - block_started,
                'error': f'{type(error).__name__}: {error}',
                'memory_error': _is_memory_error(error),
            }
            (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                       encoding='utf-8')
            print(json.dumps(state, indent=2))
            return 75 if _is_memory_error(error) else 1
        ranking, failure = experiment.parse_alias_response(
            result['text'], row['alias_to_id'], row['rrf_top10'])
        if result.get('input_tokens', 0) + experiment.MAX_NEW_TOKENS > experiment.CONTEXT_LIMIT:
            ranking, failure = list(row['rrf_top10']), 'oversized_context'
        record = {
            'user_id': uid, 'protocol_hash': protocol['protocol_hash'],
            'prompt_sha256': row['prompt_sha256'], 'raw_response': result['text'],
            'input_tokens': result.get('input_tokens'), 'output_tokens': result.get('output_tokens'),
            'generation_seconds': result.get('seconds'), 'backend': backend_name,
            'ranked_ids': ranking, 'used_rrf_fallback': failure is not None,
            'failure_reason': failure,
        }
        experiment.append_jsonl(response_path, record)
        experiment.append_jsonl(output_dir / 'generations.jsonl', {
            'user_id': uid, 'prompt_sha256': row['prompt_sha256'],
            'raw_response': result['text'], 'input_tokens': result.get('input_tokens'),
            'output_tokens': result.get('output_tokens'),
            'generation_seconds': result.get('seconds'), 'backend': backend_name,
        })
        completed.add(uid)
        generated_count += 1
        state = {
            'status': 'running', 'split': split, 'backend': backend_name,
            'last_completed_user_id': uid,
            'completed_responses': len(completed),
            'pending_responses': len(frozen_rows) - len(completed),
            'model_load_seconds_this_process': load_seconds,
            'block_generated': generated_count,
            'block_elapsed_seconds': time.perf_counter() - block_started,
            'runner_sha256': experiment.sha256_file(Path(__file__).resolve()),
            'model': getattr(generator, 'metadata', {'backend': 'pytorch',
                                                      'device': generator.device,
                                                      'dtype': 'torch.float16'}),
        }
        (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                   encoding='utf-8')
    return 0


def run_blocks(output_dir: Path, block_size: int = 5, backend_choice: str = 'torch') -> dict:
    protocol = load_json(output_dir / 'protocol.json')
    if protocol['split'] != 'main_a':
        raise ValueError('Only June main A can be launched from this runner')
    if backend_choice == 'torch':
        backend = 'torch'
    elif backend_choice in ('auto', 'mlx'):
        probe_path = output_dir / 'mlx_probe.json'
        if not probe_path.exists():
            raise FileNotFoundError('MLX/auto backend selection requires an MLX parity probe')
        probe = load_json(probe_path)
        selected = 'mlx' if probe.get('use_mlx_for_june_a') is True else 'torch'
        if backend_choice == 'mlx' and selected != 'mlx':
            raise ValueError('MLX did not pass the parity and speed gate')
        backend = selected
    else:
        raise ValueError('backend must be torch, mlx, or auto')
    if block_size < 1 or block_size > 10:
        raise ValueError('Use fresh short blocks of 1-10 users')
    initial_count = len(experiment.read_jsonl(output_dir / 'responses.jsonl'))
    worker = Path(__file__).resolve()
    all_users = experiment.read_jsonl(output_dir / 'frozen_inputs.jsonl')
    while True:
        completed = {row['user_id'] for row in experiment.read_jsonl(output_dir / 'responses.jsonl')}
        pending = [row for row in all_users if row['user_id'] not in completed]
        if not pending:
            experiment.finalize(output_dir)
            state = {
                'status': 'completed', 'split': 'main_a', 'backend': backend,
                'completed_responses': len(completed), 'pending_responses': 0,
                'new_responses_this_invocation': len(completed) - initial_count,
                'block_size': block_size, 'june_b_started': False,
            }
            (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                       encoding='utf-8')
            return state
        command = [sys.executable, str(worker), 'worker', '--output-dir', str(output_dir),
                   '--backend', backend, '--max-users', str(min(block_size, len(pending)))]
        model_dir = MLX_MODEL if backend == 'mlx' else TORCH_MODEL
        command.extend(['--model-dir', str(model_dir)])
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            completed = {row['user_id'] for row in experiment.read_jsonl(output_dir / 'responses.jsonl')}
            state = {
                'status': 'stopped_after_worker_error', 'split': 'main_a', 'backend': backend,
                'completed_responses': len(completed),
                'pending_responses': len(all_users) - len(completed),
                'new_responses_this_invocation': len(completed) - initial_count,
                'block_size': block_size, 'worker_exit_code': result.returncode,
                'june_b_started': False,
            }
            (output_dir / 'run_state.json').write_text(json.dumps(state, indent=2) + '\n',
                                                       encoding='utf-8')
            print(json.dumps(state, indent=2))
            return state


def prepare_june(output_dir: Path, data_dir: Path) -> dict:
    source_revision_file = TORCH_MODEL / '.cache' / 'huggingface' / 'download' / 'model.safetensors.metadata'
    if (not source_revision_file.exists() or
            source_revision_file.read_text(encoding='utf-8').splitlines()[0]
            != experiment.MODEL_REVISION):
        raise RuntimeError('Source Qwen snapshot does not match the frozen session1 revision')
    manifest = experiment.build_run(output_dir, data_dir, 'main_a', 100)
    model_dtypes = model_tensor_dtype(TORCH_MODEL)
    extra = {
        'source_model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'source_model_revision': experiment.MODEL_REVISION,
        'source_storage_tensor_dtypes': model_dtypes,
        'pytorch_runtime_dtype': 'torch.float16',
        'selected_backend': 'torch',
        'mlx_required': False,
        'prompt_sha256': manifest['prompt_sha256'],
        'split': 'main_a',
    }
    (output_dir / 'backend_candidates.json').write_text(
        json.dumps(extra, indent=2) + '\n', encoding='utf-8')
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze June Run A without inference')
    prepare.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'FAR-Trans')
    prepare.add_argument('--output-dir', type=Path, default=JUNE_DIR)
    probe = commands.add_parser('probe-mlx', help='Compare MLX FP16 against two saved PyTorch May calls')
    probe.add_argument('--output-dir', type=Path, default=JUNE_DIR)
    probe.add_argument('--may-dir', type=Path, default=MAY_DIR)
    probe.add_argument('--model-dir', type=Path, default=MLX_MODEL)
    run = commands.add_parser('run', help='Run June A in fresh short worker processes (PyTorch by default)')
    run.add_argument('--output-dir', type=Path, default=JUNE_DIR)
    run.add_argument('--block-size', type=int, default=5)
    run.add_argument('--backend', choices=['torch', 'mlx', 'auto'], default='torch')
    worker = commands.add_parser('worker', help=argparse.SUPPRESS)
    worker.add_argument('--output-dir', type=Path, required=True)
    worker.add_argument('--backend', choices=['torch', 'mlx'], required=True)
    worker.add_argument('--model-dir', type=Path, required=True)
    worker.add_argument('--max-users', type=int, default=5)
    worker.add_argument('--split', choices=['main_a', 'main_b'], default='main_a')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare_june(args.output_dir, args.data_dir), indent=2))
    elif args.command == 'probe-mlx':
        probe_mlx(args.output_dir, args.may_dir, args.model_dir)
    elif args.command == 'run':
        run_blocks(args.output_dir, args.block_size, args.backend)
    else:
        raise SystemExit(run_worker(args.output_dir, args.backend, args.model_dir,
                                    args.max_users, args.split))


if __name__ == '__main__':
    main()
