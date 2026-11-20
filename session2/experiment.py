#!/usr/bin/env python3
"""Session 2 implementation of the frozen-pool FAR-Trans RRF/Qwen study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_starter_dir(root: Path = ROOT) -> Path:
    candidates = (
        root / 'data' / 'upstream' / 'far-trans',
        root / 'data' / 'upstream' / 'RecSys-LLMs' / 'project' / 'far-trans',
    )
    for candidate in candidates:
        if (candidate / 'starter.py').is_file():
            return candidate
    raise FileNotFoundError(
        'Course starter not found. Clone RecSys-LLMs at commit '
        '51337032503c56e3a28ef646a24aa016782e1bd3 under data/upstream/RecSys-LLMs.')


STARTER_DIR = find_starter_dir()
PROMPT_PATH = Path(__file__).resolve().parent / 'prompts' / 'rerank.txt'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MODEL_DEFAULT = ROOT / 'data' / 'models' / 'qwen2.5-1.5b-instruct'
ALIAS_SEED = 2025
RRF_K = 60
POOL_SIZE = 50
TOP_K = 10
MAX_NEW_TOKENS = 96
CONTEXT_LIMIT = 8192


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module at {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


starter = load_module('session2_far_trans_starter', STARTER_DIR / 'starter.py')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def rrf_score(ranks: dict[str, int], k: int = RRF_K) -> float:
    return sum(1.0 / (k + int(rank)) for rank in ranks.values())


def build_pool(source_rankings: dict[str, list[str]], pool_size: int = POOL_SIZE
               ) -> tuple[list[str], dict[str, dict[str, int]], dict[str, float]]:
    """Apply source-priority quotas, deduplicate, then RRF-fill to the target size."""
    if set(source_rankings) != {'latest_buy', 'cbf', 'popularity'}:
        raise ValueError('Expected complete Latest Buy, CBF, and popularity rankings')
    universes = [set(ranking) for ranking in source_rankings.values()]
    if not universes or any(universe != universes[0] for universe in universes[1:]):
        raise ValueError('All source rankings must cover the same eligible universe')
    eligible = universes[0]
    ranks = {method: {isin: rank for rank, isin in enumerate(ranking, 1)}
             for method, ranking in source_rankings.items()}
    candidate_ranks = {isin: {method: ranks[method][isin] for method in source_rankings}
                       for isin in eligible}
    scores = {isin: rrf_score(candidate_ranks[isin]) for isin in eligible}
    quotas = (('latest_buy', 20), ('cbf', 20), ('popularity', 10))
    pool: list[str] = []
    selected: set[str] = set()
    for method, quota in quotas:
        for isin in source_rankings[method][:quota]:
            if isin not in selected:
                selected.add(isin)
                pool.append(isin)
    target = min(pool_size, len(eligible))
    fill_order = sorted(eligible - selected, key=lambda isin: (-scores[isin], isin))
    pool.extend(fill_order[:max(0, target - len(pool))])
    return pool, candidate_ranks, scores


def make_alias_map(asset_ids: list[str], rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable, per-user opaque aliases for a union of history and candidates."""
    ordered_ids = sorted(set(asset_ids))
    labels = [f'item_{index:03d}' for index in range(len(ordered_ids))]
    rng.shuffle(labels)
    id_to_alias = dict(zip(ordered_ids, labels))
    return id_to_alias, {alias: isin for isin, alias in id_to_alias.items()}


def parse_alias_response(raw: str, alias_to_id: dict[str, str], rrf_top10: list[str]
                         ) -> tuple[list[str], str | None]:
    """Strictly validate Qwen's JSON and return the RRF list for any invalid answer."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return list(rrf_top10), 'invalid_json'
    if not isinstance(value, dict) or set(value) != {'ranked_aliases'}:
        return list(rrf_top10), 'invalid_schema'
    aliases = value['ranked_aliases']
    if not isinstance(aliases, list) or len(aliases) != TOP_K:
        return list(rrf_top10), 'invalid_count'
    if any(not isinstance(alias, str) for alias in aliases):
        return list(rrf_top10), 'invalid_alias_type'
    if len(set(aliases)) != TOP_K:
        return list(rrf_top10), 'duplicate_alias'
    if not set(aliases) <= set(alias_to_id):
        return list(rrf_top10), 'out_of_pool_alias'
    ranked_ids = [alias_to_id[alias] for alias in aliases]
    try:
        starter.validate_ranking(ranked_ids, list(alias_to_id.values()))
    except ValueError:
        return list(rrf_top10), 'invalid_ranking'
    return ranked_ids, None


def make_prompt(history_rows: list[dict], candidate_rows: list[dict], template: str) -> str:
    return (template.replace('{{history_json}}', json.dumps(history_rows, ensure_ascii=False,
                                                           separators=(',', ':')))
            .replace('{{candidate_rows_json}}', json.dumps(candidate_rows, ensure_ascii=False,
                                                           separators=(',', ':'))))


def make_latest_buy_ranking(events: list[tuple[str, str, str]], eligible: set[str],
                            popularity_ranking: list[str]) -> list[str]:
    buy_last: dict[str, str] = {}
    for timestamp, isin, kind in events:
        if kind == 'Buy' and isin in eligible and (isin not in buy_last or timestamp > buy_last[isin]):
            buy_last[isin] = timestamp
    prefix = sorted(buy_last, key=lambda isin: (-date.fromisoformat(buy_last[isin]).toordinal(), isin))
    seen = set(prefix)
    return prefix + [isin for isin in popularity_ranking if isin not in seen]


def _source_hashes(data_dir: Path) -> dict[str, str]:
    return {name: sha256_file(data_dir / name) for name in
            ('transactions.csv', 'asset_information.csv', 'customer_information.csv', 'close_prices.csv')}


def _main_users(data_dir: Path, max_users: int) -> list[str]:
    assets, profiles, history, _truth, _end, _excluded = starter.load_data(
        data_dir, '2021-06-01', 30)
    return starter.select_users(profiles, history, 'warm', max_users, None)


def build_run(output_dir: Path, data_dir: Path, split: str = 'dev_may', max_users: int = 100) -> dict:
    """Prepare frozen pools/prompts; model calls are only made by the separate run command."""
    preprocessing_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if split not in ('dev_may', 'main_a', 'main_b'):
        raise ValueError('split must be dev_may, main_a, or main_b')
    if max_users != 100:
        raise ValueError('The frozen Session 1 cohort size is 100')
    cutoff = '2021-05-01' if split == 'dev_may' else '2021-06-01'
    cutoff_start = date.fromisoformat(cutoff)
    label_end = date.fromordinal(cutoff_start.toordinal() + 30).isoformat()
    assets, profiles, history, truth, end, excluded = starter.load_data(data_dir, cutoff, 30)
    all_warm_users = starter.select_users(profiles, history, 'warm', 0, None)
    main_users = _main_users(data_dir, 100)
    if split == 'dev_may':
        users = [uid for uid in all_warm_users if uid not in set(main_users)][:100]
        if len(users) != 100:
            raise ValueError(f'Only {len(users)} disjoint warm development users are available')
    else:
        users = main_users
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    popularity = starter.popularity_scores(history, assets)
    contexts, baseline = starter.contexts_for(users, assets, profiles, history, features,
                                               popularity, cutoff, POOL_SIZE, False)
    events_by_user = starter.scan_recent_events(data_dir, cutoff, assets)
    recency = starter.recency_rankings(contexts, popularity, events_by_user, k=TOP_K)
    for uid in users:
        baseline[uid].update(recency[uid])

    label_end_exclusive = end
    template = PROMPT_PATH.read_text(encoding='utf-8')
    prompt_hash = hashlib.sha256(template.encode('utf-8')).hexdigest()
    template_source_hash = sha256_file(PROMPT_PATH)
    source_hashes = _source_hashes(data_dir)
    rng = random.Random(ALIAS_SEED)
    records: list[dict] = []
    labels_records: list[dict] = []
    for uid in users:
        eligible = set(contexts[uid]['eligible_asset_ids'])
        counts = history.get(uid, Counter())
        popularity_ranking = starter.order({isin: popularity[isin] for isin in eligible})
        cbf_scores = starter.content_scores(counts, features)
        if not any(cbf_scores[isin] for isin in eligible):
            cbf_scores = popularity
        cbf_ranking = starter.order({isin: cbf_scores[isin] for isin in eligible})
        latest_buy = make_latest_buy_ranking(events_by_user.get(uid, []), eligible,
                                             popularity_ranking)
        source_rankings = {'latest_buy': latest_buy, 'cbf': cbf_ranking,
                           'popularity': popularity_ranking}
        pool, rank_by_candidate, score_by_candidate = build_pool(source_rankings)
        pool_set = set(pool)
        aliases_ids = set(pool)
        last_buys = {}
        for timestamp, isin, kind in events_by_user.get(uid, []):
            if kind == 'Buy' and isin in eligible and (isin not in last_buys or timestamp > last_buys[isin]):
                last_buys[isin] = timestamp
        history_items = [isin for isin in counts if isin in eligible and isin in last_buys]
        history_items.sort(key=lambda isin: (-date.fromisoformat(last_buys[isin]).toordinal(), isin))
        history_items = history_items[:20]
        aliases_ids.update(history_items)
        id_to_alias, alias_to_id = make_alias_map(list(aliases_ids), rng)
        history_rows = [{
            'asset_alias': id_to_alias[isin],
            'buy_count': counts[isin],
            'days_since_last_buy': (cutoff_start - date.fromisoformat(last_buys[isin])).days,
            'feature_tokens': features[isin],
        } for isin in history_items]
        candidate_rows = [{
            'candidate_alias': id_to_alias[isin],
            'feature_tokens': features[isin],
            'rank_latest_buy': rank_by_candidate[isin]['latest_buy'],
            'rank_cbf': rank_by_candidate[isin]['cbf'],
            'rank_popularity': rank_by_candidate[isin]['popularity'],
        } for isin in pool]
        if split == 'main_b':
            candidate_rows.sort(key=lambda row: (
                hashlib.sha256(f'2026:{uid}:{row["candidate_alias"]}'.encode()).hexdigest(),
                row['candidate_alias']))
        else:
            candidate_rows.sort(key=lambda row: row['candidate_alias'])
        rrf_top10 = sorted(pool, key=lambda isin: (-score_by_candidate[isin], id_to_alias[isin]))[:TOP_K]
        prompt = make_prompt(history_rows, candidate_rows, template)
        record = {
            'user_id': uid,
            'pool_ids': pool,
            'pool_aliases': [id_to_alias[isin] for isin in pool],
            'id_to_alias': id_to_alias,
            'alias_to_id': {id_to_alias[isin]: isin for isin in pool},
            'all_alias_to_id': alias_to_id,
            'source_ranks': {isin: rank_by_candidate[isin] for isin in pool},
            'rrf_scores': {isin: score_by_candidate[isin] for isin in pool},
            'rrf_top10': rrf_top10,
            'baseline_rankings': {method: baseline[uid][method] for method in
                                  ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')},
            'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        }
        records.append(record)
        relevant = set(truth.get(uid, set())) & eligible
        labels_records.append({'user_id': uid, 'relevant_ids': sorted(relevant)})
        if len(pool) != min(POOL_SIZE, len(eligible)) or not pool_set <= eligible:
            raise AssertionError(f'{uid}: invalid fixed pool size or eligible membership')

    config = {
        'study': 'session1 frozen RRF/Qwen reranking', 'split': split,
        'cutoff': cutoff, 'label_window': [cutoff, label_end],
        'label_end_exclusive_from_starter': label_end_exclusive,
        'users': users, 'main_cohort_users': main_users,
        'dev_disjoint_from_main': split != 'dev_may' or not bool(set(users) & set(main_users)),
        'candidate_pool_size': POOL_SIZE, 'candidate_quotas': {'latest_buy': 20, 'cbf': 20,
                                                                'popularity': 10},
        'rrf_k': RRF_K, 'alias_seed': ALIAS_SEED,
        'main_b_per_user_shuffle_seed': 2026 if split == 'main_b' else None,
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'model_revision': MODEL_REVISION,
        'prompt_sha256': prompt_hash, 'prompt_file_sha256': template_source_hash,
        'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': False, 'num_beams': 1,
        'context_limit': CONTEXT_LIMIT, 'history_limit': 20,
        'source_data_sha256': source_hashes, 'excluded_rows': excluded,
        'source_code_sha256': {
            'session2_experiment.py': sha256_file(Path(__file__).resolve()),
            'starter.py': sha256_file(STARTER_DIR / 'starter.py'),
        },
        'preprocessing_seconds': time.perf_counter() - preprocessing_started,
        'empty_truth_is_zero': True,
    }
    protocol_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for record in records:
        record['protocol_hash'] = protocol_hash
    manifest = dict(config, protocol_hash=protocol_hash,
                    implementation_revision=_git_revision(ROOT))
    (output_dir / 'protocol.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                         indent=2) + '\n', encoding='utf-8')
    for name, rows in (('frozen_inputs.jsonl', records), ('labels.jsonl', labels_records)):
        temp_path = output_dir / f'{name}.tmp'
        temp_path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                      for row in rows), encoding='utf-8')
        temp_path.replace(output_dir / name)
    print(f'prepared split={split} users={len(users)} pool_size={POOL_SIZE} '
          f'prompt_sha256={prompt_hash} protocol_hash={protocol_hash}')
    return manifest


def _git_revision(repo: Path) -> str | None:
    import subprocess
    try:
        return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Session1Generator:
    """Pinned local Qwen backend with FP16 and explicit CUDA/MPS GPU support."""
    def __init__(self, model_dir: Path, device: str = 'auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else (
                'mps' if torch.backends.mps.is_available() else 'unavailable')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        if device not in ('cuda', 'mps'):
            raise RuntimeError('Session 1 timing run requires a GPU (CUDA or MPS); CPU is disabled')
        metadata = model_dir / '.cache' / 'huggingface' / 'download' / 'model.safetensors.metadata'
        if not metadata.exists() or metadata.read_text(encoding='utf-8').splitlines()[0] != MODEL_REVISION:
            raise RuntimeError(f'Local model snapshot is not verified as revision {MODEL_REVISION}')
        torch.set_num_threads(4)
        self.device = device
        self.model_id = str(model_dir)
        self.revision = MODEL_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16)
        self.model.to(device).eval()
        import transformers
        self.metadata = {
            'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'local_path': str(model_dir),
            'requested_revision': MODEL_REVISION, 'resolved_revision': MODEL_REVISION,
            'device': device, 'dtype': 'torch.float16',
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'tokenizer_class': type(self.tokenizer).__name__,
            'do_sample': False, 'num_beams': 1, 'thinking': False,
            'context_limit': CONTEXT_LIMIT,
        }

    def _synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        else:
            self.torch.mps.synchronize()

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
        if not prompts:
            return []
        chat_texts = []
        for prompt in prompts:
            chat_texts.append(self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenization_started = time.perf_counter()
        inputs = self.tokenizer(chat_texts, return_tensors='pt', padding=True,
                                truncation=False).to(self.device)
        tokenization_seconds = time.perf_counter() - tokenization_started
        attention = inputs['attention_mask']
        input_tokens = attention.sum(dim=1).tolist()
        if any(count + max_new_tokens > CONTEXT_LIMIT for count in input_tokens):
            raise ValueError('oversized_context')
        self._synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id)
        self._synchronize()
        batch_seconds = time.perf_counter() - started
        padded_width = inputs['input_ids'].shape[1]
        results = []
        for index, prompt_tokens in enumerate(input_tokens):
            generated = output[index, padded_width:]
            results.append({
                'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                'input_tokens': int(prompt_tokens), 'output_tokens': len(generated),
                'seconds': batch_seconds / len(prompts),
                'batch_seconds': batch_seconds,
                'tokenization_seconds': tokenization_seconds / len(prompts),
            })
        return results


class BatchLoggedGenerator:
    """Batch-capable wrapper using the same persistent per-prompt cache key as LoggedGenerator."""
    def __init__(self, backend: Session1Generator, output_dir: Path):
        self.backend = backend
        self.output_dir = output_dir
        self.cache = output_dir / 'generation_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = output_dir / 'generations.jsonl'
        self.calls = 0
        self.batches = 0
        self.hits = 0

    def generate_many(self, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
        descriptors = [{'prompt': prompt, 'settings': self.backend.metadata,
                        'max_new_tokens': max_new_tokens} for prompt in prompts]
        keys = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                for item in descriptors]
        results: list[dict | None] = [None] * len(prompts)
        missing_indices = []
        for index, key in enumerate(keys):
            target = self.cache / (key + '.json')
            if target.exists():
                results[index] = json.loads(target.read_text(encoding='utf-8'))
                self.hits += 1
            else:
                missing_indices.append(index)
        if missing_indices:
            generated = self.backend.generate_batch(
                [prompts[index] for index in missing_indices], max_new_tokens)
            self.batches += 1
            self.calls += len(missing_indices)
            for index, value in zip(missing_indices, generated):
                results[index] = value
                target = self.cache / (keys[index] + '.json')
                target.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
        for index, (descriptor, key, result) in enumerate(zip(descriptors, keys, results)):
            hit = index not in missing_indices
            append_jsonl(self.log, dict(descriptor, response=result, cache_hit=hit,
                                        cache_key=key, batch_size=len(missing_indices)))
            result['_cache_hit'] = hit
        return results


def probe_batch(output_dir: Path, model_dir: Path = MODEL_DEFAULT, device: str = 'auto',
                batch_size: int = 2) -> dict:
    if batch_size not in (2, 4):
        raise ValueError('Probe batch size must be 2 or 4')
    manifest = json.loads((output_dir / 'protocol.json').read_text(encoding='utf-8'))
    if manifest['split'] != 'dev_may':
        raise ValueError('Batch probe is restricted to the May development split')
    inputs = read_jsonl(output_dir / 'frozen_inputs.jsonl')
    responses = {row['user_id']: row for row in read_jsonl(output_dir / 'responses.jsonl')}
    completed = [row for row in inputs if row['user_id'] in responses]
    if len(completed) < batch_size:
        raise ValueError(f'Need {batch_size} already-saved singleton examples')
    selected = completed[:batch_size]
    load_started = time.perf_counter()
    backend = Session1Generator(model_dir, device)
    model_load_seconds = time.perf_counter() - load_started
    generated = backend.generate_batch([row['prompt'] for row in selected], MAX_NEW_TOKENS)
    comparisons = []
    for row, batched in zip(selected, generated):
        single = responses[row['user_id']]
        batch_ranking, batch_failure = parse_alias_response(
            batched['text'], row['alias_to_id'], row['rrf_top10'])
        raw_match = batched['text'] == single['raw_response']
        ranking_match = batch_ranking == single['ranked_ids']
        comparisons.append({
            'user_id': row['user_id'], 'single_raw_response': single['raw_response'],
            'batch_raw_response': batched['text'], 'raw_response_match': raw_match,
            'single_top10': single['ranked_ids'], 'batch_top10': batch_ranking,
            'top10_match': ranking_match, 'batch_failure_reason': batch_failure,
            'single_generation_seconds': single['generation_seconds'],
            'batch_amortized_generation_seconds': batched['seconds'],
            'batch_wall_generation_seconds': batched['batch_seconds'],
            'batch_input_tokens': batched['input_tokens'],
            'batch_output_tokens': batched['output_tokens'],
        })
    single_total = sum(item['single_generation_seconds'] for item in comparisons)
    batch_wall = generated[0]['batch_seconds']
    result = {
        'split': manifest['split'], 'protocol_hash': manifest['protocol_hash'],
        'model': backend.metadata, 'batch_size': batch_size,
        'model_load_seconds': model_load_seconds,
        'batch_tokenization_seconds': sum(item['tokenization_seconds'] for item in generated),
        'single_generation_seconds_total': single_total,
        'batch_generation_seconds_wall': batch_wall,
        'faster_than_singletons': batch_wall < single_total,
        'all_raw_texts_match': all(item['raw_response_match'] for item in comparisons),
        'all_top10_match': all(item['top10_match'] for item in comparisons),
        'safe_to_batch': (batch_wall < single_total and
                          all(item['raw_response_match'] for item in comparisons) and
                          all(item['top10_match'] for item in comparisons)),
        'comparisons': comparisons,
    }
    (output_dir / f'batch_probe_{batch_size}.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in (
        'batch_size', 'model_load_seconds', 'single_generation_seconds_total',
        'batch_generation_seconds_wall', 'faster_than_singletons',
        'all_raw_texts_match', 'all_top10_match', 'safe_to_batch')}, indent=2))
    return result


def run_calls(output_dir: Path, model_dir: Path = MODEL_DEFAULT, device: str = 'auto',
              max_new_calls: int = 0, batch_size: int = 1) -> dict:
    manifest_path = output_dir / 'protocol.json'
    inputs_path = output_dir / 'frozen_inputs.jsonl'
    if not manifest_path.exists() or not inputs_path.exists():
        raise FileNotFoundError('Run prepare first; protocol.json and frozen_inputs.jsonl are required')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['split'] != 'dev_may':
        raise ValueError('This invocation is restricted to the May development pass; June inference is disabled')
    inputs = read_jsonl(inputs_path)
    response_path = output_dir / 'responses.jsonl'
    completed_rows = read_jsonl(response_path)
    completed = {row['user_id'] for row in completed_rows}
    if any(row.get('protocol_hash') != manifest['protocol_hash'] for row in completed_rows):
        raise ValueError('Existing response log belongs to a different frozen protocol')
    remaining = [row for row in inputs if row['user_id'] not in completed]
    if max_new_calls < 0:
        raise ValueError('max-new-calls must be >= 0 (0 means all remaining)')
    if batch_size not in (1, 2, 4):
        raise ValueError('batch-size must be 1, 2, or 4')
    chosen = remaining if max_new_calls == 0 else remaining[:max_new_calls]
    if not chosen:
        print(f'No pending users; {len(completed)} completed responses are already saved.')
        return {'completed': len(completed), 'new_calls': 0}

    load_started = time.perf_counter()
    backend = Session1Generator(model_dir, device)
    model_load_seconds = time.perf_counter() - load_started
    generator = BatchLoggedGenerator(backend, output_dir)
    run_started = time.perf_counter()
    call_durations = []
    batch_wall_durations = []
    for start in range(0, len(chosen), batch_size):
        group = chosen[start:start + batch_size]
        try:
            generated = generator.generate_many([row['prompt'] for row in group], MAX_NEW_TOKENS)
            group_error = None
            batch_wall = max((float(response.get('batch_seconds', 0.0))
                              for response in generated if not response.get('_cache_hit')), default=0.0)
            if batch_wall:
                batch_wall_durations.append(batch_wall)
        except Exception as error:  # Failed batch attempts use RRF and are not retried.
            generated = [None] * len(group)
            group_error = f'{type(error).__name__}: {error}'

        for row, response in zip(group, generated):
            uid = row['user_id']
            if group_error is not None:
                record = {
                    'user_id': uid, 'protocol_hash': manifest['protocol_hash'],
                    'prompt_sha256': row['prompt_sha256'], 'raw_response': None,
                    'generation_seconds': None, 'cache_hit': False,
                    'ranked_ids': list(row['rrf_top10']), 'used_rrf_fallback': True,
                    'failure_reason': group_error,
                }
            else:
                raw = response.get('text', '')
                ranked_ids, failure = parse_alias_response(raw, row['alias_to_id'], row['rrf_top10'])
                if response.get('input_tokens', 0) + MAX_NEW_TOKENS > CONTEXT_LIMIT:
                    ranked_ids, failure = list(row['rrf_top10']), 'oversized_context'
                seconds = float(response.get('seconds', 0.0))
                cache_hit = bool(response.get('_cache_hit'))
                if not cache_hit:
                    call_durations.append(seconds)
                record = {
                    'user_id': uid, 'protocol_hash': manifest['protocol_hash'],
                    'prompt_sha256': row['prompt_sha256'], 'raw_response': raw,
                    'input_tokens': response.get('input_tokens'),
                    'output_tokens': response.get('output_tokens'),
                    'generation_seconds': seconds,
                    'batch_generation_seconds': response.get('batch_seconds', seconds),
                    'cache_hit': cache_hit,
                    'ranked_ids': ranked_ids, 'used_rrf_fallback': failure is not None,
                    'failure_reason': failure,
                }
            append_jsonl(response_path, record)
            completed.add(uid)
            partial = {
                'split': manifest['split'], 'device': backend.device,
                'batch_size': batch_size,
                'model_load_seconds_this_process': model_load_seconds,
                'calls_in_this_process': generator.calls,
                'batches_in_this_process': generator.batches,
                'cache_hits_in_this_process': generator.hits,
                'new_responses_this_process': len(call_durations),
                'cumulative_completed_responses': len(completed),
                'pending_responses': len(inputs) - len(completed),
                'last_10_generation_seconds_per_user': call_durations[-10:],
                'batch_wall_seconds_this_process': sum(batch_wall_durations),
                'elapsed_after_load_seconds_this_process': time.perf_counter() - run_started,
                'runner_sha256': sha256_file(Path(__file__).resolve()),
                'model': backend.metadata,
            }
            (output_dir / 'timing.json').write_text(json.dumps(partial, indent=2) + '\n',
                                                   encoding='utf-8')
            print(f'{len(completed):3d}/{len(inputs)} user={uid} '
                  f'seconds={record.get("generation_seconds")} fallback={record["used_rrf_fallback"]}')
    stats = json.loads((output_dir / 'timing.json').read_text(encoding='utf-8'))
    stats['calls_in_this_process'] = generator.calls
    stats['batches_in_this_process'] = generator.batches
    stats['cache_hits_in_this_process'] = generator.hits
    stats['new_responses_this_process'] = len(chosen)
    stats['cumulative_completed_responses'] = len(completed)
    stats['pending_responses'] = len(inputs) - len(completed)
    stats['batch_wall_seconds_this_process'] = sum(batch_wall_durations)
    stats['elapsed_after_load_seconds_this_process'] = time.perf_counter() - run_started
    (output_dir / 'timing.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    if len(completed) == len(inputs):
        finalize(output_dir)
    else:
        print(f'Saved {len(chosen)} attempts; resume skips them and starts at next pending user.')
    return stats


def finalize(output_dir: Path) -> None:
    manifest = json.loads((output_dir / 'protocol.json').read_text(encoding='utf-8'))
    inputs = {row['user_id']: row for row in read_jsonl(output_dir / 'frozen_inputs.jsonl')}
    labels = {row['user_id']: set(row['relevant_ids'])
              for row in read_jsonl(output_dir / 'labels.jsonl')}
    responses = {row['user_id']: row for row in read_jsonl(output_dir / 'responses.jsonl')}
    if set(responses) != set(inputs):
        raise ValueError('Cannot finalize an incomplete pass')
    rankings: dict[str, dict[str, list[str]]] = {}
    records = []
    for uid, row in inputs.items():
        rankings[uid] = dict(row['baseline_rankings'])
        rankings[uid]['rrf'] = row['rrf_top10']
        rankings[uid]['qwen'] = responses[uid]['ranked_ids']
        relevant = labels.get(uid, set())
        for method, ranking in rankings[uid].items():
            allowed = row['pool_ids'] if method in ('rrf', 'qwen') else ranking
            starter.validate_ranking(ranking, allowed)
            records.append({'user_id': uid, 'method': method,
                            'relevant_count': len(relevant),
                            'ndcg_at_10': starter.ndcg10(ranking, relevant)})
    with (output_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for uid, per_method in rankings.items():
            for method, ranking in per_method.items():
                writer.writerows((method, uid, rank, isin) for rank, isin in enumerate(ranking, 1))
    metric_names = sorted({record['method'] for record in records})
    metrics = {method: starter.aggregate([record for record in records
                                          if record['method'] == method])
               for method in metric_names}
    for name, value in (('per_user.json', records), ('metrics.json', metrics)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    response_failures = [row for row in responses.values() if row['used_rrf_fallback']]
    run_manifest = dict(manifest, completed_responses=len(responses),
                        fallback_count=len(response_failures),
                        fallback_reasons=Counter(row['failure_reason'] for row in response_failures))
    (output_dir / 'run_manifest.json').write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Completed evaluation:')
    for method, value in metrics.items():
        print(f'  {method:20s} all={value["ndcg_at_10"]:.6f} '
              f'nonempty={value["ndcg_at_10_nonempty"]} '
              f'nonempty_users={value["nonempty_truth_users"]}')
    print(f'Qwen RRF fallbacks: {len(response_failures)}')


def check_user(output_dir: Path, user_id: str | None = None) -> dict:
    manifest = json.loads((output_dir / 'protocol.json').read_text(encoding='utf-8'))
    rows = read_jsonl(output_dir / 'frozen_inputs.jsonl')
    row = next((item for item in rows if item['user_id'] == user_id), None) if user_id else rows[0]
    if row is None:
        raise ValueError(f'Unknown user ID: {user_id}')
    candidate_ids = set(row['pool_ids'])
    rrf_ids = set(row['rrf_top10'])
    qwen_allowed_ids = {row['alias_to_id'][alias] for alias in row['pool_aliases']}
    bad_rank, failure = parse_alias_response('{not-json', row['alias_to_id'], row['rrf_top10'])
    checks = {
        'user_id': row['user_id'], 'protocol_hash': manifest['protocol_hash'],
        'candidate_pool_size': len(candidate_ids),
        'qwen_and_rrf_candidate_membership_identical': candidate_ids == qwen_allowed_ids,
        'rrf_top10_inside_pool': rrf_ids <= candidate_ids,
        'bad_response_reason': failure,
        'bad_response_falls_back_to_rrf': bad_rank == row['rrf_top10'],
        'rrf_top10': row['rrf_top10'],
        'candidate_ids': row['pool_ids'],
    }
    if not all((checks['qwen_and_rrf_candidate_membership_identical'],
                checks['rrf_top10_inside_pool'], checks['bad_response_falls_back_to_rrf'])):
        raise AssertionError(f'Common-pool/fallback check failed: {checks}')
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze candidate pools, prompts and labels')
    prepare.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'FAR-Trans')
    prepare.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'dev_may')
    prepare.add_argument('--split', choices=['dev_may', 'main_a', 'main_b'], default='dev_may')
    prepare.add_argument('--max-users', type=int, default=100)
    run = commands.add_parser('run', help='Run pending model calls; May dev only in this timing pass')
    run.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'dev_may')
    run.add_argument('--model-dir', type=Path, default=MODEL_DEFAULT)
    run.add_argument('--device', choices=['auto', 'cuda', 'mps'], default='auto')
    run.add_argument('--batch-size', type=int, choices=[1, 2, 4], default=1)
    run.add_argument('--max-new-calls', type=int, default=0,
                     help='Number of next pending users; 0 runs all remaining users')
    probe = commands.add_parser('probe-batch', help='Compare saved singleton outputs with batch inference')
    probe.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'dev_may')
    probe.add_argument('--model-dir', type=Path, default=MODEL_DEFAULT)
    probe.add_argument('--device', choices=['auto', 'cuda', 'mps'], default='auto')
    probe.add_argument('--batch-size', type=int, choices=[2, 4], default=2)
    check = commands.add_parser('check', help='Verify same candidate pool and RRF fallback')
    check.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'dev_may')
    check.add_argument('--user-id')
    args = parser.parse_args()
    if args.command == 'prepare':
        build_run(args.output_dir, args.data_dir, args.split, args.max_users)
    elif args.command == 'run':
        run_calls(args.output_dir, args.model_dir, args.device, args.max_new_calls, args.batch_size)
    elif args.command == 'probe-batch':
        probe_batch(args.output_dir, args.model_dir, args.device, args.batch_size)
    else:
        check_user(args.output_dir, args.user_id)


if __name__ == '__main__':
    main()
