import json
import random
import tempfile
import unittest
from pathlib import Path

import experiment


class PoolAndFallbackTests(unittest.TestCase):
    def test_new_clone_course_starter_path_is_discovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            starter_dir = Path(temporary) / 'data' / 'upstream' / 'RecSys-LLMs' / 'project' / 'far-trans'
            starter_dir.mkdir(parents=True)
            (starter_dir / 'starter.py').write_text('# test starter\n')
            self.assertEqual(experiment.find_starter_dir(Path(temporary)), starter_dir)

    def test_quota_pool_deduplicates_and_rrf_fills(self):
        latest = [f'a{i:02d}' for i in range(60)]
        cbf = latest[10:] + latest[:10]
        popularity = latest[20:] + latest[:20]
        pool, ranks, scores = experiment.build_pool({
            'latest_buy': latest, 'cbf': cbf, 'popularity': popularity,
        })
        self.assertEqual(len(pool), 50)
        self.assertEqual(len(set(pool)), 50)
        self.assertEqual(pool[:20], latest[:20])
        self.assertEqual(set(ranks), set(latest))
        self.assertEqual(len(scores), len(latest))

    def test_small_universe_and_deterministic_aliases(self):
        rankings = {name: ['a', 'b', 'c'] for name in
                    ('latest_buy', 'cbf', 'popularity')}
        pool, _ranks, _scores = experiment.build_pool(rankings)
        self.assertEqual(pool, ['a', 'b', 'c'])
        first = experiment.make_alias_map(['a', 'b', 'c'], random.Random(2025))
        second = experiment.make_alias_map(['c', 'a', 'b'], random.Random(2025))
        self.assertEqual(first, second)

    def test_malformed_qwen_response_falls_back_to_rrf(self):
        ids = [f'id{i}' for i in range(50)]
        aliases = {f'item_{i:03d}': isin for i, isin in enumerate(ids)}
        fallback = ids[:10]
        ranking, failure = experiment.parse_alias_response('{broken', aliases, fallback)
        self.assertEqual(ranking, fallback)
        self.assertEqual(failure, 'invalid_json')

    def test_valid_alias_order_maps_back_to_candidate_ids(self):
        aliases = {f'item_{i:03d}': f'id{i}' for i in range(50)}
        answer = json.dumps({'ranked_aliases': list(aliases)[:10]})
        ranking, failure = experiment.parse_alias_response(answer, aliases, [])
        self.assertEqual(ranking, [f'id{i}' for i in range(10)])
        self.assertIsNone(failure)


if __name__ == '__main__':
    unittest.main()
