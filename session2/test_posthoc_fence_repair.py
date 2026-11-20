import unittest

from posthoc_fence_repair import strip_markdown_fence


class FenceRemovalTests(unittest.TestCase):
    def test_removes_only_complete_json_markdown_wrapper(self):
        body = '{\n  "ranked_aliases": ["item_1"]\n}'
        raw = f'```json\n{body}\n```'
        self.assertEqual(strip_markdown_fence(raw), (body, True))

    def test_does_not_repair_unfenced_or_otherly_wrapped_text(self):
        raw = 'Answer: ```json\n{}\n```'
        self.assertEqual(strip_markdown_fence(raw), (raw, False))
        self.assertEqual(strip_markdown_fence('{}'), ('{}', False))


if __name__ == '__main__':
    unittest.main()
