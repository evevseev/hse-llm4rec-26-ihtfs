import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import june_a


class PyTorchOnlyJuneTests(unittest.TestCase):
    def test_torch_backend_does_not_require_an_mlx_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / 'protocol.json').write_text(json.dumps({'split': 'main_a'}))
            with patch.object(june_a.experiment, 'finalize') as finalize:
                state = june_a.run_blocks(output_dir, backend_choice='torch')
            finalize.assert_called_once_with(output_dir)
            self.assertEqual(state['backend'], 'torch')
            self.assertEqual(state['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
