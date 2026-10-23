"""Keep development selection isolated from the held-out surfaces."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_eval import tune_on_dev


class ScriptTests(unittest.TestCase):
    def test_tuning_never_evaluates_test_entries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"images": [
                {"image": "development.jpg", "split": "dev"},
                {"image": "held_out.jpg", "split": "test"}]}))
            seen = []

            def evaluate(bench, entries, methods, budgets, size, config):
                seen.extend(entries)
                return [{"recall": 0.5, "false_positives": 2}]

            with patch("scripts.run_eval.evaluate_entries", side_effect=evaluate):
                config, history = tune_on_dev(root, limit=2, out_dir=root / "tuning")
            self.assertEqual(len(history), 8)
            self.assertTrue(seen)
            self.assertTrue(all(item["split"] == "dev" for item in seen))
            self.assertEqual(config.positive_threshold, config.detection_threshold)


if __name__ == "__main__":
    unittest.main()
