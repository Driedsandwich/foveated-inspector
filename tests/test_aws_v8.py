"""Dev selection failure is explicit without any SDK or image processing."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aws import run_bench

ROOT = Path(__file__).resolve().parents[1]


class TuningDriverV8Tests(unittest.TestCase):
    def test_collection_preserves_complete_measurements_but_marks_missing_policy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            plan = {"job": "tune", "job_id": "fixture", "dry_run": False,
                    "collect_only": True, "merge": True, "region": "us-east-1"}
            with patch("aws.run_bench.poll_outputs", return_value=[]), patch(
                    "aws.run_bench.collect_outputs", return_value=[root / "shard_0000"]), patch(
                    "scripts.run_eval.merge_shards", return_value=([], {
                        "select_status": "no_feasible_candidate", "config_available": False})), patch(
                    "aws.run_bench.get_clients", side_effect=AssertionError("no SDK")):
                result = run_bench.execute_plan(plan, root / "out", s3=object())
            self.assertEqual(result["status"], "DONE")
            self.assertEqual(result["policy_selection_status"], "no_feasible_candidate")
            self.assertFalse(result["policy_config_available"])
            self.assertTrue(result["tradeoff_table"].endswith("dev_tradeoff.csv"))
            self.assertEqual(json.loads((root / "out/collection.json").read_text()), result)

    def test_no_feasible_tuning_exits_three_without_invoking_aws(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            result = {"status": "DONE", "policy_selection_status": "no_feasible_candidate",
                      "policy_config_available": False}
            with patch("aws.run_bench.make_plan", return_value={}), patch(
                    "aws.run_bench.execute_plan", return_value=result), contextlib.redirect_stdout(io.StringIO()) as stream:
                code = run_bench.main(["--job", "tune", "--execute", "--out", temporary])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(stream.getvalue()), result)
