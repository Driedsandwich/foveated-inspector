"""Hard-negative Lambda orchestration with fake S3 and no image analysis."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aws import run_bench
from aws.handlers import bench
from inspector.synth import TEST_SURFACES
from test_aws_bench import FakeS3

ROOT = Path(__file__).resolve().parents[1]
MODEL_PREFIX = "x132/models/" + "a" * 64 + "/"


class HardNegativeAwsTests(unittest.TestCase):
    def manifest(self):
        entries = []
        for surface in ("asphalt_floor", "concrete_pavement"):
            for index in range(4):
                entries.append({"image": f"{surface}_{index}.jpg", "labels": f"{surface}_{index}.png",
                    "surface": surface, "split": "train", "seed": index,
                    "defects": [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]})
        return {"s3_prefix": "x132/benchmarks/example/train/", "images": entries}

    def test_mining_reads_only_fit_images_from_dev_s3_namespace(self):
        from scripts.train_detector import planned_entries
        manifest = self.manifest()
        entries, _ = planned_entries(manifest, None)
        validation = {e["image"] for e in entries if e["role"] == "validation"}
        self.assertEqual(len(validation), 2)
        key = manifest["s3_prefix"] + "manifest.json"
        objects = {key: json.dumps(manifest).encode()}
        for entry in entries:
            for field in ("image", "labels"):
                objects[manifest["s3_prefix"] + entry[field]] = b"not-an-image"
        client = FakeS3(objects)
        calls = []

        def extract(dataset, output, **kwargs):
            calls.append(kwargs)
            self.assertTrue(kwargs["hard_negatives"])
            self.assertEqual(kwargs["hard_negative_fraction"], .04)
            self.assertIsNone(kwargs["holdout_surface"])
            self.assertEqual(len(json.loads((dataset / "manifest.json").read_text())["images"]), 8)
            self.assertTrue(all(not (dataset / name).exists() for name in validation))
            (output / "shard.json").write_text("{}")

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            with patch("scripts.train_detector.extract_samples", side_effect=extract), patch(
                    "aws.handlers.bench._models", return_value=Path(temporary) / "first"):
                for index in range(2):
                    result = bench.run_job({"job": "hard_negatives", "job_id": "mine", "shard_index": index,
                        "shard_count": 2, "benchmark_s3": "x132/benchmarks/example/",
                        "model_prefix": MODEL_PREFIX}, client, bucket="test-bucket", temp_root=Path(temporary))
                    self.assertEqual(result["status"], "DONE", result)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(manifest["s3_prefix"] + name not in client.downloaded for name in validation))

    def test_forged_test_training_source_is_rejected_before_image_access(self):
        manifest = self.manifest()
        manifest["images"][0]["surface"] = sorted(TEST_SURFACES)[0]
        client = FakeS3({manifest["s3_prefix"] + "manifest.json": json.dumps(manifest).encode()})
        result = bench.run_job({"job": "hard_negatives", "job_id": "minebad", "shard_count": 2,
            "benchmark_s3": "x132/benchmarks/example/", "model_prefix": MODEL_PREFIX}, client, bucket="test-bucket")
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(any(key.endswith((".jpg", ".png")) for key in client.downloaded))

    def test_refit_passes_separate_mining_manifests_and_first_model(self):
        base, extra = "x132/jobs/base/shards/0000/shard.json", "x132/jobs/mine/shards/0000/shard.json"
        client = FakeS3({base: b'{"records": []}', extra: b'{"records": []}'})
        captures = []

        def fit(source, output, **kwargs):
            captures.append(kwargs)
            self.assertEqual(len(source), 1)
            self.assertEqual(len(kwargs["hard_negative_manifests"]), 1)
            self.assertNotEqual(source[0], kwargs["hard_negative_manifests"][0])
            output.mkdir()
            (output / "metadata.json").write_text("{}")

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            first = Path(temporary) / "first"
            with patch("scripts.train_detector.fit_samples", side_effect=fit), patch("aws.handlers.bench._models", return_value=first):
                result = bench.run_job({"job": "fit_samples", "job_id": "refit", "sample_manifests": [base],
                    "hard_negatives": True, "hard_negative_manifests": [extra], "model_prefix": MODEL_PREFIX},
                    client, bucket="test-bucket", temp_root=Path(temporary))
        self.assertEqual(result["status"], "DONE", result)
        self.assertEqual(captures[0]["first_model_dir"], first)

    def test_local_mining_staging_does_not_require_or_upload_validation_files(self):
        from scripts.train_detector import planned_entries
        manifest = self.manifest()
        entries, _ = planned_entries(manifest, None)
        fit = [entry for entry in entries if entry["role"] == "fit"]
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            for entry in fit:
                for field in ("image", "labels"):
                    (root / entry[field]).write_bytes(b"not-a-decoded-image")
            staged, uploads = run_bench.stage_manifest(path, root, "x132/datasets/mine/", fit_only=True)
        self.assertEqual(len(staged["images"]), 8)
        self.assertEqual({item["name"] for item in uploads}, {e[field] for e in fit for field in ("image", "labels")})

    def test_invalid_mining_events_fail_without_s3(self):
        events = [
            {"job": "hard_negatives", "job_id": "x"},
            {"job": "hard_negatives", "job_id": "x", "model_prefix": MODEL_PREFIX, "hard_negative_fraction": .2},
            {"job": "fit_samples", "job_id": "x", "hard_negatives": True, "model_prefix": MODEL_PREFIX},
            {"job": "eval", "job_id": "x", "hard_negatives": True},
        ]
        for event in events:
            client = FakeS3()
            self.assertEqual(bench.run_job(event, client, bucket="test-bucket")["status"], "FAILED")
            self.assertEqual(client.downloaded, [])

    def test_driver_plans_mining_and_refit_without_sdk_and_keeps_save_traces(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            state = root / "state.json"
            state.write_text(json.dumps({"bucket": "fixture", "region": "us-east-1", "prefix": "x132/",
                "functions": {"bench": "x132-bench"}, "model_prefix": MODEL_PREFIX}))
            plans = []
            arguments = [
                ["--job", "hard_negatives", "--benchmark-s3", "x132/benchmarks/example", "--shards", "7"],
                ["--job", "fit_samples", "--hard-negatives", "--sample-manifests", "x132/jobs/base/shard.json",
                 "--hard-negative-manifests", "x132/jobs/mine/shard.json"],
                ["--job", "eval", "--benchmark-s3", "x132/benchmarks/example", "--shards", "7", "--save-traces"],
            ]
            for args in arguments:
                with patch("aws.run_bench.get_clients", side_effect=AssertionError("no SDK")), contextlib.redirect_stdout(io.StringIO()) as stream:
                    code = run_bench.main(["--state", str(state), "--out", str(root / "out"), "--dry-run", *args])
                self.assertEqual(code, 0, stream.getvalue())
                plans.append(json.loads(stream.getvalue()))
            self.assertFalse((root / "out").exists())
        mining, refit, evaluation = [plan["invocations"][0]["event"] for plan in plans]
        self.assertEqual(mining["hard_negative_fraction"], .04)
        self.assertIsNone(mining["holdout_surface"])
        self.assertTrue(refit["hard_negatives"])
        self.assertTrue(evaluation["save_traces"])
