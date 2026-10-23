"""Offline AWS job/driver contracts. SDKs, images and model fitting are faked."""

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aws.handlers import bench
from aws import run_bench


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class ClaimExists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.downloaded = []
        self.written = []

    def get_object(self, *, Bucket, Key):
        self.downloaded.append(Key)
        if Key not in self.objects:
            raise MissingObject()
        value = self.objects[Key]
        return {"Body": io.BytesIO(value), "ContentLength": len(value)}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClaimExists()
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.written.append(Key)
        return {}

    def upload_file(self, filename, bucket, key):
        self.objects[key] = Path(filename).read_bytes()
        self.written.append(key)


class BenchHandlerTests(unittest.TestCase):
    def manifest(self, split="train"):
        return {"version": 2, "s3_prefix": "x132/datasets/example/", "images": [
            {"image": "fit.jpg", "labels": "fit_labels.png", "surface": "asphalt_floor",
             "split": split, "seed": 1, "defects": [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]},
            {"image": "heldout.jpg", "labels": "heldout_labels.png", "surface": "concrete_pavement",
             "split": split, "seed": 2, "defects": [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]}]}

    def client(self, manifest):
        objects = {"x132/jobs/example/manifest.json": json.dumps(manifest).encode()}
        for entry in manifest["images"]:
            for field in ("image", "labels"):
                objects[manifest["s3_prefix"] + entry[field]] = b"fixture-not-a-decoded-image"
        return FakeS3(objects)

    def test_extraction_stages_only_assigned_files_but_keeps_full_metadata(self):
        manifest = self.manifest()
        client = self.client(manifest)
        event = {"job": "train_samples", "job_id": "example", "shard_index": 0, "shard_count": 2,
                 "holdout_surface": "concrete_pavement"}
        calls = []

        def fake_extract(dataset, output, **kwargs):
            staged = json.loads((dataset / "manifest.json").read_text())
            self.assertEqual(len(staged["images"]), 2)
            self.assertTrue((dataset / "fit.jpg").is_file())
            self.assertFalse((dataset / "heldout.jpg").exists())
            self.assertEqual(kwargs["shard_count"], 2)
            self.assertEqual(kwargs["calibration"], "sampled")
            calls.append(kwargs)
            (output / "samples_fixture_scale_1.npz").write_bytes(b"small-fake-npz")
            (output / "shard_0000_of_0002.json").write_text(json.dumps({"records": []}))
            return {"models_fitted": False}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("scripts.train_detector.extract_samples", side_effect=fake_extract):
                result = bench.run_job(event, client, bucket="test-bucket", temp_root=Path(tmp))
                second = bench.run_job(event, client, bucket="test-bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(second, result)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("x132/datasets/example/heldout.jpg", client.downloaded)
        self.assertTrue(any(item["name"] == "shard_0000_of_0002.json" for item in result["files"]))
        self.assertIn("x132/jobs/example/shards/0000/DONE.json", client.written)

    def test_rejects_traversal_unsupported_jobs_and_unsafe_output_prefix(self):
        for event in ({"job": "shell", "job_id": "example", "command": "secret-value"},
                      {"job": "eval", "job_id": "../escape"},
                      {"job": "eval", "job_id": "example", "output_prefix": "x132/other/"}):
            client = FakeS3()
            result = bench.run_job(event, client, bucket="test-bucket")
            self.assertEqual(result["status"], "FAILED")
            self.assertNotIn("secret-value", json.dumps(result))
            self.assertEqual(client.downloaded, [])
        manifest = self.manifest()
        manifest["images"][0]["image"] = "../escape.jpg"
        client = self.client(manifest)
        result = bench.run_job({"job": "train_samples", "job_id": "example", "shard_count": 2},
                               client, bucket="test-bucket")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("x132/jobs/example/shards/0000/FAILED.json", client.written)
        self.assertFalse(any(key.endswith("escape.jpg") for key in client.downloaded))

    def test_evaluation_outputs_match_offline_shard_contract_without_analysis(self):
        manifest = self.manifest("dev")
        client = self.client(manifest)
        event = {"job": "eval", "job_id": "example", "shard_count": 2, "shard_index": 0,
                 "detectors": ["classic"], "methods": ["agent", "exhaustive"], "budgets": [0.1]}

        def fake_evaluate(dataset, entries, methods, budgets, size, config, seed, *args, **kwargs):
            return [{"detector": "classic", "method": method, "seed": seed, "budget": budget,
                     "image": entry["image"], "n_found": 0, "n_defects": 2, "false_positives": 0}
                    for entry in entries for method in methods for budget in budgets]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("scripts.run_eval.evaluate_entries", side_effect=fake_evaluate):
                result = bench.run_job(event, client, bucket="test-bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "DONE")
        prefix = "x132/jobs/example/shards/0000/"
        metadata = json.loads(client.objects[prefix + "run.json"])
        rows = json.loads(client.objects[prefix + "results.json"])
        from scripts.run_eval import SHARD_SCHEMA, expected_record_keys, record_key
        self.assertEqual(metadata["schema"], SHARD_SCHEMA)
        self.assertEqual({record_key(row) for row in rows}, expected_record_keys(metadata["evaluation"], metadata["assigned_images"]))

    def test_more_than_four_assigned_images_is_rejected_before_image_download(self):
        manifest = self.manifest()
        manifest["images"] = [dict(manifest["images"][0], image=f"{i}.jpg", labels=f"{i}.png") for i in range(6)]
        manifest["images"][1]["surface"] = "concrete_pavement"
        client = self.client(manifest)
        result = bench.run_job({"job": "train_samples", "job_id": "example"}, client, bucket="test-bucket")
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(any(key.startswith("x132/datasets/") for key in client.downloaded))

    def test_existing_atomic_claim_does_not_recompute_or_overwrite_outputs(self):
        event = {"job": "eval", "job_id": "example"}
        validated = bench.validate_event(event, "x132/")
        digest = hashlib.sha256(json.dumps(validated, sort_keys=True, allow_nan=False).encode()).hexdigest()
        key = "x132/jobs/example/shards/0000/LOCK.json"
        client = FakeS3({key: json.dumps({"request_sha256": digest}).encode()})
        with patch("aws.handlers.bench._execute", side_effect=AssertionError("claim collision must not compute")):
            result = bench.run_job(event, client, bucket="test-bucket")
        self.assertEqual(result["status"], "RUNNING")
        self.assertEqual(client.written, [])

    def test_failed_job_never_automatically_recomputes(self):
        key = "x132/jobs/example/shards/0000/FAILED.json"
        failed = {"status": "FAILED", "error": {"code": "prior_failure"}}
        client = FakeS3({key: json.dumps(failed).encode()})
        with patch("aws.handlers.bench._execute", side_effect=AssertionError("failed job must remain stopped")):
            result = bench.run_job({"job": "eval", "job_id": "example"}, client, bucket="test-bucket")
        self.assertEqual(result, failed)
        self.assertEqual(client.written, [])

    def test_preclaim_read_failure_cannot_poison_an_existing_owners_prefix(self):
        class TransientReadFailure(FakeS3):
            def get_object(self, **kwargs):
                raise RuntimeError("temporary read failure")
        client = TransientReadFailure()
        result = bench.run_job({"job": "eval", "job_id": "example"}, client, bucket="test-bucket")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"]["code"], "preclaim_request_failed")
        self.assertEqual(client.written, [])


class BenchDriverTests(unittest.TestCase):
    def state(self, root):
        path = root / "state.json"
        path.write_text(json.dumps({"bucket": "test-bucket", "region": "us-west-2", "prefix": "x132/",
            "functions": {"bench": "x132-bench", "tool_arm64": "x132-tool-arm64", "tool_x86": "x132-tool-x86"}}))
        return path

    def test_dry_run_does_not_import_sdk_or_create_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            state = self.state(root)
            output = root / "not-created"
            with patch("aws.run_bench.get_clients", side_effect=AssertionError("SDK must not be accessed")):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = run_bench.main(["--job", "arm-benchmark", "--state", str(state),
                        "--image-ref", "s3://test-bucket/x132/images/fixture.jpg", "--detectors", "classic",
                        "--out", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(stdout.getvalue())
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["iterations"], 3)
            self.assertFalse(output.exists())

    def test_arm_collect_only_is_rejected_before_sdk_or_output_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            output = root / "not-created"
            with patch("aws.run_bench.get_clients", side_effect=AssertionError("SDK must not be accessed")):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = run_bench.main(["--job", "arm-benchmark", "--state", str(self.state(root)),
                        "--image-ref", "s3://test-bucket/x132/inputs/fixture.jpg", "--detectors", "classic",
                        "--job-id", "example", "--collect-only", "--execute", "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_new_model_directory_pins_its_content_bundle_over_older_deployed_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            state = self.state(root)
            deployed = json.loads(state.read_text())
            deployed["model_prefix"] = "x132/models/" + "0" * 64 + "/"
            state.write_text(json.dumps(deployed))
            models = root / "weights"
            models.mkdir()
            records = {scale: {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                       for scale in ("0.25", "0.5", "1")}
            (models / "metadata.json").write_text(json.dumps({"models": records}))
            for files in records.values():
                for name in files.values():
                    (models / name).write_bytes(b"tiny-fake-forest")
            (root / "image.jpg").write_bytes(b"not-decoded")
            (root / "labels.png").write_bytes(b"not-decoded")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"images": [{"image": "image.jpg", "labels": "labels.png",
                "split": "dev", "surface": "concrete", "defects": []}]}))
            with patch("aws.run_bench.get_clients", side_effect=AssertionError("SDK must not be accessed")):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = run_bench.main(["--job", "eval", "--state", str(state), "--detectors", "learned",
                        "--manifest", str(manifest), "--data-root", str(root), "--model-dir", str(models), "--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(stdout.getvalue())
            self.assertNotEqual(plan["model_prefix"], deployed["model_prefix"])
            self.assertRegex(plan["model_prefix"], r"^x132/models/[a-f0-9]{64}/$")
            self.assertEqual(plan["invocations"][0]["event"]["model_prefix"], plan["model_prefix"])
            self.assertTrue(all(upload["key"].startswith(plan["model_prefix"])
                                for upload in plan["uploads"] if upload["name"].endswith(".yml.gz")))

    def test_staging_normalizes_local_parent_paths_but_rejects_root_escape(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir()
            (root / "image.jpg").write_bytes(b"not-decoded")
            (root / "labels.png").write_bytes(b"not-decoded")
            path = root / "metadata/manifest.json"
            path.write_text(json.dumps({"images": [{"image": "../image.jpg", "labels": "../labels.png"}]}))
            normalized, uploads = run_bench.stage_manifest(path, root, "x132/datasets/id/")
            self.assertEqual(normalized["images"][0]["image"], "image.jpg")
            self.assertEqual(len(uploads), 2)
            with self.assertRaises(ValueError):
                run_bench.stage_manifest(path, root / "metadata", "x132/datasets/id/")

    def test_reused_job_id_cannot_upload_or_invoke_and_plan_is_saved_first(self):
        class NoInvoke:
            def invoke(self, **kwargs):
                raise AssertionError("a reused job must never invoke Lambda")
        guard_key = "x132/jobs/example/JOB.json"
        client = FakeS3({guard_key: b"existing-immutable-job"})
        plan = {"dry_run": False, "job": "eval", "job_id": "example", "bucket": "test-bucket",
                "region": "us-east-1", "prefix": "x132/", "collect_only": False,
                "job_guard": {"key": guard_key}, "uploads": [], "json_uploads": [], "invocations": []}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output = Path(tmp) / "recovery"
            with self.assertRaises(ClaimExists):
                run_bench.execute_plan(plan, output, s3=client, lambda_client=NoInvoke())
            self.assertEqual(json.loads((output / "plan.json").read_text())["job_id"], "example")
            self.assertEqual(client.written, [])

    def test_report_parser_does_not_fabricate_billed_duration(self):
        line = "REPORT RequestId: abc Duration: 12.34 ms Billed Duration: 13 ms Memory Size: 2048 MB"
        encoded = base64.b64encode(line.encode()).decode()
        self.assertEqual(run_bench.parse_report(encoded), {"duration_ms": 12.34, "billed_duration_ms": 13.0})
        partial = base64.b64encode(b"REPORT RequestId: abc Duration: 12.34 ms").decode()
        self.assertEqual(run_bench.parse_report(partial)["billed_duration_ms"], None)
        self.assertEqual(run_bench.parse_report(None), {"duration_ms": None, "billed_duration_ms": None})

    def test_fixed_tile_set_per_architecture_with_partial_log_coverage(self):
        class FakeLambda:
            def __init__(self):
                self.events = []

            def invoke(self, **kwargs):
                event = json.loads(kwargs["Payload"])
                self.events.append((kwargs["FunctionName"], event))
                if event["tool"] == "coarse_scan":
                    result = {"image_shape": [512, 512], "baseline": {"calibration": "sampled"}}
                else:
                    result = {"score": 0, "detections": []}
                log = "REPORT RequestId: abc Duration: 10.25 ms Billed Duration: 11 ms"
                response = {"Payload": io.BytesIO(json.dumps({"ok": True, "result": result}).encode())}
                if kwargs["FunctionName"] == "arm":
                    response["LogResult"] = base64.b64encode(log.encode()).decode()
                return response

        client = FakeLambda()
        tiles = [[0, 0, 256, 256], [256, 256, 128, 128]]
        result = run_bench.arm_benchmark(client, {"architectures": {"tool_arm64": "arm", "tool_x86": "x86"},
            "image_ref": "s3://test-bucket/x132/images/a.jpg", "detector": "classic", "rect": tiles[0], "tiles": tiles})
        self.assertEqual(len(client.events), 14)
        self.assertEqual(result["architectures"]["tool_arm64"]["duration_ms"]["n"], 6)
        self.assertEqual(result["architectures"]["tool_x86"]["round_trip_ms"]["n"], 6)
        for function in ("arm", "x86"):
            self.assertEqual([event["rect"] for name, event in client.events
                              if name == function and event["tool"] == "inspect_tile"], tiles*3)
        self.assertEqual(result["architectures"]["tool_arm64"]["coverage"], "complete")
        self.assertEqual(result["architectures"]["tool_x86"]["coverage"], "partial")
        self.assertIsNone(result["architectures"]["tool_x86"]["billed_duration_ms"]["p50"])

    def test_poll_checks_complete_identity_and_never_treats_pending_as_zero(self):
        done_key = "x132/jobs/example/shards/0000/DONE.json"
        marker = {"status": "DONE", "job_id": "example", "shard_index": 0, "shard_count": 1, "files": []}
        plan = {"bucket": "test-bucket", "job_id": "example", "shard_count": 1,
                "outputs": [{"shard_index": 0, "done_key": done_key, "failed_key": done_key.replace("DONE", "FAILED")}],
                "poll": {"timeout_seconds": 1, "interval_seconds": 0.1}}
        self.assertEqual(run_bench.poll_outputs(FakeS3({done_key: json.dumps(marker).encode()}), plan), [marker])
        with self.assertRaises(TimeoutError):
            run_bench.poll_outputs(FakeS3(), plan, sleep=lambda _: None, clock=iter([0, 2]).__next__)


if __name__ == "__main__":
    unittest.main()
