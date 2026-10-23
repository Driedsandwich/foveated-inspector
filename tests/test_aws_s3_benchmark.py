"""Metadata-only resident benchmark plans and collection; no image decoding/AWS."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aws import run_bench, upload_surfaces
from inspector.synth import DEV_SURFACES, REAL_CRACK_SURFACES, TEST_SURFACES, _example_seed

ROOT = Path(__file__).resolve().parents[1]


class Missing(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class Exists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class S3:
    def __init__(self, objects=None):
        self.objects, self.metadata = dict(objects or {}), {}
        self.reads, self.writes = [], []

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        if Key.endswith((".jpg", ".png")):
            raise AssertionError("image bodies must remain in S3")
        if Key not in self.objects:
            raise Missing()
        value = self.objects[Key]
        return {"Body": io.BytesIO(value), "ContentLength": len(value)}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise Missing()
        return {"ContentLength": len(self.objects[Key]), "Metadata": self.metadata.get(Key, {})}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise Exists()
        self.objects[Key] = Body.read() if hasattr(Body, "read") else Body
        self.metadata[Key] = kwargs.get("Metadata", {})
        self.writes.append(Key)
        return {}


def state_file(root):
    path = root / "state.json"
    path.write_text(json.dumps({"version": 1, "project": "x132", "prefix": "x132/", "suffix": "fixture",
                               "resources": [], "bucket": "test-bucket", "region": "us-west-2",
                               "functions": {"bench": "x132-bench"}}))
    return path


def generation_fixture(count=2):
    config = {"per_surface": count, "train_per_surface": count, "seed": 9,
              "size": 1024, "version": "v2", "surface_names": ["asphalt_floor"], "train_only": False}
    plan = {"job": "generate", "job_id": "fixture", "prefix": "x132/", "bucket": "test-bucket",
            "benchmark": config, "benchmark_prefix": "x132/benchmarks/fixture/", "shard_count": count}
    objects, markers = {}, []
    for index in range(count):
        files = []
        for namespace in ("eval", "train"):
            namespace_prefix = plan["benchmark_prefix"] + ("train/" if namespace == "train" else "")
            shard_prefix = namespace_prefix + (f"shard_{index:04d}_of_{count:04d}/" if count > 1 else "")
            entry = {"global_index": index, "sample_index": index, "surface": "asphalt_floor",
                     "split": "train" if namespace == "train" else "dev", "width": 1024, "height": 1024,
                     "seed": _example_seed("asphalt_floor", index, 9, 2, namespace), "defects": [],
                     "image": f"img_{index:04d}.jpg", "labels": f"img_{index:04d}_labels.png"}
            manifest = {"version": 2, "namespace": namespace, "seed": 9, "size": 1024,
                        "per_surface": count, "surfaces": ["asphalt_floor"], "n_images_total": count,
                        "shard_index": index, "shard_count": count, "s3_prefix": namespace_prefix,
                        "images": [entry]}
            data = json.dumps(manifest).encode()
            key = shard_prefix + "manifest.json"
            objects[key] = data
            files.append({"name": "manifest.json", "key": key, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            for field in ("image", "labels"):
                files.append({"name": entry[field], "key": shard_prefix + entry[field],
                              "size_bytes": 12, "sha256": hashlib.sha256(b"not-downloaded").hexdigest()})
        markers.append({"status": "DONE", "job": "generate", "job_id": "fixture",
                        "shard_index": index, "shard_count": count, "files": files})
    return plan, S3(objects), markers


class S3BenchmarkTests(unittest.TestCase):
    def dry_plan(self, root, arguments):
        output = root / "no-output"
        with patch("aws.run_bench.get_clients", side_effect=AssertionError("no SDK in dry run")):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = run_bench.main(["--state", str(state_file(root)), "--out", str(output), "--dry-run", *arguments])
        self.assertEqual(code, 0, stream.getvalue())
        self.assertFalse(output.exists())
        return json.loads(stream.getvalue())

    def test_generate_plan_has_cli_parity_and_no_local_image_uploads(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            plan = self.dry_plan(Path(temporary), ["--job", "generate", "--job-id", "generated", "--shards", "1",
                "--per-surface", "4", "--train-images", "4", "--seed", "19", "--size", "2048", "--version", "v2"])
        self.assertEqual(plan["uploads"], [])
        self.assertEqual(plan["json_uploads"], [])
        event = plan["invocations"][0]["event"]
        self.assertEqual(event["benchmark"]["train_per_surface"], 4)
        self.assertEqual(event["benchmark"]["seed"], 19)
        self.assertEqual(plan["benchmark_prefix"], "x132/benchmarks/generated/")
        self.assertIn("900", plan["runtime_note"])

    def test_resident_plan_dry_run_reads_nothing_and_execute_selection_is_metadata_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            plan = self.dry_plan(Path(temporary), ["--job", "eval", "--job-id", "evaluate", "--detectors", "classic",
                "--benchmark-s3", "x132/benchmarks/generated", "--shards", "2"])
        self.assertEqual(plan["uploads"], [])
        self.assertEqual(plan["invocations"][0]["event"]["benchmark_s3"], "x132/benchmarks/generated/")
        key = "x132/benchmarks/generated/manifest.json"
        manifest = {"s3_prefix": "x132/benchmarks/generated/", "images": [
            {"image": f"shard_{i:04d}_of_0002/img.jpg", "labels": f"shard_{i:04d}_of_0002/labels.png", "split": "dev", "surface": "concrete"}
            for i in range(2)]}
        client = S3({key: json.dumps(manifest).encode()})
        run_bench.validate_resident_selection(client, plan)
        self.assertEqual(plan["selection"]["selected_image_count"], 2)
        self.assertEqual(client.reads, [key])

    def test_1024_analysis_job_is_rejected_before_sdk_or_output_creation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            output = root / "must-not-exist"
            with patch("aws.run_bench.get_clients", side_effect=AssertionError("no SDK for invalid options")):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = run_bench.main(["--job", "eval", "--size", "1024", "--execute", "--state", str(state_file(root)),
                        "--benchmark-s3", "x132/benchmarks/generated/", "--shards", "1", "--detectors", "classic", "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_collection_mirrors_s3_layout_and_never_gets_image_bodies(self):
        plan, client, markers = generation_fixture(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            result = run_bench.collect_generation(client, plan, markers, root)
            self.assertEqual(len(result), 2)
            evaluation = json.loads((root / "manifest.json").read_text())
            train = json.loads((root / "train/manifest.json").read_text())
            self.assertEqual(evaluation["s3_prefix"], "x132/benchmarks/fixture/")
            self.assertEqual(train["s3_prefix"], "x132/benchmarks/fixture/train/")
            self.assertTrue(all(entry["image"].startswith("shard_") and ".." not in entry["image"] for entry in train["images"]))
        self.assertTrue(all(key.endswith(".json") for key in client.reads))
        self.assertIn("x132/benchmarks/fixture/manifest.json", client.writes)

    def test_single_shard_already_final_manifest_is_validated_without_overwrite(self):
        plan, client, markers = generation_fixture(1)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_bench.collect_generation(client, plan, markers, Path(temporary))
        self.assertEqual(client.writes, [])

    def test_missing_upload_descriptor_is_not_treated_as_success(self):
        plan, client, markers = generation_fixture(2)
        markers[0]["files"] = [record for record in markers[0]["files"] if not record["key"].endswith("img_0000.jpg")]
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            with self.assertRaises(ValueError):
                run_bench.collect_generation(client, plan, markers, Path(temporary))

    def test_collect_only_recovers_seed_counts_and_shards_from_job_guard(self):
        plan, client, _ = generation_fixture(2)
        invocations = [{"FunctionName": "x132-bench", "event": {"job": "generate", "job_id": "fixture",
                        "shard_index": index, "shard_count": 2, "benchmark": plan["benchmark"]}} for index in range(2)]
        key = "x132/jobs/fixture/JOB.json"
        client.objects[key] = json.dumps({"job": "generate", "job_id": "fixture", "submission": {
            "shard_count": 2, "benchmark": plan["benchmark"], "invocations": invocations}}).encode()
        plan.update(job_guard={"key": key}, benchmark={"seed": 0}, shard_count=1)
        run_bench.recover_generation_plan(client, plan)
        self.assertEqual(plan["shard_count"], 2)
        self.assertEqual(plan["benchmark"]["seed"], 9)
        self.assertTrue(plan["recovered_from_job_guard"])


class SurfaceUploadTests(unittest.TestCase):
    def fixture(self, root):
        surfaces = root / "surfaces"
        surfaces.mkdir()
        for name in DEV_SURFACES | TEST_SURFACES | REAL_CRACK_SURFACES:
            (surfaces / (name + ".jpg")).write_bytes(b"\xff\xd8tiny-source-fixture")
        return surfaces, state_file(root)

    def test_dry_run_hashes_exact_catalog_without_sdk_or_decode(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            surfaces, state = self.fixture(Path(temporary))
            with patch("aws.upload_surfaces.get_client", side_effect=AssertionError("no SDK")):
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    code = upload_surfaces.main(["--surfaces", str(surfaces), "--state", str(state), "--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(stream.getvalue())
            self.assertEqual((plan["n_clean"], plan["n_demo_only"], len(plan["uploads"])), (14, 3, 17))
            self.assertEqual(plan["images_decoded"], 0)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in plan["catalog"]["surfaces"]))
            with self.assertRaises(ValueError):
                upload_surfaces.make_plan(surfaces, state, region="wrong-region")

    def test_repeated_upload_reuses_identical_files_and_rejects_changed_content(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            surfaces, state = self.fixture(Path(temporary))
            plan = upload_surfaces.make_plan(surfaces, state, execute=True)
            client = S3()
            first = upload_surfaces.execute_plan(plan, s3=client)
            second = upload_surfaces.execute_plan(plan, s3=client)
            self.assertEqual((first["uploaded"], second["reused"]), (17, 17))
            client.metadata[plan["uploads"][0]["key"]] = {"sha256": "0" * 64}
            with self.assertRaises(ValueError):
                upload_surfaces.execute_plan(plan, s3=client)


if __name__ == "__main__":
    unittest.main()
