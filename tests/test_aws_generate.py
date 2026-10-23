"""Offline S3 generation and resident-benchmark contracts; no image processing."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aws.handlers import bench


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class ClaimExists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class FakeS3:
    def __init__(self, objects=None, fail_upload=False):
        self.objects = dict(objects or {})
        self.downloaded = []
        self.written = []
        self.fail_upload = fail_upload

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

    def upload_file(self, filename, bucket, key):
        if self.fail_upload and key.endswith(".jpg"):
            raise OSError("simulated upload failure")
        self.objects[key] = Path(filename).read_bytes()
        self.written.append(key)


class GenerateTests(unittest.TestCase):
    names = ["asphalt_floor", "brushed_concrete", "brushed_concrete_2", "concrete"]

    def client(self, fail_upload=False):
        records, objects = [], {}
        for name in self.names + ["cracked_concrete"]:
            data = ("photo-fixture-" + name).encode()
            key = f"x132/surfaces/{name}.jpg"
            records.append({"name": name, "key": key, "size_bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(), "clean": name != "cracked_concrete"})
            objects[key] = data
        objects["x132/surfaces/manifest.json"] = json.dumps({"version": 1, "surfaces": records}).encode()
        return FakeS3(objects, fail_upload)

    def event(self, **kwargs):
        return {"job": "generate", "job_id": "generation", "shard_index": 0, "shard_count": 3,
                "benchmark": {"surface_names": list(reversed(self.names)), "per_surface": 1,
                              "train_per_surface": 1, "seed": 123, "size": 1024, "version": "v2"}, **kwargs}

    def fake_generators(self, calls, *, inspect_failure_cleanup=False):
        def make(paths, output, per_surface, seed, size=4096, version="v2", shard_index=0,
                 shard_count=1, on_image_written=None, namespace="eval"):
            from inspector.synth import DEV_SURFACES
            paths = list(paths)
            self.assertEqual([path.stem for path in paths], sorted(path.stem for path in paths))
            call_record = {"namespace": namespace, "names": [path.stem for path in paths], "seed": seed,
                           "size": size, "version": version, "per_surface": per_surface}
            calls.append(call_record)
            destination = output / f"shard_{shard_index:04d}_of_{shard_count:04d}" if shard_count > 1 else output
            destination.mkdir(parents=True, exist_ok=True)
            manifest = {"namespace": namespace, "surfaces": [path.stem for path in paths],
                        "seed": seed, "size": size, "version": int(version[-1]), "per_surface": per_surface,
                        "n_images_total": len(paths) * per_surface, "shard_index": shard_index,
                        "shard_count": shard_count, "images": []}
            for surface_index, path in enumerate(paths):
                for sample in range(per_surface):
                    index = surface_index * per_surface + sample
                    if index % shard_count != shard_index:
                        continue
                    self.assertTrue(path.is_file(), "assigned input must be downloaded")
                    image = destination / f"img_{index:04d}.jpg"
                    labels = destination / f"img_{index:04d}_labels.png"
                    image.write_bytes(b"generated-image-fixture")
                    labels.write_bytes(b"generated-label-fixture")
                    entry = {"image": image.name, "labels": labels.name, "surface": path.stem,
                             "global_index": index, "sample_index": sample, "defects": [],
                             "split": "train" if namespace == "train" else "dev" if path.stem in DEV_SURFACES else "test"}
                    manifest["images"].append(entry)
                    try:
                        on_image_written(image, labels, entry)
                    except Exception:
                        if inspect_failure_cleanup:
                            call_record["failure_cleanup_ok"] = not image.exists() and not labels.exists()
                            self.assertFalse(image.exists())
                            self.assertFalse(labels.exists())
                        raise
                    self.assertFalse(image.exists(), "streamed image must not remain on disk")
                    self.assertFalse(labels.exists(), "streamed label must not remain on disk")
            (destination / "manifest.json").write_text(json.dumps(manifest))
            return manifest

        def train(*args, **kwargs):
            return make(*args, **kwargs, namespace="train")
        return make, train

    def test_generation_fetches_only_union_of_assigned_sources_and_streams_all_outputs(self):
        client, calls = self.client(), []
        make, train = self.fake_generators(calls)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("inspector.synth.make_benchmark", side_effect=make), patch("inspector.synth.make_training_benchmark", side_effect=train):
                result = bench.run_job(self.event(), client, bucket="bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "DONE")
        photos = {key for key in client.downloaded if key.startswith("x132/surfaces/") and key.endswith(".jpg")}
        self.assertEqual(photos, {"x132/surfaces/asphalt_floor.jpg", "x132/surfaces/concrete.jpg"})
        self.assertEqual(calls[0]["names"], sorted(self.names))
        self.assertEqual(calls[1]["names"], ["asphalt_floor", "brushed_concrete", "concrete"])
        self.assertTrue(all(call["seed"] == 123 and call["size"] == 1024 for call in calls))
        self.assertEqual(len(result["files"]), 8)  # Three pairs and two manifests.
        self.assertTrue(all(item["key"].startswith("x132/benchmarks/generation/") for item in result["files"]))
        self.assertIn("x132/jobs/generation/shards/0000/DONE.json", client.written)
        root = "x132/benchmarks/generation/"
        evaluation = json.loads(client.objects[root + "shard_0000_of_0003/manifest.json"])
        training = json.loads(client.objects[root + "train/shard_0000_of_0003/manifest.json"])
        self.assertEqual(evaluation["s3_prefix"], root)
        self.assertEqual(training["s3_prefix"], root + "train/")
        self.assertEqual([record["name"] for record in evaluation["source_surfaces"]], sorted(self.names))
        from inspector.synth import DEV_SURFACES
        self.assertEqual(training["source_surfaces"], [record for record in evaluation["source_surfaces"]
                                                      if record["name"] in DEV_SURFACES])
        self.assertTrue(set(training["surfaces"]).issubset(DEV_SURFACES))
        self.assertTrue({record["name"] for record in training["source_surfaces"]}.issubset(DEV_SURFACES))
        self.assertTrue({entry["surface"] for entry in training["images"]}.issubset(DEV_SURFACES))
        self.assertTrue(all("source_sha256" in entry for entry in evaluation["images"] + training["images"]))

    def test_provenance_is_identical_across_shards_even_when_sources_differ(self):
        client, calls = self.client(), []
        make, train = self.fake_generators(calls)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("inspector.synth.make_benchmark", side_effect=make), patch("inspector.synth.make_training_benchmark", side_effect=train):
                one = bench.run_job(self.event(), client, bucket="bucket", temp_root=Path(tmp))
                two = bench.run_job(self.event(shard_index=1), client, bucket="bucket", temp_root=Path(tmp))
        self.assertEqual(one["status"], "DONE")
        self.assertEqual(two["status"], "DONE")
        manifests = [json.loads(client.objects[f"x132/benchmarks/generation/shard_{index:04d}_of_0003/manifest.json"])
                     for index in (0, 1)]
        configs = [{key: value for key, value in manifest.items() if key not in ("images", "shard_index")}
                   for manifest in manifests]
        self.assertEqual(configs[0], configs[1])

    def test_hash_mismatch_stops_before_generation(self):
        client = self.client()
        client.objects["x132/surfaces/asphalt_floor.jpg"] = b"changed"
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("inspector.synth.make_benchmark", side_effect=AssertionError("no generation on hash mismatch")) as make:
                result = bench.run_job(self.event(), client, bucket="bucket", temp_root=Path(tmp))
                make.assert_not_called()
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("x132/jobs/generation/shards/0000/FAILED.json", client.written)

    def test_pair_files_are_removed_even_when_upload_fails(self):
        client, calls = self.client(fail_upload=True), []
        make, train = self.fake_generators(calls, inspect_failure_cleanup=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("inspector.synth.make_benchmark", side_effect=make), patch("inspector.synth.make_training_benchmark", side_effect=train):
                result = bench.run_job(self.event(), client, bucket="bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(any(call.get("failure_cleanup_ok", False) for call in calls))

    def test_single_shard_train_only_uses_train_root_without_shard_subdirectory(self):
        client, calls = self.client(), []
        make, train = self.fake_generators(calls)
        event = {"job": "generate", "job_id": "trainonly", "benchmark": {
            "surface_names": ["asphalt_floor"], "train_only": True, "train_images": 1,
            "size": 1024, "version": "v1"}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("inspector.synth.make_benchmark", side_effect=make) as evaluate, patch("inspector.synth.make_training_benchmark", side_effect=train):
                result = bench.run_job(event, client, bucket="bucket", temp_root=Path(tmp))
                evaluate.assert_not_called()
        self.assertEqual(result["status"], "DONE")
        self.assertEqual([call["namespace"] for call in calls], ["train"])
        self.assertIn("x132/benchmarks/trainonly/train/manifest.json", client.objects)
        self.assertFalse(any("shard_" in item["key"] for item in result["files"]))

    def test_generation_validation_rejects_bool_nonfinite_unknown_real_or_mixed_parameters(self):
        invalid = [{"per_surface": True}, {"per_surface": float("inf")}, {"seed": float("nan")},
                   {"seed": -1}, {"size": 1500}, {"size": True}, {"version": "v3"},
                   {"train_only": 1}, {"train_only": True, "train_images": 0},
                   {"train_images": True}, {"train_images": 1, "train_per_surface": 2},
                   {"unknown": 1}, {"surface_names": ["cracked_concrete"]},
                   {"surface_names": ["asphalt_floor", "asphalt_floor"]}]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                bench.validate_event({"job": "generate", "job_id": "id", "benchmark": spec}, "x132/")
        for count in (True, 0, 1025):
            with self.assertRaises(ValueError):
                bench.validate_event({"job": "generate", "job_id": "id", "shard_count": count}, "x132/")
        normalized = bench.validate_event({"job": "generate", "job_id": "id"}, "x132/")["benchmark"]
        self.assertEqual((normalized["per_surface"], normalized["train_per_surface"], normalized["version"]), (4, 4, "v2"))


class ResidentBenchmarkTests(unittest.TestCase):
    def client(self, namespace):
        prefix = "x132/benchmarks/generated/" + ("train/" if namespace == "train" else "")
        manifest = {"images": [{"image": "shard_0000_of_0001/img_0000.jpg", "labels": "shard_0000_of_0001/img_0000_labels.png",
            "surface": "asphalt_floor", "split": "train" if namespace == "train" else "dev", "seed": 1,
            "defects": [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]},
            {"image": "shard_0000_of_0001/img_0001.jpg", "labels": "shard_0000_of_0001/img_0001_labels.png",
             "surface": "concrete_pavement", "split": "train" if namespace == "train" else "dev", "seed": 2,
             "defects": [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]}]}
        objects = {prefix + "manifest.json": json.dumps(manifest).encode()}
        for entry in manifest["images"]:
            for field in ("image", "labels"):
                objects[prefix + entry[field]] = b"fixture-not-decoded"
        return FakeS3(objects), prefix

    def test_resident_eval_reads_remote_manifest_and_only_its_assigned_pair(self):
        client, prefix = self.client("eval")
        event = {"job": "eval", "job_id": "evaluate", "benchmark_s3": "x132/benchmarks/generated/",
                 "shard_count": 2, "detectors": ["classic"], "methods": ["agent"], "budgets": [0.1]}

        def evaluate(dataset, entries, methods, budgets, size, config, seed, *args, **kwargs):
            self.assertEqual(len(entries), 1)
            self.assertEqual(json.loads((dataset / "manifest.json").read_text())["s3_prefix"], prefix)
            return [{"detector": "classic", "method": "agent", "seed": seed, "budget": 0.1,
                     "image": entries[0]["image"], "n_defects": 2, "n_found": 0, "false_positives": 0}]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("scripts.run_eval.evaluate_entries", side_effect=evaluate):
                result = bench.run_job(event, client, bucket="bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "DONE")
        self.assertIn(prefix + "manifest.json", client.downloaded)
        self.assertNotIn("x132/jobs/evaluate/manifest.json", client.downloaded)
        self.assertNotIn(prefix + "shard_0000_of_0001/img_0001.jpg", client.downloaded)

    def test_resident_training_reads_train_manifest_and_retains_validation_metadata(self):
        client, prefix = self.client("train")
        event = {"job": "train_samples", "job_id": "extract", "benchmark_s3": "x132/benchmarks/generated/", "shard_count": 2,
                 "holdout_surface": "concrete_pavement"}

        def extract(dataset, output, **kwargs):
            manifest = json.loads((dataset / "manifest.json").read_text())
            self.assertEqual(len(manifest["images"]), 2)
            self.assertEqual(manifest["s3_prefix"], prefix)
            (output / "shard_0000_of_0002.json").write_text(json.dumps({"records": []}))
            return {}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("scripts.train_detector.extract_samples", side_effect=extract):
                result = bench.run_job(event, client, bucket="bucket", temp_root=Path(tmp))
        self.assertEqual(result["status"], "DONE")
        self.assertIn(prefix + "manifest.json", client.downloaded)
        self.assertNotIn("x132/jobs/extract/manifest.json", client.downloaded)

    def test_resident_mode_forbids_manifest_mixing_and_nonproject_prefixes(self):
        for source in ("x132/benchmarks/../escape/", "elsewhere/benchmarks/id/", "s3://bucket/x132/benchmarks/id/"):
            with self.assertRaises(ValueError):
                bench.validate_event({"job": "eval", "job_id": "id", "benchmark_s3": source}, "x132/")
        with self.assertRaises(ValueError):
            bench.validate_event({"job": "eval", "job_id": "id", "benchmark_s3": "x132/benchmarks/source/",
                                  "manifest_key": "x132/jobs/id/manifest.json"}, "x132/")


if __name__ == "__main__":
    unittest.main()
