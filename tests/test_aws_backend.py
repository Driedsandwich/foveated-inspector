"""Fake-client AWS boundaries. No SDK, credentials, AWS calls or large images."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aws.handlers.tool import handler, validate_request
from aws.image_validation import validate_image_bytes
from aws.runtime import MODEL_FILES, get_backend, get_clients, materialize_model_dir
from inspector.aws_backend import AwsBackend

cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[1]


class S3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}


class Lambda:
    def __init__(self):
        self.calls = []

    def invoke(self, **request):
        event = json.loads(request["Payload"])
        self.calls.append(event)
        result = {"cost_px": 16, "elapsed_ms": 0.0, "score": 0.0, "detections": []}
        return {"Payload": io.BytesIO(json.dumps({"ok": True, "result": result}).encode())}


class AwsBackendTests(unittest.TestCase):
    def setUp(self):
        success, encoded = cv2.imencode(".jpg", np.full((64, 64, 3), 130, np.uint8))
        self.assertTrue(success)
        self.data = encoded.tobytes()

    def test_one_decoded_image_cache_and_json_tools(self):
        s3 = S3({"x132/samples/a.jpg": self.data, "x132/samples/b.jpg": self.data})
        samples = {"version": 1, "samples": [{"id": "a", "key": "x132/samples/a.jpg"}]}
        backend = AwsBackend("test-bucket", s3_client=s3, samples=samples)
        image = backend.image("a")
        self.assertIs(image, backend.image("s3://test-bucket/x132/samples/a.jpg"))
        self.assertEqual(len(s3.calls), 1)
        overview = backend.coarse_scan("a", grid=(2, 2), work_width=32)
        detail = backend.inspect_tile("a", [0, 0, 32, 32], 1.0, overview["baseline"])
        json.dumps(overview, allow_nan=False)
        json.dumps(detail, allow_nan=False)
        self.assertEqual(backend.n_tool_calls, 2)
        backend.image("s3://test-bucket/x132/samples/b.jpg")
        self.assertEqual(len(s3.calls), 2)
        self.assertEqual(backend._image_ref, "s3://test-bucket/x132/samples/b.jpg")

    def test_namespace_rejected_before_fetch_or_model_loading(self):
        s3 = S3({})
        backend = AwsBackend("test-bucket", detector="learned", s3_client=s3, samples={"samples": []})
        for reference in ("s3://other-bucket/x132/a.jpg", "s3://test-bucket/other/a.jpg",
                          "s3://test-bucket/x132/../a.jpg", "/etc/passwd", "missing"):
            with self.assertRaises(ValueError):
                backend.coarse_scan(reference)
        self.assertEqual(s3.calls, [])

    def test_only_trusted_sample_namespace_receives_32_mib_read_allowance(self):
        backend = AwsBackend("test-bucket", s3_client=S3({}))
        with patch("inspector.aws_backend.read_object", return_value=self.data) as read, \
                patch("aws.image_validation.validate_image_bytes", wraps=validate_image_bytes) as decode:
            backend.image("s3://test-bucket/x132/samples/prepared-real-crack.jpg")
            self.assertEqual(read.call_args.args[3], 32 * 1024 * 1024)
            self.assertEqual(decode.call_args.kwargs["max_bytes"], 32 * 1024 * 1024)
            backend.image("s3://test-bucket/x132/inputs/frozen-upload.jpg")
            self.assertEqual(read.call_args.args[3], 8 * 1024 * 1024)
            self.assertEqual(decode.call_args.kwargs["max_bytes"], 8 * 1024 * 1024)
            self.assertEqual(decode.call_args.kwargs["max_side"], 8192)

    def test_remote_tool_calls_include_latency_and_accounting(self):
        client = Lambda()
        backend = AwsBackend("test-bucket", lambda_client=client, remote_tools=True)
        reference = "s3://test-bucket/x132/a.jpg"
        result = backend.coarse_scan(reference, grid=(2, 2))
        backend.inspect_tile(reference, [0, 0, 4, 4], 1.0, {})
        self.assertEqual([item["action"] for item in client.calls], ["coarse_scan", "inspect_tile"])
        self.assertTrue(result["transport"]["includes_network_latency"])
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertEqual((backend.coarse_cost_px, backend.fine_cost_px, backend.n_tool_calls), (16, 16, 2))

    def test_handler_strict_input_and_sanitized_error(self):
        s3 = S3({"x132/a.jpg": self.data})
        backend = AwsBackend("test-bucket", s3_client=s3)
        event = {"tool": "coarse_scan", "image_ref": "s3://test-bucket/x132/a.jpg", "grid": [2, 2], "work_width": 32}
        self.assertTrue(handler(event, backend=backend)["ok"])
        invalid = {**event, "grid": [True, 2]}
        self.assertEqual(handler(invalid, backend=backend)["error"]["code"], "invalid_request")
        invalid = {"tool": "inspect_tile", "image_ref": event["image_ref"],
                   "rect": [0, 0, 10, 10], "scale": .3, "baseline": {}}
        self.assertFalse(handler(invalid, backend=backend)["ok"])
        with patch.object(backend, "coarse_scan", side_effect=RuntimeError("secret-token-value")):
            result = handler(event, backend=backend)
        self.assertEqual(result["error"]["code"], "tool_failed")
        self.assertNotIn("secret-token-value", json.dumps(result))

    def test_injected_clients_and_verified_model_cold_cache(self):
        s3, lambda_client = S3({}), Lambda()
        self.assertEqual(get_clients(s3_client=s3, lambda_client=lambda_client), (s3, lambda_client))
        metadata = {"models": {scale: {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                               for scale in ("0.25", "0.5", "1")}}
        payloads = {name: (json.dumps(metadata).encode() if name == "metadata.json" else b"small-model-fixture")
                    for name in MODEL_FILES}
        manifest = {"files": [{"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in sorted(payloads.items())]}
        s3.objects = {"x132/models/manifest.json": json.dumps(manifest).encode(),
                      **{"x132/models/" + name: data for name, data in payloads.items()}}
        with tempfile.TemporaryDirectory(prefix="_test_aws_models_", dir=ROOT) as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"X132_MODEL_PREFIX": "x132/models/"}):
                first = materialize_model_dir(bucket="test-bucket", s3_client=s3,
                                              model_dir=root / "missing", temp_root=root / "cache")
                second = materialize_model_dir(bucket="test-bucket", s3_client=s3,
                                               model_dir=root / "missing", temp_root=root / "cache")
            self.assertEqual(first, second)
            self.assertEqual(len(s3.calls), 8)
            self.assertEqual({p.name for p in first.iterdir()}, MODEL_FILES)

            # A deployed S3 source must not silently select older bundled /opt
            # weights. This tiny directory stands in for a stale layer bundle.
            stale = root / "stale-opt-models"
            stale.mkdir()
            for name in MODEL_FILES:
                (stale / name).write_bytes(payloads[name] if name == "metadata.json" else b"stale-layer-model")
            with patch.dict(os.environ, {"X132_MODEL_SOURCE": "s3", "X132_MODEL_PREFIX": "x132/models/"}):
                selected = materialize_model_dir(bucket="test-bucket", s3_client=s3,
                                                 model_dir=stale, temp_root=root / "s3-source")
            self.assertNotEqual(selected, stale)
            self.assertEqual((selected / "scale_1_binary.yml.gz").read_bytes(), b"small-model-fixture")

            # A benchmark revision overrides both an old model environment and
            # a valid but stale local bundle, without touching that old prefix.
            revision = "x132/models/" + "a" * 64 + "/"
            s3.objects.update({revision + "manifest.json": json.dumps(manifest).encode(),
                               **{revision + name: data for name, data in payloads.items()}})
            before = len(s3.calls)
            with patch.dict(os.environ, {"X132_MODEL_SOURCE": "auto", "X132_MODEL_PREFIX": "x132/models/" + "b" * 64 + "/"}):
                selected = materialize_model_dir(bucket="test-bucket", s3_client=s3, model_dir=stale,
                                                 model_prefix=revision, temp_root=root / "explicit-revision")
            self.assertNotEqual(selected, stale)
            self.assertTrue(all(key.startswith(revision) for _, key in s3.calls[before:]))

    def test_warm_backend_model_revision_and_source_are_part_of_cache_key(self):
        s3 = S3({})
        revision_a, revision_b = "x132/models/" + "a" * 64 + "/", "x132/models/" + "b" * 64 + "/"
        with patch.dict(os.environ, {"X132_MODEL_PREFIX": revision_a, "X132_MODEL_SOURCE": "auto"}):
            first = get_backend(bucket="test-bucket", detector="classic", s3_client=s3)
            self.assertIs(first, get_backend(bucket="test-bucket", detector="classic", s3_client=s3))
            explicit = get_backend(bucket="test-bucket", detector="classic", s3_client=s3, model_prefix=revision_a)
            self.assertIsNot(first, explicit)
        with patch.dict(os.environ, {"X132_MODEL_PREFIX": revision_b, "X132_MODEL_SOURCE": "s3"}):
            changed = get_backend(bucket="test-bucket", detector="classic", s3_client=s3)
        self.assertIsNot(first, changed)
        self.assertEqual(s3.calls, [])

    def test_tool_model_prefix_is_an_immutable_same_application_revision(self):
        event = {"tool": "coarse_scan", "image_ref": "s3://test-bucket/x132/a.jpg"}
        with patch.dict(os.environ, {"X132_PREFIX": "x132/"}):
            self.assertEqual(validate_request({**event, "model_prefix": "x132/models/" + "a" * 64 + "/"}), "coarse_scan")
            for value in ("other/models/" + "a" * 64 + "/", "x132/models/", "x132/models/../secret/"):
                with self.assertRaises(ValueError):
                    validate_request({**event, "model_prefix": value})


if __name__ == "__main__":
    unittest.main()
