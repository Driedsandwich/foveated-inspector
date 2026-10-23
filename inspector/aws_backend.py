"""S3 image references and optional per-tool Lambda transport, with lazy SDK use."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import cv2

from .tools import LocalBackend, json_values

cv2.setNumThreads(2)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SAMPLE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TOOL_BYTES = 1024 * 1024


def _prefix(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ValueError("Invalid S3 prefix")
    if any(part in (".", "..") for part in value.split("/")) or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid S3 prefix")
    return value.rstrip("/") + "/"


def read_object(client, bucket: str, key: str, limit: int) -> bytes:
    """Read a bounded S3 body and always release its connection."""
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        if int(response.get("ContentLength", 0)) > limit:
            raise ValueError("S3 object exceeds its size limit")
        data = body.read(limit + 1)
        if not isinstance(data, bytes) or len(data) > limit:
            raise ValueError("S3 object exceeds its size limit")
        return data
    finally:
        if hasattr(body, "close"):
            body.close()


class AwsBackend(LocalBackend):
    """One warm decoded image, logical pixel accounting and JSON-only tool output.

    ``remote_tools=False`` runs OpenCV where this object lives, fetching only its
    image/model objects from S3. True invokes one Lambda per tool call and counts
    its complete round trip in elapsed_ms. boto3 is imported only when an actual
    uninjected client is required; importing this module never contacts AWS.
    """

    def __init__(self, bucket, prefix="x132/", detector="classic", model_dir=None,
                 s3_client=None, lambda_client=None, samples=None, remote_tools=False,
                 tool_function="x132-tool-arm64", calibration="sampled", model_prefix=None):
        if not isinstance(bucket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ValueError("Invalid configured bucket")
        if detector not in ("classic", "learned"):
            raise ValueError("Unknown detector")
        # Defer learned weights until a validated local tool call needs them.
        super().__init__(detector="classic", calibration=calibration)
        self.detector = detector
        self.bucket, self.prefix = bucket, _prefix(prefix)
        self._s3_client, self._lambda_client = s3_client, lambda_client
        self._samples = samples
        self.remote_tools = bool(remote_tools)
        self.tool_function = tool_function
        self._requested_model_dir = model_dir
        self._requested_model_prefix = model_prefix
        self._temp_root = None

    def _client(self, service):
        attr = "_s3_client" if service == "s3" else "_lambda_client"
        client = getattr(self, attr)
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("AWS SDK is unavailable") from exc
            client = boto3.client(service)
            setattr(self, attr, client)
        return client

    def sample_manifest(self):
        if self._samples is None:
            key = os.environ.get("X132_SAMPLES_KEY", self.prefix + "samples/manifest.json")
            self._validate_key(key)
            data = read_object(self._client("s3"), self.bucket, key, MAX_MANIFEST_BYTES)
            self._samples = json.loads(data)
        return self._samples

    def _validate_key(self, key):
        if (not isinstance(key, str) or not key.startswith(self.prefix) or len(key.encode()) > 1024
                or "\\" in key or any(ord(c) < 32 for c in key)
                or any(part in (".", "..") for part in key.split("/"))):
            raise ValueError("Image key is outside the configured namespace")

    def resolve_image_ref(self, image_ref):
        if not isinstance(image_ref, str) or not image_ref or len(image_ref) > 1100:
            raise ValueError("Invalid image reference")
        if image_ref.startswith("s3://"):
            parsed = urlsplit(image_ref)
            if parsed.netloc != self.bucket or parsed.query or parsed.fragment:
                raise ValueError("Image reference is outside the configured bucket")
            key = parsed.path.lstrip("/")
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", image_ref):
                raise ValueError("Invalid sample ID")
            manifest = self.sample_manifest()
            records = manifest.get("samples", []) if isinstance(manifest, dict) else manifest
            if not isinstance(records, list):
                raise ValueError("Invalid sample manifest")
            matches = [item for item in records if isinstance(item, dict) and item.get("id") == image_ref]
            if len(matches) != 1:
                raise ValueError("Unknown or duplicate sample ID")
            key = matches[0].get("key")
        self._validate_key(key)
        return f"s3://{self.bucket}/{key}"

    def image(self, image_ref):
        reference = self.resolve_image_ref(image_ref)
        if reference != self._image_ref:
            self._image, self._image_ref = None, None
            self._coarse_cache.clear()
            key = urlsplit(reference).path.lstrip("/")
            # Only administrator-prepared sample objects receive the larger
            # byte allowance; uploaded/frozen user inputs retain the 8 MiB cap.
            byte_limit = MAX_SAMPLE_IMAGE_BYTES if key.startswith(self.prefix + "samples/") else MAX_IMAGE_BYTES
            data = read_object(self._client("s3"), self.bucket, key, byte_limit)
            from aws.image_validation import validate_image_bytes
            image = validate_image_bytes(data, max_bytes=byte_limit, max_side=8192)
            self._image, self._image_ref = image, reference
        return self._image

    def _ensure_models(self):
        if self.detector == "learned" and self.models is None:
            from aws.runtime import materialize_model_dir
            from .model import DetectorModels
            directory = materialize_model_dir(bucket=self.bucket, prefix=self.prefix,
                                              model_dir=self._requested_model_dir,
                                              model_prefix=self._requested_model_prefix,
                                              s3_client=self._s3_client, temp_root=self._temp_root)
            self.models = DetectorModels(directory)

    def _invoke(self, action, image_ref, **parameters):
        started = time.perf_counter()
        event = {"action": action, "image_ref": self.resolve_image_ref(image_ref),
                 "detector": self.detector, "calibration": self.calibration, **parameters}
        if self._requested_model_prefix is not None:
            event["model_prefix"] = self._requested_model_prefix
        encoded = json.dumps(event, allow_nan=False).encode()
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("Tool request is too large")
        response = self._client("lambda").invoke(FunctionName=self.tool_function,
                                                  InvocationType="RequestResponse", Payload=encoded)
        body = response["Payload"]
        try:
            raw = body.read(MAX_TOOL_BYTES + 1)
        finally:
            if hasattr(body, "close"):
                body.close()
        if len(raw) > MAX_TOOL_BYTES or response.get("FunctionError"):
            raise RuntimeError("Remote inspection tool failed")
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or envelope.get("ok") is not True or not isinstance(envelope.get("result"), dict):
            raise RuntimeError("Remote inspection tool rejected the request")
        result = envelope["result"]
        elapsed = (time.perf_counter() - started) * 1000
        server_elapsed = float(result.get("elapsed_ms", 0))
        result["transport"] = {"mode": "lambda_per_tool", "includes_network_latency": True,
                               "round_trip_ms": elapsed, "tool_elapsed_ms": server_elapsed,
                               "transport_overhead_ms": max(0, elapsed - server_elapsed)}
        result["elapsed_ms"] = elapsed
        self.n_tool_calls += 1
        if action == "coarse_scan":
            self.coarse_cost_px += int(result["cost_px"])
            if not result.get("reused_preparation", False):
                self.physical_coarse_cost_px += int(result["cost_px"])
        else:
            self.fine_cost_px += int(result["cost_px"])
        return json_values(result)

    def coarse_scan(self, image_ref, **kw):
        if self.remote_tools:
            return self._invoke("coarse_scan", image_ref, **kw)
        started = time.perf_counter()
        reference = self.resolve_image_ref(image_ref)
        self._ensure_models()
        result = super().coarse_scan(reference, **kw)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["transport"] = {"mode": "s3_image_local_tools", "elapsed_includes_fetch": True}
        return result

    def inspect_tile(self, image_ref, rect, scale, baseline):
        if self.remote_tools:
            return self._invoke("inspect_tile", image_ref, rect=rect, scale=scale, baseline=baseline)
        started = time.perf_counter()
        reference = self.resolve_image_ref(image_ref)
        self._ensure_models()
        result = super().inspect_tile(reference, rect, scale, baseline)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["transport"] = {"mode": "s3_image_local_tools", "elapsed_includes_fetch": True}
        return result
