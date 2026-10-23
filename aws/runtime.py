"""Lazy AWS clients, warm backend reuse and bounded model materialization.

This module performs no AWS calls at import. Model downloads write only the
Lambda ephemeral directory, or an explicitly injected test temp_root.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

_CLIENTS = {}
_BACKEND = None
_MODEL_CACHE = {}
MODEL_FILES = {"metadata.json"} | {f"scale_{scale}_{kind}.yml.gz"
                                    for scale in ("0.25", "0.5", "1") for kind in ("binary", "kind")}
MAX_MODEL_FILE_BYTES = 8 * 1024 * 1024
MAX_MODEL_TOTAL_BYTES = 32 * 1024 * 1024


def get_clients(*, s3_client=None, lambda_client=None, require_lambda=False):
    def client(service):
        if service not in _CLIENTS:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("AWS SDK is unavailable") from exc
            _CLIENTS[service] = boto3.client(service)
        return _CLIENTS[service]
    return (s3_client if s3_client is not None else client("s3"),
            lambda_client if lambda_client is not None else (client("lambda") if require_lambda else None))


def _metadata_files(directory: Path):
    metadata_path = directory / "metadata.json"
    if metadata_path.stat().st_size > 256 * 1024:
        raise ValueError("Model metadata is too large")
    metadata = json.loads(metadata_path.read_text())
    models = metadata.get("models", {})
    expected = {scale: {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                for scale in ("0.25", "0.5", "1")}
    if models != expected:
        raise ValueError("Model metadata contains noncanonical filenames")
    total = 0
    for name in MODEL_FILES:
        path = directory / name
        size = path.stat().st_size
        if not 0 < size <= MAX_MODEL_FILE_BYTES:
            raise ValueError("Model file exceeds its size limit")
        total += size
    if total > MAX_MODEL_TOTAL_BYTES:
        raise ValueError("Model bundle exceeds its total size limit")


def materialize_model_dir(*, bucket=None, prefix=None, model_dir=None, model_prefix=None,
                          s3_client=None, temp_root=None):
    """Use /opt models when present, otherwise verified whitelist-only S3 files.

    ``prefix`` is the application prefix (x132/), or an explicit .../models/
    prefix. An explicit model_prefix overrides X132_MODEL_PREFIX and always
    selects S3. X132_MODEL_SOURCE=s3 also bypasses bundled /opt models. Revision
    prefixes should be immutable so warm caches cannot hide a redeployment.
    """
    from inspector.aws_backend import _prefix, read_object
    source = os.environ.get("X132_MODEL_SOURCE", "auto")
    if source not in ("auto", "local", "s3"):
        raise ValueError("Unknown model source")
    force_s3 = model_prefix is not None or source == "s3"
    local = Path(model_dir or os.environ.get("X132_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/models")))
    if not force_s3 and (local / "metadata.json").is_file():
        _metadata_files(local)
        return local
    if source == "local" and not force_s3:
        raise FileNotFoundError("Configured local model bundle is unavailable")
    bucket = bucket or os.environ.get("X132_BUCKET")
    if not bucket:
        raise ValueError("Model bucket is not configured")
    base = _prefix(prefix or os.environ.get("X132_PREFIX", "x132/"))
    namespace = base[:-len("models/")] if base.endswith("/models/") else base
    selected_prefix = _prefix(model_prefix or os.environ.get("X132_MODEL_PREFIX", namespace + "models/"))
    if not selected_prefix.startswith(namespace + "models/"):
        raise ValueError("Model prefix is outside the application model namespace")
    root = Path(temp_root) if temp_root is not None else Path("/tmp/x132-models")
    cache_key = (bucket, selected_prefix, source, str(root.resolve()), id(s3_client))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None and all((cached / name).is_file() for name in MODEL_FILES):
        return cached
    s3, _ = get_clients(s3_client=s3_client)
    encoded = read_object(s3, bucket, selected_prefix + "manifest.json", 32 * 1024)
    manifest = json.loads(encoded)
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(MODEL_FILES):
        raise ValueError("Model manifest must contain the seven whitelisted files")
    names, total = set(), 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid model file declaration")
        name, size, digest = record.get("name"), record.get("size_bytes"), record.get("sha256")
        if (name not in MODEL_FILES or name in names or not isinstance(size, int) or isinstance(size, bool)
                or not 0 < size <= MAX_MODEL_FILE_BYTES or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid model file declaration")
        names.add(name)
        total += size
    if names != MODEL_FILES or total > MAX_MODEL_TOTAL_BYTES:
        raise ValueError("Invalid model bundle size or coverage")
    identity = hashlib.sha256(bucket.encode() + selected_prefix.encode() + encoded).hexdigest()[:24]
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        data = read_object(s3, bucket, selected_prefix + record["name"], record["size_bytes"])
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Model file integrity check failed")
        target = directory / record["name"]
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
    _metadata_files(directory)
    _MODEL_CACHE[cache_key] = directory
    return directory


def get_backend(*, bucket=None, prefix=None, detector=None, model_dir=None, model_prefix=None,
                calibration=None, s3_client=None, lambda_client=None, samples=None,
                remote_tools=False, tool_function=None, temp_root=None):
    """One cached backend means at most one decoded image survives warm calls."""
    global _BACKEND
    from inspector.aws_backend import AwsBackend
    bucket = bucket or os.environ.get("X132_BUCKET")
    prefix = prefix or os.environ.get("X132_PREFIX", "x132/")
    detector = detector or os.environ.get("X132_DETECTOR", "classic")
    calibration = calibration or os.environ.get("X132_CALIBRATION", "sampled")
    model_dir = model_dir or os.environ.get("X132_MODEL_DIR", os.environ.get("MODEL_DIR"))
    function = tool_function or os.environ.get("X132_TOOL_FUNCTION", "x132-tool-arm64")
    effective_model_prefix = model_prefix or os.environ.get("X132_MODEL_PREFIX")
    model_source = os.environ.get("X132_MODEL_SOURCE", "auto")
    key = (bucket, prefix, detector, model_dir, calibration, id(s3_client), id(lambda_client),
           id(samples), bool(remote_tools), function, str(temp_root), effective_model_prefix, model_source,
           model_prefix is not None)
    if _BACKEND is None or _BACKEND[0] != key:
        backend = AwsBackend(bucket, prefix, detector, model_dir, s3_client, lambda_client,
                             samples, remote_tools, function, calibration, model_prefix=model_prefix)
        backend._temp_root = temp_root
        _BACKEND = (key, backend)
    return _BACKEND[1]
