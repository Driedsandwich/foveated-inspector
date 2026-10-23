"""Plan or upload the exact CC0 source-photo catalog; never decode photographs.

Dry run is the default and never imports an AWS SDK. Execute streams the 17
whitelisted JPEG files into the owned deployment bucket. Existing keys with
different content are rejected rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aws.deploy import APP_ROOT, load_state

MAX_SURFACE_BYTES = 32 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plan(surfaces: Path, state_path: Path, *, region=None, bucket=None, execute=False) -> dict:
    from inspector.synth import DEV_SURFACES, REAL_CRACK_SURFACES, TEST_SURFACES
    state = load_state(state_path)
    if state is None:
        raise ValueError("an owned deployment state is required")
    if region is not None and region != state.get("region"):
        raise ValueError("--region must match the owned deployment state")
    if bucket is not None and bucket != state.get("bucket"):
        raise ValueError("--bucket must match the owned deployment state")
    if not isinstance(state.get("bucket"), str) or not isinstance(state.get("region"), str):
        raise ValueError("deployment state lacks a bucket or region")
    root = surfaces.resolve()
    clean = DEV_SURFACES | TEST_SURFACES
    expected = {name + ".jpg" for name in clean | REAL_CRACK_SURFACES}
    if not root.is_dir() or {path.name for path in root.glob("*.jpg")} != expected:
        raise ValueError("source directory must contain exactly the 14 clean and three real-crack JPEG names")
    prefix = state["prefix"] + "surfaces/"
    uploads, catalog = [], []
    for filename in sorted(expected):
        path = root / filename
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("source photos must be regular files inside --surfaces")
        size = path.stat().st_size
        with path.open("rb") as stream:
            magic = stream.read(2)
        if magic != b"\xff\xd8" or not 0 < size <= MAX_SURFACE_BYTES:
            raise ValueError("source photos must be nonempty JPEG files no larger than 32 MiB")
        digest = _sha256(path)
        key = prefix + filename
        uploads.append({"local": str(path), "key": key, "sha256": digest, "size_bytes": size})
        catalog.append({"name": path.stem, "key": key, "sha256": digest,
                        "size_bytes": size, "clean": path.stem in clean})
    return {"dry_run": not execute, "bucket": state["bucket"], "region": state["region"],
            "prefix": state["prefix"], "uploads": uploads,
            "catalog_key": prefix + "manifest.json", "catalog": {"version": 1, "surfaces": catalog},
            "n_clean": len(clean), "n_demo_only": len(REAL_CRACK_SURFACES), "images_decoded": 0,
            "existing_key_policy": "Reuse identical SHA256/size; reject differing content without overwrite."}


def get_client(region):
    import boto3
    return boto3.session.Session(region_name=region).client("s3")


def _code(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code")


def _head(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        if _code(error) in ("NoSuchKey", "404", "NotFound") or isinstance(error, FileNotFoundError):
            return None
        raise


def execute_plan(plan: dict, *, s3=None) -> dict:
    if plan.get("dry_run", True):
        raise ValueError("a dry-run plan cannot upload source photos")
    s3 = get_client(plan["region"]) if s3 is None else s3
    uploaded = reused = 0
    for record in plan["uploads"]:
        path = Path(record["local"])
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError("a source photo changed after planning")
        existing = _head(s3, plan["bucket"], record["key"])
        if existing is not None:
            if (existing.get("ContentLength") != record["size_bytes"]
                    or existing.get("Metadata", {}).get("sha256") != record["sha256"]):
                raise ValueError("an existing source key has different or unverified content")
            reused += 1
            continue
        try:
            with path.open("rb") as stream:
                s3.put_object(Bucket=plan["bucket"], Key=record["key"], Body=stream, IfNoneMatch="*",
                              ContentType="image/jpeg", Metadata={"sha256": record["sha256"]})
            uploaded += 1
        except Exception as error:
            if _code(error) not in ("PreconditionFailed", "412"):
                raise
            current = _head(s3, plan["bucket"], record["key"])
            if current is None or current.get("ContentLength") != record["size_bytes"] or current.get("Metadata", {}).get("sha256") != record["sha256"]:
                raise ValueError("a concurrent source upload has different contents") from error
            reused += 1
    encoded = json.dumps(plan["catalog"], indent=2, allow_nan=False).encode()
    try:
        s3.put_object(Bucket=plan["bucket"], Key=plan["catalog_key"], Body=encoded,
                      ContentType="application/json", IfNoneMatch="*")
    except Exception as error:
        if _code(error) not in ("PreconditionFailed", "412"):
            raise
        response = s3.get_object(Bucket=plan["bucket"], Key=plan["catalog_key"])
        body = response["Body"]
        try:
            data = body.read(256 * 1024 + 1)
        finally:
            if hasattr(body, "close"):
                body.close()
        if len(data) > 256 * 1024 or json.loads(data) != plan["catalog"]:
            raise ValueError("the immutable source catalog has different contents") from error
    return {"status": "DONE", "uploaded": uploaded, "reused": reused,
            "catalog_key": plan["catalog_key"], "images_decoded": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, default=APP_ROOT.parent / "data/polyhaven_4k")
    parser.add_argument("--state", type=Path, default=APP_ROOT / "aws/state.json")
    parser.add_argument("--region")
    parser.add_argument("--bucket")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.surfaces, args.state, region=args.region, bucket=args.bucket, execute=args.execute)
        result = execute_plan(plan) if args.execute else plan
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": {"code": "surface_upload_failed", "type": type(error).__name__}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
