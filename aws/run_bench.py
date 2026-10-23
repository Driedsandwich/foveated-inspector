"""Plan or submit bounded Lambda benchmark shards; never fit a forest on this Mac.

Default mode is a credential-free dry run. --execute is required for uploads,
Lambda invocation, polling or downloads. --merge combines downloaded metadata
(or capped training arrays); fit_samples always runs in the bench Lambda.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aws.handlers.bench import MAX_IMAGES_PER_SHARD, _download, safe_key, safe_prefix, safe_relative, validate_event

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = APP_ROOT / "aws/state.json"


def generation_config(args) -> tuple[dict, int]:
    from inspector.synth import DEV_SURFACES, TEST_SURFACES
    names = sorted(getattr(args, "surface_names", None) or DEV_SURFACES | TEST_SURFACES)
    if len(names) != len(set(names)) or any(name not in DEV_SURFACES | TEST_SURFACES for name in names):
        raise ValueError("generation requires a unique clean surface selection")
    per_surface = getattr(args, "per_surface", 4)
    train_per_surface = getattr(args, "train_per_surface", 4)
    train_only = bool(getattr(args, "train_only", False))
    if type(per_surface) is not int or not 1 <= per_surface <= 64 or type(train_per_surface) is not int or not 0 <= train_per_surface <= 64:
        raise ValueError("generation counts are outside supported bounds")
    if type(args.seed) is not int or not 0 <= args.seed < 2 ** 32:
        raise ValueError("generation seed must be uint32")
    eval_total = 0 if train_only else len(names) * per_surface
    train_total = len(set(names) & DEV_SURFACES) * train_per_surface
    if eval_total + train_total == 0:
        raise ValueError("generation selection contains no examples")
    config = {"per_surface": per_surface, "train_per_surface": train_per_surface,
              "seed": args.seed, "size": args.size, "version": getattr(args, "version", "v2"),
              "surface_names": names, "train_only": train_only}
    count = 1
    while max((len(range(i, eval_total, count)) + len(range(i, train_total, count))
               for i in range(count)), default=0) > MAX_IMAGES_PER_SHARD:
        count += 1
    return config, count


def _shard_outputs(prefix: str, job_id: str, count: int) -> list[dict]:
    return [{"shard_index": index, "output_prefix": f"{prefix}jobs/{job_id}/shards/{index:04d}/",
             "done_key": f"{prefix}jobs/{job_id}/shards/{index:04d}/DONE.json",
             "failed_key": f"{prefix}jobs/{job_id}/shards/{index:04d}/FAILED.json"}
            for index in range(count)]


def _output(path: Path) -> Path:
    result = path.resolve()
    if not result.is_relative_to(APP_ROOT):
        raise ValueError("output must stay inside app/")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_manifest(manifest_path: Path, data_root: Path, dataset_prefix: str, *,
                   fit_only: bool = False, holdout_surface: str | None = None) -> tuple[dict, list[dict]]:
    """Normalize ../ references within an explicit root; no image decoding occurs."""
    root = data_root.resolve()
    original = json.loads(manifest_path.read_text())
    manifest = {**original, "s3_prefix": safe_prefix(dataset_prefix), "images": []}
    uploads = {}
    if not isinstance(original.get("images"), list) or not 1 <= len(original["images"]) <= 4096:
        raise ValueError("invalid benchmark manifest")
    for source_entry in original["images"]:
        entry = dict(source_entry)
        for field in ("image", "labels"):
            source = (manifest_path.parent / source_entry[field]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("manifest input must stay inside --data-root")
            name = safe_relative(source.relative_to(root).as_posix())
            entry[field] = name
            uploads[name] = {"local": str(source), "key": dataset_prefix + name, "name": name}
        manifest["images"].append(entry)
    if len({entry["image"] for entry in manifest["images"]}) != len(manifest["images"]):
        raise ValueError("duplicate image identity after normalization")
    if fit_only:
        from scripts.train_detector import planned_entries
        planned, _ = planned_entries(manifest, holdout_surface)
        wanted = {entry[field] for entry in planned if entry["role"] == "fit" for field in ("image", "labels")}
        uploads = {name: upload for name, upload in uploads.items() if name in wanted}
    if any(not Path(upload["local"]).is_file() for upload in uploads.values()):
        raise ValueError("selected manifest inputs must exist inside --data-root")
    return manifest, sorted(uploads.values(), key=lambda item: item["name"])


def model_uploads(model_dir: Path, model_prefix: str) -> tuple[list[dict], dict]:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    names = {"metadata.json"}
    allowed = {"metadata.json"} | {f"scale_{scale}_{kind}.yml.gz"
        for scale in ("0.25", "0.5", "1") for kind in ("binary", "kind")}
    for files in metadata.get("models", {}).values():
        names.update(files.values())
    if names != allowed:
        raise ValueError("model metadata must reference the six supported forest files")
    uploads, files = [], []
    for name in sorted(names):
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir.resolve()) or not path.is_file():
            raise ValueError("missing model file or path outside model directory")
        uploads.append({"local": str(path), "key": model_prefix + name, "name": name})
        files.append({"name": name, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    if any(not 0 < item["size_bytes"] <= 8 * 1024 * 1024 for item in files) or sum(item["size_bytes"] for item in files) > 32 * 1024 * 1024:
        raise ValueError("model files exceed the runtime bundle size limits")
    return uploads, {"files": files}


def selected_entries(manifest: dict, job: str, split: str, limit: int | None,
                     per_surface_limit: int | None) -> list[dict]:
    if job in ("train_samples", "hard_negatives"):
        entries = sorted((entry for entry in manifest["images"] if entry["split"] == "train"), key=lambda entry: entry["image"])
        from inspector.synth import DEV_SURFACES
        if any(entry.get("surface") not in DEV_SURFACES for entry in entries):
            raise ValueError("training jobs require development surfaces only")
    else:
        entries = [entry for entry in manifest["images"] if entry["split"] == ("dev" if job == "tune" else split)]
        if per_surface_limit is not None:
            counts, filtered = {}, []
            for entry in entries:
                surface = entry["surface"]
                if counts.get(surface, 0) < per_surface_limit:
                    filtered.append(entry)
                    counts[surface] = counts.get(surface, 0) + 1
            entries = filtered
        if limit is not None:
            entries = entries[:limit]
    if not entries:
        raise ValueError("no images match the requested job split")
    return entries


def make_plan(args) -> dict:
    """Build the exact SDK operations with no SDK import or credential access."""
    state = json.loads(args.state.read_text()) if args.state.is_file() else {}
    bucket = args.bucket or state.get("bucket")
    region = args.region or state.get("region", "us-east-1")
    if args.bucket and state.get("bucket") and args.bucket != state["bucket"]:
        raise ValueError("--bucket must match the deployed state")
    if args.region and state.get("region") and args.region != state["region"]:
        raise ValueError("--region must match the deployed state")
    prefix = safe_prefix(state.get("prefix", "x132/"))
    if not isinstance(bucket, str) or not bucket or "/" in bucket:
        raise ValueError("provide a deployment state or --bucket")
    functions = state.get("functions", {})
    job_id = args.job_id or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_id):
        raise ValueError("invalid job_id")
    root = f"{prefix}jobs/{job_id}/"
    if args.shards is not None and not 1 <= args.shards <= 1024:
        raise ValueError("--shards must lie in [1,1024]")
    plan = {"dry_run": not args.execute, "job": args.job, "job_id": job_id,
            "bucket": bucket, "region": region, "prefix": prefix,
            "uploads": [], "json_uploads": [], "invocations": [],
            "job_guard": {"key": root + "JOB.json", "IfNoneMatch": "*",
                          "reuse_policy": "Existing job IDs may only be collected; submissions require a fresh ID."},
            "poll": {"interval_seconds": args.poll_interval, "timeout_seconds": args.timeout}}
    model_prefix = state.get("model_prefix")
    if args.model_dir is not None:
        uploads, model_manifest = model_uploads(args.model_dir, prefix + "models/")
        bundle_sha = hashlib.sha256(json.dumps(model_manifest["files"], sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        model_prefix = f"{prefix}models/{bundle_sha}/"
        for upload in uploads:
            upload["key"] = model_prefix + upload["name"]
        plan["uploads"].extend(uploads)
        plan["json_uploads"].append({"key": model_prefix + "manifest.json", "body": model_manifest})
        plan["model_bundle_sha256"] = bundle_sha
    if model_prefix is not None:
        if not re.fullmatch(re.escape(prefix) + r"models/[a-f0-9]{64}/", model_prefix):
            if (args.job in ("eval", "tune", "arm-benchmark", "hard_negatives") or getattr(args, "hard_negatives", False)) and "learned" in args.detectors and not args.collect_only:
                raise ValueError("learned jobs require an immutable model bundle prefix")
            model_prefix = None
    if args.job in ("eval", "tune", "arm-benchmark") and "learned" in args.detectors and not args.collect_only and model_prefix is None:
        raise ValueError("learned jobs require --model-dir or a pinned deployment state model_prefix")
    if (args.job == "hard_negatives" or getattr(args, "hard_negatives", False)) and not args.collect_only and model_prefix is None:
        raise ValueError("hard-negative extraction/refit requires a pinned first-fit model")
    plan["model_prefix"] = model_prefix
    if args.job == "generate":
        config, recommended = generation_config(args)
        count = args.shards or recommended
        function = functions.get("bench")
        if not function:
            raise ValueError("deployment state is missing the bench function")
        outputs = _shard_outputs(prefix, job_id, count)
        for item in outputs:
            event = {"job": "generate", "job_id": job_id, "shard_index": item["shard_index"],
                     "shard_count": count, "benchmark": config}
            plan["invocations"].append({"FunctionName": function, "InvocationType": "Event", "event": event})
            item["request_sha256"] = hashlib.sha256(json.dumps(validate_event(event, prefix), sort_keys=True, allow_nan=False).encode()).hexdigest()
        plan.update(benchmark=config, benchmark_prefix=f"{prefix}benchmarks/{job_id}/", shard_count=count,
                    outputs=outputs, collect_only=args.collect_only, merge=True,
                    collection="Download and verify manifests only; images remain in S3.",
                    runtime_note="Explicit shard sizes, including one shard, are allowed; 900-second completion has not been measured.")
        if args.collect_only:
            plan["recover_submission_from"] = plan["job_guard"]["key"]
        return plan
    if args.job == "arm-benchmark":
        if args.collect_only:
            raise ValueError("arm-benchmark has no asynchronous collection mode")
        if not args.image_ref or not args.image_ref.startswith(f"s3://{bucket}/"):
            raise ValueError("arm-benchmark requires --image-ref in the configured S3 bucket")
        safe_key(args.image_ref[len(f"s3://{bucket}/"):], prefix)
        tiles = args.tile or [args.rect]
        if len(tiles) > 16 or any(min(tile[:2]) < 0 or min(tile[2:]) <= 0 for tile in tiles):
            raise ValueError("provide at most 16 valid fixed comparison tiles")
        plan.update(image_ref=args.image_ref, rect=args.rect, tiles=tiles, detector=args.detectors[0], iterations=3,
                    architectures={key: functions.get(key) for key in ("tool_arm64", "tool_x86")})
        if any(not value for value in plan["architectures"].values()):
            raise ValueError("deployment state is missing architecture tool functions")
        plan["operations"] = [{"architecture": architecture, "FunctionName": name,
            "coarse_scan": {"tool": "coarse_scan", "image_ref": args.image_ref, "detector": args.detectors[0]},
            "inspect_tile": {"tool": "inspect_tile", "image_ref": args.image_ref, "rect": args.rect,
                             "scale": 1.0, "detector": args.detectors[0], "baseline_from": "initial_coarse_scan.result.baseline"},
            "iterations": 3, "LogType": "Tail"} for architecture, name in plan["architectures"].items()]
        for operation in plan["operations"]:
            operation["fixed_tiles"] = tiles
        for operation in plan["operations"]:
            if model_prefix is not None:
                operation["coarse_scan"]["model_prefix"] = model_prefix
                operation["inspect_tile"]["model_prefix"] = model_prefix
        return plan
    function = functions.get("bench")
    if not isinstance(function, str) or not function:
        raise ValueError("deployment state is missing the bench function")
    manifest_key = root + "manifest.json"
    manifest = None
    resident = getattr(args, "benchmark_s3", None)
    if resident is not None:
        resident = safe_prefix(resident.rstrip("/") + "/")
        if not re.fullmatch(re.escape(prefix) + r"benchmarks/[A-Za-z0-9][A-Za-z0-9_-]{0,63}/", resident):
            raise ValueError("--benchmark-s3 must name an application benchmark root")
        if args.job not in ("train_samples", "hard_negatives", "eval", "tune") or args.manifest is not None or args.data_root is not None:
            raise ValueError("resident benchmarks cannot be combined with local staging or array jobs")
        if args.shards is None and not args.collect_only:
            raise ValueError("resident image jobs require explicit --shards; dry run never reads S3")
        manifest_key = resident + ("train/manifest.json" if args.job in ("train_samples", "hard_negatives") else "manifest.json")
        count = args.shards or 1
        plan.update(benchmark_s3=resident, resident_manifest_key=manifest_key,
                    reads=[{"key": manifest_key, "stage": "execute_only", "purpose": "Verify selection and per-shard bounds; never download image bodies."}],
                    selection={"split": "train" if args.job in ("train_samples", "hard_negatives") else "dev" if args.job == "tune" else args.split,
                               "limit": args.limit, "per_surface_limit": args.per_surface_limit,
                               "selected_image_count": "unknown until execution", "explicit_shards": count})
    elif args.manifest is not None:
        if args.data_root is None:
            raise ValueError("--manifest requires --data-root")
        manifest, uploads = stage_manifest(args.manifest, args.data_root, f"{prefix}datasets/{job_id}/",
                                          fit_only=args.job == "hard_negatives",
                                          holdout_surface=args.holdout_surface)
        selected = selected_entries(manifest, args.job, args.split, args.limit, args.per_surface_limit)
        wanted = {entry[field] for entry in selected for field in ("image", "labels")}
        plan["uploads"].extend(item for item in uploads if item["name"] in wanted)
        plan["json_uploads"].append({"key": manifest_key, "body": manifest})
        count = args.shards or max(1, math.ceil(len(selected) / MAX_IMAGES_PER_SHARD))
        if math.ceil(len(selected) / count) > MAX_IMAGES_PER_SHARD:
            raise ValueError("increase --shards: at most four images per Lambda shard")
    else:
        count = args.shards or 1
        if args.job in ("train_samples", "hard_negatives", "eval", "tune") and not args.collect_only:
            raise ValueError("image jobs require --manifest and --data-root for explicit staging")
    if not 1 <= count <= 1024:
        raise ValueError("--shards must lie in [1,1024]")
    if args.job in ("fit_samples", "merge_samples") and not args.collect_only:
        if count != 1 or not (args.sample_manifests or args.samples_key):
            raise ValueError("array jobs require one shard and --sample-manifests or --samples-key")
        for key in args.sample_manifests or [args.samples_key]:
            safe_key(key, prefix)
    outputs = []
    for index in range(count):
        output_prefix = f"{root}shards/{index:04d}/"
        event = {"job": args.job, "job_id": job_id, "shard_index": index, "shard_count": count,
                 "save_traces": bool(getattr(args, "save_traces", False)),
                 "output_prefix": output_prefix, "manifest_key": manifest_key, "split": args.split,
                 "size": args.size, "seed": args.seed, "detectors": args.detectors,
                 "methods": args.methods, "budgets": args.budgets, "random_seeds": [0, 1, 2],
                 "holdout_surface": args.holdout_surface, "max_image_patches": args.max_image_patches,
                 "calibration": args.calibration, "calibration_cap_frac": .02}
        if resident is not None:
            event.pop("manifest_key")
            event["benchmark_s3"] = resident
        if model_prefix is not None:
            event["model_prefix"] = model_prefix
        if args.limit is not None:
            event["limit"] = args.limit
        if args.per_surface_limit is not None:
            event["per_surface_limit"] = args.per_surface_limit
        if args.config:
            event["config"] = json.loads(args.config.read_text())
        if args.sample_manifests:
            event["sample_manifests"] = args.sample_manifests
        if args.samples_key:
            event["samples_key"] = args.samples_key
        if args.job == "hard_negatives":
            event["hard_negative_fraction"] = getattr(args, "hard_negative_fraction", .04)
        if getattr(args, "hard_negatives", False):
            event["hard_negatives"] = True
            event["hard_negative_manifests"] = getattr(args, "hard_negative_manifests", None)
        payload = json.dumps(event, allow_nan=False)
        if len(payload.encode()) > 128 * 1024:
            raise ValueError("job event exceeds the bounded asynchronous payload size")
        plan["invocations"].append({"FunctionName": function, "InvocationType": "Event", "event": event})
        outputs.append({"shard_index": index, "output_prefix": output_prefix,
                        "done_key": output_prefix + "DONE.json", "failed_key": output_prefix + "FAILED.json",
                        "request_sha256": hashlib.sha256(json.dumps(validate_event(event, prefix), sort_keys=True, allow_nan=False).encode()).hexdigest()})
    plan.update(shard_count=count, outputs=outputs, collect_only=args.collect_only, merge=args.merge)
    return plan


def get_clients(region: str):
    import boto3
    session = boto3.session.Session(region_name=region)
    return session.client("s3"), session.client("lambda")


def _read_marker(s3, bucket: str, key: str):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound") or isinstance(error, FileNotFoundError):
            return None
        raise
    body = response["Body"]
    try:
        data = body.read(16 * 1024 * 1024 + 1)
    finally:
        if hasattr(body, "close"):
            body.close()
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("terminal marker too large")
    return json.loads(data)


def poll_outputs(s3, plan: dict, *, sleep=time.sleep, clock=time.monotonic) -> list[dict]:
    deadline = clock() + plan["poll"]["timeout_seconds"]
    pending = {item["shard_index"]: item for item in plan["outputs"]}
    completed = {}
    while pending:
        for index, item in list(pending.items()):
            done = _read_marker(s3, plan["bucket"], item["done_key"])
            if done is not None:
                if (done.get("status") != "DONE" or done.get("job_id") != plan["job_id"]
                        or done.get("shard_index") != index or done.get("shard_count") != plan["shard_count"]):
                    raise ValueError("terminal marker identity does not match the planned shard")
                if (not plan.get("collect_only", False) and item.get("request_sha256") is not None
                        and done.get("request_sha256") != item["request_sha256"]):
                    raise ValueError("terminal marker belongs to a different request")
                completed[index] = done
                del pending[index]
            else:
                failed = _read_marker(s3, plan["bucket"], item["failed_key"])
                if failed is not None:
                    raise RuntimeError(f"remote shard {index} failed; inspect its FAILED.json")
        if pending:
            if clock() >= deadline:
                raise TimeoutError("remote jobs remain pending; no zero-result inference was made")
            sleep(min(plan["poll"]["interval_seconds"], max(0, deadline - clock())))
    return [completed[index] for index in sorted(completed)]


def collect_outputs(s3, plan: dict, markers: list[dict], output: Path) -> list[Path]:
    output = _output(output)
    if len(markers) != len(plan["outputs"]):
        raise ValueError("all planned terminal markers are required before collection")
    folders = []
    for item, marker in zip(plan["outputs"], markers):
        folder = output / f"shard_{item['shard_index']:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        for record in marker.get("files", []):
            name = safe_relative(record["name"])
            if record["key"] != item["output_prefix"] + name or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
                raise ValueError("remote output file identity is invalid")
            target = folder / name
            _download(s3, plan["bucket"], record["key"], target)
            if target.stat().st_size != record["size_bytes"] or _sha256(target) != record["sha256"]:
                raise ValueError("downloaded result disagrees with its terminal digest")
        (folder / "DONE.json").write_text(json.dumps(marker, indent=2) + "\n")
        folders.append(folder)
    return folders


def recover_generation_plan(s3, plan: dict) -> None:
    guard = _read_marker(s3, plan["bucket"], plan["job_guard"]["key"])
    if not isinstance(guard, dict) or guard.get("job") != "generate" or guard.get("job_id") != plan["job_id"]:
        raise ValueError("the generation job guard is missing or has a different identity")
    submission = guard.get("submission", {})
    count = submission.get("shard_count")
    if type(count) is not int or not 1 <= count <= 1024 or not isinstance(submission.get("benchmark"), dict):
        raise ValueError("the job guard cannot recover generation parameters")
    invocations = submission.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != count:
        raise ValueError("the job guard has incomplete invocation provenance")
    for index, invocation in enumerate(invocations):
        event = validate_event(invocation["event"], plan["prefix"])
        if (event["job"] != "generate" or event["job_id"] != plan["job_id"]
                or event["shard_index"] != index or event["shard_count"] != count
                or event["benchmark"] != submission["benchmark"]):
            raise ValueError("the job guard has inconsistent generation provenance")
    plan.update(shard_count=count, benchmark=submission["benchmark"], invocations=invocations,
                benchmark_prefix=f"{plan['prefix']}benchmarks/{plan['job_id']}/",
                outputs=_shard_outputs(plan["prefix"], plan["job_id"], count),
                recovered_from_job_guard=True)


def validate_resident_selection(s3, plan: dict) -> None:
    manifest = _read_marker(s3, plan["bucket"], plan["resident_manifest_key"])
    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list):
        raise ValueError("the resident benchmark manifest is missing or invalid")
    expected_prefix = plan["benchmark_s3"] + ("train/" if plan["job"] in ("train_samples", "hard_negatives") else "")
    if manifest.get("s3_prefix") != expected_prefix or len(manifest["images"]) > 4096:
        raise ValueError("the resident manifest is not bound to its benchmark namespace")
    for entry in manifest["images"]:
        for field in ("image", "labels"):
            safe_relative(entry[field])
    event = plan["invocations"][0]["event"]
    selected = selected_entries(manifest, plan["job"], event["split"], event.get("limit"), event.get("per_surface_limit"))
    if len({e["image"] for e in selected}) != len(selected) or math.ceil(len(selected) / plan["shard_count"]) > MAX_IMAGES_PER_SHARD:
        raise ValueError("resident selection exceeds the planned four-image shard bound")
    plan["selection"].update(selected_image_count=len(selected),
                             max_images_per_shard=math.ceil(len(selected) / plan["shard_count"]),
                             manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def _immutable_json(s3, bucket: str, key: str, value: dict) -> None:
    existing = _read_marker(s3, bucket, key)
    if existing is not None:
        if existing != value:
            raise ValueError("an immutable manifest key already has different contents")
        return
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(value, indent=2, allow_nan=False).encode(),
                      ContentType="application/json", IfNoneMatch="*")
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code not in ("PreconditionFailed", "412") or _read_marker(s3, bucket, key) != value:
            raise


def collect_generation(s3, plan: dict, markers: list[dict], output: Path) -> list[dict]:
    """Recover only verified JSON manifests; image completion comes from DONE descriptors."""
    from inspector.synth import DEV_SURFACES, merge_benchmark_shards
    output = _output(output)
    benchmark = plan["benchmark"]
    namespaces = []
    if not benchmark["train_only"]:
        namespaces.append("eval")
    if benchmark["train_per_surface"] and set(benchmark["surface_names"]) & DEV_SURFACES:
        namespaces.append("train")
    if len(markers) != plan["shard_count"]:
        raise ValueError("generation recovery needs all shard terminal markers")
    paths = {namespace: [] for namespace in namespaces}
    for index, marker in enumerate(markers):
        descriptors = {}
        for record in marker.get("files", []):
            key = safe_key(record["key"], plan["benchmark_prefix"])
            if key in descriptors or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) or type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
                raise ValueError("invalid or duplicate generation output descriptor")
            descriptors[key] = record
        for namespace in namespaces:
            namespace_prefix = plan["benchmark_prefix"] + ("train/" if namespace == "train" else "")
            shard_prefix = namespace_prefix + (f"shard_{index:04d}_of_{plan['shard_count']:04d}/" if plan["shard_count"] > 1 else "")
            key = shard_prefix + "manifest.json"
            record = descriptors.get(key)
            if record is None or record["size_bytes"] > 16 * 1024 * 1024:
                raise ValueError("a completed generation shard is missing its manifest")
            relative = safe_relative(key[len(plan["benchmark_prefix"]):])
            target = (output / relative).resolve()
            if not target.is_relative_to(output):
                raise ValueError("local manifest mirror escapes --out")
            _download(s3, plan["bucket"], key, target)
            if target.stat().st_size != record["size_bytes"] or _sha256(target) != record["sha256"]:
                raise ValueError("generation manifest digest disagrees with DONE")
            manifest = json.loads(target.read_text())
            if manifest.get("s3_prefix") != namespace_prefix:
                raise ValueError("generation manifest has a different S3 namespace")
            for entry in manifest["images"]:
                for field in ("image", "labels"):
                    relative_image = safe_relative(entry[field])
                    descriptor = descriptors.get(shard_prefix + relative_image)
                    if descriptor is None:
                        raise ValueError("a manifest image/label lacks completion evidence in DONE")
                    entry_hash = entry.get(field + "_sha256")
                    if entry_hash is None and isinstance(entry.get("sha256"), dict):
                        entry_hash = entry["sha256"].get(field)
                    if entry_hash is not None and entry_hash != descriptor["sha256"]:
                        raise ValueError("an image/label digest disagrees with its manifest")
            paths[namespace].append(target)
        marker_path = output / "_markers" / f"shard_{index:04d}_DONE.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker, indent=2, allow_nan=False))
    merged = []
    for namespace in namespaces:
        destination = output / ("train" if namespace == "train" else "")
        namespace_prefix = plan["benchmark_prefix"] + ("train/" if namespace == "train" else "")
        originals = [json.loads(path.read_text()) for path in paths[namespace]]
        with tempfile.TemporaryDirectory(prefix="_merge_metadata_", dir=destination) as temporary:
            validated = merge_benchmark_shards(paths[namespace], temporary, verify_images=False)
            for entry in validated["images"]:
                for field in ("image", "labels"):
                    source = (Path(temporary) / entry[field]).resolve()
                    entry[field] = safe_relative(source.relative_to(destination.resolve()).as_posix())
        validated["s3_prefix"] = namespace_prefix
        if plan["shard_count"] == 1:
            # The single-shard S3 manifest already occupies the final immutable
            # key. Validate it without replacing its original provenance.
            if validated["images"] != originals[0]["images"]:
                raise ValueError("single-shard image references changed during validation")
            validated = originals[0]
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(validated, indent=2, allow_nan=False))
        _immutable_json(s3, plan["bucket"], namespace_prefix + "manifest.json", validated)
        merged.append({"namespace": namespace, "manifest": str(manifest_path),
                       "manifest_key": namespace_prefix + "manifest.json", "n_images": len(validated["images"])})
    return merged


def parse_report(log_result: str | None) -> dict:
    """Extract actual Lambda REPORT timing; missing logs remain explicitly null."""
    result = {"duration_ms": None, "billed_duration_ms": None}
    if not log_result:
        return result
    try:
        log = base64.b64decode(log_result, validate=True).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return result
    for line in reversed(log.splitlines()):
        if "REPORT RequestId:" not in line:
            continue
        duration = re.search(r"(?<!Billed )\bDuration:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", line)
        billed = re.search(r"\bBilled Duration:\s*([0-9]+)\s*ms", line)
        if duration:
            result["duration_ms"] = float(duration.group(1))
        if billed:
            result["billed_duration_ms"] = float(billed.group(1))
        break
    return result


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "std": None, "n": 0}
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower, upper = math.floor(position), math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"p50": statistics.median(values), "p95": p95, "std": statistics.pstdev(values), "n": len(values)}


def _sync_call(client, function: str, event: dict) -> tuple[dict, dict]:
    started = time.perf_counter()
    response = client.invoke(FunctionName=function, InvocationType="RequestResponse", LogType="Tail",
                             Payload=json.dumps(event, allow_nan=False).encode())
    report = parse_report(response.get("LogResult"))
    if response.get("FunctionError"):
        return {"ok": False, "error": {"code": "tool_function_error"}}, report
    stream = response["Payload"]
    try:
        payload = json.loads(stream.read())
    finally:
        if hasattr(stream, "close"):
            stream.close()
    report["round_trip_ms"] = (time.perf_counter()-started)*1000
    return payload, report


def arm_benchmark(client, plan: dict) -> dict:
    architectures = {}
    tiles = plan.get("tiles", [plan["rect"]])
    n_requested = 3 * len(tiles)
    for architecture, function in plan["architectures"].items():
        base_event = {"tool": "coarse_scan", "image_ref": plan["image_ref"], "detector": plan["detector"]}
        if plan.get("model_prefix") is not None:
            base_event["model_prefix"] = plan["model_prefix"]
        try:
            initial, _ = _sync_call(client, function, base_event)
            if not initial.get("ok"):
                raise ValueError("initial_coarse_failed")
            coarse = initial["result"]
            height, width = coarse["image_shape"][:2]
            for x, y, w, h in tiles:
                if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
                    raise ValueError("comparison_tile_outside_image")
            rows = []
            for iteration in range(3):
                for tile_index, rect in enumerate(tiles):
                    event = {**base_event, "tool": "inspect_tile", "rect": rect,
                             "scale": 1.0, "baseline": coarse["baseline"]}
                    try:
                        response, timing = _sync_call(client, function, event)
                        rows.append({"iteration": iteration, "tile_index": tile_index,
                                     "rect": rect, "ok": bool(response.get("ok")), **timing})
                    except Exception:
                        rows.append({"iteration": iteration, "tile_index": tile_index, "rect": rect,
                                     "ok": False, "duration_ms": None, "billed_duration_ms": None})
            durations = [row["duration_ms"] for row in rows if row["ok"] and row["duration_ms"] is not None]
            billed = [row["billed_duration_ms"] for row in rows if row["ok"] and row["billed_duration_ms"] is not None]
            round_trips = [row["round_trip_ms"] for row in rows if row["ok"] and row.get("round_trip_ms") is not None]
            architectures[architecture] = {"coverage": "complete" if len(durations) == len(billed) == n_requested else "partial",
                "n_requested": n_requested, "n_successful": sum(row["ok"] for row in rows), "iterations": rows,
                "duration_ms": _distribution(durations), "billed_duration_ms": _distribution(billed),
                "round_trip_ms": _distribution(round_trips)}
        except Exception:
            architectures[architecture] = {"coverage": "unavailable", "error": {"code": "initial_coarse_or_tile_validation_failed"},
                "n_requested": n_requested, "n_successful": 0, "duration_ms": _distribution([]),
                "billed_duration_ms": _distribution([]), "round_trip_ms": _distribution([])}
    return {"job": "arm-benchmark", "iterations": 3, "rect": plan["rect"], "tiles": tiles, "architectures": architectures,
            "measurement": "Warm fixed-tile Lambda REPORT and client round-trip times after each architecture's initial coarse scan.",
            "limitation": "Three repetitions per tile are a small timing sample; absent REPORT/billed fields are not estimated."}


def _upload_inputs(s3, plan: dict) -> None:
    model_prefix = plan.get("model_prefix")
    model_json = next((item for item in plan["json_uploads"]
                       if model_prefix and item["key"] == model_prefix + "manifest.json"), None)
    reuse_model = False
    if model_json is not None:
        existing = _read_marker(s3, plan["bucket"], model_json["key"])
        if existing is not None:
            if existing != model_json["body"]:
                raise ValueError("an immutable model prefix has different contents")
            reuse_model = True
    for upload in plan["uploads"]:
        if reuse_model and upload["key"].startswith(model_prefix):
            continue
        s3.upload_file(upload["local"], plan["bucket"], upload["key"])
    for upload in plan["json_uploads"]:
        if reuse_model and upload is model_json:
            continue
        try:
            s3.put_object(Bucket=plan["bucket"], Key=upload["key"],
                          Body=json.dumps(upload["body"], allow_nan=False).encode(),
                          ContentType="application/json", IfNoneMatch="*")
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if (upload is not model_json or code not in ("PreconditionFailed", "412")
                    or _read_marker(s3, plan["bucket"], upload["key"]) != upload["body"]):
                raise


def execute_plan(plan: dict, output: Path, *, s3=None, lambda_client=None) -> dict:
    if plan.get("dry_run", True):
        raise ValueError("a dry-run plan cannot perform AWS operations")
    output = _output(output)
    output.mkdir(parents=True, exist_ok=True)
    # Save the generated identity before credential lookup or any external write.
    (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    if s3 is None or (lambda_client is None and not plan.get("collect_only", False)):
        actual_s3, actual_lambda = get_clients(plan["region"])
        s3 = s3 if s3 is not None else actual_s3
        lambda_client = lambda_client if lambda_client is not None else actual_lambda
    if plan["job"] == "generate" and plan.get("collect_only", False):
        recover_generation_plan(s3, plan)
    if plan.get("benchmark_s3") and not plan.get("collect_only", False):
        validate_resident_selection(s3, plan)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    if not plan.get("collect_only", False):
        identity = {key: plan.get(key) for key in ("job", "job_id", "invocations", "json_uploads", "operations")}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        # Claim the immutable job before any source/model upload or invocation.
        guard = {"job_id": plan["job_id"], "job": plan["job"], "request_sha256": digest, "immutable": True}
        if plan["job"] == "generate":
            guard["submission"] = {"shard_count": plan["shard_count"], "benchmark": plan["benchmark"],
                                   "invocations": plan["invocations"]}
        s3.put_object(Bucket=plan["bucket"], Key=plan["job_guard"]["key"], IfNoneMatch="*",
                      Body=json.dumps(guard, allow_nan=False).encode(),
                      ContentType="application/json")
    if not plan.get("collect_only", False):
        _upload_inputs(s3, plan)
        for invocation in plan["invocations"]:
            response = lambda_client.invoke(FunctionName=invocation["FunctionName"], InvocationType="Event",
                                            Payload=json.dumps(invocation["event"], allow_nan=False).encode())
            if response.get("StatusCode") != 202:
                raise RuntimeError("Lambda did not accept an asynchronous shard")
    if plan["job"] == "arm-benchmark":
        result = arm_benchmark(lambda_client, plan)
        (output / "arm_benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    markers = poll_outputs(s3, plan)
    if plan["job"] == "generate":
        manifests = collect_generation(s3, plan, markers, output)
        result = {"status": "DONE", "job_id": plan["job_id"], "n_shards": plan["shard_count"],
                  "manifests": manifests, "images_downloaded": 0, "models_fitted_locally": False}
        (output / "collection.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    folders = collect_outputs(s3, plan, markers, output)
    merge_metadata = None
    if plan["merge"]:
        if plan["job"] in ("eval", "tune"):
            from scripts.run_eval import merge_shards
            _, merge_metadata = merge_shards(folders, output / "merged")
        elif plan["job"] == "train_samples":
            from scripts.train_detector import merge_samples
            merge_samples(folders, output / "merged_samples.npz")
    result = {"status": "DONE", "job_id": plan["job_id"], "n_shards": len(folders),
              "output": str(output), "models_fitted_locally": False}
    if plan["job"] == "tune" and merge_metadata is not None:
        result.update(policy_selection_status=merge_metadata["select_status"],
                      policy_config_available=merge_metadata["config_available"],
                      tuning_report=str(output / "merged/tuning_dev.json"),
                      tradeoff_table=str(output / "merged/dev_tradeoff.csv"))
    (output / "collection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("generate", "train_samples", "hard_negatives", "eval", "tune", "merge_samples", "fit_samples", "arm-benchmark"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--bucket")
    parser.add_argument("--region")
    parser.add_argument("--job-id")
    parser.add_argument("--shards", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--benchmark-s3", help="Resident x132/benchmarks/<job-id>/ root; images are not uploaded by this driver")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--size", type=int, choices=(1024, 2048, 4096), default=4096)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--per-surface", type=int, default=4)
    parser.add_argument("--train-per-surface", "--train-images", dest="train_per_surface", type=int, default=4)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--surface-names", nargs="+")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--detectors", nargs="+", choices=("learned", "classic"), default=["learned", "classic"])
    parser.add_argument("--methods", nargs="+", choices=("agent", "exhaustive", "random", "raster", "coarse_only"), default=["agent", "exhaustive", "random", "raster", "coarse_only"])
    parser.add_argument("--budgets", type=float, nargs="+", default=[.02, .05, .1, .2, .4])
    parser.add_argument("--calibration", choices=("sampled", "whole"), default="sampled")
    parser.add_argument("--holdout-surface", help="Legacy ablation; default holds out images from every dev surface")
    parser.add_argument("--hard-negatives", action="store_true", help="fit_samples: refit once with mined negatives and the pinned first model")
    parser.add_argument("--hard-negative-manifests", nargs="+", help="fit_samples: complete S3 hard-negative shard manifests")
    parser.add_argument("--hard-negative-fraction", type=float, default=.04, help="hard_negatives job: top .03..05 of empty background patches per image/scale")
    parser.add_argument("--max-image-patches", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-surface-limit", type=int)
    parser.add_argument("--sample-manifests", nargs="+")
    parser.add_argument("--samples-key")
    parser.add_argument("--collect-only", action="store_true", help="Poll/download an existing job, without upload/invocation")
    parser.add_argument("--merge", action="store_true", help="Merge collected metadata/capped arrays; never fit locally")
    parser.add_argument("--save-traces", action="store_true", help="Ask eval shards to upload per-run JSON traces (diagnostics)")
    parser.add_argument("--out", type=Path, default=APP_ROOT / "artifacts/remote")
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--image-ref")
    parser.add_argument("--rect", type=int, nargs=4, default=[0, 0, 256, 256])
    parser.add_argument("--tile", type=int, nargs=4, action="append",
                        help="Repeat for a fixed set of ARM/x86 comparison rectangles; overrides --rect")
    args = parser.parse_args(argv)
    plan = None
    output = None
    try:
        if not 0.1 <= args.poll_interval <= 60 or not 0 < args.timeout <= 900:
            raise ValueError("poll interval must be 0.1..60 seconds; timeout must be >0 and <=900 seconds")
        if args.limit is not None and args.limit < 1 or args.per_surface_limit is not None and args.per_surface_limit < 1:
            raise ValueError("image limits must be positive")
        if args.collect_only and not args.job_id:
            raise ValueError("--collect-only requires the original --job-id")
        if args.job == "tune" and args.split != "dev":
            raise ValueError("policy tuning is dev-only")
        if args.job in ("train_samples", "hard_negatives", "eval", "tune") and args.size == 1024:
            raise ValueError("image analysis jobs require --size 2048 or 4096; 1024 is generation-only")
        if args.hard_negatives and (args.job != "fit_samples" or not args.hard_negative_manifests):
            raise ValueError("--hard-negatives requires --job fit_samples and --hard-negative-manifests")
        if args.hard_negative_manifests and not args.hard_negatives:
            raise ValueError("--hard-negative-manifests requires --hard-negatives")
        if args.job == "generate" and any(value is not None for value in (args.manifest, args.data_root, args.benchmark_s3, args.model_dir)):
            raise ValueError("generate uses only the resident surface catalog, without local input/model uploads")
        output = _output(args.out)
        plan = make_plan(args)
        if not args.execute:
            print(json.dumps(plan, indent=2, allow_nan=False))
        else:
            result = execute_plan(plan, output)
            print(json.dumps(result, indent=2, allow_nan=False))
            if result.get("policy_selection_status") == "no_feasible_candidate":
                # Completed dev measurements are retained. A missing policy is
                # not success for an automated tune -> test/deploy pipeline.
                return 3
        return 0
    except Exception as error:
        # Do not print SDK exception strings, response payloads or credentials.
        result = {"status": "FAILED", "error": {"code": "driver_failed", "type": type(error).__name__}}
        if plan is not None:
            result["job_id"] = plan["job_id"]
        if output is not None and (output / "plan.json").is_file():
            result["recovery_plan"] = str(output / "plan.json")
        print(json.dumps(result))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
