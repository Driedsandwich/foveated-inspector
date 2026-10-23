"""Bounded Lambda benchmark jobs; no shell, arbitrary module, or command dispatch.

The job manifest is a benchmark manifest with an added s3_prefix. Its image and
label paths are root-relative, normalized by aws.run_bench before upload. Full
metadata stays intact, while only the assigned image files are downloaded.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

JOBS = frozenset({"generate", "train_samples", "hard_negatives", "eval", "tune", "merge_samples", "fit_samples"})
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_IMAGES_PER_SHARD = 4
ALLOWED_METHODS = {"agent", "exhaustive", "random", "raster", "coarse_only"}


def safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(character) < 32 for character in value)):
        raise ValueError("invalid_relative_path")
    return PurePosixPath(value).as_posix()


def safe_key(value: str, prefix: str) -> str:
    key = safe_relative(value)
    if not key.startswith(prefix):
        raise ValueError("key_outside_prefix")
    return key


def safe_prefix(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("/"):
        raise ValueError("prefix_requires_trailing_slash")
    return safe_relative(value[:-1]) + "/"


def integer(value, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("integer_out_of_bounds")
    return value


def benchmark_spec(value: dict) -> dict:
    """Normalize the small generation contract without touching image files."""
    from inspector.synth import DEV_SURFACES, TEST_SURFACES
    allowed = {"per_surface", "train_per_surface", "train_images", "seed", "size",
               "version", "surface_names", "train_only"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("unknown_benchmark_parameter")
    if "train_per_surface" in value and "train_images" in value and value["train_per_surface"] != value["train_images"]:
        raise ValueError("conflicting_training_image_counts")
    # Validate both aliases even when equal (True must never alias integer 1).
    for key in ("train_per_surface", "train_images"):
        if key in value:
            integer(value[key], 0, 64)
    clean = DEV_SURFACES | TEST_SURFACES
    names = value.get("surface_names", sorted(clean))
    if (not isinstance(names, list) or not names or len(names) > len(clean)
            or any(not isinstance(name, str) or name not in clean for name in names)
            or len(names) != len(set(names))):
        raise ValueError("surface_names_must_be_unique_clean_backgrounds")
    train_only = value.get("train_only", False)
    if type(train_only) is not bool:
        raise ValueError("train_only_requires_a_boolean")
    result = {"per_surface": integer(value.get("per_surface", 4), 1, 64),
              "train_per_surface": integer(value.get("train_per_surface", value.get("train_images", 4)), 0, 64),
              "seed": integer(value.get("seed", 20261023), 0, 2**32 - 1),
              "size": integer(value.get("size", 4096), 1024, 4096),
              "version": value.get("version", "v2"), "surface_names": sorted(names), "train_only": train_only}
    if result["size"] not in (1024, 2048, 4096) or result["version"] not in ("v1", "v2"):
        raise ValueError("invalid_benchmark_size_or_version")
    if train_only and not result["train_per_surface"]:
        raise ValueError("train_only_requires_training_examples")
    if result["train_per_surface"] and not set(names).intersection(DEV_SURFACES):
        raise ValueError("training_requires_a_selected_development_surface")
    return result


def validate_event(event: dict, prefix: str) -> dict:
    if not isinstance(event, dict) or event.get("job") not in JOBS:
        raise ValueError("unsupported_job")
    job_id = event.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_id):
        raise ValueError("invalid_job_id")
    count = integer(event.get("shard_count", 1), 1, 1024)
    index = integer(event.get("shard_index", 0), 0, count - 1)
    job_root = f"{prefix}jobs/{job_id}/"
    expected_output = f"{job_root}shards/{index:04d}/"
    if event.get("output_prefix", expected_output) != expected_output:
        raise ValueError("invalid_output_prefix")
    result = dict(event, shard_index=index, shard_count=count, output_prefix=expected_output)
    if event["job"] == "generate":
        if event.get("manifest_key") is not None or event.get("benchmark_s3") is not None:
            raise ValueError("generate_does_not_accept_an_input_benchmark")
        result["benchmark"] = benchmark_spec(event.get("benchmark", {}))
    elif event["job"] in {"train_samples", "hard_negatives", "eval", "tune"}:
        if event.get("benchmark_s3") is not None:
            if event.get("manifest_key") is not None:
                raise ValueError("benchmark_s3_and_manifest_key_are_mutually_exclusive")
            source = safe_prefix(event["benchmark_s3"])
            if not re.fullmatch(re.escape(prefix) + r"benchmarks/[A-Za-z0-9][A-Za-z0-9_-]{0,63}/", source):
                raise ValueError("benchmark_s3_requires_an_application_benchmark_root")
            result["benchmark_s3"] = source
        else:
            manifest_key = event.get("manifest_key", f"{job_root}manifest.json")
            if manifest_key != f"{job_root}manifest.json":
                raise ValueError("invalid_job_manifest_key")
            result["manifest_key"] = manifest_key
    elif count != 1:
        raise ValueError("array_merge_and_fit_require_one_shard")
    if event.get("bucket") is not None or event.get("function") is not None or event.get("command") is not None:
        raise ValueError("caller_cannot_select_bucket_function_or_command")
    if event.get("calibration", "sampled") not in ("sampled", "whole"):
        raise ValueError("invalid_calibration")
    if type(event.get("hard_negatives", False)) is not bool:
        raise ValueError("hard_negatives_requires_boolean")
    if event.get("hard_negatives", False) and event["job"] != "fit_samples":
        raise ValueError("hard_negatives_refit_option_requires_fit_samples")
    extra = event.get("hard_negative_manifests")
    if extra is not None:
        if (event["job"] != "fit_samples" or not event.get("hard_negatives", False)
                or not isinstance(extra, list) or not 1 <= len(extra) <= 1024
                or any(not isinstance(key, str) for key in extra) or len(set(extra)) != len(extra)):
            raise ValueError("invalid_hard_negative_manifests")
        for key in extra:
            if not safe_key(key, prefix).endswith(".json"):
                raise ValueError("hard_negative_manifests_require_json")
    elif event.get("hard_negatives", False):
        raise ValueError("hard_negative_refit_requires_manifests")
    if event["job"] == "hard_negatives":
        fraction = event.get("hard_negative_fraction", .04)
        if type(fraction) not in (int, float) or not .03 <= fraction <= .05:
            raise ValueError("hard_negative_fraction_requires_3_to_5_percent")
        result["hard_negative_fraction"] = float(fraction)
    for key in ("limit", "per_surface_limit"):
        if event.get(key) is not None:
            integer(event[key], 1, 4096)
    if event.get("model_prefix") is not None:
        model_prefix = safe_prefix(event["model_prefix"])
        if not re.fullmatch(re.escape(prefix) + r"models/[a-f0-9]{64}/", model_prefix):
            raise ValueError("model_prefix_requires_an_immutable_bundle")
    elif event["job"] == "hard_negatives" or event.get("hard_negatives", False):
        raise ValueError("hard_negative_jobs_require_a_pinned_first_model")
    return result


def _json_object(s3, bucket: str, key: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        payload = body.read(MAX_JSON_BYTES + 1)
    finally:
        if hasattr(body, "close"):
            body.close()
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("json_input_too_large")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def _download(s3, bucket: str, key: str, target: Path) -> None:
    response = s3.get_object(Bucket=bucket, Key=key)
    if int(response.get("ContentLength", 0)) > MAX_FILE_BYTES:
        raise ValueError("input_file_too_large")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = response["Body"]
    total = 0
    try:
        with target.open("wb") as stream:
            while chunk := body.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError("input_file_too_large")
                stream.write(chunk)
    finally:
        if hasattr(body, "close"):
            body.close()


def _upload_record(s3, bucket: str, path: Path, key: str, name: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    size = path.stat().st_size
    s3.upload_file(str(path), bucket, key)
    return {"name": name, "key": key, "size_bytes": size, "sha256": digest.hexdigest()}


def _assigned_surfaces(names: list[str], per_surface: int, index: int, count: int) -> set[str]:
    from inspector.synth import assigned_sample_indices
    return {name for surface_index, name in enumerate(names)
            if assigned_sample_indices(surface_index, per_surface, index, count)}


def _surface_catalog(s3, bucket: str, prefix: str, names: list[str]) -> tuple[list[dict], str]:
    from inspector.synth import DEV_SURFACES, TEST_SURFACES, REAL_CRACK_SURFACES
    catalog = _json_object(s3, bucket, prefix + "surfaces/manifest.json")
    records = catalog.get("surfaces")
    known = DEV_SURFACES | TEST_SURFACES | REAL_CRACK_SURFACES
    if type(catalog.get("version")) is not int or catalog["version"] != 1 or not isinstance(records, list) or not 1 <= len(records) <= len(known):
        raise ValueError("invalid_surface_catalog")
    by_name = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid_surface_record")
        name = record.get("name")
        if (not isinstance(name, str) or name not in known or name in by_name
                or record.get("key") != prefix + "surfaces/" + name + ".jpg"
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
                or type(record.get("clean")) is not bool
                or record["clean"] != (name in DEV_SURFACES | TEST_SURFACES)):
            raise ValueError("invalid_surface_record")
        integer(record.get("size_bytes"), 1, MAX_FILE_BYTES)
        by_name[name] = {key: record[key] for key in ("name", "key", "sha256", "size_bytes", "clean")}
    if any(name not in by_name or not by_name[name]["clean"] for name in names):
        raise ValueError("requested_clean_surface_is_missing_from_catalog")
    digest = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return [by_name[name] for name in sorted(names)], digest


def _generate(event: dict, s3, bucket: str, prefix: str, workspace: Path, output: Path) -> tuple[Path, list[dict]]:
    """Stream generated pairs to S3 and immediately release their local files."""
    from inspector.synth import DEV_SURFACES, make_benchmark, make_training_benchmark
    spec = event["benchmark"]
    names = spec["surface_names"]
    train_names = [name for name in names if name in DEV_SURFACES]
    index, count = event["shard_index"], event["shard_count"]
    needed = set()
    if not spec["train_only"]:
        needed.update(_assigned_surfaces(names, spec["per_surface"], index, count))
    if spec["train_per_surface"]:
        needed.update(_assigned_surfaces(train_names, spec["train_per_surface"], index, count))
    descriptors, catalog_digest = _surface_catalog(s3, bucket, prefix, names)
    by_name = {record["name"]: record for record in descriptors}
    surfaces = workspace / "surfaces"
    surfaces.mkdir()
    for name in sorted(needed):
        path = surfaces / f"{name}.jpg"
        record = by_name[name]
        _download(s3, bucket, record["key"], path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("surface_content_does_not_match_catalog")
    remote_root = f"{prefix}benchmarks/{event['job_id']}/"
    local_root = output / "benchmark"
    uploaded = []

    def publish_pair(image_path: Path, label_path: Path, entry: dict) -> None:
        entry["source_sha256"] = by_name[entry["surface"]]["sha256"]
        pair = (Path(image_path), Path(label_path))
        if any(path.is_symlink() or not path.resolve().is_relative_to(local_root.resolve()) for path in pair):
            raise ValueError("generator_callback_path_outside_output")
        try:
            for path in pair:
                name = safe_relative(path.relative_to(local_root).as_posix())
                uploaded.append(_upload_record(s3, bucket, path, remote_root + name, name))
        finally:
            # Includes partial-upload failures: never retain a backlog of pairs.
            for path in pair:
                path.unlink(missing_ok=True)

    namespaces = ([] if spec["train_only"] else [("eval", names, spec["per_surface"])])
    if spec["train_per_surface"]:
        namespaces.append(("train", train_names, spec["train_per_surface"]))
    for namespace, surface_names, per_surface in namespaces:
        destination = local_root / "train" if namespace == "train" else local_root
        # The complete ordered paths preserve CLI global indices and seeds.
        # The synth generator only requires assigned sources to exist.
        paths = [surfaces / f"{name}.jpg" for name in surface_names]
        if namespace == "train":
            manifest = make_training_benchmark(paths, destination, per_surface, spec["seed"], spec["size"],
                version=spec["version"], shard_index=index, shard_count=count, on_image_written=publish_pair)
        else:
            manifest = make_benchmark(paths, destination, per_surface, spec["seed"], size=spec["size"],
                version=spec["version"], shard_index=index, shard_count=count, on_image_written=publish_pair)
        namespace_names = set(surface_names)
        namespace_sources = [record for record in descriptors if record["name"] in namespace_names]
        manifest.update(source_catalog_sha256=catalog_digest, source_surfaces=namespace_sources,
                        s3_prefix=remote_root + ("train/" if namespace == "train" else ""))
        for entry in manifest["images"]:
            entry["source_sha256"] = by_name[entry["surface"]]["sha256"]
        directory = destination / f"shard_{index:04d}_of_{count:04d}" if count > 1 else destination
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        name = safe_relative(manifest_path.relative_to(local_root).as_posix())
        try:
            uploaded.append(_upload_record(s3, bucket, manifest_path, remote_root + name, name))
        finally:
            manifest_path.unlink(missing_ok=True)
    return output, uploaded


def _stage_dataset(s3, bucket: str, prefix: str, manifest: dict,
                   entries: list[dict], target: Path) -> None:
    source_prefix = safe_prefix(manifest.get("s3_prefix", ""))
    safe_key(source_prefix + "manifest.json", prefix)
    if len(entries) > MAX_IMAGES_PER_SHARD:
        raise ValueError("too_many_images_for_one_shard")
    if not isinstance(manifest.get("images"), list) or len(manifest["images"]) > 4096:
        raise ValueError("invalid_image_manifest")
    for entry in manifest["images"]:
        for field in ("image", "labels"):
            safe_relative(entry[field])
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    for entry in entries:
        for field in ("image", "labels"):
            relative = safe_relative(entry[field])
            _download(s3, bucket, source_prefix + relative, target / relative)


def _models(s3, bucket: str, prefix: str, workspace: Path, model_prefix: str | None = None) -> Path:
    from aws.runtime import materialize_model_dir
    return materialize_model_dir(bucket=bucket, prefix=prefix, s3_client=s3, temp_root=workspace,
                                 model_prefix=model_prefix)


def _sample_inputs(event: dict, s3, bucket: str, prefix: str, root: Path) -> list[Path] | Path:
    keys = event.get("sample_manifests")
    if keys is not None:
        if not isinstance(keys, list) or not 1 <= len(keys) <= 1024 or len(keys) != len(set(keys)):
            raise ValueError("invalid_sample_manifest_list")
        paths = []
        for index, value in enumerate(keys):
            key = safe_key(value, prefix)
            manifest = _json_object(s3, bucket, key)
            folder = root / f"shard_{index:04d}"
            folder.mkdir(parents=True)
            path = folder / PurePosixPath(key).name
            path.write_text(json.dumps(manifest, allow_nan=False))
            for record in manifest.get("records", []):
                name = safe_relative(record["file"])
                if "/" in name or not name.endswith(".npz"):
                    raise ValueError("invalid_sample_filename")
                _download(s3, bucket, str(PurePosixPath(key).parent / name), folder / name)
            paths.append(path)
        return paths
    key = safe_key(event.get("samples_key", ""), prefix)
    if not key.endswith(".npz"):
        raise ValueError("merged_samples_key_requires_npz")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "merged_samples.npz"
    _download(s3, bucket, key, path)
    _download(s3, bucket, key[:-4] + ".json", path.with_suffix(".json"))
    return path


def _evaluation(event: dict, manifest: dict, backend_dir: Path | None, dataset: Path,
                bucket: str) -> tuple[dict, list[dict]]:
    from dataclasses import asdict
    from inspector.agent import InspectorConfig, REPORTING_VERSION
    from inspector.evaluate import MATCHING_VERSION, METRICS_VERSION
    from scripts.run_eval import json_fingerprint, load_manifest, model_fingerprint, shard_entries, tuning_candidates
    split = event.get("split", "dev")
    if split not in ("dev", "test") or event["job"] == "tune" and split != "dev":
        raise ValueError("invalid_evaluation_split")
    entries = load_manifest(dataset, split, event.get("limit"), event.get("per_surface_limit"))
    assigned = shard_entries(entries, event["shard_index"], event["shard_count"])
    detectors = event.get("detectors", ["learned", "classic"])
    methods = event.get("methods", ["agent", "exhaustive", "random", "raster", "coarse_only"])
    budgets = event.get("budgets", [.02, .05, .1, .2, .4])
    seeds = event.get("random_seeds", [0, 1, 2])
    if (not isinstance(detectors, list) or not detectors or any(d not in ("classic", "learned") for d in detectors)
            or len(detectors) != len(set(detectors)) or not isinstance(methods, list) or not methods
            or any(method not in ALLOWED_METHODS for method in methods) or len(methods) != len(set(methods))):
        raise ValueError("invalid_detector_or_method_list")
    if not isinstance(budgets, list) or not 1 <= len(budgets) <= 8 or len(budgets) != len(set(budgets)):
        raise ValueError("invalid_budget_list")
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 <= budget <= 1:
            raise ValueError("invalid_budget")
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(seeds) != len(set(seeds)):
        raise ValueError("invalid_random_seeds")
    for seed in seeds:
        integer(seed, 0, 2**32 - 1)
    config = InspectorConfig(**event.get("config", {}))
    if any(value > 32 for value in config.grid) or config.work_width > 1024:
        raise ValueError("policy_grid_or_overview_exceeds_remote_limit")
    calibration = event.get("calibration", "sampled")
    cap = event.get("calibration_cap_frac", .02)
    if calibration not in ("sampled", "whole") or not isinstance(cap, (int, float)) or not 0 < cap <= .02:
        raise ValueError("invalid_calibration")
    evaluation = {"bench": f"s3://{bucket}/" + manifest["s3_prefix"], "manifest_sha256": json_fingerprint(manifest),
                  "reporting_version": REPORTING_VERSION,
                  "matching_version": MATCHING_VERSION,
                  "metrics_version": METRICS_VERSION,
                  "mode": "evaluation", "split": split, "size": event.get("size", 4096),
                  "images": [entry["image"] for entry in entries], "budgets": budgets,
                  "methods": methods, "detectors": detectors, "seed": integer(event.get("seed", 0), 0, 2**32 - 1),
                  "random_seeds": seeds, "limit": event.get("limit"), "per_surface_limit": event.get("per_surface_limit"),
                  "config": asdict(config), "model_files": model_fingerprint(backend_dir) if backend_dir is not None and "learned" in detectors else None,
                  "calibration": calibration, "calibration_cap_frac": cap,
                  "resized_labels": "nearest neighbour; vanished defects retained in denominator"}
    if evaluation["size"] not in (2048, 4096):
        raise ValueError("invalid_analysis_size")
    if event["job"] == "tune":
        detector = "learned" if "learned" in detectors else "classic"
        candidates = tuning_candidates(config, detector)
        evaluation.update(mode="tuning", detectors=[detector], candidates=candidates,
                          methods=[candidate["method"] for candidate in candidates], budgets=[max(budgets)])
    return evaluation, assigned


def _execute(event: dict, s3, bucket: str, prefix: str, workspace: Path) -> Path | tuple[Path, list[dict]]:
    output = workspace / "output"
    output.mkdir()
    job = event["job"]
    if job == "generate":
        return _generate(event, s3, bucket, prefix, workspace, output)
    if job in ("merge_samples", "fit_samples"):
        from scripts.train_detector import fit_samples, merge_samples
        source = _sample_inputs(event, s3, bucket, prefix, workspace / "sample_inputs")
        if job == "merge_samples":
            if not isinstance(source, list):
                raise ValueError("merge_samples_requires_shard_manifests")
            merge_samples(source, output / "merged_samples.npz")
        else:
            supplement = {}
            if event.get("hard_negatives", False):
                extra = _sample_inputs({"sample_manifests": event["hard_negative_manifests"]}, s3, bucket,
                                       prefix, workspace / "hard_negative_inputs")
                supplement = {"hard_negative_manifests": extra,
                              "first_model_dir": _models(s3, bucket, prefix, workspace, event.get("model_prefix"))}
            fit_samples(source, output / "models", seed=integer(event.get("seed", 20261024), 0, 2**32 - 1),
                        trees=integer(event.get("trees", 64), 1, 64), depth=integer(event.get("depth", 14), 1, 14),
                        max_fit_patches=integer(event.get("max_fit_patches", 12000), 4, 12000), **supplement)
        return output
    resident = event.get("benchmark_s3")
    if resident is not None:
        source_prefix = resident + ("train/" if job in ("train_samples", "hard_negatives") else "")
        manifest = _json_object(s3, bucket, source_prefix + "manifest.json")
        if manifest.get("s3_prefix") not in (None, source_prefix):
            raise ValueError("resident_manifest_points_to_a_different_benchmark")
        manifest["s3_prefix"] = source_prefix
    else:
        manifest = _json_object(s3, bucket, event["manifest_key"])
    if not isinstance(manifest.get("images"), list) or not 1 <= len(manifest["images"]) <= 4096:
        raise ValueError("invalid_image_manifest")
    dataset = workspace / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    if job in ("train_samples", "hard_negatives"):
        from scripts.train_detector import extract_samples, planned_entries, shard_entries
        holdout = event.get("holdout_surface")
        entries, _ = planned_entries(manifest, holdout)
        selected = shard_entries(entries, event["shard_index"], event["shard_count"])
        mining = job == "hard_negatives"
        if mining:
            selected = [entry for entry in selected if entry["role"] == "fit"]
        _stage_dataset(s3, bucket, prefix, manifest, selected, dataset)
        extra_options = {}
        if mining:
            extra_options = {"hard_negatives": True,
                             "model_dir": _models(s3, bucket, prefix, workspace, event.get("model_prefix")),
                             "hard_negative_fraction": event["hard_negative_fraction"]}
        extract_samples(dataset, output, shard_index=event["shard_index"], shard_count=event["shard_count"],
                        holdout_surface=holdout, seed=integer(event.get("seed", 20261024), 0, 2**32 - 1),
                        size=integer(event.get("size", 4096), 2048, 4096),
                        calibration=event.get("calibration", "sampled"),
                        min_gt_pixels=integer(event.get("min_gt_pixels", 3), 1, 1024),
                        max_image_patches=integer(event.get("max_image_patches", 1024), 2, 4096), **extra_options)
        return output
    evaluation, selected = _evaluation(event, manifest, None, dataset, bucket)
    _stage_dataset(s3, bucket, prefix, manifest, selected, dataset)
    model_dir = _models(s3, bucket, prefix, workspace, event.get("model_prefix")) if "learned" in evaluation["detectors"] else None
    if model_dir is not None:
        from scripts.run_eval import model_fingerprint
        evaluation["model_files"] = model_fingerprint(model_dir)
    from inspector.agent import InspectorConfig
    from inspector.tools import LocalBackend
    from scripts.run_eval import SHARD_SCHEMA, evaluate_entries, evaluate_tuning_entries
    rows = []
    if job == "tune":
        rows = evaluate_tuning_entries(dataset, selected, evaluation, model_dir)
    else:
        for detector in evaluation["detectors"]:
            methods = [method for method in evaluation["methods"] if detector != "classic" or method in ("agent", "exhaustive")]
            backend = LocalBackend(max_size=evaluation["size"], detector=detector, model_dir=model_dir,
                                   calibration=evaluation["calibration"], calibration_cap_frac=evaluation["calibration_cap_frac"])
            rows.extend(evaluate_entries(dataset, selected, methods, evaluation["budgets"], evaluation["size"],
                                         InspectorConfig(**evaluation["config"]), evaluation["seed"],
                                         output / "traces" if event.get("save_traces", False) else None,
                                         detector=detector, model_dir=model_dir, random_seeds=evaluation["random_seeds"],
                                         backend=backend, calibration=evaluation["calibration"]))
    metadata = {"schema": SHARD_SCHEMA, "status": "shard_complete", "evaluation": evaluation,
                "shard_index": event["shard_index"], "shard_count": event["shard_count"],
                "assigned_images": [entry["image"] for entry in selected], "n_records": len(rows)}
    (output / "run.json").write_text(json.dumps(metadata, allow_nan=False))
    (output / "results.json").write_text(json.dumps(rows, allow_nan=False))
    return output


def run_job(event: dict, s3, *, bucket: str, prefix: str = "x132/", temp_root: Path | None = None) -> dict:
    """Injected-client boundary used by Lambda and small, offline fake-client tests."""
    prefix = safe_prefix(prefix)
    if not isinstance(bucket, str) or not bucket or "/" in bucket:
        raise ValueError("invalid_configured_bucket")
    validated = None
    request_digest = None
    owns_claim = False
    try:
        validated = validate_event(event, prefix)
        request_digest = hashlib.sha256(json.dumps(validated, sort_keys=True, allow_nan=False).encode()).hexdigest()
        try:
            previous = _json_object(s3, bucket, validated["output_prefix"] + "DONE.json")
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in ("NoSuchKey", "404", "NotFound") and not isinstance(error, FileNotFoundError):
                raise
        else:
            if previous.get("request_sha256") != request_digest:
                return {"status": "FAILED", "error": {"code": "job_identity_conflict"}}
            return previous
        try:
            failed = _json_object(s3, bucket, validated["output_prefix"] + "FAILED.json")
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in ("NoSuchKey", "404", "NotFound") and not isinstance(error, FileNotFoundError):
                raise
        else:
            return failed  # A failed job is immutable. A retry requires a new job_id.
        lock = {"request_sha256": request_digest, "job_id": validated["job_id"],
                "job": validated["job"], "shard_index": validated["shard_index"]}
        try:
            s3.put_object(Bucket=bucket, Key=validated["output_prefix"] + "LOCK.json",
                          Body=json.dumps(lock, allow_nan=False).encode(),
                          ContentType="application/json", IfNoneMatch="*")
            owns_claim = True
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise
            try:
                existing = _json_object(s3, bucket, validated["output_prefix"] + "LOCK.json")
            except Exception:
                existing = None
            if existing is not None and existing.get("request_sha256") != request_digest:
                return {"status": "FAILED", "error": {"code": "job_identity_conflict"}}
            return {"status": "RUNNING", "job_id": validated["job_id"],
                    "shard_index": validated["shard_index"], "request_sha256": request_digest,
                    "retry_policy": "Existing claim is permanent; use a new job_id after failure or timeout."}
        import cv2
        cv2.setNumThreads(2)
        with tempfile.TemporaryDirectory(prefix="x132-bench-", dir=temp_root) as temporary:
            with redirect_stdout(io.StringIO()):
                executed = _execute(validated, s3, bucket, prefix, Path(temporary))
            if isinstance(executed, tuple):
                output, files = executed
            else:
                output, files = executed, []
            for path in sorted(output.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                name = path.relative_to(output).as_posix()
                key = validated["output_prefix"] + safe_relative(name)
                files.append(_upload_record(s3, bucket, path, key, name))
        result = {"status": "DONE", "job": validated["job"], "job_id": validated["job_id"],
                  "shard_index": validated["shard_index"], "shard_count": validated["shard_count"],
                  "request_sha256": request_digest, "files": files}
        if validated["job"] == "generate":
            result["benchmark_s3"] = f"{prefix}benchmarks/{validated['job_id']}/"
            result["benchmark"] = validated["benchmark"]
        marker = validated["output_prefix"] + "DONE.json"
    except Exception as error:
        result = {"status": "FAILED", "error": {"code": "job_rejected_or_failed", "type": type(error).__name__}}
        if validated is not None:
            if not owns_claim:
                # A contender must never publish a terminal state for its owner.
                return {"status": "FAILED", "error": {"code": "preclaim_request_failed", "type": type(error).__name__}}
            result.update(job=validated["job"], job_id=validated["job_id"], shard_index=validated["shard_index"])
            if request_digest is not None:
                result["request_sha256"] = request_digest
            marker = validated["output_prefix"] + "FAILED.json"
        else:
            marker = prefix + "jobs/rejected/FAILED.json"
    try:
        s3.put_object(Bucket=bucket, Key=marker, Body=json.dumps(result, allow_nan=False).encode(), ContentType="application/json")
    except Exception:
        return {"status": "FAILED", "error": {"code": "terminal_marker_write_failed"}}
    return result


def lambda_handler(event, context):
    try:
        from aws.runtime import get_clients
        s3, _ = get_clients()
        return run_job(event, s3, bucket=os.environ["X132_BUCKET"], prefix=os.environ.get("X132_PREFIX", "x132/"))
    except Exception as error:
        return {"status": "FAILED", "error": {"code": "runtime_configuration_failed", "type": type(error).__name__}}


handler = lambda_handler
