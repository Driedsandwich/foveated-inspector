"""Separate sharded image-feature extraction from bounded, image-free forest fit.

Extraction opens split=train development images only. Each shard writes capped
per-image/scale NPZ samples and a provenance manifest. Merge verifies the complete
shard set and fits capped sample arrays without opening any source image. The
held-out development images set thresholds only. By default every fourth sorted
image of each development surface is validation. Explicit --holdout-surface
retains the older complete-surface holdout experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from inspector.features import FEATURE_NAMES, FEATURE_VERSION, PATCH_SIZE, feature_map, prepare_calibration
from inspector.model import DetectorModels, binary_probabilities, scale_key
from inspector.synth import DEV_SURFACES

SCALES = (0.25, 0.5, 1.0)
SAMPLE_VERSION = "patch_samples_v2_bound"
MAX_IMAGE_PATCHES = 4096
MAX_FIT_PATCHES = 12000


def patch_targets(labels: np.ndarray, defects: list[dict], kind_ids: dict[str, int],
                  min_pixels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Exact GT occupancy labels; dominant kind wins when a patch has two kinds."""
    height, width = labels.shape
    rows, cols = (height + 31) // 32, (width + 31) // 32
    counts = np.zeros((rows, cols, len(kind_ids)), np.int32)
    for kind, kind_id in kind_ids.items():
        ids = [int(d["id"]) for d in defects if d["kind"] == kind]
        mask = np.isin(labels, ids).astype(np.uint8)
        padded = np.pad(mask, ((0, rows * 32 - height), (0, cols * 32 - width)))
        counts[:, :, kind_id - 1] = padded.reshape(rows, 32, cols, 32).sum(axis=(1, 3))
    positive = counts.sum(axis=2) >= min_pixels
    kind = counts.argmax(axis=2).astype(np.int32) + 1
    kind[~positive] = 0
    return positive.astype(np.int32).ravel(), kind.ravel()


def _image_roles(entries: list[dict], holdout_surface: str | None) -> dict[str, str]:
    if holdout_surface is not None:
        if holdout_surface not in DEV_SURFACES:
            raise ValueError("Holdout must be a development surface")
        return {entry["image"]: "validation" if entry["surface"] == holdout_surface else "fit"
                for entry in entries}
    roles = {}
    for surface in sorted({entry["surface"] for entry in entries}):
        group = sorted((entry for entry in entries if entry["surface"] == surface), key=lambda entry: entry["image"])
        if len(group) < 2:
            raise ValueError("Image holdout needs at least two train images per surface; use explicit surface holdout for legacy data")
        validation = {entry["image"] for entry in group[3::4]}
        if not validation:
            validation = {group[-1]["image"]}
        roles.update({entry["image"]: "validation" if entry["image"] in validation else "fit" for entry in group})
    return roles


def planned_entries(manifest: dict, holdout_surface: str | None = None) -> tuple[list[dict], dict[str, int]]:
    """Pure metadata planning; non-train paths are never dereferenced.

    Image holdout selects every fourth sorted image per surface. Two- or
    three-image fixtures use their last image; singleton surfaces are rejected.
    An explicit holdout_surface retains the legacy surface-level split.
    """
    entries = sorted((dict(e) for e in manifest["images"] if e.get("split") == "train"),
                     key=lambda e: e["image"])
    if not entries or any(e["surface"] not in DEV_SURFACES for e in entries):
        raise ValueError("Extraction needs split=train development surfaces only")
    if len({e["image"] for e in entries}) != len(entries):
        raise ValueError("Training image IDs must be unique")
    roles = _image_roles(entries, holdout_surface)
    fitted = [e for e in entries if roles[e["image"]] == "fit"]
    validation = [e for e in entries if roles[e["image"]] == "validation"]
    if not fitted or not validation:
        raise ValueError("Both fitted and held-out development images are required")
    kinds = sorted({str(d["kind"]) for e in fitted for d in e["defects"]})
    if len(kinds) < 2 or any(d["kind"] not in kinds for e in validation for d in e["defects"]):
        raise ValueError("At least two fit kinds must cover every held-out kind")
    for entry in entries:
        entry["role"] = roles[entry["image"]]
    return entries, {kind: index + 1 for index, kind in enumerate(kinds)}


def shard_entries(entries: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    """Pure, deterministic image partition; every image belongs to one shard."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count>=1 and 0<=shard_index<shard_count")
    return sorted(entries, key=lambda e: e["image"])[shard_index::shard_count]


def _seed(seed: int, image_id: str, scale: str) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}:{scale}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _hardness(features: np.ndarray) -> np.ndarray:
    names = ("blackhat3_max", "blackhat9_max", "tophat9_max", "gabor_energy_max")
    return features[:, [FEATURE_NAMES.index(name) for name in names]].sum(axis=1)


def _negative_sample(features: np.ndarray, candidates: np.ndarray, count: int,
                     rng: np.random.Generator) -> np.ndarray:
    count = min(count, len(candidates))
    if count == 0:
        return candidates[:0]
    hard_count = count // 2
    order = np.argsort(_hardness(features[candidates]), kind="stable")
    hard = candidates[order[-hard_count:]] if hard_count else candidates[:0]
    remaining = np.setdiff1d(candidates, hard, assume_unique=True)
    return np.r_[hard, rng.choice(remaining, count - len(hard), replace=False)]


def select_samples(features: np.ndarray, truth: np.ndarray, kinds: np.ndarray,
                   role: str, cap: int, seed: int) -> dict[str, np.ndarray]:
    """Bound one image's arrays; validation weights restore original prevalence.

    Fit negatives are half hard-response and half random. Held-out negatives are
    purely random. Both validation strata carry inverse inclusion weights, so
    precision does not silently use the enriched sample's positive prevalence.
    """
    if role not in ("fit", "validation") or not 2 <= cap <= MAX_IMAGE_PATCHES:
        raise ValueError("Invalid role or per-image sample cap")
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(truth), np.flatnonzero(truth == 0)
    chosen_positive = rng.choice(positive, min(len(positive), cap // 2), replace=False)
    negative_count = min(len(negative), cap - len(chosen_positive))
    if role == "fit":
        negative_count = min(negative_count, max(64, len(chosen_positive) * 2))
        chosen_negative = _negative_sample(features, negative, negative_count, rng)
    else:
        chosen_negative = rng.choice(negative, negative_count, replace=False)
    chosen = np.r_[chosen_positive, chosen_negative]
    weights = np.ones(len(chosen), np.float64)
    if role == "validation":
        if len(chosen_positive):
            weights[:len(chosen_positive)] = len(positive) / len(chosen_positive)
        if len(chosen_negative):
            weights[len(chosen_positive):] = len(negative) / len(chosen_negative)
    return {"x": np.ascontiguousarray(features[chosen], dtype=np.float32),
            "y": truth[chosen].astype(np.int32), "kind": kinds[chosen].astype(np.int32),
            "weight": weights, "patch_index": chosen.astype(np.int32)}


def _source_record(entry: dict) -> dict:
    return {key: entry.get(key) for key in ("image", "surface", "seed", "split", "role")}


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_manifest_digest(manifest: dict) -> str:
    """Preserve source provenance while ignoring the S3 staging transport root."""
    identity = {key: value for key, value in manifest.items() if key != "s3_prefix"}
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_filename(image_id: str, scale: str) -> str:
    image_key = hashlib.sha256(image_id.encode()).hexdigest()[:16]
    return f"samples_{image_key}_scale_{scale}.npz"


def _sample_identity(entry: dict, scale: str, config: dict, mining: dict | None = None) -> dict:
    signature = hashlib.sha256(_canonical_json(config).encode()).hexdigest()
    result = {"sample_version": SAMPLE_VERSION, **_source_record(entry),
              "scale": scale, "config_sha256": signature}
    if mining is not None:
        result["mining"] = mining
    return result


def model_digest(model_dir: Path) -> str:
    """Hash metadata and its six forest files, compatible with deployment bundles."""
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    files = [metadata_path]
    for scale in ("0.25", "0.5", "1"):
        for kind in ("binary", "kind"):
            name = metadata["models"][scale][kind]
            if name != f"scale_{scale}_{kind}.yml.gz":
                raise ValueError("Invalid first-fit model filename")
            files.append(model_dir / name)
    if len({path.name for path in files}) != 7:
        raise ValueError("First fit requires metadata and six distinct forest files")
    entries = [{"name": path.name, "sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
               for path in sorted(files, key=lambda path: path.name)]
    return hashlib.sha256(_canonical_json(entries).encode()).hexdigest()


def _mining_binding(model_dir: Path, config: dict, fraction: float) -> dict:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    if metadata.get("hard_negative_rounds", 0) != 0:
        raise ValueError("Hard-negative mining permits only one round from an initial fit")
    if metadata.get("sample_config") != config:
        raise ValueError("First-fit model sample configuration does not match extraction provenance")
    training_digest = metadata.get("training_samples_sha256")
    if (not isinstance(training_digest, str) or len(training_digest) != 64
            or any(character not in "0123456789abcdef" for character in training_digest)):
        raise ValueError("First-fit model lacks an immutable base-sample binding")
    return {"round": 1, "model_digest": model_digest(model_dir),
            "base_config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "training_samples_sha256": training_digest, "fraction": float(fraction),
            "selection": "highest model score among strictly empty full-resolution patches"}


def empty_background_patches(labels: np.ndarray, working_shape: tuple[int, int],
                              feature_shape: tuple[int, int], integral: np.ndarray | None = None) -> np.ndarray:
    """Test full-resolution label occupancy, including thin defects lost by resize."""
    height, width = labels.shape
    work_h, work_w = working_shape
    rows, cols = feature_shape
    if rows != (work_h + PATCH_SIZE - 1) // PATCH_SIZE or cols != (work_w + PATCH_SIZE - 1) // PATCH_SIZE:
        raise ValueError("Feature grid does not cover its working image")
    if integral is None:
        integral = cv2.integral((labels != 0).astype(np.uint8), sdepth=cv2.CV_32S)
    left = np.arange(cols) * PATCH_SIZE * width // work_w
    right = (np.minimum((np.arange(cols) + 1) * PATCH_SIZE, work_w) * width + work_w - 1) // work_w
    top = np.arange(rows) * PATCH_SIZE * height // work_h
    bottom = (np.minimum((np.arange(rows) + 1) * PATCH_SIZE, work_h) * height + work_h - 1) // work_h
    occupancy = (integral[bottom[:, None], right[None, :]] - integral[top[:, None], right[None, :]]
                 - integral[bottom[:, None], left[None, :]] + integral[top[:, None], left[None, :]])
    return (occupancy == 0).ravel()


def select_hard_negatives(features: np.ndarray, empty: np.ndarray, scores: np.ndarray,
                           fraction: float, cap: int) -> dict[str, np.ndarray]:
    if (not 0 < fraction <= 1 or not 2 <= cap <= MAX_IMAGE_PATCHES
            or scores.shape != (len(features),) or empty.shape != scores.shape
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid hard-negative fraction, scores, or cap")
    candidates = np.flatnonzero(empty)
    count = min(cap, int(np.ceil(len(candidates) * fraction)))
    chosen = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
    return {"x": np.ascontiguousarray(features[chosen], dtype=np.float32),
            "y": np.zeros(count, np.int32), "kind": np.zeros(count, np.int32),
            "weight": np.ones(count, np.float64), "patch_index": chosen.astype(np.int32),
            "mining_score": scores[chosen].astype(np.float32)}


def _write_sample_pack(target: Path, arrays: dict, identity: dict) -> str:
    """Bind exact sample bytes to immutable image/config identity inside the NPZ."""
    if target.exists():
        raise FileExistsError(f"Preserving existing sample arrays: {target}")
    np.savez(target, **arrays, identity_json=np.array(_canonical_json(identity)))
    return _file_sha256(target)


def extract_samples(bench: Path, samples_dir: Path, *, shard_index: int = 0,
                    shard_count: int = 1, holdout_surface: str | None = None,
                    seed: int = 20261024, size: int = 4096, min_gt_pixels: int = 3,
                    max_image_patches: int = 1024, calibration: str = "sampled",
                    hard_negatives: bool = False, model_dir: Path | None = None,
                    hard_negative_fraction: float = .04) -> dict:
    """Image-processing stage only: no forest fitting is reachable here."""
    if min(size, min_gt_pixels) < 1 or not 2 <= max_image_patches <= MAX_IMAGE_PATCHES:
        raise ValueError("Invalid extraction size, label threshold or sample cap")
    manifest = json.loads((bench / "manifest.json").read_text())
    entries, kind_ids = planned_entries(manifest, holdout_surface)
    selected = shard_entries(entries, shard_index, shard_count)
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
              "source_manifest_digest": digest, "source_images": [_source_record(e) for e in entries],
              "source_manifest_sha256": source_manifest_digest(manifest),
              "holdout_surface": holdout_surface, "kind_ids": kind_ids, "seed": seed,
              "holdout_strategy": "surface" if holdout_surface is not None else "image_quarter",
              "size": size, "min_gt_pixels": min_gt_pixels, "max_image_patches": max_image_patches,
              "calibration": calibration}
    mining, first_model = None, None
    if hard_negatives:
        if model_dir is None or not 0 < hard_negative_fraction <= 1:
            raise ValueError("Hard-negative extraction requires first-fit model_dir and fraction in (0,1]")
        mining = _mining_binding(Path(model_dir), config, hard_negative_fraction)
        first_model = DetectorModels(model_dir)
        # Partition the complete train set first, then exclude validation. This
        # keeps shard identities identical to the original extraction stage.
        selected = [entry for entry in selected if entry["role"] == "fit"]
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = samples_dir / f"shard_{shard_index:04d}_of_{shard_count:04d}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Preserving existing shard manifest: {manifest_path}")
    records = []
    for entry in selected:
        image = cv2.imread(str(bench / entry["image"]), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(bench / entry["labels"]), cv2.IMREAD_UNCHANGED)
        if image is None or labels is None or labels.dtype != np.uint16 or labels.shape != image.shape[:2]:
            raise ValueError(f"Invalid training image/labels: {entry['image']}")
        if max(image.shape[:2]) > size:
            ratio = size / max(image.shape[:2])
            shape = (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio)))
            image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
            if not hard_negatives:
                labels = cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
        statistics, _ = prepare_calibration(image, mode=calibration)
        occupancy_integral = cv2.integral((labels != 0).astype(np.uint8), sdepth=cv2.CV_32S) if hard_negatives else None
        for scale in SCALES:
            key = scale_key(scale)
            target = samples_dir / _sample_filename(entry["image"], key)
            if target.exists():
                raise FileExistsError(f"Preserving existing sample arrays: {target}")
            shape = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
            working = image if scale == 1 else cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
            ground_truth = (None if hard_negatives else labels if scale == 1
                            else cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST))
            features, _ = feature_map(working, statistics[key])
            if hard_negatives:
                empty = empty_background_patches(labels, working.shape[:2], features.shape[:2], occupancy_integral)
                scores, _ = first_model.predict(features, scale)
                arrays = select_hard_negatives(features.reshape(-1, len(FEATURE_NAMES)), empty,
                                                scores.ravel(), hard_negative_fraction, max_image_patches)
                source_count, positive_count = len(empty), int((~empty).sum())
                extra = {"n_empty_background": int(empty.sum()), "empty_background_only": True}
            else:
                binary, kinds = patch_targets(ground_truth, entry["defects"], kind_ids, min_gt_pixels)
                arrays = select_samples(features.reshape(-1, len(FEATURE_NAMES)), binary, kinds,
                                        entry["role"], max_image_patches, _seed(seed, entry["image"], key))
                source_count, positive_count, extra = len(binary), int(binary.sum()), {}
                del binary, kinds
            digest = _write_sample_pack(target, arrays, _sample_identity(entry, key, config, mining))
            records.append({**_source_record(entry), "scale": key, "file": target.name,
                            "sha256": digest,
                            "n_samples": len(arrays["y"]), "n_source_patches": source_count,
                            "n_source_positive": positive_count, **extra})
            del working, ground_truth, features, arrays
        del image, labels, occupancy_integral
        print(json.dumps({"stage": "extract_hard_negatives" if hard_negatives else "extract", "image": entry["image"],
                          "role": entry["role"], "shard_index": shard_index}), flush=True)
    result = {"sample_version": SAMPLE_VERSION, "config": config, "shard_index": shard_index,
              "shard_count": shard_count, "records": records}
    if mining is not None:
        result["mining"] = mining
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return {"manifest": str(manifest_path), "n_images": len(selected), "n_sample_files": len(records),
            "models_fitted": False}


def merge_array_records(records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Pure deterministic concatenation; input record order/shard count is irrelevant."""
    result = {}
    for scale in SCALES:
        key = scale_key(scale)
        selected = sorted((r for r in records if r["scale"] == key), key=lambda r: r["image"])
        for role in ("fit", "validation"):
            subset = [r for r in selected if r["role"] == role]
            if not subset:
                raise ValueError(f"Missing {role} samples at scale {key}")
            result[f"{key}:{role}"] = {name: np.concatenate([r["arrays"][name] for r in subset])
                                      for name in ("x", "y", "kind", "weight", "patch_index")}
    return result


def _validate_config(config: dict) -> None:
    if (config.get("feature_version") != FEATURE_VERSION
            or config.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Incompatible extracted feature schema")
    if not 2 <= config["max_image_patches"] <= MAX_IMAGE_PATCHES:
        raise ValueError("Unbounded per-image sample declaration")
    source = config["source_images"]
    if not source or len({e["image"] for e in source}) != len(source):
        raise ValueError("Invalid source image provenance")
    holdout = config.get("holdout_surface")
    expected_strategy = "surface" if holdout is not None else "image_quarter"
    if config.get("holdout_strategy", expected_strategy) != expected_strategy:
        raise ValueError("Holdout strategy disagrees with provenance")
    roles = _image_roles(source, holdout)
    for entry in source:
        role = roles[entry["image"]]
        if entry["split"] != "train" or entry["surface"] not in DEV_SURFACES or entry["role"] != role:
            raise ValueError("Sample source is not train-only or leaks a held-out image")
    if {e["role"] for e in source} != {"fit", "validation"}:
        raise ValueError("Sample provenance requires both fit and validation images")
    kinds = config["kind_ids"]
    if len(kinds) < 2 or set(kinds.values()) != set(range(1, len(kinds) + 1)):
        raise ValueError("Invalid extracted kind-ID schema")


def load_samples(paths: list[Path]) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Verify complete shard provenance, then load bounded arrays; no image I/O."""
    manifests = []
    for path in paths:
        manifests.extend(sorted(path.glob("shard_*_of_*.json")) if path.is_dir() else [path])
    if not manifests or len(set(p.resolve() for p in manifests)) != len(manifests):
        raise ValueError("Need a nonempty, nonduplicated shard-manifest set")
    loaded = [(path, json.loads(path.read_text())) for path in manifests]
    config = loaded[0][1]["config"]
    count = int(loaded[0][1]["shard_count"])
    if {int(m["shard_index"]) for _, m in loaded} != set(range(count)) or len(loaded) != count:
        raise ValueError("Merge requires exactly one manifest for every shard")
    _validate_config(config)
    source = config["source_images"]
    expected = {e["image"]: e for e in source}
    records, seen = [], set()
    for path, manifest in loaded:
        if manifest.get("mining") is not None:
            raise ValueError("Hard-negative supplement requires the explicit one-round refit API")
        if manifest["sample_version"] != SAMPLE_VERSION or manifest["config"] != config or manifest["shard_count"] != count:
            raise ValueError("Shard configurations do not match")
        allowed = {e["image"] for e in shard_entries(source, manifest["shard_index"], count)}
        for record in manifest["records"]:
            identity = (record["image"], record["scale"])
            if identity in seen or record["image"] not in allowed or record["scale"] not in {scale_key(s) for s in SCALES}:
                raise ValueError("Duplicate, misplaced or invalid-scale sample record")
            if _source_record(record) != expected[record["image"]]:
                raise ValueError("Per-sample provenance disagrees with source manifest")
            filename = record["file"]
            if (Path(filename).name != filename
                    or filename != _sample_filename(record["image"], record["scale"])):
                raise ValueError("Sample filename does not match its image/scale identity")
            sample_path = path.parent / filename
            if record.get("sha256") != _file_sha256(sample_path):
                raise ValueError("Sample content digest disagrees with its shard manifest")
            with np.load(sample_path, allow_pickle=False) as archive:
                embedded = json.loads(str(archive["identity_json"].item()))
                if embedded != _sample_identity(expected[record["image"]], record["scale"], config):
                    raise ValueError("Sample internal image/role/config binding does not match provenance")
                arrays = {name: archive[name].copy() for name in ("x", "y", "kind", "weight", "patch_index")}
            n = len(arrays["y"])
            if (n != record["n_samples"] or n > config["max_image_patches"]
                    or arrays["x"].shape != (n, len(FEATURE_NAMES))
                    or any(arrays[name].shape != (n,) for name in ("y", "kind", "weight", "patch_index"))
                    or not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["weight"]).all()
                    or np.any(arrays["weight"] <= 0) or not np.isin(arrays["y"], (0, 1)).all()):
                raise ValueError("Malformed or unbounded extracted samples")
            positive = arrays["y"] == 1
            if (not np.isin(arrays["kind"][positive], list(config["kind_ids"].values())).all()
                    or np.any(arrays["kind"][~positive] != 0)):
                raise ValueError("Invalid extracted kind labels")
            seen.add(identity)
            records.append({**record, "arrays": arrays})
    required = {(image, scale_key(scale)) for image in expected for scale in SCALES}
    if seen != required:
        raise ValueError("Incomplete image/scale sample coverage")
    return merge_array_records(records), config


def sample_counts(samples: dict[str, dict[str, np.ndarray]]) -> dict:
    return {key: {"n_samples": len(arrays["y"]), "n_positive": int(arrays["y"].sum()),
                  "n_negative": int(len(arrays["y"]) - arrays["y"].sum()),
                  "weighted_positive": float(arrays["weight"][arrays["y"] == 1].sum()),
                  "weighted_negative": float(arrays["weight"][arrays["y"] == 0].sum())}
            for key, arrays in samples.items()}


def load_hard_negative_samples(paths: list[Path], config: dict, first_model_dir: Path
                               ) -> tuple[dict[str, dict[str, np.ndarray]], dict, list[str]]:
    """Validate one complete, fit-only supplement set without opening images."""
    _validate_config(config)
    manifests = []
    for path in paths:
        path = Path(path)
        manifests.extend(sorted(path.glob("shard_*_of_*.json")) if path.is_dir() else [path])
    if not manifests or len({path.resolve() for path in manifests}) != len(manifests):
        raise ValueError("Need a nonduplicated hard-negative shard set")
    loaded = [(path, json.loads(path.read_text())) for path in manifests]
    first = loaded[0][1]
    mining = first.get("mining", {})
    fraction = mining.get("fraction", 0)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("Invalid hard-negative mining fraction")
    expected_mining = _mining_binding(first_model_dir, config, fraction)
    count = first.get("shard_count")
    if type(count) is not int or count < 1 or len(loaded) != count:
        raise ValueError("Hard-negative refit requires every shard")
    if {manifest.get("shard_index") for _, manifest in loaded} != set(range(count)):
        raise ValueError("Missing or duplicate hard-negative shard")
    expected = {entry["image"]: entry for entry in config["source_images"] if entry["role"] == "fit"}
    records, seen = [], set()
    for path, manifest in loaded:
        if (manifest.get("sample_version") != SAMPLE_VERSION or manifest.get("config") != config
                or manifest.get("mining") != expected_mining or manifest.get("shard_count") != count):
            raise ValueError("Hard-negative first-fit/configuration binding does not match")
        allowed = {entry["image"] for entry in shard_entries(config["source_images"], manifest["shard_index"], count)
                   if entry["role"] == "fit"}
        for record in manifest["records"]:
            identity = (record["image"], record["scale"])
            if (identity in seen or record["image"] not in allowed
                    or record["scale"] not in {scale_key(scale) for scale in SCALES}
                    or _source_record(record) != expected[record["image"]]):
                raise ValueError("Hard-negative records contain held-out, duplicate, or misplaced images")
            filename = record["file"]
            if Path(filename).name != filename or filename != _sample_filename(*identity):
                raise ValueError("Hard-negative filename does not match image/scale")
            target = path.parent / filename
            if record.get("sha256") != _file_sha256(target):
                raise ValueError("Hard-negative sample digest mismatch")
            with np.load(target, allow_pickle=False) as archive:
                embedded = json.loads(str(archive["identity_json"].item()))
                if embedded != _sample_identity(expected[record["image"]], record["scale"], config, expected_mining):
                    raise ValueError("Hard-negative NPZ is not bound to its first fit and training image")
                arrays = {name: archive[name].copy() for name in
                          ("x", "y", "kind", "weight", "patch_index", "mining_score")}
            n = len(arrays["y"])
            source_count, empty_count = record.get("n_source_patches"), record.get("n_empty_background")
            if (type(source_count) is not int or source_count < 1 or type(empty_count) is not int
                    or not 0 <= empty_count <= source_count or record.get("empty_background_only") is not True
                    or record.get("n_source_positive") != source_count - empty_count
                    or n != record["n_samples"] or n != min(config["max_image_patches"], int(np.ceil(empty_count * fraction)))
                    or arrays["x"].shape != (n, len(FEATURE_NAMES)) or not np.isfinite(arrays["x"]).all()
                    or any(arrays[name].shape != (n,) for name in ("y", "kind", "weight", "patch_index", "mining_score"))
                    or np.any(arrays["y"] != 0) or np.any(arrays["kind"] != 0) or np.any(arrays["weight"] != 1)
                    or not np.isfinite(arrays["mining_score"]).all()
                    or np.any((arrays["mining_score"] < 0) | (arrays["mining_score"] > 1))
                    or not np.issubdtype(arrays["patch_index"].dtype, np.integer)
                    or len(np.unique(arrays["patch_index"])) != n
                    or np.any((arrays["patch_index"] < 0) | (arrays["patch_index"] >= source_count))):
                raise ValueError("Malformed, impure, or unbounded hard-negative samples")
            seen.add(identity)
            records.append({**record, "arrays": arrays})
    required = {(image, scale_key(scale)) for image in expected for scale in SCALES}
    if seen != required:
        raise ValueError("Incomplete fit-only hard-negative sample coverage")
    combined = {}
    for scale in SCALES:
        key = scale_key(scale)
        selected = sorted((record for record in records if record["scale"] == key), key=lambda record: record["image"])
        combined[key] = {name: np.concatenate([record["arrays"][name] for record in selected])
                         for name in ("x", "y", "kind", "weight", "patch_index", "mining_score")}
    return combined, expected_mining, sorted(_file_sha256(path) for path in manifests)


def _archive_key(group: str, field: str) -> str:
    return f"{group.replace(':', '_')}__{field}"


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256(f"{values.dtype.str}:{values.shape}".encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def training_samples_digest(samples: dict[str, dict[str, np.ndarray]]) -> str:
    """Semantic digest independent of NPZ ZIP timestamps and serialization order."""
    entries = {group: {field: _array_sha256(values) for field, values in sorted(data.items())}
               for group, data in sorted(samples.items())}
    return hashlib.sha256(_canonical_json(entries).encode()).hexdigest()


def merge_samples(paths: list[Path], target: Path) -> dict:
    """Array-only merge stage; writes one NPZ plus JSON, and never fits a model."""
    if target.suffix != ".npz":
        raise ValueError("Merged sample target must end in .npz")
    sidecar = target.with_suffix(".json")
    if target.exists() or sidecar.exists():
        raise FileExistsError("Preserving existing merged sample archive or metadata")
    samples, config = load_samples(paths)
    arrays, mapping = {}, {}
    for key, data in samples.items():
        mapping[key] = {}
        for name, values in data.items():
            archive_key = _archive_key(key, name)
            arrays[archive_key] = values
            mapping[key][name] = archive_key
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"sample_version": SAMPLE_VERSION + "_merged", "config": config,
                "arrays": mapping, "sample_counts": sample_counts(samples),
                "array_sha256": {name: _array_sha256(values) for name, values in arrays.items()}}
    embedded = _canonical_json(metadata)
    np.savez(target, **arrays, identity_json=np.array(embedded))
    sidecar_payload = {**metadata, "metadata_sha256": hashlib.sha256(embedded.encode()).hexdigest(),
                       "archive_sha256": _file_sha256(target)}
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, allow_nan=False))
    return {"archive": str(target), "metadata": str(sidecar), "sample_counts": metadata["sample_counts"],
            "source_images_opened": 0, "models_fitted": False}


def load_merged_samples(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    metadata = json.loads(path.with_suffix(".json").read_text())
    archive_digest = metadata.pop("archive_sha256", None)
    identity_digest = metadata.pop("metadata_sha256", None)
    expected_identity = _canonical_json(metadata)
    if (archive_digest != _file_sha256(path)
            or identity_digest != hashlib.sha256(expected_identity.encode()).hexdigest()):
        raise ValueError("Merged archive or sidecar integrity binding failed")
    if metadata.get("sample_version") != SAMPLE_VERSION + "_merged":
        raise ValueError("Incompatible merged sample archive")
    config = metadata["config"]
    _validate_config(config)
    expected = {f"{scale_key(scale)}:{role}" for scale in SCALES for role in ("fit", "validation")}
    if set(metadata["arrays"]) != expected:
        raise ValueError("Merged archive lacks a scale/role group")
    samples = {}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["identity_json"].item()) != expected_identity:
            raise ValueError("Merged archive internal provenance does not match its sidecar")
        for key, mapping in metadata["arrays"].items():
            if set(mapping) != {"x", "y", "kind", "weight", "patch_index"}:
                raise ValueError("Invalid merged array schema")
            if any(array_key != _archive_key(key, name) for name, array_key in mapping.items()):
                raise ValueError("Merged array keys must use canonical scale/role/field names")
            data = {name: archive[array_key].copy() for name, array_key in mapping.items()}
            if any(metadata["array_sha256"].get(mapping[name]) != _array_sha256(values)
                   for name, values in data.items()):
                raise ValueError("Merged array contents disagree with their internal digests")
            role = key.split(":")[1]
            max_rows = sum(e["role"] == role for e in config["source_images"]) * config["max_image_patches"]
            n = len(data["y"])
            if (n > max_rows or data["x"].shape != (n, len(FEATURE_NAMES))
                    or any(data[name].shape != (n,) for name in ("y", "kind", "weight", "patch_index"))
                    or not np.isfinite(data["x"]).all() or not np.isfinite(data["weight"]).all()
                    or np.any(data["weight"] <= 0) or not np.isin(data["y"], (0, 1)).all()):
                raise ValueError("Malformed or unbounded merged arrays")
            positive = data["y"] == 1
            if (not np.isin(data["kind"][positive], list(config["kind_ids"].values())).all()
                    or np.any(data["kind"][~positive] != 0)):
                raise ValueError("Invalid merged kind labels")
            samples[key] = data
    if sample_counts(samples) != metadata["sample_counts"]:
        raise ValueError("Merged archive counts disagree with its metadata")
    return samples, config


def _forest(samples: np.ndarray, labels: np.ndarray, seed: int, trees: int, depth: int):
    if len(np.unique(labels)) < 2:
        raise ValueError("Forest training requires at least two observed classes")
    cv2.setRNGSeed(int(seed % (2 ** 31 - 1)))
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(depth)
    model.setMinSampleCount(2)
    model.setMaxCategories(max(2, len(np.unique(labels))))
    model.setCalculateVarImportance(True)
    model.setActiveVarCount(max(1, int(np.sqrt(samples.shape[1]))))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, trees, 0))
    if not model.train(np.ascontiguousarray(samples, dtype=np.float32), cv2.ml.ROW_SAMPLE,
                       np.ascontiguousarray(labels, dtype=np.int32)):
        raise RuntimeError("OpenCV RTrees training failed")
    return model


def validation_metrics(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Tie-aware weighted AUC and precision points; weights undo stratified sampling."""
    truth, scores = np.asarray(truth, np.int32), np.asarray(scores, np.float64)
    weights = np.ones(len(truth), np.float64) if weights is None else np.asarray(weights, np.float64)
    if (truth.shape != scores.shape or weights.shape != truth.shape or not np.isfinite(scores).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isin(truth, (0, 1)).all()):
        raise ValueError("Invalid validation labels, scores or weights")
    positives, negatives = float(weights[truth == 1].sum()), float(weights[truth == 0].sum())
    if not positives or not negatives:
        raise ValueError("Validation requires positive and background patches")
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    pos_groups = np.add.reduceat(truth[order] * weights[order], starts)
    neg_groups = np.add.reduceat((1 - truth[order]) * weights[order], starts)
    auc = float(np.sum(pos_groups * (np.cumsum(neg_groups) - 0.5 * neg_groups)) / (positives * negatives))
    descending = np.argsort(-scores, kind="stable")
    decreasing = scores[descending]
    ends = np.r_[np.flatnonzero(np.diff(decreasing)), len(truth) - 1]
    tp = np.cumsum(truth[descending] * weights[descending])[ends]
    reported = np.cumsum(weights[descending])[ends]
    recall, precision = tp / positives, tp / reported
    operating = {}
    for target in (0.5, 0.8, 0.9):
        index = int(np.flatnonzero(recall >= target)[0])
        operating[f"{target:g}"] = {"threshold": float(decreasing[ends[index]]),
                                    "recall": float(recall[index]), "precision": float(precision[index]),
                                    "reported_patches": int(ends[index] + 1),
                                    "weighted_reported_patches": float(reported[index])}
    return {"roc_auc": auc, "n_patches": len(truth), "n_positive": int(truth.sum()),
            "n_negative": int(len(truth) - truth.sum()), "weighted_positive": positives,
            "weighted_negative": negatives, "precision_at_recall": operating}


def fit_samples(paths: list[Path] | Path, out: Path, *, seed: int = 20261024,
                trees: int = 64, depth: int = 14, max_fit_patches: int = MAX_FIT_PATCHES,
                hard_negative_manifests: list[Path] | None = None,
                first_model_dir: Path | None = None) -> dict:
    """Bounded array-only fitting stage. This function never opens source images."""
    if not 4 <= max_fit_patches <= MAX_FIT_PATCHES or not 1 <= trees <= 64 or not 1 <= depth <= 14:
        raise ValueError("Fit limits: patches 4..12000, trees 1..64, depth 1..14")
    if out.exists() and any(path.name not in {"README.md", ".gitkeep"} for path in out.iterdir()):
        raise FileExistsError(f"Preserving nonempty model output directory: {out}")
    started = time.perf_counter()
    samples, config = load_merged_samples(paths) if isinstance(paths, Path) else load_samples(paths)
    base_digest = training_samples_digest(samples)
    mining, supplement, supplement_digests, mined_starts = None, None, [], {}
    if hard_negative_manifests is not None:
        if first_model_dir is None:
            raise ValueError("One-round refit requires first_model_dir")
        supplement, mining, supplement_digests = load_hard_negative_samples(
            hard_negative_manifests, config, Path(first_model_dir))
        if mining["training_samples_sha256"] != base_digest:
            raise ValueError("Hard-negative model was not fitted from these exact base sample arrays")
        for scale in SCALES:
            key = scale_key(scale)
            base = samples[f"{key}:fit"]
            mined_starts[key] = len(base["y"])
            samples[f"{key}:fit"] = {name: np.concatenate((values, supplement[key][name]))
                                      for name, values in base.items()}
    elif first_model_dir is not None:
        raise ValueError("first_model_dir is meaningful only with hard-negative manifests")
    kind_ids = config["kind_ids"]
    if max_fit_patches < len(kind_ids):
        raise ValueError("Fit cap must allow at least one example of each kind")
    rng = np.random.default_rng(seed)
    metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "patch_size": PATCH_SIZE, "kind_names": {str(v): k for k, v in kind_ids.items()},
                "thresholds": {}, "models": {}, "seed": seed, "trees": trees, "max_depth": depth,
                "max_fit_patches": max_fit_patches, "sample_config": config,
                "training_samples_sha256": training_samples_digest(samples),
                "base_training_samples_sha256": base_digest,
                "hard_negative_rounds": 1 if mining is not None else 0,
                "calibration": config.get("calibration", "whole"),
                "sample_counts": sample_counts(samples),
                "holdout_surface": config.get("holdout_surface"), "training_images": config["source_images"],
                "holdout_strategy": config.get("holdout_strategy", "surface" if config.get("holdout_surface") is not None else "image_quarter"),
                "probability_definition": "class-1 tree vote fraction after balanced training",
                "negative_sampling": "half high morphology/Gabor response, half random",
                "validation_sampling": "class-stratified random; inverse inclusion weighted metrics"}
    if mining is not None:
        metadata["negative_sampling"] = "half the balanced negative budget reserved for model-mined background; remaining response/random"
        metadata["hard_negative_mining"] = {**mining, "manifest_sha256": supplement_digests,
                                             "reserved_negative_fraction": .5,
                                             "n_added_by_scale": {key: len(data["y"]) for key, data in supplement.items()}}
    validation = {}
    out.mkdir(parents=True, exist_ok=True)
    for scale_index, scale in enumerate(SCALES):
        key = scale_key(scale)
        fit, held_out = samples[f"{key}:fit"], samples[f"{key}:validation"]
        x, y, kind = fit["x"], fit["y"], fit["kind"]
        positive = np.flatnonzero(y)
        positive = rng.choice(positive, min(len(positive), max_fit_patches // 2), replace=False)
        negative = _fit_negative_rows(x, y, len(positive), rng, mined_starts.get(key),
                                      supplement[key]["mining_score"] if supplement is not None else None)
        chosen = np.r_[positive, negative]
        rng.shuffle(chosen)
        binary = _forest(x[chosen], y[chosen], seed + scale_index, trees, depth)
        per_kind = [np.flatnonzero((y == 1) & (kind == kind_id)) for kind_id in kind_ids.values()]
        if any(len(indices) == 0 for indices in per_kind):
            raise ValueError(f"A kind has no fit patches at scale {key}")
        per_kind_count = min(max(map(len, per_kind)), max_fit_patches // len(per_kind))
        chosen_kind = np.concatenate([rng.choice(indices, per_kind_count, replace=len(indices) < per_kind_count)
                                      for indices in per_kind])
        classifier = _forest(x[chosen_kind], kind[chosen_kind], seed + 100 + scale_index, trees, depth)
        scores = binary_probabilities(binary, held_out["x"])
        metrics = validation_metrics(held_out["y"], scores, held_out["weight"])
        metrics.update(n_fit_positive=len(positive), n_fit_negative=len(negative), n_kind_fit=len(chosen_kind),
                       n_mined_negative_fit=int(np.sum(negative >= mined_starts[key])) if key in mined_starts else 0)
        validation[key] = metrics
        metadata["thresholds"][key] = metrics["precision_at_recall"]["0.8"]["threshold"]
        files = {"binary": f"scale_{key}_binary.yml.gz", "kind": f"scale_{key}_kind.yml.gz"}
        binary.save(str(out / files["binary"]))
        classifier.save(str(out / files["kind"]))
        metadata["models"][key] = files
        print(json.dumps({"stage": "fit", "scale": key, **metrics}), flush=True)
    metadata["elapsed_seconds"] = time.perf_counter() - started
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (out / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False))
    return {"validation": validation, "elapsed_seconds": metadata["elapsed_seconds"], "source_images_opened": 0}


def _fit_negative_rows(features: np.ndarray, truth: np.ndarray, count: int,
                        rng: np.random.Generator, mined_start: int | None = None,
                        mining_scores: np.ndarray | None = None) -> np.ndarray:
    if mined_start is None:
        return _negative_sample(features, np.flatnonzero(truth == 0), count, rng)
    mined = np.arange(mined_start, len(truth), dtype=np.int64)
    if mining_scores is None or len(mining_scores) != len(mined) or np.any(truth[mined] != 0):
        raise ValueError("Invalid appended hard-negative range")
    ranked = mined[np.argsort(-mining_scores, kind="stable")]
    reserve = min(len(ranked), (count + 1) // 2)
    background = np.flatnonzero(truth[:mined_start] == 0)
    base = _negative_sample(features, background, count - reserve, rng)
    additional = min(len(ranked) - reserve, count - reserve - len(base))
    return np.r_[ranked[:reserve], base, ranked[reserve:reserve + additional]]


def fit_merged_samples(paths: Path, out: Path, **kwargs) -> dict:
    """Named merged-archive entry point; identical one-round refit contract."""
    return fit_samples(paths, out, **kwargs)


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract-only", action="store_true", help="Sharded image-feature extraction; no fitting")
    mode.add_argument("--merge-samples", type=Path, nargs="+", help="Shard manifests/directories; fit without image reads")
    mode.add_argument("--fit-samples", type=Path, help="Fit from a merged NPZ and JSON sidecar only")
    parser.add_argument("--merge-only", action="store_true", help="With --merge-samples, write a merged archive without fitting")
    parser.add_argument("--merged-out", type=Path, default=Path("samples/merged_samples.npz"))
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--holdout-surface", default=None,
                        help="Explicit legacy surface holdout; default holds out every fourth image of each surface")
    parser.add_argument("--hard-negatives", action="store_true", help="Mine or refit exactly one hard-negative round")
    parser.add_argument("--model-dir", type=Path, help="Initial-fit model directory for one-round mining/refit binding")
    parser.add_argument("--hard-negative-fraction", type=float, default=.04)
    parser.add_argument("--hard-negative-manifests", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--calibration", choices=("sampled", "whole"), default="sampled",
                        help="whole is an expensive normalization ablation")
    parser.add_argument("--min-gt-pixels", type=int, default=3)
    parser.add_argument("--max-image-patches", type=int, default=1024)
    parser.add_argument("--max-fit-patches", type=int, default=MAX_FIT_PATCHES)
    args = parser.parse_args()
    if args.merge_only and args.merge_samples is None:
        parser.error("--merge-only requires --merge-samples")
    if args.hard_negatives and (args.model_dir is None or args.merge_only):
        parser.error("--hard-negatives requires --model-dir FIRST and is incompatible with --merge-only")
    if args.hard_negative_manifests and (not args.hard_negatives or args.extract_only):
        parser.error("--hard-negative-manifests requires a --hard-negatives fit stage")
    if args.extract_only:
        if args.bench is None or args.samples_dir is None:
            parser.error("--extract-only requires --bench and --samples-dir")
        result = extract_samples(args.bench, args.samples_dir, shard_index=args.shard_index,
                                 shard_count=args.shard_count, holdout_surface=args.holdout_surface,
                                 seed=args.seed, size=args.size, min_gt_pixels=args.min_gt_pixels,
                                 max_image_patches=args.max_image_patches, calibration=args.calibration,
                                 hard_negatives=args.hard_negatives, model_dir=args.model_dir,
                                 hard_negative_fraction=args.hard_negative_fraction)
    else:
        if args.bench is not None or args.samples_dir is not None:
            parser.error("--merge-samples does not accept --bench or --samples-dir")
        if args.merge_only:
            result = merge_samples(args.merge_samples, args.merged_out)
        else:
            if args.hard_negatives and not args.hard_negative_manifests:
                parser.error("Hard-negative refit requires --hard-negative-manifests")
            source = args.fit_samples if args.fit_samples is not None else args.merge_samples
            result = fit_samples(source, args.out, seed=args.seed, trees=args.trees,
                                 depth=args.max_depth, max_fit_patches=args.max_fit_patches,
                                 hard_negative_manifests=args.hard_negative_manifests,
                                 first_model_dir=args.model_dir if args.hard_negatives else None)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
