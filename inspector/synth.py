"""Deterministic, localized defect injection on photographed surface backgrounds.

Labels describe pixels actually changed by the renderer, before JPEG compression.
The PNG label map is lossless; a saved JPEG can add compression artefacts outside
those labels. These procedural defects are a controlled benchmark, not real-world
defect ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np


DEV_SURFACES = frozenset({
    "asphalt_floor", "brushed_concrete", "concrete", "concrete_floor_01",
    "granular_concrete", "dirty_concrete", "concrete_pavement",
})
TEST_SURFACES = frozenset({
    "brushed_concrete_2", "concrete_floor_02", "rough_concrete",
    "gravel_concrete", "concrete_slab_wall_02", "grooved_concrete_driveway",
    "ribbed_concrete_wall",
})
REAL_CRACK_SURFACES = frozenset({
    "cracked_concrete", "concrete_wall_005", "worn_concrete_floor",
})
KINDS = ("crack", "scratch", "spall", "stain")
KINDS_V2 = ("crack", "spall", "efflorescence", "rust_stain")


def _validate_image(img_bgr: np.ndarray) -> None:
    if (not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8
            or img_bgr.ndim != 3 or img_bgr.shape[2] != 3):
        raise ValueError("img_bgr must be a uint8 H x W x 3 BGR image")
    if min(img_bgr.shape[:2]) < 32:
        raise ValueError("Defect injection requires image dimensions >= 32 pixels")


def _midpoint_path(start: np.ndarray, end: np.ndarray, rng: np.random.Generator,
                   roughness: float, depth: int = 5) -> np.ndarray:
    """Midpoint displacement gives a multiscale, fractal-like crack centreline."""
    points = np.stack((start, end)).astype(np.float64)
    for level in range(depth):
        expanded = [points[0]]
        for left, right in zip(points[:-1], points[1:]):
            delta = right - left
            normal = np.array([-delta[1], delta[0]])
            normal /= max(float(np.linalg.norm(normal)), 1e-9)
            middle = (left + right) * 0.5
            middle += normal * rng.normal(0.0, roughness / (1.8 ** level))
            expanded.extend((middle, right))
        points = np.asarray(expanded)
    return points


def _line_template(kind: str, rng: np.random.Generator, scale: float,
                   difficulty: float, width_override: float | None = None
                   ) -> tuple[np.ndarray, dict]:
    length = max(12.0, float(rng.uniform(180, 650) * scale))
    angle = float(rng.uniform(0, 2 * math.pi))
    width = max(1.0, float(rng.uniform(1.5, 6.0) * scale * (1 - .60 * difficulty)))
    if width_override is not None:
        width = width_override
    start = np.array([0.0, 0.0])
    end = np.array([math.cos(angle), math.sin(angle)]) * length
    roughness = length * (.035 if kind == "crack" else .004)
    main = _midpoint_path(start, end, rng, roughness, depth=5)
    paths = [main]
    if kind == "crack":
        for _ in range(int(rng.integers(0, 3))):
            branch_start = main[int(rng.integers(7, len(main) - 7))]
            branch_angle = angle + float(rng.choice((-1, 1))) * rng.uniform(.45, 1.2)
            branch_length = length * rng.uniform(.12, .32)
            branch_end = branch_start + branch_length * np.array([
                math.cos(branch_angle), math.sin(branch_angle)])
            paths.append(_midpoint_path(branch_start, branch_end, rng,
                                        branch_length * .05, depth=3))
    all_points = np.concatenate(paths)
    margin = max(4, int(math.ceil(width * 2)))
    offset = -np.floor(all_points.min(axis=0)).astype(int) + margin
    shape = np.ceil(all_points.max(axis=0) - all_points.min(axis=0)).astype(int)
    mask = np.zeros((int(shape[1]) + 2 * margin + 1,
                     int(shape[0]) + 2 * margin + 1), dtype=np.uint8)
    total_length = 0.0
    for branch_index, path in enumerate(paths):
        path = path + offset
        total_length += float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        for index, (left, right) in enumerate(zip(path[:-1], path[1:])):
            taper = 1.0 - .65 * (index / max(len(path) - 2, 1)) ** 2
            branch_factor = 1.0 if branch_index == 0 else .65
            thickness = max(1, int(round(width * taper * branch_factor)))
            cv2.line(mask, tuple(np.rint(left).astype(int)),
                     tuple(np.rint(right).astype(int)), 255, thickness, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (3, 3), .45)
    alpha[alpha < .035] = 0
    contrast = float(rng.uniform(.35, .68) if kind == "crack"
                     else rng.uniform(.12, .30)) * (1 - .65 * difficulty)
    polarity = -1 if kind == "crack" else int(rng.choice((-1, 1)))
    return alpha, {
        "nominal_length_px": length, "centreline_length_px": total_length,
        "nominal_width_px": width, "angle_rad": angle, "branches": len(paths) - 1,
        "relative_contrast": contrast, "polarity": polarity, "blur_sigma": .45,
        "roughness_px": roughness,
        "paths_local": [np.round(path + offset, 3).tolist() for path in paths],
    }


def _blob_template(kind: str, rng: np.random.Generator, scale: float,
                   difficulty: float) -> tuple[np.ndarray, dict]:
    diameter = float(rng.uniform(20, 120) if kind == "spall"
                     else rng.uniform(60, 300)) * scale
    diameter = max(6 if kind == "spall" else 15, diameter)
    side = int(math.ceil(diameter)) + 12
    centre = (side - 1) / 2
    angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    radii = diameter / 2 * rng.uniform(.65, 1.0, len(angles))
    points = np.stack((centre + radii * np.cos(angles),
                       centre + radii * np.sin(angles)), axis=1)
    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    if kind == "spall":
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (3, 3), .5)
        contrast = float(rng.uniform(.22, .42) * (1 - .65 * difficulty))
    else:
        # Large blur and smoothly interpolated modulation suppress sharp edges.
        sigma = max(2., diameter * .12)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)
        modulation = cv2.resize(rng.uniform(.65, 1, (4, 4)).astype(np.float32),
                                (side, side), interpolation=cv2.INTER_CUBIC)
        alpha *= modulation
        contrast = float(rng.uniform(.06, .16) * (1 - .65 * difficulty))
    alpha[alpha < .04] = 0
    return alpha, {"diameter_px": diameter, "relative_contrast": contrast,
                   "polarity": -1, "vertices_local": np.round(points, 3).tolist()}


def inject_defects(img_bgr: np.ndarray, rng: np.random.Generator, n_defects: int,
                   kinds: Sequence[str] | None = None, difficulty: float = .35,
                   version: str = "v1"
                   ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return a new image, exact changed-pixel labels, and generation metadata.

    Widths and sizes are specified at 4096-pixel image scale and decrease on
    smaller inputs, subject to a one-pixel rasterization floor. Placement retries
    reject any support overlap. Excessively dense requests fail explicitly.
    The caller owns the random generator; no global random state is used.
    """
    if version == "v2":
        return _inject_defects_v2(img_bgr, rng, n_defects, kinds, difficulty)
    if version != "v1":
        raise ValueError("version must be 'v1' or 'v2'")
    kinds = KINDS if kinds is None else kinds
    _validate_image(img_bgr)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not isinstance(n_defects, int) or not 0 <= n_defects <= 65535:
        raise ValueError("n_defects must be an integer in [0, 65535]")
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must lie in [0, 1]")
    if not kinds or any(kind not in KINDS for kind in kinds):
        raise ValueError(f"kinds must be a non-empty sequence from {KINDS}")
    height, width = img_bgr.shape[:2]
    scale = min(height, width) / 4096.
    output = img_bgr.copy()
    labels = np.zeros((height, width), dtype=np.uint16)
    defects = []
    for defect_id in range(1, n_defects + 1):
        kind = str(rng.choice(kinds))
        for _ in range(200):
            alpha, params = (_line_template(kind, rng, scale, difficulty)
                             if kind in ("crack", "scratch")
                             else _blob_template(kind, rng, scale, difficulty))
            patch_h, patch_w = alpha.shape
            if patch_h > height or patch_w > width:
                continue
            x = int(rng.integers(0, width - patch_w + 1))
            y = int(rng.integers(0, height - patch_h + 1))
            support = alpha > 0
            label_crop = labels[y:y + patch_h, x:x + patch_w]
            if np.any(label_crop[support]):
                continue
            source = img_bgr[y:y + patch_h, x:x + patch_w]
            source_float = source.astype(np.float32)
            relative_contrast = params["relative_contrast"]
            if kind == "spall":
                donor_x = int(rng.integers(0, width - patch_w + 1))
                donor_y = int(rng.integers(0, height - patch_h + 1))
                donor = img_bgr[donor_y:donor_y + patch_h,
                                donor_x:donor_x + patch_w].astype(np.float32)
                donor -= donor.mean(axis=(0, 1), keepdims=True)
                target = source_float * (1 - relative_contrast) + donor * .3
                rim = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT,
                                       np.ones((3, 3), dtype=np.uint8)).astype(bool)
                yy, xx = np.indices(alpha.shape)
                rim &= (xx + yy < (patch_w + patch_h) / 2) & support
                target[rim] = source_float[rim] + (255 - source_float[rim]) * .22
                params["texture_source_xy"] = [donor_x, donor_y]
                params["bright_rim_side"] = "upper_left"
            else:
                if params["polarity"] < 0:
                    target = source_float * (1 - relative_contrast)
                else:
                    target = source_float + (255 - source_float) * relative_contrast
                if kind == "stain":
                    tint = rng.uniform(.85, 1.15, 3).astype(np.float32)
                    target = source_float * (1 - relative_contrast * tint)
                    params["bgr_tint"] = tint.tolist()
            highpass = source_float - cv2.GaussianBlur(source_float, (3, 3), .8)
            noise_sigma = float(np.clip(np.std(highpass) * .08, .2, 2.0))
            noise = rng.normal(0, noise_sigma, alpha.shape).astype(np.float32)
            mixed = source_float + alpha[..., None] * (
                target - source_float + noise[..., None])
            rendered = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
            # Quantization can erase faint boundary changes. Exclude those
            # pixels so every positive label is an actual changed pixel.
            changed = support & np.any(rendered != source, axis=2)
            if not np.any(changed):
                continue
            crop = output[y:y + patch_h, x:x + patch_w]
            crop[changed] = rendered[changed]
            label_crop[changed] = defect_id
            ys, xs = np.nonzero(changed)
            area = int(changed.sum())
            length = float(params.get("centreline_length_px", 0.0))
            original_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
            rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)
            contrast = float(np.mean(np.abs(rendered_gray[changed]
                                             - original_gray[changed])) / 255)
            params.update({"generator_version": 1, "difficulty": float(difficulty),
                           "reference_size": 4096, "image_scale": scale,
                           "noise_sigma": noise_sigma, "patch_xy": [x, y]})
            defects.append({
                "id": defect_id, "kind": kind,
                "bbox": [x + int(xs.min()), y + int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "area_px": area, "length_px": length,
                "mean_width_px": float(area / length) if length > 0 else 0.0,
                "contrast": contrast, "parameters": params,
            })
            break
        else:
            raise ValueError(f"Could not place defect {defect_id} without overlap; "
                             "reduce n_defects or use a larger image")
    return output, labels, defects


def make_benchmark(surface_paths: Iterable[str | Path], out_dir: str | Path,
                   per_surface: int = 4, seed: int = 20261023, size: int = 4096,
                   version: str = "v1", split_namespace: str = "eval",
                   shard_index: int = 0, shard_count: int = 1,
                   on_image_written: Callable[[Path, Path, dict], None] | None = None) -> dict:
    """Write sequential examples without keeping multiple surface pipelines.

    Only the explicitly listed 14 clean surfaces are accepted. Real-crack and
    unknown backgrounds raise an error rather than silently tainting a split.
    Selected subsets are supported, while split membership remains surface based.
    The returned object is identical to the JSON manifest written to disk.
    Only backgrounds assigned to this shard must exist locally. The optional
    callback receives (image_path, label_path, entry) after each pair is saved
    and appended to the manifest. It may upload and unlink the files immediately;
    it may add metadata such as uploaded-file hashes to the same entry object.
    Callback failures propagate to the caller.
    """
    if not isinstance(per_surface, int) or per_surface < 1:
        raise ValueError("per_surface must be a positive integer")
    if not isinstance(size, int) or size < 32:
        raise ValueError("size must be an integer >= 32")
    if version not in ("v1", "v2"):
        raise ValueError("version must be 'v1' or 'v2'")
    if split_namespace not in ("eval", "train"):
        raise ValueError("split_namespace must be 'eval' or 'train'")
    if (type(shard_count) is not int or type(shard_index) is not int
            or shard_count < 1 or not 0 <= shard_index < shard_count):
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    if on_image_written is not None and not callable(on_image_written):
        raise TypeError("on_image_written must be callable or None")
    paths = sorted((Path(path) for path in surface_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("No surface paths were supplied")
    names = [path.stem for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Surface names must be unique")
    for path in paths:
        if path.stem in REAL_CRACK_SURFACES:
            raise ValueError(f"{path.stem} contains real cracks and is demo-only")
        if path.stem not in DEV_SURFACES | TEST_SURFACES:
            raise ValueError(f"Unknown clean benchmark surface: {path.stem}")
        if split_namespace == "train" and path.stem not in DEV_SURFACES:
            raise ValueError(f"Training may only use dev backgrounds: {path.stem}")
    for surface_index, path in enumerate(paths):
        if (assigned_sample_indices(surface_index, per_surface, shard_index, shard_count)
                and not path.is_file()):
            raise FileNotFoundError(path)
    destination = Path(out_dir)
    if shard_count > 1:
        destination /= f"shard_{shard_index:04d}_of_{shard_count:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of an existing dataset or partial run.
    if (destination / "manifest.json").exists() or any(destination.glob("img_*.jpg")):
        raise FileExistsError(f"Benchmark output already contains examples: {destination}")
    version_number = 1 if version == "v1" else 2
    manifest = {"version": version_number, "seed": int(seed), "size": size,
                "generator": f"localized_procedural_{version}",
                "namespace": split_namespace, "per_surface": per_surface,
                "surfaces": names, "n_images_total": len(paths) * per_surface,
                "shard_index": shard_index, "shard_count": shard_count,
                "label_semantics": "changed pixels before JPEG compression",
                "generator_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy_version": str(np.__version__), "opencv_version": str(cv2.__version__),
                "images": []}
    if version == "v2":
        manifest["snr_definition"] = (
            "mean absolute post-quantization grayscale difference on label pixels / "
            "max(3, original full-resolution local grayscale standard deviation)")
    for surface_index, path in enumerate(paths):
        selected_samples = assigned_sample_indices(surface_index, per_surface, shard_index, shard_count)
        if not selected_samples:
            continue
        background = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if background is None:
            raise ValueError(f"Cannot decode surface image: {path}")
        if background.shape[:2] != (size, size):
            background = cv2.resize(background, (size, size), interpolation=cv2.INTER_AREA)
        for sample_index in selected_samples:
            # Surface-name entropy makes each example independent of the input
            # subset and its ordering, without Python's randomized hash().
            image_seed = _example_seed(path.stem, sample_index, seed, version_number,
                                       split_namespace)
            rng = np.random.default_rng(image_seed)
            n_defects = int(rng.integers(5, 10) if version == "v1"
                            else rng.integers(3, 9))
            difficulty = float(rng.uniform(.15, .80))
            image, labels, defects = inject_defects(background, rng, n_defects,
                                                    difficulty=difficulty,
                                                    version=version)
            index = surface_index * per_surface + sample_index
            image_name = f"img_{index:04d}.jpg"
            label_name = f"img_{index:04d}_labels.png"
            if not cv2.imwrite(str(destination / image_name), image,
                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Cannot write {image_name}")
            if not cv2.imwrite(str(destination / label_name), labels):
                raise OSError(f"Cannot write {label_name}")
            manifest["images"].append({
                "image": image_name, "labels": label_name, "surface": path.stem,
                "global_index": index, "sample_index": sample_index,
                "split": ("train" if split_namespace == "train" else
                          "dev" if path.stem in DEV_SURFACES else "test"),
                "seed": image_seed, "defects": defects, "width": size, "height": size,
                "difficulty": difficulty, "n_defects": n_defects,
            })
            # Upload callbacks need files and metadata only. Release the large
            # rendered arrays before any SDK buffering or transfer begins.
            del image, labels
            if on_image_written is not None:
                on_image_written(destination / image_name, destination / label_name,
                                 manifest["images"][-1])
        del background
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    return manifest


def make_training_benchmark(surface_paths: Iterable[str | Path], out_dir: str | Path,
                            images_per_surface: int, seed: int = 20261023,
                            size: int = 4096, version: str = "v2",
                            shard_index: int = 0, shard_count: int = 1,
                            on_image_written: Callable[[Path, Path, dict], None] | None = None) -> dict:
    """Generate N *per dev surface*, isolated from evaluation seed streams.

    Test surfaces are rejected rather than silently filtered. The script selects
    dev paths before calling this helper; callers can deliberately pass a subset.
    """
    return make_benchmark(surface_paths, out_dir, images_per_surface, seed, size,
                          version=version, split_namespace="train",
                          shard_index=shard_index, shard_count=shard_count,
                          on_image_written=on_image_written)


def assigned_sample_indices(surface_index: int, per_surface: int,
                             shard_index: int = 0, shard_count: int = 1) -> list[int]:
    """List image samples needed from one surface in the globally sorted list.

    Callers can use this before downloading source photos. Pass the *full*
    surface list to make_benchmark so global filenames and seeds remain stable.
    """
    if (type(surface_index) is not int or surface_index < 0 or type(per_surface) is not int
            or per_surface < 1 or type(shard_count) is not int or shard_count < 1
            or type(shard_index) is not int or not 0 <= shard_index < shard_count):
        raise ValueError("Invalid surface, sample count, or shard indices")
    return [sample for sample in range(per_surface)
            if (surface_index * per_surface + sample) % shard_count == shard_index]


def _example_seed(surface: str, sample_index: int, seed: int, version: int,
                   namespace: str) -> int:
    entropy = [int(seed), sample_index, *surface.encode("utf-8")]
    if version == 2 or namespace == "train":
        entropy = [version, 2718 if namespace == "eval" else 314159, *entropy]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def merge_benchmark_shards(manifest_paths: Iterable[str | Path], out_dir: str | Path,
                            verify_images: bool = True) -> dict:
    """Join complete shard manifests without decoding or copying image data.

    Paths in the new manifest are relative references to the original shards;
    keep those shard directories alongside the merged manifest. All shards must
    share an identical generation configuration and together cover every index.
    Set verify_images=False to merge metadata mirrored from S3 without downloading
    its images. Relative references then preserve the mirrored shard layout.
    """
    if type(verify_images) is not bool:
        raise ValueError("verify_images must be a bool")
    paths = [Path(path) / "manifest.json" if Path(path).is_dir() else Path(path)
             for path in manifest_paths]
    if not paths:
        raise ValueError("No shard manifests supplied")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = manifests[0]
    required = {"surfaces", "n_images_total", "shard_index", "shard_count",
                "namespace", "version", "seed", "size", "per_surface", "images"}
    if any(not required <= manifest.keys() for manifest in manifests):
        raise ValueError("A manifest lacks sharding configuration")
    config = {key: value for key, value in first.items() if key not in ("images", "shard_index")}
    count = first["shard_count"]
    if type(count) is not int or count < 1:
        raise ValueError("Invalid shard_count")
    surfaces = first["surfaces"]
    if (not surfaces or surfaces != sorted(set(surfaces))
            or any(surface not in DEV_SURFACES | TEST_SURFACES for surface in surfaces)):
        raise ValueError("Invalid clean surface list")
    if first["namespace"] not in ("eval", "train") or first["version"] not in (1, 2):
        raise ValueError("Invalid version or namespace")
    if first["namespace"] == "train" and any(surface not in DEV_SURFACES for surface in surfaces):
        raise ValueError("Training shards may only use dev backgrounds")
    per_surface = first["per_surface"]
    if type(per_surface) is not int or per_surface < 1:
        raise ValueError("Invalid per_surface")
    total = len(surfaces) * per_surface
    if first["n_images_total"] != total:
        raise ValueError("Global image count does not match surfaces and per_surface")
    destination = Path(out_dir)
    combined, seen_indices, seen_shards = [], set(), set()
    for path, manifest in zip(paths, manifests):
        candidate_config = {key: value for key, value in manifest.items()
                            if key not in ("images", "shard_index")}
        if candidate_config != config:
            raise ValueError("Shard generation configurations differ")
        shard = manifest["shard_index"]
        if type(shard) is not int or not 0 <= shard < count or shard in seen_shards:
            raise ValueError("Invalid or duplicate shard index")
        seen_shards.add(shard)
        for entry in manifest["images"]:
            index = entry.get("global_index")
            if (type(index) is not int or not 0 <= index < total
                    or index % count != shard or index in seen_indices):
                raise ValueError("Invalid or duplicate global image index")
            surface_index, sample_index = divmod(index, per_surface)
            surface = surfaces[surface_index]
            expected_split = ("train" if first["namespace"] == "train" else
                              "dev" if surface in DEV_SURFACES else "test")
            expected_seed = _example_seed(surface, sample_index, first["seed"],
                                           first["version"], first["namespace"])
            if (entry.get("surface") != surface or entry.get("sample_index") != sample_index
                    or entry.get("split") != expected_split or entry.get("seed") != expected_seed
                    or entry.get("width") != first["size"] or entry.get("height") != first["size"]):
                raise ValueError("Shard entry surface, split, size, or seed is inconsistent")
            converted = dict(entry)
            for key, expected_name in (("image", f"img_{index:04d}.jpg"),
                                       ("labels", f"img_{index:04d}_labels.png")):
                raw_path = entry.get(key)
                if (not isinstance(raw_path, str) or not raw_path
                        or "\\" in raw_path or ":" in raw_path or "\x00" in raw_path
                        or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts
                        or (not verify_images and raw_path != expected_name)
                        or Path(raw_path).name != expected_name):
                    raise ValueError(f"Unsafe or incorrectly named shard {key} path")
                referenced = path.parent / raw_path
                if referenced.is_symlink():
                    raise ValueError(f"Unsafe symlink in shard {key} path")
                if verify_images and not referenced.is_file():
                    raise ValueError(f"Missing or incorrectly named shard {key} file")
                reference = referenced.resolve() if verify_images else referenced.absolute()
                base = destination.resolve() if verify_images else destination.absolute()
                converted[key] = os.path.relpath(reference, base)
            combined.append(converted)
            seen_indices.add(index)
    if seen_shards != set(range(count)) or seen_indices != set(range(total)):
        raise ValueError("Shards are incomplete; all shard and image indices are required")
    result = dict(config)
    result.update({"shard_index": 0, "shard_count": 1,
                   "source_shard_count": count,
                   "images": sorted(combined, key=lambda entry: entry["global_index"])})
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "manifest.json"
    if output.exists():
        raise FileExistsError(f"Merged manifest already exists: {output}")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _deposit_template(kind: str, rng: np.random.Generator, scale: float
                      ) -> tuple[np.ndarray, dict]:
    """Chalk deposits and gravity-oriented rust trails have irregular soft edges."""
    diameter = max(10., float(rng.uniform(90, 270) * scale))
    side = int(math.ceil(diameter)) + 20
    mask = np.zeros((side, side), dtype=np.float32)
    center = (side - 1) / 2
    if kind == "efflorescence":
        # Overlapping lobes produce an incomplete, powdery mineral deposit.
        lobes = int(rng.integers(4, 10))
        for _ in range(lobes):
            location = tuple(np.rint(rng.normal(center, diameter * .15, 2)).astype(int))
            axes = tuple(np.maximum(2, np.rint(rng.uniform(.12, .28, 2) * diameter)).astype(int))
            cv2.ellipse(mask, location, axes, float(rng.uniform(0, 180)), 0, 360,
                        float(rng.uniform(.6, 1)), -1, cv2.LINE_AA)
        sigma = max(.7, diameter * .025)
        params = {"lobes": lobes, "achromatic": True}
    else:
        # Deposits collect under an origin and follow several short drips.
        drips = int(rng.integers(2, 5))
        top = max(5, int(center - diameter * .35))
        for _ in range(drips):
            start_x = int(rng.normal(center, diameter * .13))
            end_y = min(side - 5, top + int(rng.uniform(.4, .85) * diameter))
            points = np.array([[start_x, top],
                               [start_x + rng.normal(0, diameter * .035), (top + end_y) / 2],
                               [start_x + rng.normal(0, diameter * .05), end_y]])
            cv2.polylines(mask, [np.rint(points).astype(np.int32)], False,
                          float(rng.uniform(.65, 1)), max(2, int(diameter * rng.uniform(.04, .12))),
                          cv2.LINE_AA)
        cv2.ellipse(mask, (int(center), top), (max(3, int(diameter * .25)),
                    max(2, int(diameter * .1))), 0, 0, 360, 1., -1)
        sigma = max(.8, diameter * .035)
        params = {"drips": drips, "gravity_direction": [0, 1]}
    alpha = cv2.GaussianBlur(mask, (0, 0), sigma)
    texture = cv2.resize(rng.uniform(.75, 1., (6, 6)).astype(np.float32),
                         (side, side), interpolation=cv2.INTER_CUBIC)
    alpha *= texture
    alpha[alpha < .12] = 0
    params.update({"diameter_px": diameter, "blur_sigma": sigma})
    return alpha, params


def _visible_scale(kind: str, area: int, width: float) -> float:
    for scale in (.25, .5, 1.):
        if (width * scale >= .8 if kind == "crack" else area * scale ** 2 >= 40):
            return scale
    return 1.


def _inject_defects_v2(img_bgr: np.ndarray, rng: np.random.Generator,
                       n_defects: int, kinds: Sequence[str] | None,
                       difficulty: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Render v2 with measured post-injection SNR >= 1.3, or fail explicitly.

    Local texture noise is the original grayscale standard deviation in a padded
    full-resolution rectangle. A 3-gray-level floor avoids unbounded SNR on flat
    backgrounds. The denominator is fixed before rendering; it is never reduced
    to pass the acceptance test. Candidate amplitude is adjusted, then placement
    is retried if clipping makes the visibility requirement unattainable.
    """
    _validate_image(img_bgr)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not isinstance(n_defects, int) or not 0 <= n_defects <= 65535:
        raise ValueError("n_defects must be an integer in [0, 65535]")
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must lie in [0, 1]")
    kinds = KINDS_V2 if kinds is None else tuple(kinds)
    if not kinds or any(kind not in KINDS_V2 for kind in kinds):
        raise ValueError(f"v2 kinds must be a non-empty sequence from {KINDS_V2}")
    weights = np.array([.5 if kind == "crack" else 1 / 6 for kind in kinds])
    weights /= weights.sum()
    height, width = img_bgr.shape[:2]
    image_scale = min(height, width) / 4096.
    output = img_bgr.copy()
    labels = np.zeros((height, width), dtype=np.uint16)
    defects = []
    for defect_id in range(1, n_defects + 1):
        kind = str(rng.choice(kinds, p=weights))
        for placement_attempt in range(250):
            if kind == "crack":
                width_4096 = float(rng.uniform(1.5, 6.))
                alpha, params = _line_template("crack", rng, image_scale, 0.,
                                               width_override=width_4096 * image_scale)
                params["nominal_width_4096"] = width_4096
                # Discard the most transparent blur fringe before measuring SNR.
                alpha[alpha < .12] = 0
            elif kind == "spall":
                alpha, params = _blob_template("spall", rng, image_scale, 0.)
                alpha[alpha < .12] = 0
            else:
                alpha, params = _deposit_template(kind, rng, image_scale)
            patch_h, patch_w = alpha.shape
            if patch_h > height or patch_w > width:
                continue
            x = int(rng.integers(0, width - patch_w + 1))
            y = int(rng.integers(0, height - patch_h + 1))
            placement_relation = {"type": "random"}
            if kind == "efflorescence":
                # Moisture can carry mineral deposits down from an existing
                # fissure. Prefer a nearby location below one when space allows.
                cracks = [defect for defect in defects if defect["kind"] == "crack"]
                if cracks and rng.random() < .7:
                    crack = cracks[int(rng.integers(len(cracks)))]
                    cx, cy, cw, ch = crack["bbox"]
                    gap = int(rng.integers(2, max(3, int(80 * image_scale))))
                    proposed_y = cy + ch + gap
                    if proposed_y + patch_h <= height:
                        center_x = cx + cw * .5 + rng.normal(0, max(2., cw * .2))
                        x = int(np.clip(round(center_x - patch_w / 2), 0, width - patch_w))
                        y = proposed_y
                        placement_relation = {"type": "below_crack", "defect_id": crack["id"]}
            params["placement_relation"] = placement_relation
            support = alpha > 0
            label_crop = labels[y:y + patch_h, x:x + patch_w]
            if not np.any(support) or np.any(label_crop[support]):
                continue
            source = img_bgr[y:y + patch_h, x:x + patch_w]
            source_float = source.astype(np.float32)
            original_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
            padding = max(16, int(max(patch_h, patch_w) * .15))
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1, y1 = min(width, x + patch_w + padding), min(height, y + patch_h + padding)
            neighborhood = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            local_std = float(np.std(neighborhood, dtype=np.float64))
            denominator = max(3., local_std)
            target_k = float(np.clip(4. - 2.4 * difficulty + rng.uniform(-.15, .15), 1.6, 4.))
            amplitude = target_k * denominator / max(float(alpha[support].mean()), .15)
            noise = rng.normal(0, min(2., denominator * .035), alpha.shape).astype(np.float32)
            detail = np.zeros_like(source_float)
            if kind == "spall":
                donor_x = int(rng.integers(0, width - patch_w + 1))
                donor_y = int(rng.integers(0, height - patch_h + 1))
                donor = img_bgr[donor_y:donor_y + patch_h,
                                donor_x:donor_x + patch_w].astype(np.float32)
                coarsening = float(rng.uniform(3., 5.))
                coarse_size = (max(2, int(round(patch_w / coarsening))),
                               max(2, int(round(patch_h / coarsening))))
                coarse_texture = cv2.resize(donor, coarse_size, interpolation=cv2.INTER_AREA)
                coarse_texture = cv2.resize(coarse_texture, (patch_w, patch_h),
                                             interpolation=cv2.INTER_CUBIC)
                centered = coarse_texture - coarse_texture.mean(axis=(0, 1), keepdims=True)
                texture_gain = max(2., denominator * .6) / max(float(np.std(centered)), 1.)
                detail = centered * texture_gain
                params["texture_source_xy"] = [donor_x, donor_y]
                params["texture_coarsening"] = coarsening
                params["texture_gain"] = float(texture_gain)
            if kind == "efflorescence":
                direction = np.ones(3, dtype=np.float32)
            elif kind == "rust_stain":
                # BGR deltas retain orange-brown chroma while luminance drops.
                direction = np.array([-1.25, -.85, -.25], dtype=np.float32)
            else:
                direction = -np.ones(3, dtype=np.float32)
            snr = 0.
            for adjustment in range(7):
                delta = amplitude * direction + detail + noise[..., None]
                if kind == "spall":
                    rim = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT,
                                          np.ones((3, 3), dtype=np.uint8)).astype(bool)
                    yy, xx = np.indices(alpha.shape)
                    rim &= support & (xx + yy < (patch_w + patch_h) / 2)
                    delta[rim] = -amplitude * 1.65 + detail[rim] * .3
                    params["shadow_rim_side"] = "upper_left"
                mixed = source_float + alpha[..., None] * delta
                rendered = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
                changed = support & np.any(rendered != source, axis=2)
                if not np.any(changed):
                    amplitude *= 1.5
                    continue
                rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)
                mean_difference = float(np.mean(np.abs(rendered_gray[changed]
                                                       - original_gray[changed])))
                snr = mean_difference / denominator
                if snr >= 1.3:
                    break
                amplitude *= max(1.25, 1.4 / max(snr, .1))
            if snr < 1.3:
                continue
            crop = output[y:y + patch_h, x:x + patch_w]
            crop[changed] = rendered[changed]
            label_crop[changed] = defect_id
            ys, xs = np.nonzero(changed)
            area = int(changed.sum())
            length = float(params.get("centreline_length_px", 0.))
            nominal_width = float(params.get("nominal_width_px", 0.))
            visible_at = _visible_scale(kind, area, nominal_width)
            params.update({
                "generator_version": 2, "difficulty": float(difficulty),
                "reference_size": 4096, "image_scale": image_scale,
                "patch_xy": [x, y], "local_statistics_rect": [x0, y0, x1 - x0, y1 - y0],
                "local_fullres_std": local_std, "std_floor": 3.,
                "snr_denominator": denominator, "target_k": target_k,
                "actual_mean_luminance_difference": mean_difference,
                "amplitude": float(amplitude), "amplitude_adjustments": adjustment,
                "placement_attempt": placement_attempt,
                "noise_sigma": min(2., denominator * .035),
            })
            defects.append({
                "id": defect_id, "kind": kind,
                "bbox": [x + int(xs.min()), y + int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "area_px": area, "length_px": length,
                "mean_width_px": float(area / length) if length > 0 else 0.,
                "contrast": mean_difference / 255., "snr": float(snr),
                "visible_at": visible_at, "params": params, "parameters": params,
            })
            break
        else:
            raise ValueError(f"Cannot place v2 {kind} defect {defect_id} with SNR >= 1.3; "
                             "the background may be too dark, bright, noisy, or crowded")
    return output, labels, defects
