"""Streaming evaluation, deterministic image shards, and metadata-only merging.

No inference runs in --merge mode. Learned comparisons use all selected
policies; classic is a reference for agent/exhaustive only in v2.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from inspector.agent import REPORTING_VERSION, ActiveInspector, InspectorConfig
from inspector.baselines import run_baseline
from inspector.evaluate import METRICS_VERSION, MATCHING_VERSION, evaluate_result, write_summaries
from inspector.tools import LocalBackend

APP_ROOT = Path(__file__).resolve().parents[1]
SHARD_SCHEMA = "inspection-eval-shard-v5"


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def model_fingerprint(model_dir: Path) -> dict:
    """Fingerprint metadata/referenced files without loading or running forests."""
    metadata = json.loads((model_dir / "metadata.json").read_text())
    names = {"metadata.json"}
    for files in metadata.get("models", {}).values():
        names.update(str(filename) for filename in files.values())
    digests = {}
    for name in sorted(names):
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir.resolve()):
            raise ValueError("model metadata references a file outside model_dir")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[name] = digest.hexdigest()
    return digests


def app_output(path: Path) -> Path:
    output = path.resolve()
    if not output.is_relative_to(APP_ROOT):
        raise ValueError("output must be inside app/")
    return output


def shard_entries(entries: list, shard_index: int, shard_count: int) -> list:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("require shard_count >=1 and 0 <= shard_index < shard_count")
    return entries[shard_index::shard_count]


def load_manifest(bench: Path, split: str, limit: int | None,
                  per_surface_limit: int | None = None) -> list[dict]:
    manifest = json.loads((bench / "manifest.json").read_text())
    entries = [item for item in manifest["images"] if item["split"] == split]
    if per_surface_limit is not None:
        if per_surface_limit < 1:
            raise ValueError("per_surface_limit must be positive")
        counts: dict[str, int] = {}
        selected = []
        for entry in entries:
            surface = entry["surface"]
            if counts.get(surface, 0) < per_surface_limit:
                selected.append(entry)
                counts[surface] = counts.get(surface, 0) + 1
        entries = selected
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError(f"No {split} images selected")
    images = [entry["image"] for entry in entries]
    if len(images) != len(set(images)):
        raise ValueError("manifest selection contains duplicate image IDs")
    return entries


def resized_truth(bench: Path, entry: dict, shape: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    labels = cv2.imread(str(bench / entry["labels"]), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.dtype != np.uint16 or labels.ndim != 2:
        raise ValueError(f"Missing or invalid uint16 labels for {entry['image']}")
    source_shape = labels.shape
    ratio_x, ratio_y = shape[1] / source_shape[1], shape[0] / source_shape[0]
    linear_ratio = min(ratio_x, ratio_y)
    if labels.shape != shape:
        labels = cv2.resize(labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    defects = []
    for original in entry["defects"]:
        defect = dict(original)
        ys, xs = np.nonzero(labels == defect["id"])
        defect["area_px"] = int(xs.size)
        defect["bbox"] = ([int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1),
                           int(ys.max()-ys.min()+1)] if xs.size else [0, 0, 0, 0])
        if source_shape != shape:
            for key in ("length_px", "mean_width_px"):
                if key in defect:
                    defect[key] = float(defect[key]) * linear_ratio
            for parameter_key in ("params", "parameters"):
                if parameter_key not in original:
                    continue
                parameters = dict(original[parameter_key])
                for key, value in list(parameters.items()):
                    if key.endswith("_px") and isinstance(value, (int, float)):
                        parameters[key] = float(value) * linear_ratio
                if "blur_sigma" in parameters:
                    parameters["blur_sigma"] = float(parameters["blur_sigma"]) * linear_ratio
                if "image_scale" in parameters:
                    parameters["image_scale"] = float(parameters["image_scale"]) * linear_ratio
                if "paths_local" in parameters:
                    parameters["paths_local"] = [[[float(x) * ratio_x, float(y) * ratio_y] for x, y in path]
                                                  for path in parameters["paths_local"]]
                if "patch_xy" in parameters:
                    x, y = parameters["patch_xy"]
                    parameters["patch_xy"] = [x * ratio_x, y * ratio_y]
                if "local_statistics_rect" in parameters:
                    x, y, w, h = parameters["local_statistics_rect"]
                    parameters["local_statistics_rect"] = [x * ratio_x, y * ratio_y, w * ratio_x, h * ratio_y]
                parameters["analysis_resize_ratio"] = [ratio_x, ratio_y]
                defect[parameter_key] = parameters
            # Reapply v2's geometric visibility criterion in analysis coordinates.
            # Legacy manifests without visible_at remain unstratified.
            if "visible_at" in original:
                defect["source_visible_at"] = original["visible_at"]
                parameters = defect.get("params", defect.get("parameters", {}))
                width = float(parameters.get("nominal_width_px", defect.get("mean_width_px", 0)))
                defect["visible_at"] = 1.0
                for scale in (0.25, 0.5, 1.0):
                    visible = width * scale >= 0.8 if defect["kind"] == "crack" else xs.size * scale**2 >= 40
                    if visible:
                        defect["visible_at"] = scale
                        break
        defects.append(defect)
    return labels, defects


def evaluate_entries(bench: Path, entries: list[dict], methods: list[str], budgets: list[float],
                     size: int, config: InspectorConfig, seed: int = 0,
                     save_traces: Path | None = None, *, detector: str = "classic",
                     model_dir: Path | None = None,
                     random_seeds: list[int] | None = None,
                     backend: LocalBackend | None = None, calibration: str = "sampled") -> list[dict]:
    """Evaluate one detector, preserving the original positional call contract.

    Prepare once per image outside policy timers. Every standalone method still
    pays full logical coarse/calibration cost. Policy time plus shared preparation
    time is explicitly an uncached estimate, not a separately measured execution.
    """
    records = []
    backend = backend or LocalBackend(max_size=size, detector=detector, model_dir=model_dir,
                                      calibration=calibration)
    for entry in entries:
        image_ref = str(bench / entry["image"])
        labels, defects = resized_truth(bench, entry, backend.image(image_ref).shape[:2])
        prepared = backend.coarse_scan(image_ref, grid=config.grid, work_width=config.work_width)
        for method in methods:
            seeds = random_seeds if method == "random" and random_seeds is not None else [seed]
            for run_seed in seeds:
                exhaustive_result = None
                for budget in budgets:
                    backend.reset_accounting()
                    if method == "exhaustive" and exhaustive_result is not None:
                        result = exhaustive_result
                    elif method == "agent":
                        result = ActiveInspector(backend, budget, config).run(image_ref)
                    else:
                        result = run_baseline(method, backend, image_ref, budget,
                                              seed=run_seed + int(entry.get("seed", 0)), config=config)
                        if method == "exhaustive":
                            exhaustive_result = result
                    result = {**result, "preparation_elapsed_ms": prepared.get("elapsed_ms", 0),
                              "preparation_included_in_elapsed": False,
                              "calibration_cost_px": prepared.get("calibration_cost_px", 0)}
                    metrics = evaluate_result(result, labels, defects)
                    records.append({"detector": detector, "method": method, "seed": run_seed,
                                    "budget": budget, "image": entry["image"],
                                    "surface": entry["surface"], "split": entry["split"], **metrics})
                    if save_traces is not None:
                        save_traces.mkdir(parents=True, exist_ok=True)
                        token = hashlib.sha256(entry["image"].encode()).hexdigest()[:8]
                        name = f"{Path(entry['image']).stem}_{token}_{detector}_{method}_s{run_seed}_{budget:g}.json"
                        (save_traces / name).write_text(json.dumps(result, indent=2, allow_nan=False))
        del labels, prepared
    return records


def tune_on_dev(bench: Path, *, limit: int = 2, size: int = 2048,
                budget: float = 0.1, base: InspectorConfig | None = None,
                detector: str = "classic", model_dir: Path | None = None,
                out_dir: Path | None = None) -> tuple[InspectorConfig | None, list[dict]]:
    """Select policy settings on dev only; never fit or select on test images."""
    entries = load_manifest(bench, "dev", limit)
    base = base or InspectorConfig()
    history = []
    for candidate in tuning_candidates(base, detector):
        config = InspectorConfig(**candidate["config"])
        # Keep the six-positional-argument seam used by existing test fixtures.
        kwargs = {} if detector == "classic" and model_dir is None else {
            "detector": detector, "model_dir": model_dir}
        records = evaluate_entries(bench, entries, ["agent"], [budget], size, config, **kwargs)
        history.append(_candidate_tradeoff(candidate, records, detector))
    best, status = _select_tradeoff(history)
    report = _tuning_report(history, best, status, detector, [entry["image"] for entry in entries], [budget])
    write_tuning_report(report, app_output(out_dir or APP_ROOT / "artifacts/tuning_dev"))
    return InspectorConfig(**best["config"]) if best else None, history


MAX_FP_PER_IMAGE = 10.0  # 2026-10-24: chosen from the dev trade-off table only (5 removed every usable dev operating point)
FALLBACK_MAX_FP_PER_IMAGE = 50.0
TUNING_OBJECTIVE_TEXT = (f"highest mean image recall among candidates with <= {MAX_FP_PER_IMAGE:g} "
                         f"FP/image; otherwise highest mean image recall with <= {FALLBACK_MAX_FP_PER_IMAGE:g} "
                         "FP/image; otherwise no feasible candidate")


def tuning_objective(recall: float, false_positives_per_image: float) -> float | None:
    """Encode primary/fallback tiers; candidates above both limits are unranked."""
    if false_positives_per_image <= MAX_FP_PER_IMAGE:
        return 2.0 + float(recall)
    if false_positives_per_image <= FALLBACK_MAX_FP_PER_IMAGE:
        return float(recall)
    return None


def _candidate_tradeoff(candidate: dict, rows: list[dict], detector: str) -> dict:
    if not rows:
        raise ValueError("each tuning candidate requires dev observations")
    recalls = [row.get("recall") for row in rows if row.get("recall") is not None]
    recall = float(np.mean(recalls)) if recalls else 0.0
    fp = float(np.mean([row["false_positives_per_image"] if "false_positives_per_image" in row else
                        row["false_positives"] / row.get("n_images", 1) for row in rows]))
    if not np.isfinite(recall) or not 0 <= recall <= 1 or not np.isfinite(fp) or fp < 0:
        raise ValueError("tuning recall/false positives must be finite and valid")
    observations = sum(row.get("n_images", 1) for row in rows)
    n_images = len({row["image"] for row in rows}) if all("image" in row for row in rows) else observations
    entry = {**candidate, "detector": detector, "n_images": n_images, "n_observations": observations,
             "macro_recall": recall, "recall": recall, "false_positives_per_image": fp,
             "objective": tuning_objective(recall, fp)}
    for key in ("fine_cost_px", "coarse_cost_px", "calibration_cost_px", "fine_cost_frac", "coarse_cost_frac",
                "calibration_cost_frac", "total_cost_frac", "fraction_pixels_processed", "n_tool_calls",
                "human_review_count", "wall_time_ms", "preparation_time_ms", "uncached_wall_estimate_ms",
                "flagged_area_frac", "defect_cell_recall", "flagged_cell_frac", "cell_precision", "precision"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if key in ("flagged_area_frac", "flagged_cell_frac") and any(
                not np.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"{key} must be an image-level fraction in [0,1]")
        entry[key] = sum(values) / len(values) if values else None
    return entry


def _select_tradeoff(history: list[dict]) -> tuple[dict | None, str]:
    for limit, status in ((MAX_FP_PER_IMAGE, "selected_primary"),
                          (FALLBACK_MAX_FP_PER_IMAGE, "selected_fallback")):
        feasible = [entry for entry in history if entry["false_positives_per_image"] <= limit]
        if feasible:
            return max(feasible, key=lambda entry: entry["macro_recall"]), status
    return None, "no_feasible_candidate"


def _tuning_report(history: list[dict], best: dict | None, status: str, detector: str,
                   images: list[str], budgets: list[float]) -> dict:
    return {"selection_split": "dev", "selection_budgets": budgets,
            "objective": TUNING_OBJECTIVE_TEXT, "metrics_version": METRICS_VERSION,
            "matching_version": MATCHING_VERSION, "reporting_version": REPORTING_VERSION,
            "select_status": status, "primary_fp_limit": MAX_FP_PER_IMAGE,
            "config_available": best is not None,
            "fallback_fp_limit": FALLBACK_MAX_FP_PER_IMAGE, "detector": detector,
            "fallback_used": status == "selected_fallback",
            "fp_limit_used": (MAX_FP_PER_IMAGE if status == "selected_primary" else
                              FALLBACK_MAX_FP_PER_IMAGE if status == "selected_fallback" else None),
            "selection_reason": ("No candidate satisfies either dev FP limit." if best is None else
                                 "Primary limit has no feasible candidate; use the explicitly wider dev FP limit."
                                 if status == "selected_fallback" else "Choose highest recall within the primary dev FP limit."),
            "selected_method": best["method"] if best else None,
            "selected_config": best["config"] if best else None, "images": images, "history": history}


def write_tuning_report(report: dict, output: Path) -> None:
    """Always save the complete dev cost/recall trade-off, including infeasibility."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "tuning_dev.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    history = report["history"]
    (output / "dev_tradeoff.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    columns = ["method", "detector", "n_images", "n_observations", "macro_recall", "false_positives_per_image", "objective",
               "fine_cost_px", "coarse_cost_px", "calibration_cost_px", "fine_cost_frac", "coarse_cost_frac",
               "calibration_cost_frac", "total_cost_frac", "fraction_pixels_processed", "n_tool_calls",
               "human_review_count", "wall_time_ms", "preparation_time_ms", "uncached_wall_estimate_ms",
               "flagged_area_frac", "defect_cell_recall", "flagged_cell_frac", "cell_precision", "precision", "config"]
    with (output / "dev_tradeoff.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**entry, "config": json.dumps(entry["config"], sort_keys=True)} for entry in history)
    lines = [f"Selection: {report['select_status']}", "", report["objective"], "",
             "| Candidate | Recall | FP/image | Fine fraction | Coarse fraction | Calibration fraction | Total fraction | Tool calls | Reviews | Wall ms | Preparation ms | Uncached estimate ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for entry in history:
        lines.append("| " + " | ".join(str(entry.get(key, "")) for key in
                     ("method", "macro_recall", "false_positives_per_image", "fine_cost_frac", "coarse_cost_frac",
                      "calibration_cost_frac", "total_cost_frac", "n_tool_calls", "human_review_count", "wall_time_ms",
                      "preparation_time_ms", "uncached_wall_estimate_ms")) + " |")
    (output / "dev_tradeoff.md").write_text("\n".join(lines) + "\n")
    config_path = output / "config.json"
    if report["selected_config"] is not None:
        config_path.write_text(json.dumps(report["selected_config"], indent=2) + "\n")
    elif config_path.exists():
        # Do not leave a previous selected config looking current after failure.
        raise ValueError("no feasible candidate; use a fresh output directory to avoid stale config.json")


def record_key(row: dict) -> tuple:
    return (str(row.get("detector", "classic")), str(row["method"]), str(row.get("seed", 0)),
            float(row.get("budget_frac", row.get("budget"))), str(row["image"]))


def tuning_candidates(base: InspectorConfig, detector: str) -> list[dict]:
    """Freeze policy alternatives, including validation-floor-safe multipliers."""
    # Learned thresholds span the per-scale validation thresholds (about 0.72-0.84).
    thresholds = (0.6, 0.75, 0.85, 0.92) if detector == "learned" else (0.7, 0.8)
    follows = (2.0,) if detector == "learned" else (2.0, 4.0)
    multipliers = sorted({max(base.threshold_multiplier, value) for value in (1.0, 1.05, 1.1)}) if detector == "learned" else [base.threshold_multiplier]
    candidates = []
    for index, (initial, positive, follow, multiplier) in enumerate(itertools.product(
            (0.25, 0.5), thresholds, follows, multipliers)):
        config = replace(base, initial_scale=initial, positive_threshold=positive,
                         detection_threshold=positive, follow_weight=follow, threshold_multiplier=multiplier,
                         ambiguous_low=min(base.ambiguous_low, 0.25) if detector == "learned" else base.ambiguous_low)
        candidates.append({"method": f"agent_candidate_{index:04d}", "config": asdict(config)})
    return candidates


def evaluate_tuning_entries(bench: Path, entries: list[dict], evaluation: dict,
                            model_dir: Path, save_traces: Path | None = None) -> list[dict]:
    """Evaluate candidate policies on assigned dev images, reusing one preparation."""
    if evaluation["split"] != "dev" or any(entry.get("split") != "dev" for entry in entries):
        raise ValueError("policy tuning is restricted to dev images")
    detector = evaluation["detectors"][0]
    backend = LocalBackend(max_size=evaluation["size"], detector=detector, model_dir=model_dir,
                           calibration=evaluation.get("calibration", "sampled"))
    rows = []
    for entry in entries:
        for candidate in evaluation["candidates"]:
            trace_dir = save_traces / candidate["method"] if save_traces is not None else None
            candidate_rows = evaluate_entries(bench, [entry], ["agent"], evaluation["budgets"],
                                               evaluation["size"], InspectorConfig(**candidate["config"]),
                                               evaluation["seed"], trace_dir, detector=detector,
                                               model_dir=model_dir, backend=backend)
            for row in candidate_rows:
                row["method"] = candidate["method"]
            rows.extend(candidate_rows)
    return rows


def select_tuning_config(records: list[dict], evaluation: dict) -> tuple[dict | None, dict]:
    """Choose from complete saved dev rows; no inference or test-data access."""
    if evaluation.get("mode") != "tuning" or evaluation.get("split") != "dev":
        raise ValueError("configuration selection requires dev tuning records")
    names = [candidate["method"] for candidate in evaluation["candidates"]]
    if (not names or len(names) != len(set(names)) or set(names) != set(evaluation["methods"])
            or len(evaluation["detectors"]) != 1 or not evaluation["images"] or not evaluation["budgets"]):
        raise ValueError("invalid dev candidate specification")
    if any(row.get("split", "dev") != "dev" for row in records):
        raise ValueError("test records cannot select a policy")
    if evaluation.get("metrics_version", METRICS_VERSION) != METRICS_VERSION or any(
            row.get("metrics_version", METRICS_VERSION) != METRICS_VERSION for row in records):
        raise ValueError("incompatible tuning metrics version")
    keys = [record_key(row) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected_record_keys(evaluation, evaluation["images"]):
        raise ValueError("configuration selection requires the complete dev candidate grid")
    history = []
    for candidate in evaluation["candidates"]:
        rows = [row for row in records if row["method"] == candidate["method"]]
        history.append(_candidate_tradeoff(candidate, rows, evaluation["detectors"][0]))
    best, status = _select_tradeoff(history)
    report = _tuning_report(history, best, status, evaluation["detectors"][0],
                            evaluation["images"], evaluation["budgets"])
    return best["config"] if best else None, report


def expected_record_keys(evaluation: dict, images: list[str]) -> set[tuple]:
    keys = set()
    for detector in evaluation["detectors"]:
        for method in evaluation["methods"]:
            if evaluation.get("mode") != "tuning" and detector == "classic" and method not in ("agent", "exhaustive"):
                continue
            seeds = evaluation["random_seeds"] if method == "random" else [evaluation["seed"]]
            for seed, budget, image in itertools.product(seeds, evaluation["budgets"], images):
                keys.add((detector, method, str(seed), float(budget), image))
    return keys


def merge_shards(shard_paths: list[Path], out_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Validate complete coverage/configuration and merge JSON; never read images."""
    if not shard_paths:
        raise ValueError("at least one shard output is required")
    metadata, all_records, seen_keys = [], [], set()
    for source in shard_paths:
        folder = source if source.is_dir() else source.parent
        meta = json.loads((folder / "run.json").read_text())
        if meta.get("schema") != SHARD_SCHEMA:
            raise ValueError("incompatible evaluation shard schema")
        rows = json.loads((folder / "results.json").read_text())
        evaluation = meta["evaluation"]
        if evaluation.get("metrics_version", METRICS_VERSION) != METRICS_VERSION or any(
                row.get("metrics_version", METRICS_VERSION) != METRICS_VERSION for row in rows):
            raise ValueError("incompatible shard metrics version")
        assigned = shard_entries(evaluation["images"], int(meta["shard_index"]), int(meta["shard_count"]))
        if meta["assigned_images"] != assigned:
            raise ValueError("shard image assignment does not match the manifest partition")
        keys = [record_key(row) for row in rows]
        if len(keys) != len(set(keys)) or seen_keys.intersection(keys):
            raise ValueError("duplicate evaluation records in shard outputs")
        if set(keys) != expected_record_keys(evaluation, assigned):
            raise ValueError("shard records do not cover configured image/method/seed/budget grid")
        seen_keys.update(keys)
        metadata.append(meta)
        all_records.extend(rows)
    reference = metadata[0]
    count = int(reference["shard_count"])
    if len(metadata) != count or {int(meta["shard_index"]) for meta in metadata} != set(range(count)):
        raise ValueError("merge requires each shard index exactly once")
    if any(meta["evaluation"] != reference["evaluation"] or int(meta["shard_count"]) != count
           for meta in metadata):
        raise ValueError("shards have different configurations, models, manifests or partition counts")
    image_order = {image: index for index, image in enumerate(reference["evaluation"]["images"])}
    all_records.sort(key=lambda row: (image_order[row["image"]], record_key(row)[:-1]))
    merged = {"schema": SHARD_SCHEMA, "status": "merged", "evaluation": reference["evaluation"],
              "source_shards": count, "n_records": len(all_records), "inference_rerun": False,
              "timing_note": "Cached policy wall time; uncached wall is an estimate adding measured shared preparation."}
    selected_config, tuning_report = reference["evaluation"]["config"], None
    if reference["evaluation"].get("mode") == "tuning":
        selected_config, tuning_report = select_tuning_config(all_records, reference["evaluation"])
        merged["selected_method"] = tuning_report["selected_method"]
        merged["select_status"] = tuning_report["select_status"]
    merged["config_available"] = selected_config is not None
    if out_dir is not None:
        output = app_output(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_summaries(all_records, output)
        (output / "run.json").write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
        if tuning_report is not None:
            write_tuning_report(tuning_report, output)
        else:
            (output / "config.json").write_text(json.dumps(selected_config, indent=2) + "\n")
    return all_records, merged


def main(argv: list[str] | None = None) -> int:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+", help="Merge shard outputs without image/model inference")
    parser.add_argument("--out", type=Path, default=APP_ROOT / "artifacts/eval")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--methods", nargs="+", choices=("agent", "exhaustive", "random", "raster", "coarse_only"),
                        default=["agent", "exhaustive", "random", "raster", "coarse_only"])
    parser.add_argument("--detectors", nargs="+", choices=("classic", "learned"), default=["learned", "classic"])
    parser.add_argument("--model-dir", "--models", dest="model_dir", type=Path, default=APP_ROOT / "models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-surface-limit", type=int)
    parser.add_argument("--size", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--calibration", choices=("sampled", "whole"), default="sampled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tune", action="store_true", help="Shardable dev-only candidate search; merge exports the chosen config")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        output = app_output(args.out)
        if args.merge:
            if args.bench is not None or args.tune:
                parser.error("--merge cannot be combined with --bench or --tune")
            records, metadata = merge_shards(args.merge, output)
            print(json.dumps({"status": metadata["status"], "n_records": len(records), "inference_rerun": False,
                              "select_status": metadata.get("select_status"),
                              "config_available": metadata.get("config_available")}))
            return 3 if metadata.get("select_status") == "no_feasible_candidate" else 0
        if args.bench is None:
            parser.error("--bench is required unless --merge is used")
        shard_entries([], args.shard_index, args.shard_count)
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.per_surface_limit is not None and args.per_surface_limit < 1:
            parser.error("--per-surface-limit must be positive")
        if any(not np.isfinite(b) or not 0 <= b <= 1 for b in args.budgets):
            parser.error("budgets must be finite values in [0,1]")
        if any(len(values) != len(set(values)) for values in
               (args.budgets, args.methods, args.detectors, args.random_seeds)):
            parser.error("budget, method, detector and random-seed lists must not contain duplicates")
        if args.tune and args.split != "dev":
            parser.error("--tune requires --split dev; test data never selects a policy")
        config = InspectorConfig(**json.loads(args.config.read_text())) if args.config else InspectorConfig()
        output.mkdir(parents=True, exist_ok=True)
        entries = load_manifest(args.bench, args.split, args.limit, args.per_surface_limit)
        assigned = shard_entries(entries, args.shard_index, args.shard_count)
        manifest = json.loads((args.bench / "manifest.json").read_text())
        evaluation = {"bench": str(args.bench.resolve()), "manifest_sha256": json_fingerprint(manifest),
                      "mode": "evaluation", "calibration": args.calibration, "reporting_version": REPORTING_VERSION,
                      "matching_version": MATCHING_VERSION,
                      "metrics_version": METRICS_VERSION,
                      "split": args.split, "size": args.size, "images": [entry["image"] for entry in entries],
                      "budgets": args.budgets, "methods": args.methods, "detectors": args.detectors,
                      "seed": args.seed, "random_seeds": args.random_seeds,
                      "limit": args.limit, "per_surface_limit": args.per_surface_limit,
                      "config": asdict(config), "model_files": model_fingerprint(args.model_dir) if "learned" in args.detectors else None,
                      "opencv_version": cv2.__version__, "numpy_version": np.__version__,
                      "resized_labels": "nearest neighbour; vanished defects retained in denominator"}
        records = []
        if args.tune:
            detector = "learned" if "learned" in args.detectors else "classic"
            candidates = tuning_candidates(config, detector)
            evaluation.update(mode="tuning", detectors=[detector], candidates=candidates,
                              methods=[candidate["method"] for candidate in candidates], budgets=[max(args.budgets)])
            if assigned:
                records = evaluate_tuning_entries(args.bench, assigned, evaluation, args.model_dir,
                                                   output / "traces" if args.save_traces else None)
        else:
            for detector in args.detectors:
                methods = [method for method in args.methods if detector != "classic" or method in ("agent", "exhaustive")]
                if methods and assigned:
                    records.extend(evaluate_entries(args.bench, assigned, methods, args.budgets, args.size,
                                                    config, args.seed, output / "traces" if args.save_traces else None,
                                                    detector=detector, model_dir=args.model_dir,
                                                    random_seeds=args.random_seeds, calibration=args.calibration))
        metadata = {"schema": SHARD_SCHEMA, "status": "complete" if args.shard_count == 1 else "shard_complete",
                    "evaluation": evaluation, "shard_index": args.shard_index, "shard_count": args.shard_count,
                    "assigned_images": [entry["image"] for entry in assigned], "n_records": len(records),
                    "timing_note": "Preparation measured once/image/detector outside policy timers; uncached wall is an estimate."}
        (output / "results.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        config_name = "base_config.json" if args.tune else "config.json"
        (output / config_name).write_text(json.dumps(asdict(config), indent=2) + "\n")
        if args.shard_count == 1:
            if args.tune:
                selected, report = select_tuning_config(records, evaluation)
                write_tuning_report(report, output)
                metadata.update(select_status=report["select_status"], config_available=selected is not None)
                (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            print(json.dumps(write_summaries(records, output), indent=2, allow_nan=False))
        else:
            print(json.dumps({"status": "shard_complete", "shard_index": args.shard_index,
                              "shard_count": args.shard_count, "n_images": len(assigned), "n_records": len(records)}))
        return 3 if metadata.get("select_status") == "no_feasible_candidate" else 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
