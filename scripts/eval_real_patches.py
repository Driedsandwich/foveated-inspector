"""Evaluate a frozen detector on independent Positive/Negative JPEG patches.

This is image classification, not the synthetic benchmark's pixel-grounded
localisation evaluation. No model is fitted or threshold selected on real data.
A missing dataset is a normal, explicitly reported skipped result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from scripts.run_eval import app_output, model_fingerprint, shard_entries

APP_ROOT = Path(__file__).resolve().parents[1]
SHARD_SCHEMA = "real-patches-shard-v2"


def select_patches(root: Path, per_class: int = 1000, seed: int = 2026) -> tuple[list[tuple[Path, int]], dict]:
    """Recursively select deterministic samples from Positive/Negative folders."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    paths: dict[int, list[Path]] = {0: [], 1: []}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        parts = {part.lower() for part in path.relative_to(root).parts[:-1]}
        positive, negative = "positive" in parts, "negative" in parts
        if positive != negative:
            paths[int(positive)].append(path)
    rng = np.random.default_rng(seed)
    selected = []
    counts = {}
    for label, name in ((0, "negative"), (1, "positive")):
        candidates = sorted(paths[label], key=lambda path: path.relative_to(root).as_posix())
        sample_size = min(per_class, len(candidates))
        indices = sorted(rng.choice(len(candidates), sample_size, replace=False).tolist()) if sample_size else []
        selected.extend((candidates[index], label) for index in indices)
        counts[name] = {"available": len(candidates), "selected": sample_size}
    return selected, counts


def evaluate_patches(
    root: Path, model_dir: Path, *, per_class: int = 1000, seed: int = 2026,
    detector=None, feature_extractor: Callable | None = None,
    shard_index: int = 0, shard_count: int = 1,
) -> tuple[dict, list[dict]]:
    """Return a JSON-safe report and per-image predictions, holding one image."""
    if not root.is_dir():
        return {"status": "skipped", "reason": "root_missing", "root": str(root)}, []
    selected, counts = select_patches(root, per_class, seed)
    if not selected or any(value["selected"] == 0 for value in counts.values()):
        return {"status": "skipped", "reason": "positive_or_negative_images_missing",
                "root": str(root), "class_counts": counts}, []
    all_selected = selected
    selected = shard_entries(all_selected, shard_index, shard_count)
    if detector is None:
        from inspector.model import DetectorModels
        detector = DetectorModels(model_dir)
    if feature_extractor is None:
        from inspector.features import feature_map
        feature_extractor = feature_map
    threshold = float(detector.threshold(1.0))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("the frozen scale-1 detector threshold must lie in [0,1]")
    started = time.perf_counter()
    predictions = []
    for path, label in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            predictions.append({"image": path.relative_to(root).as_posix(), "label": label,
                                "status": "unreadable", "score": None, "prediction": None,
                                "pixels_processed": 0, "n_feature_cells": 0})
            continue
        from inspector.features import prepare_calibration
        statistics, calibration_info = prepare_calibration(image)
        features, _ = feature_extractor(image, stats=statistics["1"])
        probabilities, _ = detector.predict(features, 1.0)
        probabilities = np.asarray(probabilities)
        if probabilities.size == 0 or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid detector probabilities for {path.name}")
        score = float(np.max(probabilities))
        if not 0 <= score <= 1:
            raise ValueError(f"probabilities outside [0,1] for {path.name}")
        predictions.append({"image": path.relative_to(root).as_posix(), "label": label,
                            "status": "ok", "score": score, "prediction": int(score >= threshold),
                            "pixels_processed": int(image.shape[0] * image.shape[1]) + calibration_info["total_cost_px"],
                            "n_feature_cells": int(probabilities.size)})
        del image, features, probabilities
    metadata = {
        "evaluation_level": "image_classification", "root": str(root.resolve()),
        "model_dir": str(model_dir.resolve()), "seed": seed, "requested_per_class": per_class,
        "class_counts": counts, "threshold": threshold,
        "selected_images": [{"image": path.relative_to(root).as_posix(), "label": label}
                            for path, label in all_selected],
        "shard_index": shard_index, "shard_count": shard_count,
        "assigned_images": [path.relative_to(root).as_posix() for path, _ in selected],
    }
    return summarize_predictions(predictions, metadata, (time.perf_counter() - started) * 1000), predictions


def summarize_predictions(predictions: list[dict], metadata: dict, wall_time_ms: float = 0.0) -> dict:
    """Compute image-level classification metrics from saved rows, without inference."""
    valid = [row for row in predictions if row["status"] == "ok"]
    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in valid)
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in valid)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in valid)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in valid)
    summary = {
        **metadata,
        "status": "complete" if len(valid) == len(predictions) else "partial",
        "n_images": len(valid), "n_unreadable": len(predictions) - len(valid),
        "threshold_source": "frozen model dev validation; scale 1.0",
        "score_rule": "maximum probability over 32-pixel feature cells at scale 1.0",
        "normalization": "per-image robust statistics, without ground truth",
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(valid) if valid else None,
        "pixels_processed": sum(int(row.get("pixels_processed", 0)) for row in valid),
        "n_feature_cells": sum(int(row.get("n_feature_cells", 0)) for row in valid),
        "wall_time_ms": wall_time_ms,
        "trained_on_real_data": False,
        "limitation": "Image labels do not provide defect localisation ground truth; these metrics are not synthetic detection recall.",
    }
    return summary


def write_patch_outputs(summary: dict, predictions: list[dict], output: Path) -> None:
    output = app_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("predictions.json", predictions)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label", "status", "score", "prediction",
                                                    "pixels_processed", "n_feature_cells"])
        writer.writeheader()
        writer.writerows(predictions)


def merge_real_shards(shard_paths: list[Path], out_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Merge a complete deterministic image partition, validating saved predictions."""
    if not shard_paths:
        raise ValueError("at least one real-patch shard is required")
    runs, rows, seen = [], [], set()
    total_wall_time = 0.0
    for source in shard_paths:
        folder = source if source.is_dir() else source.parent
        run = json.loads((folder / "run.json").read_text())
        if run.get("schema") != SHARD_SCHEMA:
            raise ValueError("incompatible real-patch shard schema")
        evaluation = run["evaluation"]
        selected = evaluation["selected_images"]
        assigned = shard_entries(selected, int(run["shard_index"]), int(run["shard_count"]))
        expected = {item["image"]: int(item["label"]) for item in assigned}
        if len(expected) != len(assigned) or run["assigned_images"] != [item["image"] for item in assigned]:
            raise ValueError("real-patch shard assignment does not match its sampling manifest")
        predictions = json.loads((folder / "predictions.json").read_text())
        names = [row["image"] for row in predictions]
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("duplicate real-patch image predictions")
        if set(names) != set(expected):
            raise ValueError("real-patch shard has missing or unexpected image predictions")
        for row in predictions:
            if int(row["label"]) != expected[row["image"]] or row["status"] not in ("ok", "unreadable"):
                raise ValueError("invalid real-patch prediction label or status")
            if row["status"] == "ok":
                score = float(row["score"])
                if not np.isfinite(score) or not 0 <= score <= 1 or int(row["prediction"]) != int(score >= evaluation["threshold"]):
                    raise ValueError("real-patch prediction is inconsistent with the frozen threshold")
        seen.update(names)
        rows.extend(predictions)
        runs.append(run)
        total_wall_time += float(run.get("wall_time_ms", 0))
    reference = runs[0]
    count = int(reference["shard_count"])
    if len(runs) != count or {int(run["shard_index"]) for run in runs} != set(range(count)):
        raise ValueError("merge requires every real-patch shard index exactly once")
    if any(run["evaluation"] != reference["evaluation"] or int(run["shard_count"]) != count for run in runs):
        raise ValueError("real-patch shards have different models, samples, thresholds or configurations")
    order = {item["image"]: index for index, item in enumerate(reference["evaluation"]["selected_images"])}
    rows.sort(key=lambda row: order[row["image"]])
    summary = summarize_predictions(rows, reference["evaluation"], total_wall_time)
    summary.update(source_shards=count, inference_rerun=False,
                   timing_note="Sum of measured shard compute times; not elapsed parallel wall time.")
    if out_dir is not None:
        write_patch_outputs(summary, rows, out_dir)
        (app_output(out_dir) / "run.json").write_text(json.dumps({"schema": SHARD_SCHEMA, "status": "merged",
                                                               "evaluation": reference["evaluation"],
                                                               "source_shards": count}, indent=2) + "\n")
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--model-dir", "--models", dest="model_dir", type=Path,
                        default=APP_ROOT / "models")
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=APP_ROOT / "artifacts/real_patches")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    try:
        output = app_output(args.out)
        if args.merge:
            if args.root is not None:
                parser.error("--merge cannot be combined with --root")
            summary, _ = merge_real_shards(args.merge, output)
            print(json.dumps(summary, indent=2, allow_nan=False))
            return 0
        if args.root is None:
            parser.error("--root is required unless --merge is used")
        shard_entries([], args.shard_index, args.shard_count)
        summary, predictions = evaluate_patches(args.root, args.model_dir,
                                               per_class=args.per_class, seed=args.seed,
                                               shard_index=args.shard_index, shard_count=args.shard_count)
        if summary["status"] != "skipped":
            evaluation = {key: summary[key] for key in ("evaluation_level", "root", "model_dir", "seed",
                                                       "requested_per_class", "class_counts", "threshold", "selected_images")}
            evaluation["model_files"] = model_fingerprint(args.model_dir)
            run = {"schema": SHARD_SCHEMA, "evaluation": evaluation,
                   "shard_index": args.shard_index, "shard_count": args.shard_count,
                   "assigned_images": summary["assigned_images"], "wall_time_ms": summary["wall_time_ms"]}
            write_patch_outputs(summary, predictions, output)
            (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    except (ImportError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "error", "reason": str(error)}, allow_nan=False))
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
