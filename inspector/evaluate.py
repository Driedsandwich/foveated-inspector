"""Pixel-grounded matching and streaming-friendly benchmark summaries.

Bounding boxes are ``[x, y, width, height]`` in the label map's coordinates.
Primary matching requires bbox IoU >=0.1, or >=30% of the GT mask pixels with
a report area no larger than nine GT bbox areas. Every unmatched report is an
FP. Legacy columns apply the former pixel rule to the SAME current merged
boxes; they do not reproduce an older merging policy.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .agent import REPORTING_VERSION, flatten_cell_reports, merge_reports

DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.40)
METRICS_VERSION = "inspection-cell-and-localization-v1"
MATCHING_VERSION = "bbox-iou0.1-or-maskcoverage0.3-area9-v1"
LEGACY_MATCHING_VERSION = "mask-smallerarea0.3-or-40pixels-zerooverlap-fp-v1"


def _duplicate_indices(matching: dict) -> list[int]:
    """Count later reports that match an already found GT ID, once per report."""
    seen: set[int] = set()
    duplicates = []
    for index, matched in enumerate(matching["matches"]):
        if seen.intersection(matched):
            duplicates.append(index)
        seen.update(matched)
    return duplicates


def _match_detections(
    detections: Sequence[dict], label_map: np.ndarray, defects: Sequence[dict], *, legacy: bool
) -> dict:
    """Use actual mask areas/bounds, including after resizing, rather than metadata."""
    if label_map.ndim != 2 or not np.issubdtype(label_map.dtype, np.integer):
        raise ValueError("label_map must be a two-dimensional integer array")
    if label_map.size == 0 or int(label_map.min()) < 0:
        raise ValueError("label_map must be non-empty and nonnegative")
    ids = {int(item["id"]) for item in defects}
    if 0 in ids or len(ids) != len(defects):
        raise ValueError("defect IDs must be unique positive integers")
    areas = dict(zip(*np.unique(label_map, return_counts=True)))
    present_ids = {int(value) for value in areas if value != 0}
    if present_ids - ids:
        raise ValueError("label_map contains IDs missing from defect metadata")
    truth_boxes = {}
    if not legacy and detections:
        for defect_id in sorted(present_ids):
            mask = label_map == defect_id
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            truth_boxes[defect_id] = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
            del mask
    height, width = label_map.shape
    found: set[int] = set()
    false_positive_indices: list[int] = []
    matches: list[list[int]] = []
    partial_overlap_indices: list[int] = []
    overlapping_indices: list[int] = []
    for index, detection in enumerate(detections):
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
            raise ValueError("each detection requires bbox=[x,y,w,h]")
        x, y, box_width, box_height = map(float, bbox)
        if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
            raise ValueError("detection bounding boxes must be finite")
        if box_width <= 0 or box_height <= 0:
            raise ValueError("detection bounding boxes must have positive area")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right = min(width, math.ceil(x + box_width))
        bottom = min(height, math.ceil(y + box_height))
        matched: list[int] = []
        intersections = {}
        if right > left and bottom > top:
            values, counts = np.unique(label_map[top:bottom, left:right], return_counts=True)
            for value, overlap in zip(values, counts):
                defect_id, intersection = int(value), int(overlap)
                if defect_id == 0:
                    continue
                intersections[defect_id] = intersection
        if intersections:
            overlapping_indices.append(index)
        if legacy:
            for defect_id, intersection in intersections.items():
                smaller_area = min(box_width * box_height, int(areas[defect_id]))
                if intersection >= 0.30 * smaller_area or intersection >= 40:
                    matched.append(defect_id)
        else:
            report_area = box_width * box_height
            for defect_id, (gx, gy, gr, gb) in truth_boxes.items():
                truth_bbox_area = (gr - gx) * (gb - gy)
                bbox_intersection = max(0.0, min(x + box_width, gr) - max(x, gx)) * max(0.0, min(y + box_height, gb) - max(y, gy))
                iou = bbox_intersection / (report_area + truth_bbox_area - bbox_intersection)
                covered = intersections.get(defect_id, 0) >= 0.30 * int(areas[defect_id])
                if iou >= 0.1 or (covered and report_area <= 9 * truth_bbox_area):
                    matched.append(defect_id)
        found.update(matched)
        if (not intersections if legacy else not matched):
            false_positive_indices.append(index)
        if intersections and not matched:
            partial_overlap_indices.append(index)
        matches.append(sorted(matched))
    return {
        "found_ids": sorted(found),
        "missed_ids": sorted(ids - found),
        "matches": matches,
        "false_positive_indices": false_positive_indices,
        "partial_overlap_indices": partial_overlap_indices,
        "overlapping_indices": overlapping_indices,
    }


def match_detections(detections: Sequence[dict], label_map: np.ndarray, defects: Sequence[dict]) -> dict:
    """Match IoU>=0.1 OR (GT-pixel coverage>=0.3 AND area<=9*GT-bbox area).

    Geometric areas use the original, unclipped report box. Pixel intersections
    are clipped to the image. Vanished GT IDs remain missed. A report with no
    match is an FP even if it touches a few GT pixels.
    """
    return _match_detections(detections, label_map, defects, legacy=False)


def legacy_match_detections(detections: Sequence[dict], label_map: np.ndarray, defects: Sequence[dict]) -> dict:
    """Former smaller-area/40-pixel rule, with FP only for zero GT-pixel overlap."""
    return _match_detections(detections, label_map, defects, legacy=True)


def flagged_area_frac(detections: Sequence[dict], shape: tuple[int, int]) -> float:
    """Exact clipped integer-pixel bbox union; no image-sized extra mask."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    boxes = []
    for detection in detections:
        x, y, w, h = map(float, detection["bbox"])
        if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError("detection bounding boxes must have finite positive area")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        if left < right and top < bottom:
            boxes.append((left, top, right, bottom))
    edges = sorted({edge for left, _, right, _ in boxes for edge in (left, right)})
    area = 0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((top, bottom) for x, top, r, bottom in boxes if x < right and r > left)
        covered, end = 0, 0
        for top, bottom in intervals:
            covered += max(0, bottom - max(top, end))
            end = max(end, bottom)
        area += (right - left) * covered
    return area / (height * width)


def cell_metrics(detections: Sequence[dict], result: dict, label_map: np.ndarray,
                 defects: Sequence[dict]) -> dict:
    """Flag only reported cells, using provenance or bbox/grid intersection.

    A defect is found only if ONE flagged cell contains >=30% of its GT pixels;
    coverage from multiple cells is never added. Cell precision is the fraction
    of flagged cells containing any GT pixels. Candidates, reviews, and merely
    inspected cells do not flag anything. With no grid, cell metrics are unknown.
    """
    rects = result.get("coarse", {}).get("cell_rects", result.get("cell_rects", []))
    height, width = label_map.shape
    flagged = set()
    for detection in detections:
        provenance = detection.get("cells") or []
        if not provenance and detection.get("cell") is not None:
            provenance = [detection["cell"]]
        valid = {int(cell) for cell in provenance if isinstance(cell, (int, np.integer))
                 and not isinstance(cell, bool) and 0 <= int(cell) < len(rects)}
        if valid:
            flagged.update(valid)
            continue
        x, y, w, h = map(float, detection["bbox"])
        for cell, (cx, cy, cw, ch) in enumerate(rects):
            if min(x + w, cx + cw, width) > max(x, cx, 0) and min(y + h, cy + ch, height) > max(y, cy, 0):
                flagged.add(cell)
    areas = dict(zip(*np.unique(label_map, return_counts=True)))
    found, positive_cells = set(), 0
    for cell in sorted(flagged):
        x, y, w, h = map(float, rects[cell])
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        counts = {}
        if left < right and top < bottom:
            counts = {int(value): int(count) for value, count in zip(*np.unique(
                label_map[top:bottom, left:right], return_counts=True)) if value != 0}
        positive_cells += bool(counts)
        found.update(defect_id for defect_id, count in counts.items() if count >= .30 * int(areas[defect_id]))
    n_defects = len(defects) if rects else 0
    return {"cell_n_defects": n_defects, "cell_n_found": len(found),
            "defect_cell_recall": len(found) / n_defects if n_defects else None,
            "n_grid_cells": len(rects), "n_flagged_cells": len(flagged),
            "n_true_flagged_cells": positive_cells,
            "flagged_cell_frac": len(flagged) / len(rects) if rects else None,
            "cell_precision": positive_cells / len(flagged) if flagged else None,
            "flagged_cells": sorted(flagged), "cell_found_ids": sorted(found)}


def evaluate_result(result: dict, label_map: np.ndarray, defects: Sequence[dict]) -> dict:
    """Return JSON-safe metrics for one image and one method/budget run.

    Pixel costs count analysed image samples once per tool invocation, including
    repeated inspection. Fine cost and coarse overhead are both exposed. Per-kind
    false positives use the detector's ``kind_guess``; unknown guesses stay under
    ``unknown`` rather than being silently charged to a ground-truth kind.
    Primary precision is matched reports/all reports. Legacy precision preserves
    the old any-pixel-overlap definition, including its unmatched partial hits.
    Legacy statistics change only the matcher, not report merging or geometry.
    """
    # Rebuild from current raw evidence, never from historical union boxes.
    # Legacy scale-missing and low-resolution reports are unconfirmed, matching
    # the UI convention. The pixel-level matcher itself remains schema-agnostic.
    coarse_only = result.get("report_mode") == "coarse_only"
    source = (result.get("reported_defects", []) if coarse_only else
              flatten_cell_reports(result["raw_reports_by_cell"])
              if "raw_reports_by_cell" in result else
              result.get("raw_reported_defects", result.get("reported_defects", [])))
    raw_detections = [dict(item) for item in source if item.get("scale") == (0.25 if coarse_only else 1.0)]
    detections = raw_detections if coarse_only else merge_reports(raw_detections)
    cells = cell_metrics(detections, result, label_map, defects)
    matching = match_detections(detections, label_map, defects)
    legacy_matching = legacy_match_detections(detections, label_map, defects)
    raw_matching = match_detections(raw_detections, label_map, defects)
    duplicates = _duplicate_indices(matching)
    raw_duplicates = _duplicate_indices(raw_matching)
    found = set(matching["found_ids"])
    legacy_found = set(legacy_matching["found_ids"])
    area = int(label_map.size)
    fine_cost = float(result.get("fine_cost_px", result.get("budget_used_frac", 0.0) * area))
    coarse_cost = float(result.get("coarse_cost_px", result.get("coarse", {}).get("cost_px", 0.0)))
    calibration_cost = float(result.get("calibration_cost_px", 0.0))
    per_kind: dict[str, dict] = {}
    kinds = {str(item["kind"]) for item in defects}
    kinds.update(str(item.get("kind_guess", "unknown")) for item in detections)
    for kind in sorted(kinds):
        kind_ids = {int(item["id"]) for item in defects if item["kind"] == kind}
        found_count = len(kind_ids & found)
        false_positives = sum(
            str(detections[i].get("kind_guess", "unknown")) == kind
            for i in matching["false_positive_indices"]
        )
        n_reports = sum(str(item.get("kind_guess", "unknown")) == kind for item in detections)
        matched_reports = n_reports - false_positives
        overlap_reports = sum(str(detections[i].get("kind_guess", "unknown")) == kind
                              for i in matching["overlapping_indices"])
        legacy_fp = sum(str(detections[i].get("kind_guess", "unknown")) == kind
                        for i in legacy_matching["false_positive_indices"])
        per_kind[kind] = {
            "n_defects": len(kind_ids), "n_found": found_count,
            "recall": found_count / len(kind_ids) if kind_ids else None,
            "false_positives": int(false_positives),
            "n_reports": n_reports, "n_overlapping_reports": overlap_reports,
            "n_matched_reports": matched_reports,
            "precision": matched_reports / n_reports if n_reports else None,
            "legacy_n_found": len(kind_ids & legacy_found),
            "legacy_recall": len(kind_ids & legacy_found) / len(kind_ids) if kind_ids else None,
            "legacy_false_positives": legacy_fp,
            "legacy_n_matched_reports": n_reports - legacy_fp,
            "legacy_precision": (n_reports - legacy_fp) / n_reports if n_reports else None,
            "duplicate_reports": sum(str(detections[i].get("kind_guess", "unknown")) == kind for i in duplicates),
            "cell_n_defects": len(kind_ids) if cells["n_grid_cells"] else 0,
            "cell_n_found": len(kind_ids.intersection(cells["cell_found_ids"])),
            "defect_cell_recall": (len(kind_ids.intersection(cells["cell_found_ids"])) / len(kind_ids)
                                   if cells["n_grid_cells"] and kind_ids else None),
        }
    per_visible_at: dict[str, dict] = {}
    for defect in defects:
        visibility = str(defect.get("visible_at", "unspecified"))
        entry = per_visible_at.setdefault(visibility, {"n_defects": 0, "n_found": 0})
        entry["n_defects"] += 1
        entry["n_found"] += int(int(defect["id"]) in found)
    for entry in per_visible_at.values():
        entry["recall"] = entry["n_found"] / entry["n_defects"]
    preparation_time = float(result.get("preparation_elapsed_ms", 0.0))
    wall_time = float(result.get("elapsed_ms", 0.0))
    preparation_included = bool(result.get(
        "preparation_included_in_elapsed", not result.get("coarse", {}).get("reused_preparation", False)))
    return {
        "metrics_version": METRICS_VERSION,
        "report_mode": "coarse_only" if coarse_only else "confirmed",
        "reporting_version": REPORTING_VERSION,
        "matching_version": MATCHING_VERSION,
        "legacy_matching_version": LEGACY_MATCHING_VERSION,
        "n_images": 1,
        "n_defects": len(defects),
        "n_found": len(found),
        "recall": len(found) / len(defects) if defects else None,
        "false_positives": len(matching["false_positive_indices"]),
        "false_positives_per_image": float(len(matching["false_positive_indices"])),
        "n_reports": len(detections),
        "n_overlapping_reports": len(matching["overlapping_indices"]),
        "n_matched_reports": len(detections) - len(matching["false_positive_indices"]),
        "precision": ((len(detections) - len(matching["false_positive_indices"])) / len(detections)
                      if detections else None),
        "duplicate_reports": len(duplicates),
        "flagged_area_frac": flagged_area_frac(detections, label_map.shape),
        "legacy_n_found": len(legacy_found),
        "legacy_recall": len(legacy_found) / len(defects) if defects else None,
        "legacy_false_positives": len(legacy_matching["false_positive_indices"]),
        "legacy_false_positives_per_image": float(len(legacy_matching["false_positive_indices"])),
        "legacy_n_matched_reports": len(detections) - len(legacy_matching["false_positive_indices"]),
        "legacy_precision": ((len(detections) - len(legacy_matching["false_positive_indices"])) / len(detections)
                             if detections else None),
        "legacy_duplicate_reports": len(_duplicate_indices(legacy_matching)),
        "false_positives_after_merge": len(matching["false_positive_indices"]),
        "raw_n_reports": len(raw_detections),
        "raw_false_positives": len(raw_matching["false_positive_indices"]),
        "raw_duplicate_reports": len(raw_duplicates),
        "unconfirmed_report_count": len(source) - len(raw_detections),
        "candidate_count": len(result.get("coarse_candidates", [])),
        "fine_cost_px": fine_cost,
        "coarse_cost_px": coarse_cost,
        # Calibration is already part of coarse_cost, not an additional charge.
        "calibration_cost_px": calibration_cost,
        "fine_cost_frac": fine_cost / area,
        "coarse_cost_frac": coarse_cost / area,
        "calibration_cost_frac": calibration_cost / area,
        "total_cost_frac": (fine_cost + coarse_cost) / area,
        "fraction_pixels_processed": (fine_cost + coarse_cost) / area,
        "n_tool_calls": int(result.get("n_tool_calls", 0)),
        "human_review_count": len(result.get("human_review_queue", [])),
        "wall_time_ms": wall_time,
        "preparation_time_ms": preparation_time,
        "preparation_included_in_elapsed": preparation_included,
        "uncached_wall_estimate_ms": wall_time if preparation_included else wall_time + preparation_time,
        "per_kind": per_kind,
        "per_visible_at": per_visible_at,
        "matching": matching,
        "raw_matching": raw_matching,
        "legacy_matching": legacy_matching,
        **cells,
    }


def recall_vs_budget(
    methods: Mapping[str, Callable[[float], dict]],
    label_map: np.ndarray,
    defects: Sequence[dict],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
) -> list[dict]:
    """Evaluate budget->result callables without retaining their image pipelines."""
    records = []
    for budget in budgets:
        if not 0 <= float(budget) <= 1:
            raise ValueError("budgets must be in [0, 1]")
        for method, runner in methods.items():
            result = runner(float(budget))
            metrics = evaluate_result(result, label_map, defects)
            records.append({"method": method, "budget_frac": float(budget), **metrics})
    return records


def _bootstrap_recall(
    rows: Sequence[dict], *, subset: str | None = None, category: str | None = None,
    n_bootstrap: int = 2000, seed: int = 0,
) -> dict:
    """Percentile 95% CI from image-cluster resampling, never defect resampling.

    Duplicate image IDs (for example repeat runs) share one resampling cluster.
    The point estimate remains micro recall over defect trials in those rows.
    One-image intervals are degenerate and are explicitly marked by n_ci_images.
    """
    clusters: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        image_id = str(row.get("image", row.get("image_ref", f"row_{index}")))
        value = row if subset is None else row.get(subset, {}).get(category, {})
        cluster = clusters.setdefault(image_id, [0.0, 0.0])
        cluster[0] += float(value.get("n_found", 0))
        cluster[1] += float(value.get("n_defects", 0))
    counts = np.asarray(list(clusters.values()), dtype=np.float64)
    metadata = {"confidence_level": 0.95, "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": seed, "bootstrap_unit": "image", "n_ci_images": len(clusters)}
    if not len(counts) or counts[:, 1].sum() == 0:
        return {"recall_ci_low": None, "recall_ci_high": None, **metadata}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(counts), size=(n_bootstrap, len(counts)))
    draws = counts[sampled].sum(axis=1)
    nonempty = draws[:, 1] > 0
    ratios = draws[nonempty, 0] / draws[nonempty, 1]
    low, high = np.percentile(ratios, [2.5, 97.5])
    return {"recall_ci_low": float(low), "recall_ci_high": float(high), **metadata}


def _pool_random_seeds(records: Sequence[dict]) -> list[dict]:
    """Average random repeats per photograph before any image resampling."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in records:
        if row["method"] == "random" and str(row.get("seed", "")) not in ("", "pooled"):
            groups[(str(row.get("detector", "classic")),
                    float(row.get("budget_frac", row.get("budget"))))].append(row)
    pooled = []
    for (detector, budget), group in groups.items():
        seeds = {str(row["seed"]) for row in group}
        if len(seeds) < 2:
            continue
        images: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            if "image" not in row and "image_ref" not in row:
                raise ValueError("pooling random seeds requires image IDs")
            images[str(row.get("image", row.get("image_ref")))].append(row)
        for image, repeats in images.items():
            # Each photograph must have the same complete seed set. Missing
            # runs must not silently change the intended equal-seed estimator.
            if {str(row["seed"]) for row in repeats} != seeds or len(repeats) != len(seeds):
                raise ValueError("pooling random seeds requires one run per image and seed")
            average = {"detector": detector, "method": "random", "seed": "pooled",
                       "image": image, "budget_frac": budget, "n_images": 1,
                       "n_seeds": len(seeds), "seed_aggregation": "mean_per_image"}
            keys = {key for row in repeats for key, value in row.items()
                    if isinstance(value, (int, float)) and key not in ("seed", "budget", "budget_frac", "n_images")}
            for key in keys:
                values = [float(row[key]) for row in repeats if row.get(key) is not None]
                # Precision-like ratios are None when a seed made no reports; average the defined ones.
                average[key] = sum(values) / len(values) if values else None
            for subset in ("per_kind", "per_visible_at"):
                average[subset] = {}
                categories = {category for row in repeats for category in row.get(subset, {})}
                for category in categories:
                    counts = {key: sum(float(row.get(subset, {}).get(category, {}).get(key, 0))
                                       for row in repeats) / len(repeats)
                              for key in ("n_found", "n_defects", "false_positives", "n_reports",
                                          "n_overlapping_reports", "n_matched_reports", "duplicate_reports",
                                          "legacy_n_found", "legacy_false_positives", "legacy_n_matched_reports",
                                          "cell_n_found", "cell_n_defects")}
                    average[subset][category] = counts
            pooled.append(average)
    return pooled


def _image_mean(rows: Sequence[dict], key: str, *, unknown_if_absent: bool = False) -> float | None:
    """Average image-level means by n_images; never sum fractional coverage.

    None is unknown (excluded), whereas zero is an observed zero (included).
    This distinction also keeps zero-report precision pooling intact.
    """
    weighted, count = 0.0, 0
    for row in rows:
        value = row.get(key, None if unknown_if_absent else 0.0)
        if value is None:
            continue
        weight = int(row.get("n_images", 1))
        weighted += float(value) * weight
        count += weight
    return weighted / count if count else None


def aggregate_metrics(records: Sequence[dict], *, pool_random_seeds: bool = True) -> list[dict]:
    """Aggregate by detector, method, seed and budget, without mixing detectors.

    Recall is defect-weighted and costs are image-averaged. The 95% interval is
    a deterministic 2,000-replicate image bootstrap. Different random seeds stay
    separate. An additional explicitly named pooled random estimate averages
    seeds within each photograph before resampling photographs.
    """
    groups: dict[tuple[str, str, str, float], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("metrics_version", METRICS_VERSION) != METRICS_VERSION:
            raise ValueError("incompatible metrics version")
        if (isinstance(record.get("n_images", 1), bool) or
                not isinstance(record.get("n_images", 1), int) or record.get("n_images", 1) < 1):
            raise ValueError("metric rows require positive integer n_images")
        for fraction in ("flagged_area_frac", "flagged_cell_frac"):
            if fraction not in record or record[fraction] is None:
                continue
            value = record[fraction]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{fraction} must be a finite image-level fraction in [0,1]")
        budget = record.get("budget_frac", record.get("budget"))
        if budget is None:
            raise ValueError("each metric record requires budget_frac or budget")
        key = (str(record.get("detector", "classic")), str(record["method"]),
               str(record.get("seed", "")), float(budget))
        groups[key].append(record)
    summaries = []
    for (detector, method, seed, budget), rows in sorted(groups.items()):
        count = sum(int(row.get("n_images", 1)) for row in rows)
        total = sum(row["n_defects"] for row in rows)
        found = sum(row["n_found"] for row in rows)
        fp = sum(row["false_positives"] for row in rows)
        summary = {
            "metrics_version": METRICS_VERSION,
            "report_mode": rows[0].get("report_mode", "confirmed"),
            "reporting_version": REPORTING_VERSION,
            "matching_version": MATCHING_VERSION,
            "legacy_matching_version": LEGACY_MATCHING_VERSION,
            "detector": detector, "method": method, "seed": seed,
            "budget_frac": budget, "n_images": count,
            "n_defects": total, "n_found": found,
            "recall": found / total if total else None,
            "false_positives": fp, "false_positives_per_image": fp / count,
        }
        summary.update(_bootstrap_recall(rows))
        for key in ("n_reports", "n_overlapping_reports", "n_matched_reports", "duplicate_reports", "raw_n_reports",
                    "raw_false_positives", "raw_duplicate_reports", "unconfirmed_report_count", "candidate_count",
                    "legacy_n_found", "legacy_false_positives", "legacy_n_matched_reports", "legacy_duplicate_reports",
                    "cell_n_found", "cell_n_defects", "n_flagged_cells", "n_grid_cells", "n_true_flagged_cells"):
            summary[key] = sum(row.get(key, 0) for row in rows)
        summary["precision"] = summary["n_matched_reports"] / summary["n_reports"] if summary["n_reports"] else None
        summary["legacy_recall"] = summary["legacy_n_found"] / total if total else None
        summary["legacy_precision"] = summary["legacy_n_matched_reports"] / summary["n_reports"] if summary["n_reports"] else None
        summary["legacy_false_positives_per_image"] = summary["legacy_false_positives"] / count
        summary["false_positives_after_merge"] = fp
        summary["defect_cell_recall"] = summary["cell_n_found"] / summary["cell_n_defects"] if summary["cell_n_defects"] else None
        summary["cell_precision"] = summary["n_true_flagged_cells"] / summary["n_flagged_cells"] if summary["n_flagged_cells"] else None
        summary["flagged_cell_frac"] = _image_mean(rows, "flagged_cell_frac", unknown_if_absent=True)
        if seed == "pooled":
            summary.update(n_seeds=max(int(row["n_seeds"]) for row in rows),
                           seed_aggregation="mean_per_image")
        for key in ("fine_cost_frac", "coarse_cost_frac", "calibration_cost_frac", "total_cost_frac",
                    "fraction_pixels_processed", "n_tool_calls", "human_review_count", "wall_time_ms",
                    "preparation_time_ms", "uncached_wall_estimate_ms", "flagged_area_frac"):
            summary[key] = _image_mean(rows, key)
        summary["per_kind"] = {}
        kinds = sorted({kind for row in rows for kind in row.get("per_kind", {})})
        for kind in kinds:
            entries = [row.get("per_kind", {}).get(kind, {}) for row in rows]
            kind_total = sum(entry.get("n_defects", 0) for entry in entries)
            kind_found = sum(entry.get("n_found", 0) for entry in entries)
            kind_fp = sum(entry.get("false_positives", 0) for entry in entries)
            kind_reports = sum(entry.get("n_reports", 0) for entry in entries)
            kind_overlap = sum(entry.get("n_overlapping_reports", 0) for entry in entries)
            kind_matched = sum(entry.get("n_matched_reports", 0) for entry in entries)
            legacy_kind_found = sum(entry.get("legacy_n_found", 0) for entry in entries)
            legacy_kind_matched = sum(entry.get("legacy_n_matched_reports", 0) for entry in entries)
            legacy_kind_fp = sum(entry.get("legacy_false_positives", 0) for entry in entries)
            cell_kind_total = sum(entry.get("cell_n_defects", 0) for entry in entries)
            cell_kind_found = sum(entry.get("cell_n_found", 0) for entry in entries)
            summary["per_kind"][kind] = {
                "n_defects": kind_total, "n_found": kind_found,
                "recall": kind_found / kind_total if kind_total else None,
                "false_positives": kind_fp, "false_positives_per_image": kind_fp / count,
                "n_reports": kind_reports, "n_overlapping_reports": kind_overlap,
                "n_matched_reports": kind_matched,
                "precision": kind_matched / kind_reports if kind_reports else None,
                "legacy_n_found": legacy_kind_found,
                "legacy_recall": legacy_kind_found / kind_total if kind_total else None,
                "legacy_n_matched_reports": legacy_kind_matched,
                "legacy_precision": legacy_kind_matched / kind_reports if kind_reports else None,
                "legacy_false_positives": legacy_kind_fp,
                "legacy_false_positives_per_image": legacy_kind_fp / count,
                "cell_n_defects": cell_kind_total, "cell_n_found": cell_kind_found,
                "defect_cell_recall": cell_kind_found / cell_kind_total if cell_kind_total else None,
                "duplicate_reports": sum(entry.get("duplicate_reports", 0) for entry in entries),
                **_bootstrap_recall(rows, subset="per_kind", category=kind),
                # Shared acquisition cost: these are not per-kind-attributable costs.
                **{key: summary[key] for key in ("fine_cost_frac", "total_cost_frac",
                                                "n_tool_calls", "human_review_count", "wall_time_ms",
                                                "preparation_time_ms", "uncached_wall_estimate_ms")},
            }
        summary["per_visible_at"] = {}
        visibilities = sorted({value for row in rows for value in row.get("per_visible_at", {})})
        for visibility in visibilities:
            entries = [row.get("per_visible_at", {}).get(visibility, {}) for row in rows]
            visible_total = sum(entry.get("n_defects", 0) for entry in entries)
            visible_found = sum(entry.get("n_found", 0) for entry in entries)
            summary["per_visible_at"][visibility] = {
                "n_defects": visible_total, "n_found": visible_found,
                "recall": visible_found / visible_total if visible_total else None,
                **_bootstrap_recall(rows, subset="per_visible_at", category=visibility),
            }
        summaries.append(summary)
    if pool_random_seeds:
        pooled = _pool_random_seeds(records)
        if pooled:
            summaries.extend(aggregate_metrics(pooled, pool_random_seeds=False))
    return summaries


def write_summaries(records: Sequence[dict], out_dir: str | Path) -> list[dict]:
    """Write detailed JSON, aggregate JSON/CSV, per-kind CSV and an SVG chart."""
    from .viz import write_recall_chart

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_metrics(records)
    for filename, value in (("results.json", list(records)), ("summary.json", summaries)):
        (out / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    confidence_columns = ["recall_ci_low", "recall_ci_high", "confidence_level", "bootstrap_replicates",
                          "bootstrap_seed", "bootstrap_unit", "n_ci_images"]
    legacy_columns = ["legacy_n_found", "legacy_recall", "legacy_n_matched_reports", "legacy_precision",
                      "legacy_false_positives", "legacy_false_positives_per_image"]
    cell_columns = ["cell_n_defects", "cell_n_found", "defect_cell_recall", "n_grid_cells", "n_flagged_cells",
                    "n_true_flagged_cells", "flagged_cell_frac", "cell_precision"]
    columns = ["metrics_version", "report_mode", "reporting_version", "matching_version", "legacy_matching_version", "detector", "method", "seed", "budget_frac", "n_images", "n_defects", "n_found", "recall",
               "precision", "n_reports", "n_overlapping_reports", "n_matched_reports", "duplicate_reports", "false_positives_after_merge",
               "raw_n_reports", "raw_false_positives", "raw_duplicate_reports", "unconfirmed_report_count", "candidate_count",
               "flagged_area_frac", *cell_columns, *legacy_columns, "legacy_duplicate_reports",
               "false_positives", "false_positives_per_image", "fine_cost_frac", "coarse_cost_frac",
               "calibration_cost_frac",
               "total_cost_frac", "fraction_pixels_processed", "n_tool_calls", "human_review_count",
               "wall_time_ms", "preparation_time_ms", "uncached_wall_estimate_ms",
               "n_seeds", "seed_aggregation", *confidence_columns]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "surface", "split", *columns], extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "budget_frac": record.get("budget_frac", record.get("budget"))})
    kind_columns = ["detector", "method", "seed", "budget_frac", "kind", "n_defects", "n_found", "recall",
                    "cell_n_defects", "cell_n_found", "defect_cell_recall",
                    "precision", "n_reports", "n_overlapping_reports", "n_matched_reports", "duplicate_reports", *legacy_columns,
                    "false_positives", "false_positives_per_image", "fine_cost_frac", "total_cost_frac",
                    "n_tool_calls", "human_review_count", "wall_time_ms", "preparation_time_ms",
                    "uncached_wall_estimate_ms", *confidence_columns]
    with (out / "per_kind.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=kind_columns)
        writer.writeheader()
        for summary in summaries:
            for kind, values in summary["per_kind"].items():
                writer.writerow({"detector": summary["detector"], "method": summary["method"],
                                 "seed": summary["seed"], "budget_frac": summary["budget_frac"],
                                 "kind": kind, **values})
    visibility_columns = ["detector", "method", "seed", "budget_frac", "visible_at", "n_defects",
                          "n_found", "recall", *confidence_columns]
    with (out / "per_visibility.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=visibility_columns)
        writer.writeheader()
        for summary in summaries:
            for visible_at, values in summary["per_visible_at"].items():
                writer.writerow({"detector": summary["detector"], "method": summary["method"],
                                 "seed": summary["seed"], "budget_frac": summary["budget_frac"],
                                 "visible_at": visible_at, **values})
    write_recall_chart(summaries, out / "recall_vs_budget.svg")
    write_recall_chart(summaries, out / "recall_vs_total_cost.svg", cost_key="total_cost_frac")
    surface_rows = []
    for surface in sorted({str(row.get("surface", "unspecified")) for row in records}):
        rows = [row for row in records if str(row.get("surface", "unspecified")) == surface]
        surface_rows.extend(dict(summary, surface=surface) for summary in aggregate_metrics(rows))
    (out / "surface_summary.json").write_text(json.dumps(surface_rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "surface_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surface", *columns], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(surface_rows)
    table = ["| Surface | Detector | Method | Seed | Budget | Recall | Precision | FP/image | Flagged area fraction | Defect-cell recall | Flagged-cell fraction | Cell precision | Legacy recall | Legacy FP/image |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in surface_rows:
        values = [row.get(key) for key in ("surface", "detector", "method", "seed", "budget_frac", "recall", "precision", "false_positives_per_image", "flagged_area_frac", "defect_cell_recall", "flagged_cell_frac", "cell_precision", "legacy_recall", "legacy_false_positives_per_image")]
        table.append("| " + " | ".join("—" if value is None else str(value).replace("|", "\\|") for value in values) + " |")
    (out / "surface_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return summaries
