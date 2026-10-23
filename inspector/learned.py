"""Learned tools with a paid overview and bounded sampled surface calibration."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .features import PATCH_SIZE, feature_map, prepare_calibration
from .components import clip_bbox, component_gate, spatial_candidate_gate
from .model import DetectorModels, effective_threshold, scale_key


def _patch_components(probability: np.ndarray, kinds: np.ndarray, *,
                      work_shape: tuple[int, int], rect: list[int], scale: float,
                      threshold: float, kind_names: dict, patch_origin=(0, 0),
                      cell: int | None = None, clip_rect: list[int] | None = None,
                      follow_endpoints: bool = True, confirmation_candidates: list | None = None,
                      candidate_margin_px: float = 32, linear_min_length_px: float = 128,
                      image_shape: tuple[int, int] | None = None) -> tuple[list[dict], list[dict]]:
    """Localize classified patch components without re-analysing image pixels.

    ``patch_origin`` locates a cell's map slice in the original working image.
    Each patch belongs to exactly one overview cell, by its centre. Geometry
    comes from physical patch corners, including partial edge patches, and
    may be clipped to that owning cell. Counts always remain actual classified
    component sizes, rather than bbox area or a synthetic whole-cell seed.
    """
    if probability.size == 0:
        return [], []
    if confirmation_candidates is not None:
        if not isinstance(confirmation_candidates, list):
            raise ValueError("confirmation_candidates must be a list or null")
        if (isinstance(linear_min_length_px, bool) or not isinstance(linear_min_length_px, (int, float))
                or not math.isfinite(linear_min_length_px) or linear_min_length_px < 0):
            raise ValueError("linear_min_length_px must be finite and nonnegative")
    work_h, work_w = work_shape
    x, y, width, height = rect
    sx, sy = width/work_w, height/work_h
    origin_x, origin_y = patch_origin
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    detections, endpoints = [], []
    for component in range(1, count):
        bx, by, bw, bh, patch_count = map(int, stats[component])
        if patch_count < 2:
            continue
        mask = labels[by:by+bh, bx:bx+bw] == component
        probabilities = probability[by:by+bh, bx:bx+bw][mask]
        component_kinds = kinds[by:by+bh, bx:bx+bw][mask]
        kind_id = int(np.bincount(component_kinds, weights=probabilities).argmax())
        py, px = np.nonzero(mask)
        points = np.stack((px+bx+origin_x, py+by+origin_y), axis=1).astype(np.float32)
        lefts, tops = points[:, 0]*PATCH_SIZE, points[:, 1]*PATCH_SIZE
        rights = np.minimum(lefts+PATCH_SIZE, work_w)
        bottoms = np.minimum(tops+PATCH_SIZE, work_h)
        corners = np.stack((np.stack((lefts, tops), axis=1), np.stack((rights, tops), axis=1),
                            np.stack((rights, bottoms), axis=1), np.stack((lefts, bottoms), axis=1)), axis=1)
        corners *= np.array([sx, sy], np.float32)
        (_, _), sides, _ = cv2.minAreaRect(corners.reshape(-1, 1, 2))
        long_side, short_side = max(1.0, max(sides)), max(1.0, min(sides))
        elongation = float(long_side/short_side)
        elongated = elongation >= 1.8
        fx0, fy0 = x+int(lefts.min()*sx), y+int(tops.min()*sy)
        fx1 = min(x+width, x+math.ceil(float(rights.max()*sx)))
        fy1 = min(y+height, y+math.ceil(float(bottoms.max()*sy)))
        bbox = [fx0, fy0, fx1-fx0, fy1-fy0]
        if clip_rect is not None:
            clipped = clip_bbox(bbox, clip_rect)
            if clipped is None:
                continue
            bbox = [int(value) for value in clipped]
        detection = {"bbox": bbox, "kind_guess": kind_names.get(kind_id, "unknown"),
                     "score": float(probabilities.max()), "mean_probability": float(probabilities.mean()),
                     "threshold": float(threshold), "scale": float(scale), "localized": True,
                     "component_id": component, "patch_count": patch_count,
                     "patch_count_source": "classified_grid_component", "elongation": elongation,
                     "length_px": float(long_side) if elongated else 0.0,
                     "width_px": float(min(short_side, bbox[2], bbox[3]))}
        if cell is not None:
            detection["cell"] = int(cell)
        detections.append(detection)
        if not elongated or not follow_endpoints:
            continue
        physical_centres = np.stack(((lefts+rights)*.5*sx, (tops+bottoms)*.5*sy), axis=1)
        _, eigenvectors = cv2.PCACompute(physical_centres, mean=None)
        direction = eigenvectors[0]
        for edge_mask, ax, ay in (
                (points[:, 0] == 0, -1, 0),
                (points[:, 0] == probability.shape[1]-1, 1, 0),
                (points[:, 1] == 0, 0, -1),
                (points[:, 1] == probability.shape[0]-1, 0, 1)):
            if not np.any(edge_mask):
                continue
            vector = direction.copy()
            if vector[0]*ax+vector[1]*ay < 0:
                vector *= -1
            if vector[0]*ax+vector[1]*ay < .2:
                continue
            point = physical_centres[edge_mask].mean(axis=0)
            endpoint_x = float(min(x+width-1, x+point[0]))
            endpoint_y = float(min(y+height-1, y+point[1]))
            if ax:
                endpoint_x = float(x if ax < 0 else x+width-1)
            if ay:
                endpoint_y = float(y if ay < 0 else y+height-1)
            endpoints.append({"x": endpoint_x, "y": endpoint_y, "dx": ax, "dy": ay,
                              "tangent_dx": float(vector[0]), "tangent_dy": float(vector[1]),
                              "component_id": component})
    if confirmation_candidates is not None:
        # Spatial support comes before top-k: three unrelated high-score
        # components must not hide a fourth component inside the candidate.
        supported = spatial_candidate_gate(detections, confirmation_candidates,
                                           margin_px=candidate_margin_px, image_shape=image_shape)
        supported_ids = {detection["component_id"] for detection in supported}
        supported.extend(detection for detection in detections
                         if detection["component_id"] not in supported_ids
                         and detection["elongation"] >= 4
                         and detection["length_px"] >= linear_min_length_px)
        detections = supported
    detections = component_gate(detections)
    retained = {detection["component_id"] for detection in detections}
    return detections, [endpoint for endpoint in endpoints if endpoint["component_id"] in retained]


def learned_coarse_scan(image: np.ndarray, models: DetectorModels,
                        grid: tuple[int, int] = (16, 16), *, calibration: str = "sampled",
                        calibration_cap_frac: float = 0.02) -> dict:
    started = time.perf_counter()
    height, width = image.shape[:2]
    rows, cols = grid
    if not (1 <= rows <= height and 1 <= cols <= width):
        raise ValueError("grid must fit the image")
    overview_w, overview_h = max(1, width // 4), max(1, height // 4)
    work = cv2.resize(image, (overview_w, overview_h), interpolation=cv2.INTER_AREA)
    overview_features, overview_stats = feature_map(work)
    overview_probability, overview_kinds = models.predict(overview_features, 0.25)
    del work
    # Patch centres locate the evidence in original-image coordinates. No
    # full-resolution classifier map is retained or used by the active policy.
    ph, pw = overview_probability.shape
    start_x, start_y = np.arange(pw)*PATCH_SIZE, np.arange(ph)*PATCH_SIZE
    centre_x = (start_x + np.minimum(start_x+PATCH_SIZE, overview_w))*.5*width/overview_w
    centre_y = (start_y + np.minimum(start_y+PATCH_SIZE, overview_h))*.5*height/overview_h
    thresholds = {scale_key(scale): float(models.threshold(scale)) for scale in (.25, .5, 1.0)}
    for floor in thresholds.values():
        effective_threshold(floor)
    rects, prior, overview_candidates = [], [], []
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        values = []
        for col in range(cols):
            x0, x1 = col * width // cols, (col + 1) * width // cols
            patch_rows = np.flatnonzero((centre_y >= y0) & (centre_y < y1))
            patch_cols = np.flatnonzero((centre_x >= x0) & (centre_x < x1))
            selection = overview_probability[np.ix_(patch_rows, patch_cols)]
            if selection.size:
                probability = float(selection.max())
            else:
                probability = float(overview_probability[
                    min(ph - 1, int((y0+y1)/2 * overview_h/height / PATCH_SIZE)),
                    min(pw - 1, int((x0+x1)/2 * overview_w/width / PATCH_SIZE))])
            values.append(probability)
            if selection.size:
                candidates, _ = _patch_components(
                    selection, overview_kinds[np.ix_(patch_rows, patch_cols)],
                    work_shape=(overview_h, overview_w), rect=[0, 0, width, height],
                    scale=.25, threshold=thresholds["0.25"], kind_names=models.kind_names,
                    patch_origin=(int(patch_cols[0]), int(patch_rows[0])), cell=len(rects),
                    clip_rect=[x0, y0, x1-x0, y1-y0], follow_endpoints=False)
                overview_candidates.extend({**candidate, "source": "overview", "status": "candidate"}
                                           for candidate in candidates)
            rects.append([x0, y0, x1-x0, y1-y0])
        prior.append(values)
    statistics, calibration_info = prepare_calibration(
        image, overview_stats=overview_stats, overview_features=overview_features,
        cell_rects=rects, grid=grid, mode=calibration, cap_frac=calibration_cap_frac,
        extractor=feature_map)
    del overview_features
    declared = getattr(models, "metadata", {}).get("calibration", "unspecified")
    if isinstance(declared, dict):
        declared = declared.get("mode", "unspecified")
    calibration_cost = calibration_info["total_cost_px"]
    return {"prior": prior, "cell_rects": rects, "grid": list(grid),
            "overview_candidates": overview_candidates,
            "image_shape": [height, width],
            "baseline": {"detector": "learned", "feature_stats": statistics,
                         "detector_thresholds": thresholds,
                         "calibration": calibration, "calibration_details": calibration_info,
                         "model_calibration": declared,
                         "model_calibration_compatible": declared in ("unspecified", calibration)},
            "cost_px": calibration_cost, "calibration_cost_px": calibration_info["additional_cost_px"],
            "overview_cost_px": max(1, int(width*.25))*max(1, int(height*.25)),
            "high_scale_calibration_cost_px": calibration_info["additional_cost_px"],
            "elapsed_ms": (time.perf_counter()-started)*1000}


def learned_inspect_tile(image: np.ndarray, rect: list[int], scale: float,
                         baseline: dict, models: DetectorModels) -> dict:
    started = time.perf_counter()
    key = scale_key(scale)
    floors = {scale_key(value): float(models.threshold(value)) for value in (.25, .5, 1.0)}
    multiplier = baseline.get("threshold_multiplier", 1.0)
    threshold = effective_threshold(floors[key], multiplier)
    if "effective_thresholds" in baseline:
        declared = baseline["effective_thresholds"]
        if not isinstance(declared, dict) or key not in declared:
            raise ValueError("Effective threshold map must include the inspected scale")
        # The additive policy map is already multiplied. Validate it, but never
        # apply the multiplier twice or allow a caller to undercut model floors.
        requested = effective_threshold(floors[key], policy_floor=declared[key])
        threshold = max(threshold, requested)
    x, y, width, height = map(int, rect)
    image_h, image_w = image.shape[:2]
    if (len(rect) != 4 or any(int(v) != v for v in rect) or x < 0 or y < 0
            or width < 1 or height < 1 or x+width > image_w or y+height > image_h):
        raise ValueError("rect must be an integer rectangle inside the image")
    work_w, work_h = max(1, int(width*scale)), max(1, int(height*scale))
    work = cv2.resize(image[y:y+height, x:x+width], (work_w, work_h),
                      interpolation=cv2.INTER_AREA)
    features, _ = feature_map(work, baseline["feature_stats"][key])
    probability, kinds = models.predict(features, scale)
    detections, endpoints = _patch_components(
        probability, kinds, work_shape=(work_h, work_w), rect=[x, y, width, height],
        scale=scale, threshold=threshold, kind_names=models.kind_names,
        confirmation_candidates=baseline.get("confirmation_candidates") if scale == 1. else None,
        candidate_margin_px=baseline.get("candidate_margin_px", 32),
        linear_min_length_px=baseline.get("linear_min_length_px", 128),
        image_shape=image.shape[:2])
    # The tile's maximum probability drives escalation even below the model's
    # segmentation threshold. Only segmented components can become reports.
    score = float(probability.max())
    return {"score": score, "confidence": float(abs(score-.5)*2),
            "detections": detections, "cost_px": work_w*work_h,
            "threshold": threshold, "detector_thresholds": floors,
            "elapsed_ms": (time.perf_counter()-started)*1000, "endpoints": endpoints}
