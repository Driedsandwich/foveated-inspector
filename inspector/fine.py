"""Tile-local OpenCV defect evidence with explicit sample cost and coordinates."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .coarse import _robust_stats
from .components import CLASSIFIER_PATCH_SIZE


def _odd(value: float, minimum: int = 3) -> int:
    return max(minimum, int(round(value)) | 1)


def _length(mask: np.ndarray) -> float:
    """Estimate skeleton arc length, counting diagonal links as sqrt(2)."""
    # Tight component boxes often have foreground on every border. Both
    # thinning and distanceTransform need explicit exterior background pixels.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skeleton = cv2.ximgproc.thinning(padded)[1:-1, 1:-1] > 0
        # Count links once. This is exact for a one-pixel unbranched polyline.
        horizontal = np.count_nonzero(skeleton[:, 1:] & skeleton[:, :-1])
        vertical = np.count_nonzero(skeleton[1:, :] & skeleton[:-1, :])
        diagonal = (np.count_nonzero(skeleton[1:, 1:] & skeleton[:-1, :-1])
                    + np.count_nonzero(skeleton[1:, :-1] & skeleton[:-1, 1:]))
        return max(1.0, float(horizontal + vertical + math.sqrt(2) * diagonal))
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    positive = distance[mask > 0]
    return float(np.count_nonzero(mask) / max(1.0, 2 * np.median(positive)))


def _occupied_patch_count(mask: np.ndarray, x: int, y: int, work_width: int) -> int:
    """Classic proxy: count 32px working-grid blocks touched by actual foreground.

    This is not a learned classifier count. Using occupied blocks, rather than
    component area, preserves long one-pixel cracks without counting empty
    blocks inside a ragged component's bounding box.
    """
    py, px = np.nonzero(mask)
    columns = (work_width+CLASSIFIER_PATCH_SIZE-1)//CLASSIFIER_PATCH_SIZE
    blocks = ((py+y)//CLASSIFIER_PATCH_SIZE)*columns+(px+x)//CLASSIFIER_PATCH_SIZE
    return int(np.unique(blocks).size)


def inspect_tile(img_bgr: np.ndarray, rect: list[int] | tuple[int, int, int, int],
                 scale: float, baseline: dict | None, *, detector: str = "classic", models=None) -> dict:
    """Inspect one crop at 1/4, 1/2 or full resolution and return JSON-safe evidence.

    Costs count actual resized pixels (floored dimensions). The whole crop is
    charged on every call, including repeat inspections at a higher scale.
    Per-tile robust response statistics handle scale-dependent background noise;
    overview colour/texture estimates are additional conservative noise floors.
    """
    if detector == "learned":
        from .learned import learned_inspect_tile
        if models is None:
            from pathlib import Path
            from .model import DetectorModels
            models = DetectorModels(Path(__file__).resolve().parents[1] / "models")
        return learned_inspect_tile(img_bgr, list(rect), scale, baseline, models)
    if detector != "classic":
        raise ValueError("detector must be classic or learned")
    started = time.perf_counter()
    if scale not in (0.25, 0.5, 1.0):
        raise ValueError("scale must be one of 0.25, 0.5, 1.0")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be a uint8 BGR image")
    if len(rect) != 4 or any(int(v) != v for v in rect):
        raise ValueError("rect must contain four integer coordinates")
    x, y, width, height = map(int, rect)
    image_h, image_w = img_bgr.shape[:2]
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > image_w or y + height > image_h:
        raise ValueError("rect must lie within the image")
    work_w, work_h = max(1, int(width * scale)), max(1, int(height * scale))
    crop = img_bgr[y:y + height, x:x + width]
    work = cv2.resize(crop, (work_w, work_h), interpolation=cv2.INTER_AREA)
    sx, sy = width / work_w, height / work_h
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(13 * scale),) * 2)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    raw = np.maximum(dark, light)
    eq_response = np.maximum(cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel),
                             cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel))
    stats = _robust_stats(raw, 2.0)
    baseline = baseline or {}
    texture = baseline.get("responses", {})
    overview_floor = max(float(texture.get("blackhat", {}).get("q95", 0)),
                         float(texture.get("tophat", {}).get("q95", 0)))
    threshold = max(6.0, stats["median"] + 3.5 * stats["sigma"], overview_floor * 0.85)
    binary = ((raw > threshold) & (eq_response > 5)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    detections: list[dict] = []
    endpoints: list[dict] = []
    count, labels, components, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for component in range(1, count):
        bx, by, bw, bh, area = map(int, components[component])
        if area < max(4, int(10 * scale * scale)) or max(bw, bh) < max(5, int(12 * scale)):
            continue
        mask = (labels[by:by + bh, bx:bx + bw] == component).astype(np.uint8) * 255
        points = cv2.findNonZero(mask)
        (_, _), sides, _ = cv2.minAreaRect(points)
        long_side, short_side = max(sides) + 1, min(sides) + 1
        elongation = long_side / short_side
        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 3)[1:-1, 1:-1]
        thickness = max(1.0, float(2 * np.median(distance[mask > 0])))
        length = _length(mask)
        component_response = raw[by:by + bh, bx:bx + bw][mask > 0]
        strength = float(np.median(component_response)) / max(1.0, threshold)
        # A ragged compact texture component can have a very long skeleton.
        # Require geometric elongation and a sustained span before calling it
        # a line; skeleton length remains a measurement, not evidence of shape.
        pad = max(2, int(5 * scale))
        left, top = max(0, bx - pad), max(0, by - pad)
        right, bottom = min(work_w, bx + bw + pad), min(work_h, by + bh + pad)
        component_mask = (labels[top:bottom, left:right] == component).astype(np.uint8)
        ring = ((cv2.dilate(component_mask, np.ones((2 * pad + 1,) * 2, np.uint8)) > 0)
                & (component_mask == 0))
        neighbourhood = gray_f[top:bottom, left:right]
        surrounding = neighbourhood[ring]
        component_values = neighbourhood[component_mask > 0]
        if surrounding.size:
            contrast = abs(float(np.median(component_values)) - float(np.median(surrounding)))
            contrast_z = contrast / _robust_stats(surrounding, 3.0)["sigma"]
        else:
            contrast_z = 0.0
        line_shape = min(1.0, max(0.0, (elongation - 2.0) / 6))
        span_support = min(1.0, long_side / max(16.0, 48 * scale))
        line_score = min(0.99, 0.30 + 0.24 * line_shape + 0.18 * span_support
                         + 0.16 * min(contrast_z / 2, 1) + 0.10 * min(strength / 1.8, 1))
        is_line = (elongation >= 3 and long_side >= max(12, 24 * scale)
                   and long_side / thickness >= 5 and contrast_z >= 0.6)
        compact = (area >= max(10, 70 * scale * scale)
                   and area / (bw * bh) > 0.25 and contrast_z >= 0.6)
        blob_score = min(0.92, 0.30 + 0.17 * min(strength, 2) + 0.12 * min(area / 100, 1))
        if not is_line and not compact:
            continue
        score = float(line_score if is_line else blob_score)
        if score < 0.48:
            continue
        fx0, fy0 = x + int(bx * sx), y + int(by * sy)
        fx1 = min(x + width, x + math.ceil((bx + bw) * sx))
        fy1 = min(y + height, y + math.ceil((by + bh) * sy))
        darker = float(np.mean(dark[by:by + bh, bx:bx + bw][mask > 0])) >= float(
            np.mean(light[by:by + bh, bx:bx + bw][mask > 0]))
        detections.append({"bbox": [fx0, fy0, fx1 - fx0, fy1 - fy0],
                           "kind_guess": ("crack" if darker else "scratch") if is_line else "spall",
                           "score": score, "length_px": float(length * (sx + sy) / 2),
                           "width_px": float(thickness * (sx + sy) / 2),
                           "elongation": float(elongation),
                           "patch_count": _occupied_patch_count(mask, bx, by, work_w),
                           "patch_count_source": "occupied_analysis_grid_proxy",
                           "component_id": f"edge_{component}", "localized": True,
                           "scale": float(scale)})
        if is_line:
            py, px = np.nonzero(mask)
            px, py = px + bx, py + by
            margin = max(2, int(3 * scale))
            for edge_mask, dx, dy in ((px <= margin, -1, 0), (px >= work_w - 1 - margin, 1, 0),
                                       (py <= margin, 0, -1), (py >= work_h - 1 - margin, 0, 1)):
                if np.any(edge_mask):
                    endpoints.append({"x": float(x + np.mean(px[edge_mask]) * sx),
                                      "y": float(y + np.mean(py[edge_mask]) * sy),
                                      "dx": dx, "dy": dy, "component_id": f"edge_{component}"})
    # Stains are broad low-frequency deviations. Difference of Gaussian filters
    # removes gradual illumination; a robust tile threshold avoids flat noise.
    sigma = max(2.0, 10 * scale)
    smooth = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    broad = np.abs(smooth - cv2.GaussianBlur(smooth, (0, 0), max(4.0, 40 * scale)))
    broad_stats = _robust_stats(broad, 1.0)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_score = np.zeros_like(broad)
    for channel in (1, 2):
        channel_smooth = cv2.GaussianBlur(lab[:, :, channel], (0, 0), sigma)
        residual = np.abs(channel_smooth - cv2.GaussianBlur(channel_smooth, (0, 0), max(4.0, 40 * scale)))
        color_floor = max(1.5, float(baseline.get("lab", [{}, {}, {}])[channel].get("sigma", 2.0)) * 0.35)
        np.maximum(color_score, residual / color_floor, out=color_score)
    broad_threshold = max(4.0, broad_stats["median"] + 4 * broad_stats["sigma"],
                          float(baseline.get("broad", {}).get("q95", 0)) * 1.25)
    blob_mask = ((broad > broad_threshold) | (color_score > 4.0)).astype(np.uint8) * 255
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(blob_mask, connectivity=8)
    for component in range(1, n_blobs):
        bx, by, bw, bh, area = map(int, blob_stats[component])
        if area < max(20, int(180 * scale * scale)) or area > work_w * work_h * 0.6:
            continue
        if min(bw, bh) < max(4, int(10 * scale)) or max(bw, bh) / min(bw, bh) > 4:
            continue
        bbox = [x + int(bx * sx), y + int(by * sy),
                min(width - int(bx * sx), math.ceil(bw * sx)),
                min(height - int(by * sy), math.ceil(bh * sy))]
        if any(_bbox_overlap(bbox, d["bbox"]) > 0.5 for d in detections):
            continue
        strength = float(np.percentile(broad[by:by + bh, bx:bx + bw], 75)) / broad_threshold
        score = min(0.85, 0.45 + 0.12 * strength)
        mask = (blob_labels[by:by+bh, bx:bx+bw] == component).astype(np.uint8)
        (_, _), sides, _ = cv2.minAreaRect(cv2.findNonZero(mask))
        elongation = (max(sides)+1)/(min(sides)+1)
        detections.append({"bbox": bbox, "kind_guess": "stain", "score": float(score),
                           "length_px": 0.0, "width_px": float(min(bw * sx, bh * sy)),
                           "elongation": float(elongation),
                           "patch_count": _occupied_patch_count(mask, bx, by, work_w),
                           "patch_count_source": "occupied_analysis_grid_proxy",
                           "component_id": f"blob_{component}", "localized": True,
                           "scale": float(scale)})
    detections.sort(key=lambda d: (-d["score"], d["bbox"]))
    # Bound transport size on exceptionally busy surfaces; keep strongest evidence.
    detections = detections[:64]
    retained = {detection["component_id"] for detection in detections}
    endpoints = [endpoint for endpoint in endpoints if endpoint["component_id"] in retained]
    score = max((d["score"] for d in detections), default=0.0)
    confidence = abs(score - 0.5) * 2
    return {"score": float(score), "confidence": float(confidence), "detections": detections,
            "cost_px": work_w * work_h, "elapsed_ms": (time.perf_counter() - started) * 1000,
            "endpoints": endpoints[:128]}


def _bbox_overlap(a: list[int], b: list[int]) -> float:
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return intersection / max(1, min(a[2] * a[3], b[2] * b[3]))
