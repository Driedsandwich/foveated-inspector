"""Self-calibrating overview analysis; no full-resolution response maps are built."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _robust_stats(values: np.ndarray, floor: float = 1.0) -> dict:
    """Return JSON-safe location/scale estimates, with a noise floor for flat views."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - median)))
    return {"median": median, "mad": mad, "sigma": max(floor, 1.4826 * mad),
            "q95": float(np.percentile(values, 95))}


def _positive_z(values: np.ndarray, stats: dict) -> np.ndarray:
    return np.maximum(0.0, (values.astype(np.float32) - stats["median"]) / stats["sigma"])


def coarse_scan(img_bgr: np.ndarray, grid: tuple[int, int] = (16, 16),
                work_width: int = 1024, *, detector: str = "classic", models=None,
                calibration: str = "sampled", calibration_cap_frac: float = 0.02) -> dict:
    """Rank grid cells from a reduced overview using robust per-image statistics.

    ``grid`` is (rows, columns). Rectangle coordinates are integer full-image
    [x, y, width, height] values in row-major order. ``cost_px`` counts image
    samples in this working view, rather than CPU operations across filters.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be a uint8 BGR image")
    if detector == "learned":
        from .learned import learned_coarse_scan
        if models is None:
            from pathlib import Path
            from .model import DetectorModels
            models = DetectorModels(Path(__file__).resolve().parents[1] / "models")
        return learned_coarse_scan(img_bgr, models, grid, calibration=calibration,
                                   calibration_cap_frac=calibration_cap_frac)
    if detector != "classic":
        raise ValueError("detector must be classic or learned")
    height, width = img_bgr.shape[:2]
    rows, cols = map(int, grid)
    if not (1 <= rows <= height and 1 <= cols <= width) or work_width < 1:
        raise ValueError("grid and work_width must be positive and fit the image")
    ratio = min(1.0, work_width / width)
    work_w = max(cols, min(width, int(width * ratio)))
    work_h = max(rows, min(height, int(height * ratio)))
    work = cv2.resize(img_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    white = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (9, 9))
    variance = cv2.boxFilter(gray_f * gray_f, -1, (9, 9)) - mean * mean
    local_std = np.sqrt(np.maximum(variance, 0))
    gabor = np.zeros_like(gray_f)
    for angle in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        wave = cv2.getGaborKernel((13, 13), 2.5, angle, 6.0, 0.6, 0,
                                  ktype=cv2.CV_32F)
        wave -= wave.mean()
        wave /= max(float(np.sum(np.abs(wave))), 1e-6)
        response = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, wave))
        np.maximum(gabor, response, out=gabor)
    responses = {"blackhat": black, "tophat": white,
                 "local_std": local_std, "gabor": gabor}
    statistics = {key: _robust_stats(value) for key, value in responses.items()}
    normalised = {key: _positive_z(value, statistics[key])
                  for key, value in responses.items()}
    combined = (0.40 * np.maximum(normalised["blackhat"], normalised["tophat"])
                + 0.35 * normalised["gabor"] + 0.25 * normalised["local_std"])
    # Broad low-frequency changes complement edge responses for stains.
    low = cv2.GaussianBlur(gray_f, (0, 0), 5)
    broad = np.abs(low - cv2.GaussianBlur(low, (0, 0), 21))
    broad_stats = _robust_stats(broad, 1.0)
    combined = np.maximum(combined, 0.5 * _positive_z(broad, broad_stats))
    priors: list[list[float]] = []
    rects: list[list[int]] = []
    for row in range(rows):
        row_prior = []
        y0, y1 = row * height // rows, (row + 1) * height // rows
        wy0, wy1 = row * work_h // rows, (row + 1) * work_h // rows
        for col in range(cols):
            x0, x1 = col * width // cols, (col + 1) * width // cols
            wx0, wx1 = col * work_w // cols, (col + 1) * work_w // cols
            value = float(np.percentile(combined[wy0:wy1, wx0:wx1], 97))
            row_prior.append(float(1.0 - math.exp(-value / 2.5)))
            rects.append([x0, y0, x1 - x0, y1 - y0])
        priors.append(row_prior)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    baseline = {"gray": _robust_stats(gray), "responses": statistics,
                "lab": [_robust_stats(lab[:, :, channel], 2.0) for channel in range(3)],
                "broad": broad_stats, "working_scale": work_w / width,
                "working_shape": [work_h, work_w]}
    return {"prior": priors, "cell_rects": rects, "grid": [rows, cols],
            "image_shape": [height, width], "baseline": baseline,
            "cost_px": work_h * work_w}
