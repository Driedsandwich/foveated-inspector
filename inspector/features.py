"""Vectorized, surface-normalized OpenCV features for nonoverlapping 32 px cells.

Each filter runs on the complete image or requested tile. Only the small pooled
feature tensor survives a filter family. RMS is an explicit tail-response proxy;
it is not presented as a percentile. Calibration statistics must be computed on
the complete surface at the same scale, then reused for all its tile calls.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

PATCH_SIZE = 32
FEATURE_VERSION = "opencv_patch32_v2_1"
_CHANNELS = (
    ("blackhat3", ("mean", "rms", "max")),
    ("tophat3", ("mean", "rms", "max")),
    ("blackhat9", ("mean", "rms", "max")),
    ("tophat9", ("mean", "rms", "max")),
    ("hessian_ridge", ("rms", "max")),
    ("gabor_energy", ("rms", "max")),
    ("gabor_anisotropy", ("mean", "max")),
    ("local_std_ratio", ("mean", "max")),
    ("canny", ("mean",)),
    ("lab_l", ("mean", "rms")),
    ("lab_a", ("mean", "rms")),
    ("lab_b", ("mean", "rms")),
    ("chroma_delta", ("mean", "rms")),
)
FEATURE_NAMES = tuple(f"{channel}_{pool}" for channel, pools in _CHANNELS for pool in pools)


def _stats(values: np.ndarray, floor: float = 1.0) -> dict:
    if values.dtype == np.uint8:
        # Exact 8-bit median and MAD from a 256-bin histogram avoid a full-size
        # floating-point deviation map for morphology, gray and Lab channels.
        histogram = cv2.calcHist([values], [0], None, [256], [0, 256]).ravel().astype(np.int64)
        total = int(histogram.sum())

        def weighted_median(locations: np.ndarray, weights: np.ndarray) -> float:
            order = np.argsort(locations, kind="stable")
            cumulative = np.cumsum(weights[order])
            lower = np.searchsorted(cumulative, (total - 1) // 2 + 1)
            upper = np.searchsorted(cumulative, total // 2 + 1)
            return float((locations[order[lower]] + locations[order[upper]]) / 2)

        locations = np.arange(256, dtype=np.float64)
        median = weighted_median(locations, histogram)
        mad = weighted_median(np.abs(locations - median), histogram)
        return {"median": median, "mad": mad, "sigma": max(floor, 1.4826 * mad)}
    median = float(np.median(values))
    deviation = values.astype(np.float32, copy=True)
    deviation -= median
    np.abs(deviation, out=deviation)
    mad = float(np.median(deviation))
    return {"median": median, "mad": mad, "sigma": max(floor, 1.4826 * mad)}


def _pool(values: np.ndarray, modes: tuple[str, ...]) -> list[np.ndarray]:
    height, width = values.shape
    rows, cols = math.ceil(height / PATCH_SIZE), math.ceil(width / PATCH_SIZE)
    bottom, right = rows * PATCH_SIZE - height, cols * PATCH_SIZE - width
    padded = (values if bottom == 0 and right == 0
              else np.pad(values, ((0, bottom), (0, right)), constant_values=0))
    blocks = padded.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE)
    row_sizes = np.minimum(PATCH_SIZE, height - np.arange(rows) * PATCH_SIZE)
    col_sizes = np.minimum(PATCH_SIZE, width - np.arange(cols) * PATCH_SIZE)
    counts = np.outer(row_sizes, col_sizes).astype(np.float32)
    result = []
    for mode in modes:
        if mode == "mean":
            result.append(blocks.sum(axis=(1, 3)) / counts)
        elif mode == "rms":
            squared = np.square(padded)
            pooled = squared.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE).sum(axis=(1, 3)) / counts
            result.append(np.sqrt(pooled))
        elif mode == "max":
            # These channels are nonnegative; exterior zero padding is neutral.
            result.append(blocks.max(axis=(1, 3)))
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    return result


def feature_map(image: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return (ceil(H/32) x ceil(W/32) x 29 float32 features, JSON stats).

    Supplied statistics are treated as immutable. Filters at tile boundaries
    naturally differ from a whole-image filtering pass; only normalization and
    colour reference are shared. This API does not cache whole-image features.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a nonempty uint8 BGR array")
    fitting = stats is None
    if fitting:
        calibration = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                       "patch_size": PATCH_SIZE, "channels": {}}
    else:
        if stats.get("feature_version") != FEATURE_VERSION or stats.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Incompatible feature calibration metadata")
        calibration = stats
    channels = calibration["channels"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if fitting:
        calibration["gray"] = _stats(gray, 2.0)
    columns: list[np.ndarray] = []

    def emit(name: str, response: np.ndarray, *, signed: bool = False,
             floor: float = 1.0, raw: bool = False) -> None:
        if fitting:
            channels[name] = _stats(response, floor)
        reference = channels[name]
        normal = response.astype(np.float32)
        if not raw:
            normal -= float(reference["median"])
            normal /= float(reference["sigma"])
            if not signed:
                np.maximum(normal, 0, out=normal)
        np.clip(normal, -32, 32, out=normal)
        columns.extend(_pool(normal, dict(_CHANNELS)[name]))

    for size in (3, 9):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        emit(f"blackhat{size}", cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
        emit(f"tophat{size}", cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))

    dxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    difference = dxx - dyy
    dxx += dyy
    del dyy
    np.square(difference, out=difference)
    dxy *= 2
    np.square(dxy, out=dxy)
    difference += dxy
    del dxy
    np.sqrt(difference, out=difference)
    dxx += difference
    del difference
    dxx *= 0.5
    np.maximum(dxx, 0, out=dxx)
    emit("hessian_ridge", dxx)
    del dxx

    energy_max = np.zeros(gray.shape, np.float32)
    energy_sum = np.zeros(gray.shape, np.float32)
    for angle in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        kernel = cv2.getGaborKernel((9, 9), 2, angle, 5, 0.6, 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        kernel /= max(float(np.abs(kernel).sum()), 1e-6)
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        np.abs(response, out=response)
        energy_sum += response
        np.maximum(energy_max, response, out=energy_max)
    del response
    emit("gabor_energy", energy_max)
    energy_sum *= 0.25
    energy_sum += 1e-3
    energy_max /= energy_sum
    del energy_sum
    emit("gabor_anisotropy", energy_max, floor=0.1)
    del energy_max

    gray_float = gray.astype(np.float32)
    local_mean = cv2.boxFilter(gray_float, -1, (9, 9))
    np.square(gray_float, out=gray_float)
    variance = cv2.boxFilter(gray_float, -1, (9, 9))
    del gray_float
    np.square(local_mean, out=local_mean)
    variance -= local_mean
    del local_mean
    np.maximum(variance, 0, out=variance)
    np.sqrt(variance, out=variance)
    variance /= float(calibration["gray"]["sigma"])
    emit("local_std_ratio", variance, floor=0.1)
    del variance
    noise = float(calibration["gray"]["sigma"])
    edges = cv2.Canny(gray, max(10, 0.7 * noise), max(30, 2 * noise))
    emit("canny", edges.astype(np.float32) / 255, raw=True)
    del edges, gray

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for index, name in enumerate(("lab_l", "lab_a", "lab_b")):
        emit(name, lab[:, :, index], signed=True, floor=2.0)
    delta_a = lab[:, :, 1].astype(np.float32)
    delta_a -= float(channels["lab_a"]["median"])
    delta_a /= float(channels["lab_a"]["sigma"])
    delta_b = lab[:, :, 2].astype(np.float32)
    delta_b -= float(channels["lab_b"]["median"])
    delta_b /= float(channels["lab_b"]["sigma"])
    del lab
    np.square(delta_a, out=delta_a)
    np.square(delta_b, out=delta_b)
    delta_a += delta_b
    del delta_b
    np.sqrt(delta_a, out=delta_a)
    emit("chroma_delta", delta_a, floor=0.5)
    features = np.stack(columns, axis=-1).astype(np.float32)
    return features, calibration


def background_prior_from_features(features: np.ndarray, grid: tuple[int, int] = (16, 16)) -> list:
    """Unsupervised quarter-view texture evidence shared by fit and inference."""
    indices = [FEATURE_NAMES.index(name) for name in
               ("blackhat3_max", "blackhat9_max", "gabor_energy_max")]
    evidence = features[:, :, indices].mean(axis=2)
    prior = 1 - np.exp(-np.maximum(evidence, 0) / 4)
    return cv2.resize(prior, (int(grid[1]), int(grid[0])), interpolation=cv2.INTER_AREA).tolist()


def prepare_calibration(image: np.ndarray, *, overview_stats: dict | None = None,
                        overview_features: np.ndarray | None = None,
                        prior=None, cell_rects: list[list[int]] | None = None,
                        grid: tuple[int, int] = (16, 16), mode: str = "sampled",
                        cap_frac: float = 0.02, tile_count: int = 8,
                        analysis_side: int = 128, extractor=None) -> tuple[dict, dict]:
    """Prepare per-scale normalization without inspecting every high-res pixel.

    The paid quarter overview is reused when supplied. At higher scales, eight
    spatial strata each select a low-prior cell. Crops are resized individually
    before assembly into a bounded mosaic. Its gray/Lab distribution contains
    the sampled pixels exactly; filter statistics are approximate because mosaic
    joins introduce synthetic neighbours. No per-tile median averaging is used.
    ``whole`` explicitly enables the expensive full-image calibration ablation.
    """
    if mode not in ("sampled", "whole") or not 0 <= cap_frac <= 0.02:
        raise ValueError("calibration must be sampled/whole with cap_frac in [0,.02]")
    if tile_count != 8 or analysis_side < 1:
        raise ValueError("Calibration uses eight spatial strata and a positive tile size")
    extract = feature_map if extractor is None else extractor
    height, width = image.shape[:2]
    area = height * width
    quarter_w, quarter_h = max(1, width // 4), max(1, height // 4)
    overview_cost = quarter_w * quarter_h
    newly_paid_overview = overview_stats is None
    if overview_stats is None:
        overview = cv2.resize(image, (quarter_w, quarter_h), interpolation=cv2.INTER_AREA)
        overview_features, overview_stats = extract(overview)
        del overview
    elif prior is None and overview_features is None:
        overview = cv2.resize(image, (quarter_w, quarter_h), interpolation=cv2.INTER_AREA)
        overview_features, _ = extract(overview, overview_stats)
        del overview
    prior_source = "supplied_prior" if prior is not None else "low_response_quarter_features"
    if prior is None:
        prior = background_prior_from_features(overview_features, grid)
    statistics = {"0.25": overview_stats}
    regions = []
    additional = 0
    if mode == "whole":
        for scale in (0.5, 1.0):
            work_w, work_h = max(1, int(width * scale)), max(1, int(height * scale))
            working = image if scale == 1 else cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
            _, statistics[f"{scale:g}"] = extract(working)
            additional += work_w * work_h
            del working
    else:
        rows, cols = map(int, grid)
        if not (1 <= rows <= height and 1 <= cols <= width):
            raise ValueError("Calibration grid must fit image")
        if cell_rects is None:
            cell_rects = [[c * width // cols, r * height // rows,
                           (c + 1) * width // cols - c * width // cols,
                           (r + 1) * height // rows - r * height // rows]
                          for r in range(rows) for c in range(cols)]
        belief = np.zeros(rows * cols) if prior is None else np.asarray(prior, dtype=np.float64).ravel()
        if len(cell_rects) != rows * cols or belief.size != rows * cols or not np.isfinite(belief).all():
            raise ValueError("Calibration prior/rectangles do not match grid")
        selected = []
        for stratum_y in range(2):
            for stratum_x in range(4):
                centre_x, centre_y = (stratum_x + 0.5) * width / 4, (stratum_y + 0.5) * height / 2
                candidates = [index for index, (x, y, w, h) in enumerate(cell_rects)
                              if stratum_x * width / 4 <= x + w / 2 < (stratum_x + 1) * width / 4
                              and stratum_y * height / 2 <= y + h / 2 < (stratum_y + 1) * height / 2]
                if not candidates:
                    candidates = list(range(len(cell_rects)))
                selected.append(min(candidates, key=lambda index: (
                    belief[index], (cell_rects[index][0] + cell_rects[index][2] / 2 - centre_x) ** 2
                    + (cell_rects[index][1] + cell_rects[index][3] / 2 - centre_y) ** 2, index)))
        cap = int(area * cap_frac)
        side = min(analysis_side, int(math.sqrt(cap // (2 * tile_count))),
                   max(1, width // 2), max(1, height // 2))
        if side == 0:
            # Tiny/zero-cap inputs reuse the already-paid overview statistics.
            statistics.update({"0.5": overview_stats, "1": overview_stats})
        else:
            for scale in (0.5, 1.0):
                mosaic = np.empty((2 * side, 4 * side, 3), dtype=np.uint8)
                for index, cell_index in enumerate(selected):
                    cell = cell_rects[cell_index]
                    cx, cy = cell[0] + cell[2] / 2, cell[1] + cell[3] / 2
                    crop_side = int(side / scale)
                    x = max(0, min(width - crop_side, int(cx - crop_side / 2)))
                    y = max(0, min(height - crop_side, int(cy - crop_side / 2)))
                    crop = image[y:y + crop_side, x:x + crop_side]
                    tile = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
                    r, c = divmod(index, 4)
                    mosaic[r * side:(r + 1) * side, c * side:(c + 1) * side] = tile
                    regions.append({"scale": scale, "rect": [x, y, crop_side, crop_side],
                                    "work_shape": [side, side], "source_cell": selected[index]})
                _, statistics[f"{scale:g}"] = extract(mosaic)
                additional += mosaic.shape[0] * mosaic.shape[1]
                del mosaic
    return statistics, {"mode": mode, "overview_cost_px": overview_cost,
                        "overview_newly_processed_px": overview_cost if newly_paid_overview else 0,
                        "additional_cost_px": additional, "total_cost_px": overview_cost + additional,
                        "high_scale_cap_px": int(area * cap_frac), "sample_regions": regions,
                        "prior_source": prior_source,
                        "approximation": "sample mosaic; filter seams may affect robust response statistics"
                        if mode == "sampled" else "whole image at each scale"}
