"""Small, shared evidence gates; no image analysis or synthetic boxes are added."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

CLASSIFIER_PATCH_SIZE = 32


def _rectangle(bbox: Sequence[float]) -> tuple[float, float, float, float] | None:
    try:
        if len(bbox) != 4 or any(isinstance(value, bool) for value in bbox):
            return None
        x, y, width, height = map(float, bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)) or min(width, height) <= 0:
        return None
    right, bottom = x+width, y+height
    if not math.isfinite(right) or not math.isfinite(bottom):
        return None
    return x, y, right, bottom


def clip_bbox(bbox: Sequence[float], rect: Sequence[float]) -> list[float] | None:
    """Intersect two [x,y,w,h] rectangles without rounding or growing evidence."""
    box, bounds = _rectangle(bbox), _rectangle(rect)
    if box is None or bounds is None:
        return None
    left, top = max(box[0], bounds[0]), max(box[1], bounds[1])
    right, bottom = min(box[2], bounds[2]), min(box[3], bounds[3])
    if right <= left or bottom <= top:
        return None
    return [left, top, right-left, bottom-top]


def component_gate(detections: Iterable[dict], *, min_patches: int = 2,
                   max_per_cell: int = 3) -> list[dict]:
    """Keep >=2 classified patches and the strongest <=3 components per cell.

    Learned tools supply actual connected-component patch counts. Classic
    detectors explicitly mark their occupied-analysis-grid proxy. Missing/invalid counts
    fail closed; a thin shape never exempts a single-patch component. Inputs
    are copied, not mutated. A call without cell metadata is one cell's output.
    """
    if isinstance(min_patches, bool) or not isinstance(min_patches, int) or min_patches < 2:
        raise ValueError("min_patches must be an integer >= 2")
    if isinstance(max_per_cell, bool) or not isinstance(max_per_cell, int) or not 1 <= max_per_cell <= 3:
        raise ValueError("max_per_cell must be an integer in [1,3]")
    valid = []
    for detection in detections:
        bbox = _rectangle(detection.get("bbox"))
        count, score = detection.get("patch_count", 0), detection.get("score", 0)
        if bbox is None or isinstance(count, bool) or isinstance(score, bool):
            continue
        try:
            count, score = float(count), float(score)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(count) or count < min_patches or not math.isfinite(score):
            continue
        key = (-score, *bbox, str(detection.get("kind_guess", "unknown")),
               str(detection.get("cell", "")), str(detection.get("component_id", "")))
        valid.append((key, detection))
    selected, per_cell = [], {}
    for _, detection in sorted(valid, key=lambda entry: entry[0]):
        cell = detection.get("cell")
        if per_cell.get(cell, 0) >= max_per_cell:
            continue
        per_cell[cell] = per_cell.get(cell, 0)+1
        selected.append(dict(detection))
    return selected


def spatial_candidate_gate(detections: Iterable[dict], candidates: Iterable[dict], *,
                           margin_px: float = CLASSIFIER_PATCH_SIZE,
                           image_shape: Sequence[int] | None = None) -> list[dict]:
    """Keep detections intersecting a localized candidate plus one patch margin.

    The default margin is one confirming full-scale patch: 32 full-image pixels,
    independent of the candidate's scale. Candidate expansion
    is only an eligibility test: output boxes remain actual detected boxes,
    optionally clipped to the image. No full-cell fallback boxes are created.
    Apply ``component_gate`` separately after any policy thin-shape exception.
    """
    if isinstance(margin_px, bool) or not isinstance(margin_px, (int, float)) or not math.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be a finite nonnegative number")
    image_rect = None
    if image_shape is not None:
        if len(image_shape) < 2:
            raise ValueError("image_shape must include height and width")
        image_rect = [0, 0, image_shape[1], image_shape[0]]
        if _rectangle(image_rect) is None:
            raise ValueError("image_shape dimensions must be finite and positive")
    expanded = []
    for candidate in candidates:
        if candidate.get("localized") is False:
            continue
        bbox = _rectangle(candidate.get("bbox"))
        scale = candidate.get("scale")
        if bbox is None or isinstance(scale, bool) or scale not in (.25, .5, 1.):
            continue
        expanded.append([bbox[0]-margin_px, bbox[1]-margin_px,
                         bbox[2]-bbox[0]+2*margin_px, bbox[3]-bbox[1]+2*margin_px])
    selected = []
    for detection in detections:
        bbox = detection.get("bbox")
        if _rectangle(bbox) is None:
            continue
        if image_rect is not None:
            bbox = clip_bbox(bbox, image_rect)
            if bbox is None:
                continue
        if any(clip_bbox(bbox, candidate) is not None for candidate in expanded):
            selected.append({**detection, "bbox": list(bbox)})
    return selected
