"""Transport-neutral tools and a single-image local implementation."""

from pathlib import Path
import copy
import time
from typing import Any, Protocol

import cv2
import numpy as np

from .coarse import coarse_scan
from .fine import inspect_tile


class ToolBackend(Protocol):
    """Implementations return JSON values; image_ref is opaque to the policy."""

    def coarse_scan(self, image_ref: str, **kw: Any) -> dict: ...

    def inspect_tile(self, image_ref: str, rect: list[int], scale: float,
                     baseline: dict) -> dict: ...


def json_values(value: Any) -> Any:
    """Remove NumPy containers/scalars at the transport boundary."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    return value


class LocalBackend:
    """Decode/cache one image. Costs count input pixels, not filter operations.

    max_size optionally defines a smaller evaluation coordinate system. All
    rectangles, labels and cost denominators must then use that same system.
    Instances are intended for one sequential inspection stream, not threads.
    """

    def __init__(self, max_size: int | None = None, *, detector: str = "classic",
                 model_dir: str | Path | None = None, calibration: str = "sampled",
                 calibration_cap_frac: float = 0.02):
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        if detector not in ("classic", "learned"):
            raise ValueError("detector must be classic or learned")
        self.detector = detector
        if calibration not in ("sampled", "whole") or not 0 <= calibration_cap_frac <= 0.02:
            raise ValueError("Invalid calibration mode or pixel cap")
        self.calibration = calibration
        self.calibration_cap_frac = calibration_cap_frac
        self.models = None
        if detector == "learned":
            from .model import DetectorModels
            self.models = DetectorModels(model_dir or Path(__file__).resolve().parents[1]/"models")
        self._coarse_cache: dict = {}
        self._image_ref: str | None = None
        self._image: np.ndarray | None = None
        self.reset_accounting()

    def reset_accounting(self) -> None:
        self.coarse_cost_px = 0
        self.physical_coarse_cost_px = 0
        self.fine_cost_px = 0
        self.n_tool_calls = 0

    def image(self, image_ref: str) -> np.ndarray:
        """Read-only by contract; callers must not mutate the cached array."""
        ref = str(Path(image_ref).resolve())
        if ref != self._image_ref:
            # Release the previous pipeline before decoding another 4K image.
            self._image = None
            self._image_ref = None
            self._coarse_cache.clear()
            img = cv2.imread(ref, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot decode image: {image_ref}")
            if self.max_size and max(img.shape[:2]) > self.max_size:
                ratio = self.max_size / max(img.shape[:2])
                img = cv2.resize(img, (max(1, int(img.shape[1] * ratio)),
                                       max(1, int(img.shape[0] * ratio))),
                                 interpolation=cv2.INTER_AREA)
            self._image, self._image_ref = img, ref
        assert self._image is not None
        return self._image

    def coarse_scan(self, image_ref: str, **kw: Any) -> dict:
        image = self.image(image_ref)
        kw.setdefault("calibration", self.calibration)
        kw.setdefault("calibration_cap_frac", self.calibration_cap_frac)
        key = (tuple(kw.get("grid", (16, 16))), kw.get("work_width", 1024),
               kw["calibration"], kw["calibration_cap_frac"])
        reused = key in self._coarse_cache
        if not reused:
            started = time.perf_counter()
            result = json_values(coarse_scan(image, **kw, detector=self.detector, models=self.models))
            result["elapsed_ms"] = (time.perf_counter()-started)*1000
            self._coarse_cache[key] = result
            self.physical_coarse_cost_px += int(result["cost_px"])
        result = copy.deepcopy(self._coarse_cache[key])
        result["reused_preparation"] = reused
        self.coarse_cost_px += int(result["cost_px"])
        self.n_tool_calls += 1
        return result

    def inspect_tile(self, image_ref: str, rect: list[int], scale: float,
                     baseline: dict) -> dict:
        result = json_values(inspect_tile(self.image(image_ref), rect, scale, baseline,
                                         detector=self.detector, models=self.models))
        self.fine_cost_px += int(result["cost_px"])
        self.n_tool_calls += 1
        return result
