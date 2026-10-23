"""Inference-only loader for independent, per-scale OpenCV random forests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .features import FEATURE_NAMES, FEATURE_VERSION


def scale_key(scale: float) -> str:
    if scale not in (0.25, 0.5, 1.0):
        raise ValueError("scale must be 0.25, 0.5 or 1.0")
    return f"{scale:g}"


def effective_threshold(model_floor: float, multiplier: float = 1.0,
                        policy_floor: float | None = None, cap: float = .97) -> float:
    """Raise a calibrated floor without allowing a policy/cap to lower it."""
    for value in (model_floor, multiplier, cap):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Threshold parameters must be finite numbers")
    if not 0 <= model_floor <= 1 or multiplier < 1 or not 0 <= cap <= 1:
        raise ValueError("Invalid model floor, multiplier or threshold cap")
    if policy_floor is None:
        policy_floor = model_floor
    if isinstance(policy_floor, bool) or not isinstance(policy_floor, (int, float)) or not math.isfinite(policy_floor) or not 0 <= policy_floor <= 1:
        raise ValueError("Policy floor must lie in [0,1]")
    return float(max(model_floor, min(cap, max(model_floor, policy_floor) * multiplier)))


def binary_probabilities(model, samples: np.ndarray) -> np.ndarray:
    """Class-1 vote fraction. Row zero of RTrees.getVotes lists class labels."""
    votes = model.getVotes(np.ascontiguousarray(samples, dtype=np.float32), 0)
    if votes.ndim != 2 or votes.shape[0] != len(samples) + 1:
        raise ValueError("Unexpected RTrees vote-table shape")
    classes = votes[0].astype(np.int32)
    locations = np.flatnonzero(classes == 1)
    if locations.size != 1:
        raise ValueError("Binary detector must include class 1")
    counts = votes[1:].astype(np.float32)
    return counts[:, int(locations[0])] / np.maximum(counts.sum(axis=1), 1)


class DetectorModels:
    """Load trained YAML forests without importing any training implementation."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text())
        if (self.metadata.get("feature_version") != FEATURE_VERSION
                or self.metadata.get("feature_names") != list(FEATURE_NAMES)):
            raise ValueError("Model feature schema does not match this extractor")
        self.kind_names = {int(key): str(value) for key, value in self.metadata["kind_names"].items()}
        raw_thresholds = self.metadata.get("thresholds")
        if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != {"0.25", "0.5", "1"}:
            raise ValueError("Models must declare one calibrated threshold per scale")
        self.thresholds = {}
        for key, value in raw_thresholds.items():
            effective_threshold(value)
            self.thresholds[key] = float(value)
        self._binary = {}
        self._kind = {}
        for scale in (0.25, 0.5, 1.0):
            key = scale_key(scale)
            files = self.metadata["models"][key]
            self._binary[key] = cv2.ml.RTrees_load(str(self.model_dir / files["binary"]))
            self._kind[key] = cv2.ml.RTrees_load(str(self.model_dir / files["kind"]))
            for model in (self._binary[key], self._kind[key]):
                if not model.isTrained() or model.getVarCount() != len(FEATURE_NAMES):
                    raise ValueError(f"Invalid trained forest for scale {key}")
            if key not in self.thresholds or not 0 <= self.thresholds[key] <= 1:
                raise ValueError(f"Invalid probability threshold for scale {key}")

    def threshold(self, scale: float) -> float:
        return self.thresholds[scale_key(scale)]

    @property
    def detector_thresholds(self) -> dict[str, float]:
        """JSON-safe calibrated floors, copied so callers cannot mutate weights' metadata."""
        return {scale_key(scale): self.threshold(scale) for scale in (.25, .5, 1.0)}

    def effective_threshold(self, scale: float, multiplier: float = 1.0,
                            policy_floor: float | None = None) -> float:
        return effective_threshold(self.threshold(scale), multiplier, policy_floor)

    def predict(self, features: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        if features.ndim != 3 or features.shape[2] != len(FEATURE_NAMES) or not np.isfinite(features).all():
            raise ValueError("features must be a finite H x W x D feature map")
        key = scale_key(scale)
        samples = np.ascontiguousarray(features.reshape(-1, features.shape[-1]), dtype=np.float32)
        probability = binary_probabilities(self._binary[key], samples)
        _, kinds = self._kind[key].predict(samples)
        return probability.reshape(features.shape[:2]), kinds.astype(np.int32).reshape(features.shape[:2])
