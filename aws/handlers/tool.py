"""Strict direct-Lambda inspection tool boundary. No raw errors/events are logged."""

from __future__ import annotations

import json
import math
import os
import re

from aws.runtime import get_backend

MAX_BASELINE_BYTES = 128 * 1024
_COMMON = {"action", "tool", "image_ref", "detector", "calibration", "calibration_cap_frac", "model_prefix"}


def _integer(value, low, high):
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def validate_request(event):
    if not isinstance(event, dict) or ("action" in event) == ("tool" in event):
        raise ValueError("Exactly one action discriminator is required")
    action = event.get("action", event.get("tool"))
    action = {"coarse": "coarse_scan", "inspect": "inspect_tile"}.get(action, action)
    if action not in ("coarse_scan", "inspect_tile"):
        raise ValueError("Unknown tool")
    allowed = _COMMON | ({"grid", "work_width"} if action == "coarse_scan" else {"rect", "scale", "baseline"})
    if set(event) - allowed:
        raise ValueError("Unknown request fields")
    reference = event.get("image_ref")
    if not isinstance(reference, str) or not 1 <= len(reference) <= 1100:
        raise ValueError("Invalid image reference")
    if event.get("detector", "classic") not in ("classic", "learned"):
        raise ValueError("Unknown detector")
    if event.get("calibration", "sampled") not in ("sampled", "whole"):
        raise ValueError("Unknown calibration")
    if "model_prefix" in event:
        namespace = os.environ.get("X132_PREFIX", "x132/").rstrip("/") + "/"
        value = event["model_prefix"]
        if not isinstance(value, str) or not re.fullmatch(re.escape(namespace) + r"models/[0-9a-f]{64}/", value):
            raise ValueError("Model prefix must identify an immutable application revision")
    cap = event.get("calibration_cap_frac", .02)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 <= cap <= .02:
        raise ValueError("Invalid calibration cap")
    if action == "coarse_scan":
        grid = event.get("grid", [16, 16])
        if not isinstance(grid, (list, tuple)) or len(grid) != 2 or not all(_integer(v, 1, 64) for v in grid):
            raise ValueError("Invalid grid")
        if not _integer(event.get("work_width", 1024), 1, 2048):
            raise ValueError("Invalid overview width")
    else:
        rect = event.get("rect")
        if (not isinstance(rect, list) or len(rect) != 4 or not all(_integer(v, 0, 8192) for v in rect)
                or rect[2] < 1 or rect[3] < 1 or rect[0] + rect[2] > 8192 or rect[1] + rect[3] > 8192):
            raise ValueError("Invalid rectangle")
        scale = event.get("scale")
        if isinstance(scale, bool) or scale not in (.25, .5, 1.0):
            raise ValueError("Invalid scale")
        baseline = event.get("baseline")
        if not isinstance(baseline, dict) or len(json.dumps(baseline, allow_nan=False).encode()) > MAX_BASELINE_BYTES:
            raise ValueError("Invalid or oversized baseline")
    return action


def handler(event, context=None, *, backend=None, s3_client=None, lambda_client=None):
    try:
        action = validate_request(event)
        backend = backend or get_backend(detector=event.get("detector", "classic"),
                                          calibration=event.get("calibration", "sampled"),
                                          model_prefix=event.get("model_prefix"),
                                          s3_client=s3_client, lambda_client=lambda_client)
        # Resolve bucket/prefix/sample identity before learned model loading.
        reference = backend.resolve_image_ref(event["image_ref"])
        if action == "coarse_scan":
            result = backend.coarse_scan(reference, grid=tuple(event.get("grid", [16, 16])),
                                         work_width=event.get("work_width", 1024),
                                         calibration=event.get("calibration", "sampled"),
                                         calibration_cap_frac=event.get("calibration_cap_frac", .02))
        else:
            result = backend.inspect_tile(reference, event["rect"], event["scale"], event["baseline"])
        return {"ok": True, "result": result}
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return {"ok": False, "error": {"code": "invalid_request", "message": "The inspection request is invalid."}}
    except Exception:
        return {"ok": False, "error": {"code": "tool_failed", "message": "The inspection tool could not complete."}}


lambda_handler = handler
