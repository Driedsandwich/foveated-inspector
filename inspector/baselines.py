"""Comparators sharing confirmation thresholds and raw-evidence spatial merging."""

import math
import time

import numpy as np

from .agent import (REPORTING_VERSION, InspectorConfig, flatten_cell_reports, policy_baseline,
                    replace_cell_reports, tile_cost)
from .tools import ToolBackend
from .components import component_gate


def run_baseline(method: str, backend: ToolBackend, image_ref: str,
                 budget_frac: float = 0.1, seed: int = 0,
                 config: InspectorConfig | None = None,
                 coarse_threshold: float = 0.75) -> dict:
    """Full-resolution comparators confirm reports; coarse_only reports overview boxes.

    Exhaustive ignores the fine budget. Random/raster stop before overspending.
    Every method pays for common calibration. Coarse-only reports are explicitly
    marked as low-resolution evidence, without full-resolution spatial merging.
    Classic tools lacking overview components use an explicit prior-cell fallback.
    """
    if method not in {"exhaustive", "random", "raster", "coarse_only"}:
        raise ValueError(f"Unknown baseline: {method}")
    if not math.isfinite(budget_frac) or not 0 <= budget_frac <= 1:
        raise ValueError("budget_frac must be in [0, 1]")
    if not 0 <= coarse_threshold <= 1:
        raise ValueError("coarse_threshold must be in [0, 1]")
    cfg = config or InspectorConfig()
    started = time.perf_counter()
    coarse = backend.coarse_scan(image_ref, grid=cfg.grid, work_width=cfg.work_width)
    baseline = policy_baseline(coarse["baseline"], cfg)
    coarse = {**coarse, "baseline": baseline}
    height, width = coarse["image_shape"][:2]
    area = height * width
    limit = area if method == "exhaustive" else math.floor(area * budget_frac)
    rects = coarse["cell_rects"]
    prior = np.asarray(coarse["prior"]).reshape(-1)
    order = np.arange(len(rects))
    if method == "random":
        np.random.default_rng(seed).shuffle(order)
    reports, reviews, candidates, raw_by_cell = [], [], [], {}
    used, calls = 0, 1
    trace = [{"step": 0, "action": "coarse_scan", "cell": None, "scale": None,
              "reason": "Obtain common surface calibration.",
              "observation": {"score": float(prior.max()), "n_detections": 0},
              "belief_before": None, "belief_after": None, "cost_px": coarse["cost_px"],
              "cum_cost_frac": 0.0, "elapsed_ms": (time.perf_counter()-started)*1000}]
    overview_by_cell = {}
    if method == "coarse_only" and "overview_candidates" in coarse:
        threshold = max(coarse_threshold, baseline["effective_thresholds"]["0.25"])
        for item in component_gate(coarse["overview_candidates"]):
            if float(item["score"]) >= threshold:
                overview_by_cell.setdefault(int(item["cell"]), []).append(item)
    for cell_np in order:
        cell = int(cell_np)
        rect = rects[cell]
        if method == "coarse_only":
            threshold = max(coarse_threshold, baseline["effective_thresholds"]["0.25"])
            evidence = overview_by_cell.get(cell, [])
            if "overview_candidates" not in coarse and prior[cell] >= threshold:
                evidence = [{"bbox": list(rect), "kind_guess": "unknown", "score": float(prior[cell]),
                             "localized": False, "source": "prior_cell_fallback",
                             "overview_working_scale": float(baseline.get("working_scale", min(1.0, cfg.work_width / width)))}]
            cell_reports = [dict(item, cell=cell, cells=[cell], scale=.25, status="coarse_report",
                                 threshold=threshold,
                                 reason=("Prior-cell fallback: this backend supplies no localized overview components."
                                         if item.get("source") == "prior_cell_fallback" else
                                         "Report the overview component without full-resolution confirmation."))
                            for item in evidence]
            reports.extend(cell_reports)
            if cell_reports:
                trace.append({"step": len(trace), "action": "report_defect", "cell": cell, "rect": rect,
                              "scale": .25, "reason": cell_reports[0]["reason"],
                              "observation": {"score": max(item["score"] for item in cell_reports),
                                              "n_detections": len(cell_reports)},
                              "belief_before": float(prior[cell]), "belief_after": float(prior[cell]),
                              "cost_px": 0, "cum_cost_frac": 0.0, "elapsed_ms": 0.0,
                              "detections": cell_reports})
            continue
        cost = tile_cost(rect, 1.0)
        if used + cost > limit:
            continue
        observation = backend.inspect_tile(image_ref, rect, 1.0, baseline)
        if int(observation["cost_px"]) != cost:
            raise ValueError("Backend violated the fine raster cost contract")
        calls += 1
        used += cost
        score = float(observation["score"])
        threshold = baseline["effective_thresholds"]["1"]
        confirmed = (component_gate(dict(item, cell=cell) for item in observation["detections"])
                     if score >= threshold else [])
        reports = replace_cell_reports(raw_by_cell, cell, confirmed, threshold)
        trace.append({"step": len(trace), "action": "inspect_tile", "cell": cell, "rect": rect,
                      "scale": 1.0, "reason": f"Inspect in {method} order.",
                      "observation": {"score": score, "n_detections": len(observation["detections"])},
                      "belief_before": float(prior[cell]), "belief_after": score, "cost_px": cost,
                      "cum_cost_frac": used/area, "elapsed_ms": observation.get("elapsed_ms", 0.0),
                      "clear_candidates": True, "merged_reports": reports})
        if raw_by_cell[cell]:
            trace.append({**trace[-1], "step": len(trace), "action": "report_defect", "cost_px": 0,
                          "elapsed_ms": 0.0, "reason": "Report detections confirmed at full resolution.",
                          "detections": raw_by_cell[cell], "merged_reports": reports})
        elif score >= cfg.ambiguous_low:
            review = {"cell": cell, "rect": rect, "scale": 1.0,
                      "reason": "Ambiguous full-resolution evidence.", "score": score}
            reviews.append(review)
            trace.append({**trace[-1], "step": len(trace), "action": "request_human_review", "cost_px": 0,
                          "elapsed_ms": 0.0, "reason": review["reason"], "review": review})
    return {"reporting_version": REPORTING_VERSION,
            "report_mode": "coarse_only" if method == "coarse_only" else "full_resolution",
            "reported_defects": reports, "raw_reports_by_cell": {str(cell): items for cell, items in raw_by_cell.items()},
            "raw_reported_defects": flatten_cell_reports(raw_by_cell), "coarse_candidates": candidates,
            "human_review_queue": reviews, "budget_used_frac": used/area, "fine_cost_px": used,
            "coarse_cost_px": coarse["cost_px"], "n_tool_calls": calls,
            "calibration_cost_px": coarse.get("calibration_cost_px", 0),
            "preparation_elapsed_ms": coarse.get("elapsed_ms", 0),
            "elapsed_ms": (time.perf_counter()-started)*1000, "trace": trace, "coarse": coarse}


def exhaustive(backend: ToolBackend, image_ref: str, **kw) -> dict:
    return run_baseline("exhaustive", backend, image_ref, **kw)


def random(backend: ToolBackend, image_ref: str, budget_frac: float, **kw) -> dict:
    return run_baseline("random", backend, image_ref, budget_frac, **kw)


def raster(backend: ToolBackend, image_ref: str, budget_frac: float, **kw) -> dict:
    return run_baseline("raster", backend, image_ref, budget_frac, **kw)


def coarse_only(backend: ToolBackend, image_ref: str, **kw) -> dict:
    return run_baseline("coarse_only", backend, image_ref, **kw)
