"""Deterministic inspection with full-resolution confirmation and spatial merging."""

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .tools import ToolBackend
from .components import clip_bbox, component_gate, spatial_candidate_gate

REPORTING_VERSION = "v8-spatial-confirmation-bounded-pairs"


@dataclass(frozen=True)
class InspectorConfig:
    """Policy thresholds/weights; learned validation thresholds are hard floors."""

    grid: tuple[int, int] = (16, 16)
    work_width: int = 1024
    initial_scale: float = 0.5
    ambiguous_low: float = 0.35
    positive_threshold: float = 0.75
    stop_threshold: float = 0.12
    follow_boost: float = 0.8
    follow_weight: float = 4.0
    escalation_weight: float = 1.8
    detection_threshold: float = 0.75
    threshold_multiplier: float = 1.0
    linear_min_length_px: float = 128.0
    candidate_margin_px: float = 32.0

    def __post_init__(self) -> None:
        if self.initial_scale not in (0.25, 0.5):
            raise ValueError("initial_scale must be 0.25 or 0.5")
        if not 0 <= self.ambiguous_low < self.positive_threshold <= 1:
            raise ValueError("invalid score thresholds")
        if not 0 <= self.stop_threshold <= 1:
            raise ValueError("invalid stop threshold")
        if (len(self.grid) != 2 or any(not isinstance(v, int) or v < 1 for v in self.grid)
                or not isinstance(self.work_width, int) or self.work_width < 1):
            raise ValueError("grid and work_width must be positive")
        if (not 0 <= self.follow_boost <= 1 or not math.isfinite(self.follow_weight)
                or not math.isfinite(self.escalation_weight) or self.follow_weight <= 0
                or self.escalation_weight <= 0 or not 0 <= self.detection_threshold <= 1):
            raise ValueError("invalid policy weights")
        if (isinstance(self.threshold_multiplier, bool) or not math.isfinite(self.threshold_multiplier)
                or self.threshold_multiplier < 1):
            raise ValueError("threshold_multiplier must be finite and >= 1")
        if (isinstance(self.linear_min_length_px, bool) or not math.isfinite(self.linear_min_length_px)
                or self.linear_min_length_px <= 0):
            raise ValueError("linear_min_length_px must be finite and positive")
        if (isinstance(self.candidate_margin_px, bool) or not math.isfinite(self.candidate_margin_px)
                or self.candidate_margin_px < 0):
            raise ValueError("candidate_margin_px must be finite and nonnegative")


def tile_cost(rect: list[int], scale: float) -> int:
    """Same floor-rounded raster dimensions as the fine tool."""
    return max(1, int(rect[2] * scale)) * max(1, int(rect[3] * scale))


def threshold_floor(baseline: dict, scale: float) -> float:
    values = baseline.get("detector_thresholds", {})
    floor = float(values.get(f"{scale:g}", values.get(str(scale), 0.0)))
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("invalid detector validation threshold")
    return floor


def effective_threshold(baseline: dict, scale: float, legacy: float, multiplier: float = 1.0) -> float:
    """Raise the learned floor/legacy threshold, capped without lowering the floor."""
    if (isinstance(multiplier, bool) or not math.isfinite(multiplier) or multiplier < 1
            or not math.isfinite(legacy) or not 0 <= legacy <= 1):
        raise ValueError("thresholds must be finite; multiplier must be >= 1")
    floor = threshold_floor(baseline, scale)
    return max(floor, min(0.97, max(floor, legacy) * multiplier))


def policy_baseline(baseline: dict, config: InspectorConfig) -> dict:
    """Prepare additive tool metadata without mutating cached calibration."""
    legacy = max(config.positive_threshold, config.detection_threshold)
    return {**baseline, "threshold_multiplier": config.threshold_multiplier,
            "effective_thresholds": {f"{scale:g}": effective_threshold(baseline, scale, legacy, config.threshold_multiplier)
                                     for scale in (0.25, 0.5, 1.0)}}


def merge_reports(detections: list[dict]) -> list[dict]:
    """Merge disjoint pairs of overlapping originals; never chain unions.

    A pair needs IoU >= .3 or intersection/minimum member area >= .7.
    Its union is capped at twice the larger member area and 1024px per side.
    Highest-IoU pairs win, then coverage and deterministic geometry/score ties.
    Each original participates once; merged results are never eligible again,
    including when this helper receives an already merged result. More than
    two duplicate observations may consequently remain as separate reports.
    Cell replacement must always rebuild from the original per-cell evidence.
    """
    boxes = []
    for detection in detections:
        if detection.get("scale") != 1.0:
            raise ValueError("only explicitly full-resolution reports can be merged")
        x, y, w, h = map(float, detection["bbox"])
        if not all(math.isfinite(value) for value in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("report boxes must have finite positive area")
        boxes.append((x, y, x + w, y + h))
    def identity(index):
        return (*boxes[index], -float(detections[index].get("score", 0)),
                int(detections[index].get("cell", -1)), str(detections[index].get("kind_guess", "unknown")))

    pairs = []
    for index, (left, top, right, bottom) in enumerate(boxes):
        if detections[index].get("n_merged_components", 1) != 1:
            continue
        for other in range(index):
            if detections[other].get("n_merged_components", 1) != 1:
                continue
            a, b, c, d = boxes[other]
            intersection = max(0.0, min(right, c) - max(left, a)) * max(0.0, min(bottom, d) - max(top, b))
            area, other_area = (right-left)*(bottom-top), (c-a)*(d-b)
            iou = intersection / (area + other_area - intersection)
            coverage = intersection / min(area, other_area)
            if iou < .3 and coverage < .7:
                continue
            union_w = math.ceil(max(right, c)) - math.floor(min(left, a))
            union_h = math.ceil(max(bottom, d)) - math.floor(min(top, b))
            if max(union_w, union_h) > 1024 or union_w * union_h > 2 * max(area, other_area):
                continue
            pair_key = tuple(sorted((identity(index), identity(other))))
            pairs.append((-iou, -coverage, pair_key, index, other))
    groups, used = [], set()
    for _, _, _, index, other in sorted(pairs):
        if index not in used and other not in used:
            groups.append([index, other])
            used.update((index, other))
    groups.extend([index] for index in range(len(boxes)) if index not in used)
    merged = []
    for indices in groups:
        best = min(indices, key=lambda index: (-float(detections[index].get("score", 0)), identity(index)))
        detection = dict(detections[best])
        left = math.floor(min(boxes[index][0] for index in indices))
        top = math.floor(min(boxes[index][1] for index in indices))
        right = math.ceil(max(boxes[index][2] for index in indices))
        bottom = math.ceil(max(boxes[index][3] for index in indices))
        cells = sorted({int(cell) for index in indices
                        for cell in detections[index].get("cells", [detections[index].get("cell", -1)])})
        # An unmerged box must retain fractional coordinates: rounding two
        # separate boxes outward can manufacture overlap on the next call.
        bbox = (list(detections[best]["bbox"]) if len(indices) == 1
                else [left, top, right - left, bottom - top])
        detection.update(bbox=bbox,
                         score=float(detections[best].get("score", 0)), cell=int(detection.get("cell", cells[0])),
                         cells=cells, scale=1.0, status="confirmed",
                         n_merged_components=sum(int(detections[index].get("n_merged_components", 1)) for index in indices))
        merged.append(detection)
    return sorted(merged, key=lambda item: (item["bbox"][1], item["bbox"][0], item["cell"]))


def flatten_cell_reports(raw_by_cell: dict) -> list[dict]:
    return [dict(item) for cell in sorted(raw_by_cell, key=int) for item in raw_by_cell[cell]]


def replace_cell_reports(raw_by_cell: dict, cell: int, detections: list[dict], threshold: float) -> list[dict]:
    """Replace one cell's raw evidence, then reconstruct all unions from scratch."""
    raw_by_cell[cell] = [dict(item, cell=cell, scale=1.0, status="confirmed")
                         for item in detections if float(item["score"]) >= threshold]
    return merge_reports(flatten_cell_reports(raw_by_cell))


def add_reports(reports: list[dict], detections: list[dict], threshold: float) -> None:
    """Merge explicit scale-1 reports; never promote unverified observations.

    Policies use replace_cell_reports instead, retaining raw per-cell lineage.
    """
    reports[:] = merge_reports([*reports, *(dict(item) for item in detections if item["score"] >= threshold)])


def confirmed_components(detections: list[dict], candidates: list[dict],
                         config: InspectorConfig, image_shape: list[int], cell: int) -> list[dict]:
    """Confirm candidate locations, with a narrow exception for long thin lines.

    Localized candidates expand by one confirmation-scale patch: 32 fullres px.
    Score-only candidates do not provide spatial evidence. The linear exception
    still requires two contributing patches and shares the per-cell top-three cap.
    """
    matched = spatial_candidate_gate(detections, candidates, margin_px=config.candidate_margin_px,
                                     image_shape=image_shape)
    matched_boxes = {tuple(item["bbox"]) for item in matched}
    for item in detections:
        try:
            elongation = float(item.get("elongation", 0))
            length = float(item.get("length_px", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        bbox = clip_bbox(item.get("bbox", []), [0, 0, image_shape[1], image_shape[0]])
        if (math.isfinite(elongation) and math.isfinite(length) and elongation >= 4
                and length >= config.linear_min_length_px and bbox is not None and tuple(bbox) not in matched_boxes):
            matched.append(dict(item, bbox=bbox))
            matched_boxes.add(tuple(bbox))
    return component_gate(dict(item, cell=cell) for item in matched)


class ActiveInspector:
    """Spend a hard fine-pixel budget; report only after full-resolution evidence."""

    def __init__(self, backend: ToolBackend, budget_frac: float, config: InspectorConfig | None = None):
        if not math.isfinite(budget_frac) or not 0 <= budget_frac <= 1:
            raise ValueError("budget_frac must be finite and in [0, 1]")
        self.backend, self.budget_frac = backend, budget_frac
        self.config = config or InspectorConfig()

    def run(self, image_ref: str, *, time_limit_ms: float | None = None) -> dict:
        started = time.perf_counter()
        if time_limit_ms is not None and (not math.isfinite(time_limit_ms) or time_limit_ms <= 0):
            raise ValueError("time_limit_ms must be finite and positive")
        cfg = self.config
        coarse_start = time.perf_counter()
        coarse = self.backend.coarse_scan(image_ref, grid=cfg.grid, work_width=cfg.work_width)
        coarse_ms = (time.perf_counter() - coarse_start) * 1000
        baseline = policy_baseline(coarse["baseline"], cfg)
        coarse = {**coarse, "baseline": baseline}
        rects = coarse["cell_rects"]
        height, width = coarse["image_shape"][:2]
        area, limit = height * width, math.floor(height * width * self.budget_frac)
        belief = np.asarray(coarse["prior"], dtype=float).reshape(-1).copy()
        initial_belief = belief.tolist()
        pending = {cell: cfg.initial_scale for cell in range(len(rects))}
        following: set[int] = set()
        candidates_by_cell, raw_by_cell, reviews_by_cell = {}, {}, {}
        reports = []
        trace = [{"step": 0, "action": "coarse_scan", "cell": None, "scale": None,
                  "reason": "Initialise beliefs from the overview.",
                  "observation": {"score": float(belief.max()), "n_detections": 0},
                  "belief_before": None, "belief_after": None, "cost_px": coarse["cost_px"],
                  "cum_cost_frac": 0.0, "elapsed_ms": coarse_ms}]
        used, calls = 0, 1
        rows, cols = cfg.grid

        def record(action, cell, scale, reason, observation, before, cost=0):
            trace.append({"step": len(trace), "action": action, "cell": cell, "rect": rects[cell], "scale": scale,
                          "reason": reason, "observation": {"score": observation["score"],
                          "n_detections": len(observation["detections"]), "endpoints": observation.get("endpoints", [])},
                          "belief_before": before, "belief_after": float(belief[cell]), "cost_px": cost,
                          "cum_cost_frac": used / area, "elapsed_ms": observation.get("elapsed_ms", 0.0) if cost else 0.0})

        time_stopped = False
        while pending:
            if time_limit_ms is not None and (time.perf_counter() - started) * 1000 >= time_limit_ms:
                time_stopped = True
                break
            actions = []
            for cell, scale in pending.items():
                cost = tile_cost(rects[cell], scale)
                confirming = cell in candidates_by_cell and scale == 1.0
                if cost > limit - used or not confirming and belief[cell] < cfg.stop_threshold:
                    continue
                weight = cfg.escalation_weight if scale > cfg.initial_scale else 1.0
                if cell in following:
                    weight *= cfg.follow_weight
                actions.append((int(confirming), float(belief[cell]) * weight / cost, -cell, cell))
            if not actions:
                break
            cell = max(actions)[-1]
            scale = pending.pop(cell)
            is_follow = cell in following
            following.discard(cell)
            before = float(belief[cell])
            tool_baseline = baseline
            if scale == 1.0:
                # Let the detector apply location eligibility before its top-three
                # cap. Never mutate the image's cached calibration metadata.
                tool_baseline = {**baseline,
                                 "confirmation_candidates": [dict(item) for item in candidates_by_cell.get(cell, [])
                                                             if item.get("localized") is not False],
                                 "candidate_margin_px": cfg.candidate_margin_px,
                                 "linear_min_length_px": cfg.linear_min_length_px}
            observation = self.backend.inspect_tile(image_ref, rects[cell], scale, tool_baseline)
            actual_cost = int(observation["cost_px"])
            if actual_cost != tile_cost(rects[cell], scale):
                raise ValueError("Backend violated the agreed fine raster cost contract")
            used += actual_cost
            calls += 1
            score = float(observation["score"])
            belief[cell] = score
            reason = ("Confirm the candidate in the same cell at full resolution." if scale == 1.0 else
                      "Follow border evidence into the neighbouring cell." if is_follow else
                      "Inspect the highest expected evidence per pixel.")
            record("inspect_tile", cell, scale, reason, observation, before, actual_cost)
            inspection = trace[-1]
            inspection["belief_updates"] = []
            row, col = divmod(cell, cols)
            for endpoint in observation.get("endpoints", []):
                dx, dy = endpoint["dx"], endpoint["dy"]
                dr, dc = ((0, 1 if dx > 0 else -1) if abs(dx) >= abs(dy) else (1 if dy > 0 else -1, 0))
                nr, nc = row + dr, col + dc
                neighbour = nr * cols + nc
                if 0 <= nr < rows and 0 <= nc < cols and neighbour in pending:
                    neighbour_before = float(belief[neighbour])
                    belief[neighbour] = min(1.0, max(belief[neighbour], cfg.follow_boost))
                    following.add(neighbour)
                    inspection["belief_updates"].append({"cell": neighbour, "belief_before": neighbour_before,
                        "belief_after": float(belief[neighbour]), "reason": "Observed line touches the shared tile border."})

            threshold = baseline["effective_thresholds"][f"{scale:g}"]
            # The validation floor protects confirmation, not uncertainty: weak
            # evidence can still warrant a zoom or human review below that floor.
            ambiguous = cfg.ambiguous_low
            if scale < 1.0:
                if score >= ambiguous:
                    candidates = [dict(item, cell=cell, scale=scale, status="candidate", threshold=threshold,
                                       localized=item.get("localized") is not False,
                                       reason="Requires full-resolution confirmation.")
                                  for item in component_gate(dict(item, cell=cell) for item in observation["detections"])
                                  if float(item["score"]) >= ambiguous]
                    if not candidates:
                        candidates = [{"bbox": list(rects[cell]), "cell": cell, "scale": scale, "score": score,
                                       "kind_guess": "unknown", "status": "candidate", "threshold": threshold,
                                       "localized": False,
                                       "reason": "Unlocalized evidence requires full-resolution confirmation."}]
                    candidates_by_cell[cell] = candidates
                    pending[cell] = 1.0
                    record("record_candidate", cell, scale, "Keep coarse evidence pending full-resolution confirmation.", observation, before)
                    trace[-1].update(candidates=candidates, detections=candidates)
                continue

            spatial_evidence = candidates_by_cell.pop(cell, [])
            inspection["clear_candidates"] = True
            inspection["confirmation_candidates"] = spatial_evidence
            confirmed = (confirmed_components(observation["detections"], spatial_evidence, cfg,
                                              coarse["image_shape"], cell) if score >= threshold else [])
            reports = replace_cell_reports(raw_by_cell, cell, confirmed, threshold)
            inspection["merged_reports"] = reports
            reviews_by_cell.pop(cell, None)
            if raw_by_cell[cell]:
                record("report_defect", cell, 1.0, "Report detections confirmed at full resolution.", observation, before)
                trace[-1].update(detections=raw_by_cell[cell], merged_reports=reports)
            elif score >= ambiguous:
                review = {"cell": cell, "rect": rects[cell], "scale": 1.0, "score": score,
                          "reason": "Ambiguous evidence remains at full resolution."}
                reviews_by_cell[cell] = review
                record("request_human_review", cell, 1.0, review["reason"], observation, before)
                trace[-1]["review"] = review

        return {"reporting_version": REPORTING_VERSION,
                "report_mode": "full_resolution",
                "reported_defects": reports, "raw_reports_by_cell": {str(cell): items for cell, items in raw_by_cell.items()},
                "raw_reported_defects": flatten_cell_reports(raw_by_cell),
                "coarse_candidates": flatten_cell_reports(candidates_by_cell),
                "human_review_queue": [reviews_by_cell[cell] for cell in sorted(reviews_by_cell)],
                "budget_used_frac": used / area, "fine_cost_px": used, "coarse_cost_px": coarse["cost_px"],
                "n_tool_calls": calls, "calibration_cost_px": coarse.get("calibration_cost_px", 0),
                "preparation_elapsed_ms": coarse.get("elapsed_ms", coarse_ms),
                "elapsed_ms": (time.perf_counter() - started) * 1000, "trace": trace, "coarse": coarse,
                "final_belief": belief.reshape(cfg.grid).tolist(), "initial_belief": initial_belief, "config": asdict(cfg),
                "stop_reason": ("Time limit reached; partial inspection saved." if time_stopped else
                                "No eligible action fits the remaining budget or belief threshold.")}
