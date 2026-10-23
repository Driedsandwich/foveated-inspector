"""OpenCV overlays, evidence-trace frames, and a dependency-free SVG chart."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

SCALE_COLOURS = {0.25: (235, 180, 40), 0.5: (40, 190, 245), 1.0: (180, 70, 220)}
CANDIDATE_COLOUR = (255, 197, 80)
REPORT_COLOUR = (70, 255, 70)
REVIEW_COLOUR = (0, 145, 255)
LEGACY_COLOUR = (170, 170, 170)
FOOTER_COLOUR = (32, 39, 45)
GAZE_LIMIT = 40


def _cell_rects(result: dict) -> list[list[int]]:
    coarse = result.get("coarse", {})
    rects = result.get("cell_rects", coarse.get("cell_rects", []))
    return np.asarray(rects, dtype=int).reshape(-1, 4).tolist() if len(rects) else []


def _inspection_steps(result: dict) -> list[dict]:
    return [step for step in result.get("trace", [])
            if step.get("scale") is not None and step.get("cell") is not None
            and step.get("action") in ("inspect_tile", "inspect", "fine_scan")]


def _dashed_box(canvas, top_left, bottom_right, colour, thickness=2):
    x0, y0 = top_left
    x1, y1 = bottom_right
    for start in range(x0, x1 + 1, 14):
        cv2.line(canvas, (start, y0), (min(start + 8, x1), y0), colour, thickness)
        cv2.line(canvas, (start, y1), (min(start + 8, x1), y1), colour, thickness)
    for start in range(y0, y1 + 1, 14):
        cv2.line(canvas, (x0, start), (x0, min(start + 8, y1)), colour, thickness)
        cv2.line(canvas, (x1, start), (x1, min(start + 8, y1)), colour, thickness)


def _trace_candidates(prefix: list[dict]) -> list[dict]:
    by_cell = {}
    for step in prefix:
        cell = step.get("cell")
        if step.get("clear_candidates") and cell is not None:
            by_cell.pop(cell, None)
        if step.get("action") == "record_candidate":
            candidates = step.get("candidates", step.get("detections", []))
            if cell is not None:
                by_cell[cell] = candidates
    return [candidate for candidates in by_cell.values() for candidate in candidates]


def _put_text_inside(canvas: np.ndarray, text: str, origin: tuple[int, int],
                     colour: tuple[int, int, int], font_scale: float = .32) -> None:
    """Keep the entire glyph box, including its baseline, inside this drawing ROI."""
    height, width = canvas.shape[:2]
    if width < 4 or height < 4:
        return
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    font_scale *= min(1.0, (width-3)/max(1, size[0]),
                      (height-3)/max(1, size[1]+baseline))
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    if size[0]+2 >= width or size[1]+baseline+2 >= height:
        return
    x = max(1, min(int(origin[0]), width-size[0]-2))
    y = max(size[1]+1, min(int(origin[1]), height-baseline-2))
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, colour, 1, cv2.LINE_AA)


def _wrap_legend(text: str, width: int, font_scale: float) -> list[str]:
    """Wrap by measured glyph width, splitting words for narrow preview exports."""
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        for character in word:
            candidate = current + character
            if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > width:
                lines.append(current)
                current = character
            else:
                current = candidate
    if current:
        lines.append(current)
    return lines


def _legend_footer(width: int, show_heatmap: bool) -> np.ndarray:
    """Lay out legend chips in a separate image; no legend touches the photograph."""
    entries = [
        ("0.25 scale: inspected", SCALE_COLOURS[.25], "thin"),
        ("0.5 scale: inspected", SCALE_COLOURS[.5], "thin"),
        ("1.0 scale: inspected", SCALE_COLOURS[1.], "thin"),
        ("Gaze: last 40 inspections", (220, 220, 220), "gaze"),
        ("Confirmed model report", REPORT_COLOUR, "bold"),
        ("Unconfirmed candidate", CANDIDATE_COLOUR, "dashed"),
        ("Coarse-only report: low scale", CANDIDATE_COLOUR, "dashed"),
        ("Human review", REVIEW_COLOUR, "bold"),
        ("Legacy: full scale unverified", LEGACY_COLOUR, "dashed"),
        (f"Belief heatmap: {'on' if show_heatmap else 'off'}", (190, 190, 190), "thin"),
    ]
    padding = min(8, width//8)
    swatch = min(15, max(0, width-2*padding-12))
    gap, font_scale, line_height = 6, .34, 17
    text_width = max(1, width-2*padding-swatch-gap)
    x, y, row_height = padding, padding, 0
    chips = []
    for label, colour, style in entries:
        lines = _wrap_legend(label, text_width, font_scale)
        measured = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
                       for line in lines)
        chip_width = min(width-2*padding, swatch+gap+measured+10)
        chip_height = line_height*len(lines)
        if x > padding and x+chip_width > width-padding:
            x, y, row_height = padding, y+row_height+4, 0
        chips.append((x, y, lines, colour, style))
        row_height = max(row_height, chip_height)
        x += chip_width
    footer = np.full((max(1, y+row_height+padding), width, 3), FOOTER_COLOUR, np.uint8)
    for x, y, lines, colour, style in chips:
        if swatch >= 4:
            left, right = min(width-1, x), min(width-1, x+swatch-1)
            top, bottom = y+4, y+11
            if style == "dashed":
                _dashed_box(footer, (left, top), (right, bottom), colour, 1)
            elif style == "gaze":
                cv2.line(footer, (left, top+3), (right, top+3), colour, 1)
                cv2.circle(footer, ((left+right)//2, top+3), 2, colour, -1)
            else:
                cv2.rectangle(footer, (left, top), (right, bottom), colour,
                              2 if style == "bold" else 1)
        for index, line in enumerate(lines):
            _put_text_inside(footer, line, (x+swatch+gap, y+12+index*line_height),
                             (238, 238, 238), font_scale)
    return footer


def overlay(img_bgr: np.ndarray, result: dict, max_width: int = 1024) -> np.ndarray:
    """Draw low-opacity visited cells, recent gaze, evidence, and a separate legend.

    The independent BGR output includes a footer below the resized photograph,
    so its height exceeds the image region. Tiny images gain right-side margin
    for readable legend wrapping; output width never exceeds ``max_width``.
    Heat is opt-in via ``result['show_heatmap']``. Numbered dots and a faint
    polyline show only the last 40 inspections, while thin scale outlines retain
    all visited cells. The source image is never modified or displayed.
    """
    if (img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or not img_bgr.shape[0]
            or not img_bgr.shape[1] or not isinstance(max_width, int) or max_width <= 0):
        raise ValueError("expected a BGR image and a positive max_width")
    ratio = min(1.0, max_width / img_bgr.shape[1])
    photo_width = max(1, round(img_bgr.shape[1]*ratio))
    photo_height = max(1, round(img_bgr.shape[0]*ratio))
    canvas = cv2.resize(img_bgr, (photo_width, photo_height), interpolation=cv2.INTER_AREA)
    sx, sy = photo_width/img_bgr.shape[1], photo_height/img_bgr.shape[0]
    rects = _cell_rects(result)

    def box(rect: Sequence[float]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        try:
            x, y, width, height = map(float, rect)
        except (TypeError, ValueError):
            return None
        if not all(map(math.isfinite, (x, y, width, height))) or width <= 0 or height <= 0:
            return None
        if x+width <= 0 or y+height <= 0 or x >= img_bgr.shape[1] or y >= img_bgr.shape[0]:
            return None
        x0, y0 = math.floor(max(0, x)*sx), math.floor(max(0, y)*sy)
        x1 = math.ceil(min(img_bgr.shape[1], x+width)*sx)-1
        y1 = math.ceil(min(img_bgr.shape[0], y+height)*sy)-1
        return ((max(0, x0), max(0, y0)),
                (min(photo_width-1, x1), min(photo_height-1, y1)))

    def item_box(item: dict, field: str = "bbox"):
        rect = item.get(field)
        cell = item.get("cell")
        if rect is None and cell is not None and 0 <= int(cell) < len(rects):
            rect = rects[int(cell)]
        return box(rect) if rect is not None else None

    priors = np.asarray(result.get("coarse", {}).get("prior", []), dtype=float).ravel()
    beliefs = {i: float(value) for i, value in enumerate(priors)}
    for step in result.get("trace", []):
        if step.get("cell") is not None and step.get("belief_after") is not None:
            beliefs[int(step["cell"])] = float(step["belief_after"])
        for update in step.get("belief_updates", []):
            beliefs[int(update["cell"])] = float(update["belief_after"])
    show_heatmap = bool(result.get("show_heatmap", False))
    if show_heatmap and rects:
        heatmap = canvas.copy()
        for cell, rect in enumerate(rects):
            corners = box(rect)
            if corners is None:
                continue
            probability = np.clip(beliefs.get(cell, 0.0), 0.0, 1.0)
            colour = (int(210*(1-probability)), int(140*(1-probability)), int(245*probability))
            cv2.rectangle(heatmap, *corners, colour, thickness=-1)
        cv2.addWeighted(heatmap, .18, canvas, .82, 0.0, dst=canvas)
        del heatmap

    visited = canvas.copy()
    gaze = []
    for order, step in enumerate(_inspection_steps(result), start=1):
        cell = int(step["cell"])
        rect = step.get("rect")
        if rect is None and 0 <= cell < len(rects):
            rect = rects[cell]
        if rect is None:
            continue
        corners = box(rect)
        if corners is None:
            continue
        top_left, bottom_right = corners
        centre = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        colour = SCALE_COLOURS.get(float(step["scale"]), (255, 255, 255))
        cv2.rectangle(visited, top_left, bottom_right, colour, 1)
        gaze.append((order, centre, colour))
    cv2.addWeighted(visited, .22, canvas, .78, 0.0, dst=canvas)
    del visited
    recent = gaze[-GAZE_LIMIT:]
    if len(recent) >= 2:
        path = np.asarray([centre for _, centre, _ in recent], np.int32).reshape(-1, 1, 2)
        gaze_layer = canvas.copy()
        cv2.polylines(gaze_layer, [path], False, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.addWeighted(gaze_layer, .28, canvas, .72, 0.0, dst=canvas)
        del gaze_layer
    radius = min(3, max(0, (min(photo_width, photo_height)-1)//2))
    for order, centre, colour in recent:
        # Clamp the dot centre as well as its box so edge cells remain in-frame.
        point = (min(photo_width-radius-1, max(radius, centre[0])),
                 min(photo_height-radius-1, max(radius, centre[1])))
        cv2.circle(canvas, point, radius, colour, -1, cv2.LINE_AA)
        _put_text_inside(canvas, str(order), (point[0]+5, point[1]-4), (245, 245, 245), .28)
    for candidate in result.get("coarse_candidates", []):
        corners = item_box(candidate)
        if corners is not None:
            _dashed_box(canvas, *corners, CANDIDATE_COLOUR)
    for detection in result.get("reported_defects", []):
        corners = item_box(detection)
        if corners is None:
            continue
        if detection.get("status") == "coarse_report" or detection.get("report_mode") == "coarse_only":
            _dashed_box(canvas, *corners, CANDIDATE_COLOUR)
        elif detection.get("scale") == 1.0:
            cv2.rectangle(canvas, *corners, REPORT_COLOUR, 3)
        else:
            _dashed_box(canvas, *corners, LEGACY_COLOUR)
    for review in result.get("human_review_queue", []):
        corners = item_box(review, "rect")
        if corners is not None:
            cv2.rectangle(canvas, *corners, REVIEW_COLOUR, 3)
    output_width = max(photo_width, min(320, max_width))
    footer = _legend_footer(output_width, show_heatmap)
    output = np.full((photo_height+footer.shape[0], output_width, 3), FOOTER_COLOUR, np.uint8)
    output[:photo_height, :photo_width] = canvas
    output[photo_height:] = footer
    return output


def trace_frames(
    img_bgr: np.ndarray, result: dict, out_dir: str | Path, max_width: int = 1024,
    *, frame_stride: int = 1, max_frames: int | None = None,
) -> list[str]:
    """Write one numbered PNG per trace entry plus an initial belief overview.

    Reports/reviews are shown only after their cell's decision action has
    occurred. The final frame shows all reports, including traces with abbreviated
    observations. Image arrays are released after each write. Optional sampling
    always includes the initial and final states; filenames retain original
    prefix-step indices. Sampling does not modify or shorten the JSON trace.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = result.get("trace", [])
    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    minimum_frames = 2 if trace else 1
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < minimum_frames):
        raise ValueError(f"max_frames must be at least {minimum_frames}")
    ends = list(range(0, len(trace) + 1, frame_stride))
    if ends[-1] != len(trace):
        ends.append(len(trace))
    if max_frames is not None and len(ends) > max_frames:
        indices = np.linspace(0, len(ends) - 1, max_frames, dtype=int)
        ends = [ends[int(index)] for index in indices]
    paths = []
    for end in ends:
        prefix = trace[:end]
        reported_cells = {entry.get("cell") for entry in prefix if entry.get("action") == "report_defect"}
        review_cells = {entry.get("cell") for entry in prefix if entry.get("action") == "request_human_review"}
        reports_at_step = []
        for entry in prefix:
            if "merged_reports" in entry:
                reports_at_step = entry["merged_reports"]
            elif entry.get("action") == "report_defect":
                reports_at_step = reports_at_step + entry.get("detections", [])
        reviews_at_step = [entry["review"] for entry in prefix if "review" in entry]
        explicit_reports = any("merged_reports" in entry or
                               (entry.get("action") == "report_defect" and "detections" in entry)
                               for entry in trace)
        explicit_reviews = any("review" in entry for entry in trace)
        frame_result = {
            **result, "trace": prefix,
            "coarse_candidates": _trace_candidates(prefix),
            "reported_defects": reports_at_step if explicit_reports else [
                item for item in result.get("reported_defects", []) if item.get("cell") in reported_cells],
            "human_review_queue": reviews_at_step if explicit_reviews else [
                item for item in result.get("human_review_queue", []) if item.get("cell") in review_cells],
        }
        if end == len(trace):
            frame_result = result
        path = out / f"frame_{end:04d}.png"
        if not cv2.imwrite(str(path), overlay(img_bgr, frame_result, max_width=max_width)):
            raise OSError(f"could not write trace frame: {path}")
        paths.append(str(path))
    return paths


def write_recall_chart(
    summaries: Sequence[dict], path: str | Path, *, cost_key: str = "fine_cost_frac"
) -> None:
    """Write recall and image-bootstrap CIs against measured pixel cost.

    ``total_cost_frac`` includes the overview and its calibration work exactly
    once. Detector and random-seed variants always have separate legend entries.
    """
    if cost_key not in ("fine_cost_frac", "total_cost_frac"):
        raise ValueError("cost_key must be fine_cost_frac or total_cost_frac")
    width, height = 980, 500
    left, top, plot_width, plot_height = 75, 45, 535, 365
    valid = [row for row in summaries if row.get("recall") is not None]
    x_max = max(0.05, max((float(row[cost_key]) for row in valid), default=1.0))
    x_max = min(1.0, x_max * 1.05) if x_max <= 1 else x_max * 1.05
    colours = ["#1665b5", "#c13f38", "#33995d", "#9864af", "#cc8c12", "#328f99"]

    def point(x: float, y: float) -> tuple[float, float]:
        return left + x / x_max * plot_width, top + (1 - y) * plot_height

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#222">',
           '<text x="75" y="25" font-size="17">Defect recall vs measured inspection cost</text>']
    for index in range(6):
        y = index / 5
        _, screen_y = point(0, y)
        svg.append(f'<line x1="{left}" y1="{screen_y:.1f}" x2="{left + plot_width}" y2="{screen_y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left - 9}" y="{screen_y + 4:.1f}" text-anchor="end">{100*y:.0f}%</text>')
        x = x_max * index / 5
        screen_x, _ = point(x, 0)
        svg.append(f'<text x="{screen_x:.1f}" y="{top + plot_height + 22}" text-anchor="middle">{100*x:.1f}%</text>')
    svg.append(f'<path d="M{left},{top} V{top + plot_height} H{left + plot_width}" fill="none" stroke="#444"/>')
    axis_label = ("Fine pixels analysed / full image pixels (repeats counted)" if cost_key == "fine_cost_frac"
                  else "Total pixels analysed / image pixels (overview + calibration + fine)")
    svg.append(f'<text x="{left + plot_width/2}" y="{height - 40}" text-anchor="middle">{axis_label}</text>')
    svg.append(f'<text x="18" y="{top + plot_height/2}" transform="rotate(-90 18 {top + plot_height/2})" text-anchor="middle">Ground-truth defects found</text>')
    def variant(row: dict) -> tuple[str, str, str]:
        return str(row.get("detector", "classic")), str(row["method"]), str(row.get("seed", ""))

    methods = sorted({variant(row) for row in valid})
    for index, method in enumerate(methods):
        colour = colours[index % len(colours)]
        rows = sorted((row for row in valid if variant(row) == method),
                      key=lambda row: (float(row[cost_key]), float(row.get("budget_frac", 0))))
        points = [point(float(row[cost_key]), float(row["recall"])) for row in rows]
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for row, (x, y) in zip(rows, points):
            low, high = row.get("recall_ci_low"), row.get("recall_ci_high")
            if low is not None and high is not None:
                _, low_y = point(float(row[cost_key]), float(low))
                _, high_y = point(float(row[cost_key]), float(high))
                svg.append(f'<path d="M{x-3:.2f},{low_y:.2f} H{x+3:.2f} M{x:.2f},{low_y:.2f} V{high_y:.2f} M{x-3:.2f},{high_y:.2f} H{x+3:.2f}" stroke="{colour}" opacity="0.5" fill="none"/>')
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colour}"/>')
        y = top + 20 * index + 10
        svg.append(f'<line x1="635" y1="{y}" x2="655" y2="{y}" stroke="{colour}" stroke-width="3"/>')
        label = f"{method[0]}/{method[1]}" + (f" (seed {method[2]})" if method[2] else "")
        svg.append(f'<text x="662" y="{y+4}">{html.escape(label)}</text>')
    note = ("Coarse/calibration cost is separate in CSV/JSON and total-cost chart."
            if cost_key == "fine_cost_frac" else "Calibration is part of coarse cost; it is not charged twice.")
    svg.append(f'<text x="75" y="485" fill="#555">{note} Bars: 95% image bootstrap.</text></g></svg>')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
