"""Evidence states and trace replay remain distinct on small synthetic images."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.viz import (CANDIDATE_COLOUR, LEGACY_COLOUR, REPORT_COLOUR,
                           REVIEW_COLOUR, overlay, trace_frames)

cv2.setNumThreads(2)


class EvidenceVisualizationTests(unittest.TestCase):
    def test_coarse_reports_are_blue_and_do_not_look_like_confirmed_or_legacy_reports(self):
        image = np.full((80, 128, 3), 127, np.uint8)
        result = {"reported_defects": [
            {"bbox": [10, 10, 24, 24], "scale": .25, "status": "coarse_report"},
            {"bbox": [60, 10, 24, 24], "scale": .5, "report_mode": "coarse_only"},
        ]}
        drawn = overlay(image, result)
        np.testing.assert_array_equal(drawn[10, 10], CANDIDATE_COLOUR)
        np.testing.assert_array_equal(drawn[10, 60], CANDIDATE_COLOUR)

    def test_legend_is_below_an_unmodified_photo_and_heat_is_opt_in(self):
        image = np.full((64, 96, 3), 127, np.uint8)
        result = {"coarse": {"prior": [[.95]], "cell_rects": [[0, 0, 96, 64]]}}
        drawn = overlay(image, result)
        self.assertGreater(drawn.shape[0], image.shape[0])
        self.assertGreaterEqual(drawn.shape[1], image.shape[1])
        self.assertLessEqual(drawn.shape[1], 1024)
        np.testing.assert_array_equal(drawn[:64, :96], image)
        heated = overlay(image, {**result, "show_heatmap": True})
        self.assertFalse(np.array_equal(heated[32, 48], image[32, 48]))
        np.testing.assert_array_equal(image, np.full_like(image, 127))

    def test_only_last_forty_inspections_have_gaze_dots_and_numbers(self):
        image = np.full((200, 360, 3), 127, np.uint8)
        trace = [{"action": "inspect_tile", "cell": index, "scale": (.25, .5, 1.)[index % 3],
                  "rect": [(index % 9)*40, (index//9)*40, 40, 40]}
                 for index in range(45)]
        original_text, original_path = cv2.putText, cv2.polylines
        with patch("inspector.viz.cv2.putText", wraps=original_text) as text, \
                patch("inspector.viz.cv2.polylines", wraps=original_path) as paths, \
                patch("inspector.viz.cv2.arrowedLine", side_effect=AssertionError("Gaze has no arrows")):
            overlay(image, {"trace": trace})
        numbers = [int(call.args[1]) for call in text.call_args_list if call.args[1].isdigit()]
        self.assertEqual(numbers, list(range(6, 46)))
        self.assertEqual(paths.call_count, 1)
        points = paths.call_args.args[1][0].reshape(-1, 2)
        self.assertEqual(len(points), 40)
        self.assertTrue(np.all((points >= 0) & (points < [360, 200])))

    def test_clipped_annotations_and_narrow_footer_stay_in_drawing_bounds(self):
        image = np.full((64, 96, 3), 127, np.uint8)
        result = {
            "trace": [{"action": "inspect_tile", "cell": 0, "scale": .5,
                       "rect": [-20, -10, 140, 90]}],
            "coarse_candidates": [{"bbox": [-20, 40, 50, 80]}],
            "reported_defects": [{"bbox": [70, -10, 90, 80], "scale": 1.},
                                 {"bbox": [200, 200, 20, 20], "scale": 1.}],
            "human_review_queue": [{"rect": [-20, -10, 140, 90]}],
        }
        original_rectangle, original_line, original_text = cv2.rectangle, cv2.line, cv2.putText

        def bounded_shape(draw, canvas, first, second, *args, **kwargs):
            height, width = canvas.shape[:2]
            for x, y in (first, second):
                self.assertTrue(0 <= x < width and 0 <= y < height)
            return draw(canvas, first, second, *args, **kwargs)

        def bounded_text(canvas, label, origin, font, scale, colour, thickness, *args):
            (width, height), baseline = cv2.getTextSize(label, font, scale, thickness)
            x, y = origin
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y-height, 0)
            self.assertLess(x+width, canvas.shape[1])
            self.assertLess(y+baseline, canvas.shape[0])
            return original_text(canvas, label, origin, font, scale, colour, thickness, *args)

        with patch("inspector.viz.cv2.rectangle", side_effect=lambda *args, **kwargs: bounded_shape(original_rectangle, *args, **kwargs)), \
                patch("inspector.viz.cv2.line", side_effect=lambda *args, **kwargs: bounded_shape(original_line, *args, **kwargs)), \
                patch("inspector.viz.cv2.putText", side_effect=bounded_text):
            drawn = overlay(image, result, max_width=80)
        self.assertEqual(drawn.shape[1], 80)
        self.assertGreater(drawn.shape[0], round(64*80/96))
        tiny = overlay(np.full((2, 3, 3), 127, np.uint8), {}, max_width=96)
        self.assertEqual(tiny.shape[1], 96)
        np.testing.assert_array_equal(tiny[:2, :3], np.full((2, 3, 3), 127, np.uint8))

    def test_candidates_reports_reviews_and_legacy_have_distinct_styles(self):
        image = np.full((160, 240, 3), 127, np.uint8)
        result = {
            "coarse_candidates": [{"bbox": [10, 90, 35, 35], "scale": .5}],
            "reported_defects": [{"bbox": [65, 90, 35, 35], "scale": 1.},
                                 {"bbox": [120, 90, 35, 35]}],
            "human_review_queue": [{"rect": [175, 90, 35, 35]}],
        }
        drawn = overlay(image, result)
        for point, colour in (((90, 10), CANDIDATE_COLOUR), ((90, 65), REPORT_COLOUR),
                              ((90, 120), LEGACY_COLOUR), ((90, 175), REVIEW_COLOUR)):
            np.testing.assert_array_equal(drawn[point], colour)
        # The gap between blue dashes remains background; a solid report does not.
        np.testing.assert_array_equal(drawn[90, 21], [127, 127, 127])
        np.testing.assert_array_equal(drawn[90, 76], REPORT_COLOUR)
        self.assertTrue(np.all(image == 127))

    def test_replay_clears_candidates_and_replaces_merged_report_snapshots(self):
        candidate = {"cell": 0, "bbox": [0, 0, 16, 16], "scale": .5}
        first = {"cell": 0, "bbox": [0, 0, 16, 16], "scale": 1.}
        merged = {"cell": 0, "cells": [0, 1], "bbox": [0, 0, 32, 16], "scale": 1.}
        trace = [
            {"action": "record_candidate", "cell": 0, "candidates": [candidate],
             "detections": [candidate]},
            {"action": "inspect_tile", "cell": 0, "scale": 1., "clear_candidates": True},
            {"action": "report_defect", "cell": 0, "merged_reports": [first]},
            {"action": "report_defect", "cell": 1, "merged_reports": [merged]},
            {"action": "inspect_tile", "cell": 0, "scale": 1., "clear_candidates": True,
             "merged_reports": []},
            {"action": "request_human_review", "cell": 0,
             "review": {"cell": 0, "rect": [0, 0, 16, 16], "reason": "Ambiguous."}},
        ]
        result = {"trace": trace, "reported_defects": [], "coarse_candidates": [],
                  "human_review_queue": [trace[-1]["review"]]}
        frames = []

        def render(image, state, max_width):
            frames.append(state)
            return np.zeros((8, 8, 3), np.uint8)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with patch("inspector.viz.overlay", side_effect=render), \
                    patch("inspector.viz.cv2.imwrite", return_value=True):
                trace_frames(np.zeros((32, 32, 3), np.uint8), result, temporary)
        self.assertEqual(frames[1]["coarse_candidates"], [candidate])
        self.assertEqual(frames[1]["reported_defects"], [])
        self.assertEqual(frames[2]["coarse_candidates"], [])
        self.assertEqual(frames[3]["reported_defects"], [first])
        self.assertEqual(frames[4]["reported_defects"], [merged])
        self.assertEqual(frames[5]["reported_defects"], [])
        self.assertEqual(frames[6]["human_review_queue"], [trace[-1]["review"]])


if __name__ == "__main__":
    unittest.main()
