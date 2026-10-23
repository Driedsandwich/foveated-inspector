"""Small exact-label examples for the benchmark's explicit matching rule."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from inspector.evaluate import (aggregate_metrics, evaluate_result, legacy_match_detections, match_detections,
                                recall_vs_budget, write_summaries)
from inspector.viz import overlay, trace_frames

cv2.setNumThreads(2)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((100, 100), np.uint16)
        self.labels[10:20, 10:20] = 1
        self.labels[60:70, 60:70] = 2
        self.defects = [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "stain"}]

    def test_found_not_found_false_positive(self):
        detections = [
            {"bbox": [10, 10, 10, 10], "kind_guess": "crack"},
            {"bbox": [0, 0, 3, 3], "kind_guess": "scratch"},
            # Tiny overlap is an FP under the new localization rule.
            {"bbox": [51, 51, 10, 10], "kind_guess": "stain"},
        ]
        matched = match_detections(detections, self.labels, self.defects)
        self.assertEqual(matched["found_ids"], [1])
        self.assertEqual(matched["missed_ids"], [2])
        self.assertEqual(matched["false_positive_indices"], [1, 2])
        self.assertEqual(matched["partial_overlap_indices"], [2])
        confirmed = [dict(item, scale=1.0, cell=index) for index, item in enumerate(detections)]
        metrics = evaluate_result({"reported_defects": confirmed, "fine_cost_px": 500,
                                   "coarse_cost_px": 100, "elapsed_ms": 2}, self.labels, self.defects)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["false_positives"], 2)
        self.assertEqual(metrics["raw_false_positives"], 2)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["legacy_false_positives"], 1)
        self.assertEqual(metrics["fine_cost_frac"], 0.05)
        self.assertEqual(metrics["total_cost_frac"], 0.06)
        self.assertEqual(metrics["per_kind"]["crack"]["recall"], 1)
        self.assertEqual(metrics["per_kind"]["stain"]["recall"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_legacy_forty_pixel_and_smaller_area_rules_are_separate(self):
        # 40 pixels / 100 = 40%, and huge box makes the absolute rule material.
        large_labels = np.zeros((100, 100), np.uint16)
        large_labels[20:40, 20:40] = 1
        defects = [{"id": 1, "kind": "spall"}]
        report = [{"bbox": [0, 0, 100, 22]}]
        found = legacy_match_detections(report, large_labels, defects)
        self.assertEqual(found["found_ids"], [1])  # 40/400 <30%, but 40 pixels.
        self.assertEqual(match_detections(report, large_labels, defects)["found_ids"], [])
        report = [{"bbox": [19, 19, 3, 3]}]
        found = legacy_match_detections(report, large_labels, defects)
        self.assertEqual(found["found_ids"], [1])  # 4/9 >30%, fewer than 40 pixels.
        self.assertEqual(match_detections(report, large_labels, defects)["found_ids"], [])

    def test_bbox_clipping_and_invalid_inputs(self):
        matched = match_detections([{"bbox": [-5, -5, 2, 2]}], self.labels, self.defects)
        self.assertEqual(matched["false_positive_indices"], [0])
        with self.assertRaises(ValueError):
            match_detections([{"bbox": [0, 0, -1, 2]}], self.labels, self.defects)
        with self.assertRaises(ValueError):
            match_detections([], self.labels, [{"id": 1, "kind": "crack"}])

    def test_duplicates_do_not_inflate_recall_and_vanished_ids_remain_missed(self):
        small = np.zeros((20, 20), np.uint16)
        small[5:10, 5:10] = 1
        detections = [{"bbox": [5, 5, 5, 5], "scale": 1.0, "cell": 0},
                      {"bbox": [5, 5, 5, 5], "scale": 1.0, "cell": 1}]
        metrics = evaluate_result({"reported_defects": detections}, small, self.defects)
        self.assertEqual(metrics["n_found"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["matching"]["missed_ids"], [2])
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["raw_duplicate_reports"], 1)
        self.assertEqual(metrics["duplicate_reports"], 0)
        self.assertEqual(metrics["n_reports"], 1)

    def test_aggregation_writers_and_budget_runner(self):
        def runner(budget):
            return {"reported_defects": [{"bbox": [10, 10, 10, 10], "scale": 1.0, "cell": 0}],
                    "budget_used_frac": budget, "n_tool_calls": 2}
        records = recall_vs_budget({"agent": runner}, self.labels, self.defects, [0.05, 0.2])
        summaries = aggregate_metrics(records)
        self.assertEqual([row["recall"] for row in summaries], [0.5, 0.5])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            write_summaries(records, tmp)
            self.assertEqual(len(json.loads((Path(tmp) / "summary.json").read_text())), 2)
            svg = (Path(tmp) / "recall_vs_budget.svg").read_text()
            self.assertIn("<svg", svg)
            self.assertIn("measured inspection cost", svg)
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            self.assertTrue((Path(tmp) / "per_kind.csv").exists())
            self.assertTrue((Path(tmp) / "surface_summary.csv").exists())
            self.assertTrue((Path(tmp) / "surface_summary.md").exists())

    def test_unverified_reports_are_excluded_and_raw_cells_are_authoritative(self):
        confirmed = {"bbox": [10, 10, 10, 10], "scale": 1.0, "cell": 0, "kind_guess": "crack"}
        legacy = {"bbox": [60, 60, 10, 10]}
        low_res = dict(legacy, scale=0.5)
        metrics = evaluate_result({"reported_defects": [confirmed, legacy, low_res]}, self.labels, self.defects)
        self.assertEqual(metrics["n_found"], 1)
        self.assertEqual(metrics["unconfirmed_report_count"], 2)
        replaced = evaluate_result({"reported_defects": [confirmed], "raw_reports_by_cell": {"0": []}},
                                   self.labels, self.defects)
        self.assertEqual(replaced["n_reports"], 0)
        self.assertEqual(replaced["n_found"], 0)

    def test_precision_and_surface_tables_use_merged_reports(self):
        labels = np.zeros((256, 256), np.uint16)
        labels[10:20, 10:20] = 1
        reports = [{"bbox": [10, 10, 10, 10], "scale": 1.0, "cell": 0, "kind_guess": "crack"},
                   {"bbox": [200, 200, 10, 10], "scale": 1.0, "cell": 1, "kind_guess": "crack"}]
        metrics = evaluate_result({"reported_defects": reports}, labels, [{"id": 1, "kind": "crack"}])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["per_kind"]["crack"]["precision"], 0.5)
        rows = [{**metrics, "surface": "a", "image": "a.jpg", "method": "agent", "budget": 0.2}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            write_summaries(rows, tmp)
            summary = json.loads((Path(tmp) / "surface_summary.json").read_text())[0]
            self.assertEqual(summary["surface"], "a")
            self.assertEqual(summary["precision"], 0.5)
            self.assertEqual(summary["false_positives_after_merge"], 1)

    def test_overlay_and_trace_frames_preserve_source(self):
        image = np.full((100, 100, 3), 127, np.uint8)
        result = {"coarse": {"prior": [[0.2, 0.9]], "cell_rects": [[0, 0, 50, 100], [50, 0, 50, 100]]},
                  "trace": [{"action": "inspect_tile", "cell": 1, "scale": 0.5, "belief_after": 0.8}],
                  "reported_defects": [{"bbox": [60, 60, 10, 10]}], "human_review_queue": []}
        drawn = overlay(image, result)
        self.assertEqual(drawn.shape[2], image.shape[2])
        self.assertGreaterEqual(drawn.shape[1], image.shape[1])
        self.assertGreater(drawn.shape[0], image.shape[0])  # footer follows the intact image region
        self.assertLessEqual(drawn.shape[1], 1024)
        self.assertTrue(np.all(image == 127))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            paths = trace_frames(image, result, tmp)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).exists() for path in paths))

    def test_neighbour_belief_updates_and_report_timing(self):
        image = np.full((100, 100, 3), 127, np.uint8)
        coarse = {"prior": [[0.2, 0.1]], "cell_rects": [[0, 0, 50, 100], [50, 0, 50, 100]]}
        result = {"coarse": coarse, "trace": [{"action": "inspect_tile", "cell": 0,
                  "scale": 0.5, "belief_after": 0.2,
                  "belief_updates": [{"cell": 1, "belief_after": 0.8}]}]}
        expected = {"coarse": {**coarse, "prior": [[0.2, 0.8]]},
                    "trace": [{"action": "inspect_tile", "cell": 0, "scale": 0.5,
                               "belief_after": 0.2}]}
        np.testing.assert_array_equal(overlay(image, result), overlay(image, expected))
        first = {"cell": 0, "bbox": [1, 1, 4, 4]}
        later = {"cell": 0, "bbox": [3, 3, 8, 8]}
        result["trace"].extend([
            {"action": "report_defect", "cell": 0, "detections": [first]},
            {"action": "report_defect", "cell": 0, "detections": [later]}])
        result["reported_defects"] = [later]
        frames = []

        def render(source, state, max_width):
            frames.append(state)
            return np.zeros((8, 8, 3), np.uint8)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with patch("inspector.viz.overlay", side_effect=render):
                trace_frames(image, result, temporary)
        self.assertEqual(frames[1]["reported_defects"], [])
        self.assertEqual(frames[2]["reported_defects"], [first])
        self.assertEqual(frames[3]["reported_defects"], [later])


if __name__ == "__main__":
    unittest.main()
