"""Localization-aware matching, transparent legacy comparison and bbox coverage."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inspector.evaluate import (MATCHING_VERSION, evaluate_result, flagged_area_frac,
                                legacy_match_detections, match_detections, write_summaries)
from scripts.run_eval import SHARD_SCHEMA

cv2.setNumThreads(2)


class LocalizationMetricsTests(unittest.TestCase):
    def test_iou_can_find_sparse_defect_without_pixel_overlap(self):
        labels = np.zeros((128, 128), np.uint16)
        labels[10, 10] = labels[109, 109] = 1
        # Actual GT bbox is 100x100; metadata is deliberately stale.
        defects = [{"id": 1, "kind": "crack", "bbox": [0, 0, 1, 1], "area_px": 1000}]
        report = {"bbox": [40, 40, 40, 40], "scale": 1.0, "cell": 0}
        matching = match_detections([report], labels, defects)
        self.assertEqual(matching["found_ids"], [1])  # bbox IoU=.16
        self.assertEqual(matching["overlapping_indices"], [])
        self.assertEqual(matching["false_positive_indices"], [])
        self.assertEqual(legacy_match_detections([report], labels, defects)["found_ids"], [])
        metrics = evaluate_result({"reported_defects": [report]}, labels, defects)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["legacy_precision"], 0)

    def test_gt_pixel_coverage_path_and_boundary(self):
        labels = np.zeros((128, 128), np.uint16)
        indices = np.arange(10, 110)
        labels[indices, indices] = 1
        defects = [{"id": 1, "kind": "crack"}]
        found = match_detections([{"bbox": [10, 10, 30, 30]}], labels, defects)
        self.assertEqual(found["found_ids"], [1])  # 30/100 GT pixels, bbox IoU .09
        missed = match_detections([{"bbox": [10, 10, 29, 29]}], labels, defects)
        self.assertEqual(missed["found_ids"], [])
        self.assertEqual(missed["false_positive_indices"], [0])
        self.assertEqual(missed["partial_overlap_indices"], [0])

    def test_area_limit_rejects_full_image_box_legacy_is_explicit(self):
        labels = np.zeros((128, 128), np.uint16)
        labels[10:20, 10:20] = 1
        defects = [{"id": 1, "kind": "spall"}]
        report = {"bbox": [0, 0, 128, 128], "scale": 1.0, "cell": 0, "kind_guess": "spall"}
        metrics = evaluate_result({"reported_defects": [report]}, labels, defects)
        self.assertEqual(metrics["matching_version"], MATCHING_VERSION)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["raw_false_positives"], 1)
        self.assertEqual(metrics["legacy_recall"], 1)
        self.assertEqual(metrics["legacy_false_positives"], 0)
        self.assertEqual(metrics["per_kind"]["spall"]["legacy_recall"], 1)
        self.assertEqual(metrics["flagged_area_frac"], 1)
        self.assertEqual(SHARD_SCHEMA, "inspection-eval-shard-v5")

    def test_coverage_area_limit_includes_nine_but_rejects_larger_overhang(self):
        labels = np.zeros((128, 128), np.uint16)
        positions = np.arange(10, 110)
        labels[positions, positions] = 1
        defects = [{"id": 1, "kind": "crack"}]
        # The original bbox area, including its out-of-image overhang, is used.
        # 30/100 mask pixels, IoU<.1, report area = 9*10000 at the boundary.
        accepted = match_detections([{"bbox": [-260, 0, 300, 300]}], labels, defects)
        rejected = match_detections([{"bbox": [-260, 0, 301, 300]}], labels, defects)
        self.assertEqual(accepted["found_ids"], [1])
        self.assertEqual(rejected["found_ids"], [])
        self.assertEqual(rejected["false_positive_indices"], [0])

    def test_flagged_area_is_clipped_union_and_is_averaged_per_surface(self):
        boxes = [{"bbox": [-5, 0, 15, 10]}, {"bbox": [5, 0, 10, 10]},
                 {"bbox": [30, 30, 2, 2]}]
        self.assertEqual(flagged_area_frac(boxes, (20, 20)), 150 / 400)
        labels = np.zeros((20, 20), np.uint16)
        reports = [dict(item, cell=index, scale=1.0) for index, item in enumerate(boxes)]
        first = evaluate_result({"reported_defects": reports}, labels, [])
        second = evaluate_result({}, labels, [])
        self.assertEqual(first["flagged_area_frac"], 150 / 400)
        rows = [dict(metric, method="agent", image=f"{index}.jpg", surface="flat", budget=.2)
                for index, metric in enumerate((first, second))]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            summaries = write_summaries(rows, temporary)
            self.assertEqual(summaries[0]["flagged_area_frac"], 75 / 400)
            output = Path(temporary)
            surface = json.loads((output / "surface_summary.json").read_text())[0]
            self.assertEqual(surface["flagged_area_frac"], 75 / 400)
            self.assertEqual(surface["legacy_false_positives"], 3)
            for name in ("summary.csv", "surface_summary.csv"):
                with (output / name).open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(float(row["flagged_area_frac"]), 75 / 400)
                self.assertIn("legacy_recall", row)
                self.assertEqual(row["matching_version"], MATCHING_VERSION)
            with (output / "results.csv").open() as stream:
                image_rows = list(csv.DictReader(stream))
            self.assertEqual(len(image_rows), 2)
            self.assertEqual(float(image_rows[0]["legacy_false_positives"]), 3)
            self.assertEqual(float(image_rows[0]["flagged_area_frac"]), 150 / 400)
            self.assertIn("Flagged area fraction", (output / "surface_summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
