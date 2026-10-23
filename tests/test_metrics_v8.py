"""Report-derived cell metrics, bounded fraction means and dev selection tiers."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.agent import InspectorConfig
from inspector.evaluate import METRICS_VERSION, aggregate_metrics, evaluate_result, write_summaries
from scripts.run_eval import (select_tuning_config, tune_on_dev, tuning_candidates, write_tuning_report)

cv2.setNumThreads(2)


class CellMetricsTests(unittest.TestCase):
    def test_only_reported_cells_are_flagged_and_single_cell_coverage_is_required(self):
        labels = np.zeros((20, 100), np.uint16)
        labels[5:10, :] = 1  # each of five cells contains just 20% of GT pixels
        rects = [[x, 0, 20, 20] for x in range(0, 100, 20)]
        report = {"bbox": [0, 0, 1, 1], "cells": [0, 1], "cell": 0, "scale": 1.0}
        result = {"reported_defects": [report], "coarse": {"cell_rects": rects},
                  "coarse_candidates": [{"cell": 2}], "human_review_queue": [{"cell": 3}],
                  "trace": [{"action": "inspect_tile", "cell": 4, "scale": 1.0}]}
        metrics = evaluate_result(result, labels, [{"id": 1, "kind": "crack"}])
        self.assertEqual(metrics["flagged_cells"], [0, 1])
        self.assertEqual(metrics["flagged_cell_frac"], .4)
        self.assertEqual(metrics["cell_precision"], 1)
        self.assertEqual(metrics["cell_n_found"], 0)  # 40% in total is insufficient
        self.assertEqual(metrics["defect_cell_recall"], 0)

    def test_bbox_fallback_cell_recall_precision_and_empty_reports(self):
        labels = np.zeros((20, 40), np.uint16)
        labels[5:10, 5:10] = 1
        rects = [[0, 0, 20, 20], [20, 0, 20, 20]]
        report = {"bbox": [5, 5, 30, 5], "scale": 1.0}
        result = {"reported_defects": [report], "coarse": {"cell_rects": rects}}
        metrics = evaluate_result(result, labels, [{"id": 1, "kind": "spall"}])
        self.assertEqual(metrics["flagged_cells"], [0, 1])
        self.assertEqual(metrics["defect_cell_recall"], 1)
        self.assertEqual(metrics["cell_precision"], .5)
        empty = evaluate_result({"coarse": {"cell_rects": rects}}, labels, [{"id": 1, "kind": "spall"}])
        self.assertEqual(empty["n_flagged_cells"], 0)
        self.assertEqual(empty["flagged_cell_frac"], 0)
        self.assertIsNone(empty["cell_precision"])

    def test_low_resolution_reports_require_explicit_coarse_only_mode(self):
        labels = np.zeros((20, 20), np.uint16)
        labels[5:10, 5:10] = 1
        report = {"bbox": [5, 5, 5, 5], "scale": .25, "cell": 0}
        result = {"reported_defects": [report], "raw_reports_by_cell": {},
                  "coarse": {"cell_rects": [[0, 0, 20, 20]]}}
        defects = [{"id": 1, "kind": "spall"}]
        self.assertEqual(evaluate_result(result, labels, defects)["n_reports"], 0)
        coarse = evaluate_result(dict(result, report_mode="coarse_only"), labels, defects)
        self.assertEqual(coarse["n_reports"], 1)
        self.assertEqual(coarse["recall"], 1)
        self.assertEqual(coarse["defect_cell_recall"], 1)
        self.assertEqual(coarse["metrics_version"], METRICS_VERSION)

    def test_area_means_zero_report_seed_and_invalid_raw_sum_are_not_confused(self):
        labels = np.zeros((20, 20), np.uint16)
        rows = []
        for index, width in enumerate((20, 10, 0)):
            reports = [{"bbox": [0, 0, width, 20], "scale": 1.0, "cell": 0}] if width else []
            metric = evaluate_result({"reported_defects": reports}, labels, [])
            rows.append(dict(metric, method="agent", image=f"{index}.jpg", surface="flat", budget=.2))
        self.assertEqual([row["flagged_area_frac"] for row in rows], [1, .5, 0])
        self.assertEqual(aggregate_metrics(rows)[0]["flagged_area_frac"], .5)
        weighted = [dict(rows[0], n_images=2), rows[-1]]
        self.assertAlmostEqual(aggregate_metrics(weighted)[0]["flagged_area_frac"], 2 / 3)
        with self.assertRaisesRegex(ValueError, "flagged_area_frac"):
            aggregate_metrics([dict(rows[0], flagged_area_frac=7.19)])
        with self.assertRaisesRegex(ValueError, "metrics version"):
            aggregate_metrics([dict(rows[0], metrics_version="older")])
        random_rows = [dict(row, method="random", image="same.jpg", seed=index)
                       for index, row in enumerate(rows)]
        pooled = next(row for row in aggregate_metrics(random_rows) if row["seed"] == "pooled")
        self.assertEqual(pooled["flagged_area_frac"], .5)
        self.assertEqual(pooled["precision"], 0)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            write_summaries(rows, temporary)
            out = Path(temporary)
            surface = json.loads((out / "surface_summary.json").read_text())[0]
            self.assertEqual(surface["flagged_area_frac"], .5)
            with (out / "surface_summary.csv").open() as stream:
                self.assertEqual(float(next(csv.DictReader(stream))["flagged_area_frac"]), .5)


class DevSelectionTests(unittest.TestCase):
    def test_primary_fallback_and_no_feasible_share_the_same_selection(self):
        candidates = tuning_candidates(InspectorConfig(), "classic")
        evaluation = {"mode": "tuning", "split": "dev", "images": ["dev.jpg"],
                      "detectors": ["classic"], "methods": [item["method"] for item in candidates],
                      "candidates": candidates, "seed": 0, "random_seeds": [0, 1, 2], "budgets": [.2]}
        cases = [(10, 20, "selected_primary", 0), (11, 50, "selected_fallback", 1),
                 (51, 60, "no_feasible_candidate", None)]
        for first_fp, second_fp, status, chosen in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                root = Path(temporary)
                records = []
                for index, candidate in enumerate(candidates):
                    records.append({"method": candidate["method"], "detector": "classic", "seed": 0,
                                    "budget": .2, "image": "dev.jpg", "n_defects": 2,
                                    "n_found": 2 if index == 1 else 1, "recall": 1 if index == 1 else .5,
                                    "false_positives": first_fp if index == 0 else second_fp if index == 1 else 51,
                                    "fine_cost_frac": .1, "total_cost_frac": .12, "wall_time_ms": 3})
                selected, report = select_tuning_config(records, evaluation)
                self.assertEqual(report["select_status"], status)
                self.assertEqual(report["fallback_used"], status == "selected_fallback")
                self.assertEqual(selected, candidates[chosen]["config"] if chosen is not None else None)
                write_tuning_report(report, root / "direct")
                self.assertEqual((root / "direct/config.json").exists(), chosen is not None)
                for name in ("tuning_dev.json", "dev_tradeoff.json", "dev_tradeoff.csv", "dev_tradeoff.md"):
                    self.assertTrue((root / "direct" / name).exists())
                self.assertEqual(report["history"][0]["total_cost_frac"], .12)
                (root / "manifest.json").write_text(json.dumps({"images": [{"image": "dev.jpg", "split": "dev"}]}))
                responses = iter(records)
                with patch("scripts.run_eval.evaluate_entries", side_effect=lambda *args, **kwargs: [next(responses)]):
                    helper_config, history = tune_on_dev(root, budget=.2, out_dir=root / "helper")
                self.assertEqual(helper_config, InspectorConfig(**selected) if selected is not None else None)
                helper = json.loads((root / "helper/tuning_dev.json").read_text())
                self.assertEqual(helper["select_status"], status)
                self.assertEqual(len(history), len(candidates))


if __name__ == "__main__":
    unittest.main()
