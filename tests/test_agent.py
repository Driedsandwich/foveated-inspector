"""Policy tests use controlled visual observations, independent of detection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.agent import (ActiveInspector, InspectorConfig, effective_threshold,
                             merge_reports, policy_baseline, replace_cell_reports)
from inspector.baselines import run_baseline
from inspector.tools import LocalBackend


class ConstructedBackend:
    def __init__(self, score=0.1, endpoints=False):
        self.calls = []
        self.score = score
        self.endpoints = endpoints

    def coarse_scan(self, image_ref, **kw):
        return {"prior": [[0.9, 0.15, 0.8]], "cell_rects": [[0, 0, 32, 32],
                [32, 0, 32, 32], [64, 0, 32, 32]], "image_shape": [32, 96],
                "baseline": {}, "cost_px": 768}

    def inspect_tile(self, image_ref, rect, scale, baseline):
        self.calls.append((rect[0] // 32, scale))
        return {"score": self.score, "confidence": 0.8, "detections": [],
                "cost_px": int(rect[2]*scale)*int(rect[3]*scale), "elapsed_ms": 0,
                "endpoints": [{"x": 31, "y": 16, "dx": 1, "dy": 0}]
                if self.endpoints and rect[0] == 0 else []}


class PolicyTests(unittest.TestCase):
    def config(self):
        return InspectorConfig(grid=(1, 3), initial_scale=0.5, stop_threshold=0.1)

    def test_never_exceeds_budget(self):
        for budget in (0, 0.001, 0.05, 0.2, 0.51, 1):
            backend = ConstructedBackend(score=0.5)
            result = ActiveInspector(backend, budget, self.config()).run("example")
            self.assertLessEqual(result["fine_cost_px"], int(3072 * budget))
            self.assertEqual(result["n_tool_calls"], len(backend.calls) + 1)
            json.dumps(result, allow_nan=False)

    def test_border_evidence_changes_next_action(self):
        backend = ConstructedBackend(endpoints=True)
        result = ActiveInspector(backend, 0.2, self.config()).run("example")
        self.assertEqual([cell for cell, _ in backend.calls], [0, 1])
        self.assertIn("neighbour", result["trace"][2]["reason"])
        control = ConstructedBackend(endpoints=False)
        ActiveInspector(control, 0.2, self.config()).run("example")
        self.assertEqual([cell for cell, _ in control.calls], [0, 2])

    def test_escalation_and_human_queue(self):
        backend = ConstructedBackend(score=0.5)
        result = ActiveInspector(backend, 1, self.config()).run("example")
        self.assertIn((0, 1.0), backend.calls)
        self.assertTrue(result["human_review_queue"])

    def test_baseline_costs(self):
        for method in ("random", "raster", "exhaustive", "coarse_only"):
            result = run_baseline(method, ConstructedBackend(), "example", 0.4, config=self.config())
            expected = 1 if method == "exhaustive" else 0 if method == "coarse_only" else 1/3
            self.assertAlmostEqual(result["budget_used_frac"], expected)
            json.dumps(result, allow_nan=False)

    def test_low_resolution_positive_is_only_a_candidate(self):
        backend = ConstructedBackend(score=.95)
        result = ActiveInspector(backend, .1, self.config()).run("example")
        self.assertEqual(backend.calls, [(0, .5)])
        self.assertEqual(result["reported_defects"], [])
        self.assertTrue(result["coarse_candidates"])
        self.assertTrue(any(step["action"] == "record_candidate" for step in result["trace"]))

    def test_candidate_is_confirmed_in_same_cell_before_search_continues(self):
        class EvidenceBackend(ConstructedBackend):
            def inspect_tile(self, image_ref, rect, scale, baseline):
                observation = super().inspect_tile(image_ref, rect, scale, baseline)
                observation["detections"] = [{"bbox": list(rect), "score": self.score, "kind_guess": "crack", "patch_count": 2}]
                return observation

        backend = EvidenceBackend(score=.95)
        result = ActiveInspector(backend, .5, self.config()).run("example")
        self.assertEqual(backend.calls[:2], [(0, .5), (0, 1.0)])
        self.assertTrue(result["reported_defects"])
        self.assertTrue(all(report["scale"] == 1 and "cell" in report for report in result["reported_defects"]))
        inspection = next(step for step in result["trace"] if step["action"] == "inspect_tile" and step["scale"] == 1)
        self.assertTrue(inspection["clear_candidates"])
        self.assertTrue(inspection["confirmation_candidates"])
        self.assertEqual(result["raw_reports_by_cell"]["0"][0]["scale"], 1)

    def test_full_resolution_negative_clears_candidate(self):
        class NegativeConfirmation(ConstructedBackend):
            def inspect_tile(self, image_ref, rect, scale, baseline):
                observation = super().inspect_tile(image_ref, rect, scale, baseline)
                observation["score"] = .95 if scale < 1 else .05
                return observation

        result = ActiveInspector(NegativeConfirmation(), .42, self.config()).run("example")
        self.assertEqual(result["reported_defects"], [])
        self.assertEqual(result["coarse_candidates"], [])
        self.assertEqual(result["raw_reports_by_cell"]["0"], [])
        self.assertTrue(any(step.get("merged_reports") == [] and step.get("clear_candidates")
                            for step in result["trace"]))

    def test_model_validation_thresholds_are_hard_floors(self):
        baseline = {"detector_thresholds": {"0.25": .8, "0.5": .85, "1": .99}}
        config = InspectorConfig(positive_threshold=.6, detection_threshold=.6, threshold_multiplier=1.1)
        prepared = policy_baseline(baseline, config)
        self.assertAlmostEqual(prepared["effective_thresholds"]["0.25"], .88)
        self.assertAlmostEqual(prepared["effective_thresholds"]["0.5"], .935)
        self.assertEqual(prepared["effective_thresholds"]["1"], .99)
        self.assertNotIn("effective_thresholds", baseline)
        self.assertEqual(effective_threshold(baseline, 1, .1), .99)
        for multiplier in (.9, float("nan"), True):
            with self.assertRaises(ValueError):
                InspectorConfig(threshold_multiplier=multiplier)

    def test_policy_and_baseline_cannot_report_below_model_floor(self):
        class FloorBackend(ConstructedBackend):
            def coarse_scan(self, image_ref, **kw):
                result = super().coarse_scan(image_ref, **kw)
                result["baseline"]["detector_thresholds"] = {"0.25": .9, "0.5": .9, "1": .9}
                return result

            def inspect_tile(self, image_ref, rect, scale, baseline):
                self.asserted_threshold = baseline["effective_thresholds"][f"{scale:g}"]
                result = super().inspect_tile(image_ref, rect, scale, baseline)
                result["detections"] = [{"bbox": list(rect), "score": self.score, "kind_guess": "crack", "patch_count": 2}]
                return result

        config = InspectorConfig(grid=(1, 3), positive_threshold=.6, detection_threshold=.6)
        backend = FloorBackend(score=.85)
        result = ActiveInspector(backend, .42, config).run("example")
        self.assertEqual(result["reported_defects"], [])
        self.assertEqual(result["coarse_candidates"], [])
        self.assertTrue(result["human_review_queue"])
        self.assertEqual(backend.asserted_threshold, .9)
        baseline = run_baseline("exhaustive", FloorBackend(score=.85), "example", config=config)
        self.assertEqual(baseline["reported_defects"], [])
        self.assertTrue(baseline["human_review_queue"])

    def test_uncertainty_threshold_is_independent_of_validation_floor(self):
        floor = .84375

        class BoundaryBackend(ConstructedBackend):
            def coarse_scan(self, image_ref, **kw):
                result = super().coarse_scan(image_ref, **kw)
                result["baseline"].update(detector="learned", detector_thresholds={
                    "0.25": floor, "0.5": floor, "1": floor})
                return result

            def inspect_tile(self, image_ref, rect, scale, baseline):
                observation = super().inspect_tile(image_ref, rect, scale, baseline)
                # Mimic segmentation: probabilities below the validation floor
                # produce tile-level uncertainty without a localized bbox.
                if self.score >= floor:
                    observation["detections"] = [{"bbox": list(rect), "score": self.score, "kind_guess": "crack", "patch_count": 2}]
                return observation

        config = InspectorConfig(grid=(1, 3), ambiguous_low=.35,
                                 positive_threshold=.92, detection_threshold=.92)
        for score, expected in ((.34, "negative"), (.35, "review"), (.80, "review"),
                                (floor, "review"), (.88, "review"), (.92, "confirmed"), (.95, "confirmed")):
            with self.subTest(score=score):
                backend = BoundaryBackend(score=score)
                result = ActiveInspector(backend, .42, config).run("example")
                self.assertEqual(bool(result["reported_defects"]), expected == "confirmed")
                self.assertEqual(bool(result["human_review_queue"]), expected == "review")
                self.assertEqual(result["coarse_candidates"], [])
                if expected == "negative":
                    self.assertNotIn((0, 1.0), backend.calls)
                else:
                    self.assertEqual(backend.calls[:2], [(0, .5), (0, 1.0)])
                baseline = run_baseline("exhaustive", BoundaryBackend(score=score), "example", config=config)
                self.assertEqual(bool(baseline["reported_defects"]), expected == "confirmed")
                self.assertEqual(bool(baseline["human_review_queue"]), expected == "review")
        unresolved = ActiveInspector(BoundaryBackend(score=.80), .1, config).run("example")
        self.assertEqual(unresolved["reported_defects"], [])
        self.assertEqual(unresolved["coarse_candidates"][0]["kind_guess"], "unknown")
        self.assertEqual(unresolved["coarse_candidates"][0]["scale"], .5)

    def test_overlapping_pairs_replace_raw_cell_reports_without_chaining(self):
        def report(x, y=0, score=.9):
            return {"bbox": [x, y, 20, 10], "scale": 1.0, "cell": int(x), "score": score}
        merged = merge_reports([report(0), report(10, score=.95), report(20)])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["bbox"], [0, 0, 30, 10])
        self.assertEqual(merged[0]["score"], .95)
        self.assertEqual(merge_reports(merged), merged)
        self.assertEqual(len(merge_reports([report(0), report(21)])), 2)
        raw = {}
        replace_cell_reports(raw, 0, [report(0)], .8)
        self.assertEqual(replace_cell_reports(raw, 1, [report(10)], .8)[0]["bbox"], [0, 0, 30, 10])
        self.assertEqual(replace_cell_reports(raw, 1, [report(200)], .8)[0]["bbox"], [0, 0, 20, 10])
        self.assertEqual(len(replace_cell_reports(raw, 1, [], .8)), 1)
        self.assertEqual(replace_cell_reports(raw, 0, [], .8), [])
        for item in ({"bbox": [0, 0, 5, 5]}, {"bbox": [0, 0, 5, 5], "scale": .5}):
            with self.assertRaises(ValueError):
                merge_reports([item])

    def test_long_crack_stays_in_bounded_pairs(self):
        reports = [{"bbox": [x, 100, 64, 10], "scale": 1.0, "cell": i, "score": .95}
                   for i, x in enumerate(range(0, 449, 32))]
        merged = merge_reports(reports)
        self.assertEqual(len(merged), 8)
        self.assertTrue(all(item["bbox"][2] <= 96 and item["bbox"][3] == 10 for item in merged))
        self.assertTrue(all(item["n_merged_components"] <= 2 for item in merged))
        self.assertEqual(merge_reports(list(reversed(reports))), merged)
        self.assertEqual(merge_reports(merged), merged)

    def test_pair_coverage_and_area_and_side_caps(self):
        def report(box, cell=0):
            return {"bbox": box, "scale": 1.0, "cell": cell, "score": .9}
        # Containment qualifies despite IoU=.01; the union remains the larger box.
        self.assertEqual(len(merge_reports([report([0, 0, 100, 100]), report([5, 5, 10, 10], 1)])), 1)
        self.assertEqual(len(merge_reports([report([0, 0, 100, 100]), report([93, 0, 10, 100], 1)])), 1)
        self.assertEqual(len(merge_reports([report([0, 0, 100, 100]), report([94, 0, 10, 100], 1)])), 2)
        # IoU > .3, but the cross-shaped pair's enclosing box exceeds 2x4700.
        self.assertEqual(len(merge_reports([report([0, 0, 100, 47]), report([0, 0, 47, 100], 1)])), 2)
        # Pure bbox arithmetic only: no >512px image is created or processed.
        self.assertEqual(len(merge_reports([report([0, 0, 1000, 100]), report([100, 0, 1000, 100], 1)])), 2)
        capped = merge_reports([report([0, 0, 900, 100]), report([124, 0, 900, 100], 1)])
        self.assertEqual(capped[0]["bbox"], [0, 0, 1024, 100])

    def test_unmerged_fractional_boxes_do_not_gain_overlap_when_rounded(self):
        reports = [{"bbox": box, "scale": 1.0, "cell": index, "score": .9}
                   for index, box in enumerate(([.49, 0, 1, 1], [1.05, 0, 1, 1]))]
        merged = merge_reports(reports)
        self.assertEqual(len(merged), 2)
        self.assertEqual([item["bbox"] for item in merged], [item["bbox"] for item in reports])
        self.assertEqual(merge_reports(merged), merged)

    def test_coarse_only_explicitly_reports_prior_fallback_without_confirmation(self):
        result = run_baseline("coarse_only", ConstructedBackend(), "example", config=self.config())
        self.assertEqual(result["report_mode"], "coarse_only")
        self.assertTrue(result["reported_defects"])
        self.assertTrue(all(item["status"] == "coarse_report" and item["scale"] == .25
                            and item["source"] == "prior_cell_fallback" and not item["localized"]
                            for item in result["reported_defects"]))
        self.assertEqual(result["raw_reported_defects"], [])
        self.assertEqual(result["fine_cost_px"], 0)

    def test_coarse_only_reports_real_overview_components_without_cell_unions(self):
        class OverviewBackend(ConstructedBackend):
            def coarse_scan(self, image_ref, **kw):
                result = super().coarse_scan(image_ref, **kw)
                result["overview_candidates"] = [
                    {"bbox": [2, 3, 8, 12], "cell": 0, "score": .94, "scale": .25, "patch_count": 2},
                    {"bbox": [3, 4, 8, 12], "cell": 0, "score": .93, "scale": .25, "patch_count": 2}]
                return result

        result = run_baseline("coarse_only", OverviewBackend(), "example", config=self.config())
        self.assertEqual([item["bbox"] for item in result["reported_defects"]], [[2, 3, 8, 12], [3, 4, 8, 12]])
        self.assertTrue(all(item["scale"] == .25 and item["cells"] == [0] for item in result["reported_defects"]))
        self.assertEqual(result["coarse_candidates"], [])

    def test_validation(self):
        for budget in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                ActiveInspector(ConstructedBackend(), budget)

    def test_time_guard_saves_partial_trace_before_another_tool_call(self):
        backend = ConstructedBackend(score=.9)
        with patch("inspector.agent.time.perf_counter", side_effect=[0, 0, .01, 2, 2]):
            result = ActiveInspector(backend, .4, self.config()).run("example", time_limit_ms=1000)
        self.assertEqual(backend.calls, [])
        self.assertEqual(result["n_tool_calls"], 1)
        self.assertEqual(len(result["trace"]), 1)
        self.assertIn("partial", result["stop_reason"])

    def test_real_tools_json_and_accounting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp)/"flat.png"
            cv2.imwrite(str(path), np.full((128, 128, 3), 128, np.uint8))
            backend = LocalBackend()
            coarse = backend.coarse_scan(str(path), grid=(2, 2), work_width=64)
            fine = backend.inspect_tile(str(path), [0, 0, 64, 64], 0.5, coarse["baseline"])
            json.dumps(coarse, allow_nan=False)
            json.dumps(fine, allow_nan=False)
            self.assertEqual(backend.fine_cost_px, 1024)
            self.assertEqual(backend.coarse_cost_px, coarse["cost_px"])
            self.assertEqual(backend.n_tool_calls, 2)
            self.assertIs(backend.image(str(path)), backend.image(str(path)))


if __name__ == "__main__":
    unittest.main()
