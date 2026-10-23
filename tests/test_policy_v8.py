"""Spatial confirmation gates use controlled components, without image analysis."""

import json
import unittest

from inspector.agent import ActiveInspector, InspectorConfig
from inspector.baselines import run_baseline


def component(box, score=.95, **features):
    return {"bbox": box, "score": score, "patch_count": 2, "kind_guess": "spall", **features}


class SpatialBackend:
    """Two 256px cells permit one zoom+confirmation within a unit budget."""

    def __init__(self, coarse_detections, fine_detections, overview=None):
        self.coarse_detections = coarse_detections
        self.fine_detections = fine_detections
        self.overview = overview
        self.calls = []
        self.baselines = []

    def coarse_scan(self, image_ref, **kw):
        result = {"prior": [[.95, .01]], "cell_rects": [[0, 0, 256, 256], [256, 0, 256, 256]],
                  "image_shape": [256, 512], "baseline": {}, "cost_px": 8192}
        if self.overview is not None:
            result["overview_candidates"] = self.overview
        return result

    def inspect_tile(self, image_ref, rect, scale, baseline):
        self.calls.append((rect[0], scale))
        self.baselines.append(baseline)
        return {"score": .95, "detections": self.fine_detections if scale == 1 else self.coarse_detections,
                "endpoints": [], "cost_px": int(rect[2]*scale)*int(rect[3]*scale), "elapsed_ms": 0}


class SpatialPolicyTests(unittest.TestCase):
    def config(self, scale=.5, **kw):
        return InspectorConfig(grid=(1, 2), initial_scale=scale, **kw)

    def test_candidate_margin_is_one_confirmation_patch_at_both_scales(self):
        candidate = component([16, 32, 64, 32])
        touching = component([112, 40, 32, 32], .99)
        overlapping = component([111, 40, 32, 32], .94)
        unrelated = component([180, 150, 64, 32], .98)
        for scale in (.25, .5):
            with self.subTest(scale=scale):
                backend = SpatialBackend([candidate], [touching, overlapping, unrelated])
                result = ActiveInspector(backend, .625, self.config(scale)).run("fake")
                self.assertEqual([item["bbox"] for item in result["reported_defects"]], [overlapping["bbox"]])
                inspection = next(step for step in result["trace"]
                                  if step["action"] == "inspect_tile" and step["scale"] == 1)
                self.assertEqual(inspection["confirmation_candidates"][0]["bbox"], candidate["bbox"])
                self.assertTrue(inspection["clear_candidates"])
                self.assertEqual(result["coarse_candidates"], [])
                self.assertEqual(backend.baselines[1]["confirmation_candidates"][0]["bbox"], candidate["bbox"])
                self.assertEqual(backend.baselines[1]["candidate_margin_px"], 32)
                self.assertNotIn("confirmation_candidates", backend.baselines[0])
                self.assertNotIn("confirmation_candidates", result["coarse"]["baseline"])
                json.dumps(result, allow_nan=False)

    def test_score_only_candidate_cannot_authorize_unrelated_blob(self):
        backend = SpatialBackend([], [component([180, 150, 64, 32])])
        unresolved = ActiveInspector(backend, .125, self.config()).run("fake")
        self.assertFalse(unresolved["coarse_candidates"][0]["localized"])
        result = ActiveInspector(backend, .625, self.config()).run("fake")
        self.assertEqual(result["reported_defects"], [])
        self.assertTrue(result["human_review_queue"])
        self.assertEqual(backend.baselines[-1]["confirmation_candidates"], [])

    def test_spatial_gate_precedes_top_three_selection(self):
        candidate = component([0, 0, 64, 64])
        supported = component([20, 20, 32, 32], .91)
        unrelated = [component([180, 20+i*60, 32, 32], .99-i*.01) for i in range(3)]
        result = ActiveInspector(SpatialBackend([candidate], [*unrelated, supported]), .625,
                                 self.config()).run("fake")
        self.assertEqual([item["bbox"] for item in result["reported_defects"]], [supported["bbox"]])

    def test_unseeded_thin_exception_requires_length_shape_and_two_patches(self):
        eligible = component([180, 20, 64, 4], elongation=4, length_px=128, kind_guess="crack")
        short = component([180, 60, 64, 4], elongation=10, length_px=127)
        wide = component([180, 100, 64, 4], elongation=3.99, length_px=200)
        single = component([180, 140, 64, 4], elongation=10, length_px=200, patch_count=1)
        result = ActiveInspector(SpatialBackend([], [short, wide, single, eligible]), .625,
                                 self.config()).run("fake")
        self.assertEqual([item["bbox"] for item in result["reported_defects"]], [eligible["bbox"]])

    def test_minimum_patch_count_and_top_three_are_enforced_for_fake_tools(self):
        candidate = component([0, 0, 240, 240])
        detections = [component([20 + i*40, 100, 24, 24], round(.91+i*.01, 2), cell=i) for i in range(5)]
        detections.append(component([20, 20, 24, 24], .999, patch_count=1))
        for method in ("agent", "exhaustive", "random", "raster"):
            with self.subTest(method=method):
                backend = SpatialBackend([candidate], detections)
                if method == "agent":
                    result = ActiveInspector(backend, .625, self.config()).run("fake")
                else:
                    # Raster/random/exhaustive have no candidate spatial gate.
                    result = run_baseline(method, backend, "fake", .5, config=self.config())
                raw = result["raw_reports_by_cell"]
                for items in raw.values():
                    self.assertEqual(len(items), 3)
                    self.assertEqual(sorted(item["score"] for item in items), [.93, .94, .95])
                    self.assertTrue(all(item["patch_count"] >= 2 for item in items))

    def test_fullres_baselines_do_not_apply_active_candidate_gate(self):
        unrelated = component([180, 150, 64, 32])
        result = run_baseline("raster", SpatialBackend([], [unrelated]), "fake", .5, config=self.config())
        self.assertEqual(result["reported_defects"][0]["bbox"], unrelated["bbox"])

    def test_present_empty_overview_does_not_fall_back_to_cell_prior(self):
        result = run_baseline("coarse_only", SpatialBackend([], [], overview=[]), "fake", config=self.config())
        self.assertEqual(result["reported_defects"], [])
        self.assertEqual(result["report_mode"], "coarse_only")
        self.assertEqual(result["n_tool_calls"], 1)

    def test_linear_and_margin_config_are_validated(self):
        for length in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.config(linear_min_length_px=length)
        for margin in (-1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.config(candidate_margin_px=margin)


if __name__ == "__main__":
    unittest.main()
