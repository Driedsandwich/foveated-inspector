"""Bounded component counts, spatial confirmation, and paid-overview localization."""

import json
import unittest

import cv2
import numpy as np

from inspector.coarse import coarse_scan
from inspector.components import clip_bbox, component_gate, spatial_candidate_gate
from inspector.learned import _patch_components

cv2.setNumThreads(2)


def detection(x, *, cell=0, count=2, score=.9, component_id=None):
    return {"bbox": [x, 10, 12, 24], "cell": cell, "score": score,
            "patch_count": count, "elongation": 2., "length_px": 24.,
            "component_id": component_id}


class ComponentGateTests(unittest.TestCase):
    def test_minimum_two_is_not_waived_for_a_thin_shape(self):
        thin = {**detection(0, count=1), "elongation": 20., "length_px": 500.}
        missing = detection(20)
        missing.pop("patch_count")
        self.assertEqual(component_gate([thin, missing]), [])
        with self.assertRaises(ValueError):
            component_gate([thin], min_patches=1)

    def test_top_three_per_cell_are_deterministic_and_do_not_mutate_inputs(self):
        records = [detection(x, score=.8, component_id=x) for x in (40, 10, 30, 20)]
        records.append(detection(60, cell=1, score=.99))
        snapshot = json.dumps(records, sort_keys=True)
        selected = component_gate(records)
        self.assertEqual(selected, component_gate(reversed(records)))
        self.assertEqual([item["bbox"][0] for item in selected], [60, 10, 20, 30])
        self.assertEqual(json.dumps(records, sort_keys=True), snapshot)
        json.dumps(selected, allow_nan=False)

    def test_spatial_margin_is_full_resolution_32_not_scaled_candidate_patch(self):
        candidate = {"bbox": [100, 100, 10, 10], "scale": .25, "localized": True}
        near = {**detection(130), "bbox": [130, 100, 5, 5]}
        far = {**detection(160), "bbox": [160, 100, 5, 5]}
        touching = {**detection(142), "bbox": [142, 100, 5, 5]}
        selected = spatial_candidate_gate([far, near, touching], [candidate])
        self.assertEqual(selected, [near])
        self.assertEqual(spatial_candidate_gate([near], [{**candidate, "localized": False}]), [])
        self.assertEqual(spatial_candidate_gate([near], []), [])

    def test_spatial_gate_clips_real_detections_without_replacing_them_by_seed_boxes(self):
        candidate = {"bbox": [0, 0, 20, 20], "scale": .5, "localized": True}
        actual = {**detection(0), "bbox": [-10, -8, 20, 18]}
        selected = spatial_candidate_gate([actual], [candidate], image_shape=(64, 64))
        self.assertEqual(selected[0]["bbox"], [0, 0, 10, 10])
        self.assertEqual(actual["bbox"], [-10, -8, 20, 18])
        self.assertIsNone(clip_bbox([100, 100, 10, 10], [0, 0, 64, 64]))
        self.assertEqual(component_gate([{**actual, "bbox": [0, 0, float("nan"), 10]}]), [])


class ClassifiedComponentTests(unittest.TestCase):
    def components(self, probability, **kwargs):
        height, width = probability.shape
        options = {"work_shape": (height*32, width*32), "rect": [0, 0, width*32, height*32],
                   "scale": 1., "threshold": .6, "kind_names": {1: "crack"}}
        options.update(kwargs)
        return _patch_components(probability, np.ones(probability.shape, np.int32), **options)

    def test_patch_count_is_actual_component_size_not_bbox_area(self):
        probability = np.full((3, 3), .1, np.float32)
        probability[0, :] = .9
        probability[:, 0] = .9
        records, _ = self.components(probability)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["patch_count"], 5)
        self.assertEqual(records[0]["bbox"], [0, 0, 96, 96])
        self.assertGreaterEqual(records[0]["elongation"], 1.)
        self.assertIn("length_px", records[0])

    def test_supported_fourth_component_survives_and_only_retained_endpoints_remain(self):
        probability = np.full((16, 16), .1, np.float32)
        for column, score in ((0, .99), (4, .98), (8, .97), (12, .90)):
            probability[:2, column] = score
        all_records, all_endpoints = self.components(probability)
        self.assertEqual(len(all_records), 3)
        self.assertNotIn(384, [item["bbox"][0] for item in all_records])
        self.assertTrue(all(endpoint["component_id"] in {item["component_id"] for item in all_records}
                            for endpoint in all_endpoints))
        records, endpoints = self.components(
            probability, confirmation_candidates=[{"bbox": [380, 0, 42, 70], "scale": .5,
                                                     "localized": True}])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["bbox"], [384, 0, 32, 64])
        self.assertEqual(records[0]["patch_count"], 2)
        self.assertTrue(endpoints)
        self.assertTrue(all(endpoint["component_id"] == records[0]["component_id"] for endpoint in endpoints))

    def test_thin_exception_still_requires_two_patches(self):
        probability = np.full((4, 2), .1, np.float32)
        probability[:, 0] = .9
        records, _ = self.components(probability, confirmation_candidates=[])
        self.assertEqual(records[0]["patch_count"], 4)
        self.assertGreaterEqual(records[0]["elongation"], 4.)
        self.assertGreaterEqual(records[0]["length_px"], 128.)
        # A single partial patch can be 4x128 full-image pixels and very thin,
        # but it cannot waive the mandatory two-classified-patch minimum.
        partial = np.array([[.1, .9]], np.float32)
        records, _ = self.components(partial, work_shape=(32, 33), rect=[0, 0, 132, 128],
                                     scale=.25, confirmation_candidates=[])
        self.assertEqual(records, [])

    def test_overview_uses_paid_map_once_and_splits_components_by_owning_cell(self):
        class Predictor:
            kind_names = {1: "crack"}

            def __init__(self):
                self.calls = []
                self.cross_boundary = False

            def threshold(self, scale):
                return .6

            def predict(self, features, scale):
                self.calls.append(scale)
                probability = np.full(features.shape[:2], .1, np.float32)
                if self.cross_boundary:
                    probability[1, 1:3] = .9
                else:
                    probability[:2, 0] = .9
                    probability[0, 2] = .99
                return probability, np.ones(features.shape[:2], np.int32)

        image = np.full((512, 512, 3), 127, np.uint8)
        models = Predictor()
        overview = coarse_scan(image, grid=(1, 2), detector="learned", models=models,
                               calibration_cap_frac=0)
        self.assertEqual(models.calls, [.25])
        self.assertEqual(overview["cost_px"], 128**2)
        candidates = overview["overview_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["cell"], 0)
        self.assertEqual(candidates[0]["bbox"], [0, 0, 128, 256])
        self.assertEqual(candidates[0]["patch_count"], 2)
        self.assertEqual(candidates[0]["scale"], .25)
        self.assertTrue(candidates[0]["localized"])
        json.dumps(overview, allow_nan=False)
        models.cross_boundary = True
        separated = coarse_scan(image, grid=(2, 2), detector="learned", models=models,
                                calibration_cap_frac=0)
        self.assertEqual(separated["overview_candidates"], [])


if __name__ == "__main__":
    unittest.main()
