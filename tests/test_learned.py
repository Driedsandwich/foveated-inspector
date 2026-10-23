"""Small tool-contract checks independent of any production model weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.coarse import coarse_scan
from inspector.fine import inspect_tile
from inspector.tools import LocalBackend

cv2.setNumThreads(2)


class TinyPredictor:
    kind_names = {1: "crack"}

    def threshold(self, scale):
        return 0.6

    def predict(self, features, scale):
        probability = np.full(features.shape[:2], .1, np.float32)
        probability[:, 0] = .9
        return probability, np.ones(features.shape[:2], np.int32)


class LearnedToolTests(unittest.TestCase):
    def test_json_coordinates_cost_and_border_direction(self):
        image = np.full((128, 128, 3), 135, np.uint8)
        models = TinyPredictor()
        overview = coarse_scan(image, grid=(2, 2), detector="learned", models=models,
                               calibration="whole")
        self.assertEqual(overview["calibration_cost_px"], 128**2+64**2)
        self.assertEqual(overview["cost_px"], overview["calibration_cost_px"]+32**2)
        self.assertEqual(overview["baseline"]["detector_thresholds"],
                         {"0.25": .6, "0.5": .6, "1": .6})
        observation = inspect_tile(image, [32, 32, 64, 64], 1.0, overview["baseline"],
                                   detector="learned", models=models)
        self.assertEqual(observation["cost_px"], 4096)
        self.assertEqual(observation["detections"][0]["bbox"], [32, 32, 32, 64])
        self.assertTrue(any(endpoint["dy"] > .9 for endpoint in observation["endpoints"]))
        json.dumps(overview, allow_nan=False)
        json.dumps(observation, allow_nan=False)

    def test_segmentation_uses_model_floor_and_policy_multiplier(self):
        image = np.full((64, 64, 3), 135, np.uint8)
        models = TinyPredictor()
        baseline = coarse_scan(image, detector="learned", models=models,
                               grid=(2, 2))["baseline"]
        baseline["effective_thresholds"] = {"0.25": .1, "0.5": .1, "1": .1}
        observation = inspect_tile(image, [0, 0, 64, 64], 1., baseline,
                                   detector="learned", models=models)
        self.assertEqual(observation["threshold"], .6)
        self.assertTrue(observation["detections"])
        baseline["threshold_multiplier"] = 1.6
        observation = inspect_tile(image, [0, 0, 64, 64], 1., baseline,
                                   detector="learned", models=models)
        self.assertAlmostEqual(observation["threshold"], .96)
        self.assertEqual(observation["detections"], [])
        # A policy-provided threshold is already multiplied, so .84 stays .84.
        baseline["threshold_multiplier"] = 1.2
        baseline["effective_thresholds"]["1"] = .84
        observation = inspect_tile(image, [0, 0, 64, 64], 1., baseline,
                                   detector="learned", models=models)
        self.assertAlmostEqual(observation["threshold"], .84)
        self.assertTrue(observation["detections"])

    def test_backend_cache_reports_physical_and_standalone_costs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            image_path = Path(temporary)/"surface.png"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, np.uint8))
            with patch("inspector.model.DetectorModels", return_value=TinyPredictor()):
                backend = LocalBackend(detector="learned")
            first = backend.coarse_scan(str(image_path), grid=(2, 2))
            second = backend.coarse_scan(str(image_path), grid=(2, 2))
            self.assertFalse(first["reused_preparation"])
            self.assertTrue(second["reused_preparation"])
            self.assertEqual(backend.physical_coarse_cost_px, first["cost_px"])
            self.assertEqual(backend.coarse_cost_px, 2*first["cost_px"])
            json.dumps(second, allow_nan=False)

    def test_empty_confirmation_context_cannot_confirm_an_unrelated_short_component(self):
        image = np.full((64, 64, 3), 135, np.uint8)
        models = TinyPredictor()
        baseline = coarse_scan(image, detector="learned", models=models)["baseline"]
        baseline["confirmation_candidates"] = []
        baseline["candidate_margin_px"] = 32
        baseline["linear_min_length_px"] = 128
        observation = inspect_tile(image, [0, 0, 64, 64], 1., baseline,
                                   detector="learned", models=models)
        self.assertEqual(observation["detections"], [])
        self.assertEqual(observation["endpoints"], [])

    def test_single_patch_positive_is_not_reported(self):
        image = np.full((128, 128, 3), 135, np.uint8)
        models = TinyPredictor()
        baseline = coarse_scan(image, grid=(2, 2), detector="learned", models=models)["baseline"]
        result = inspect_tile(image, [10, 20, 61, 59], .25, baseline,
                              detector="learned", models=models)
        self.assertEqual(result["detections"], [])
        self.assertGreater(result["score"], .8)
        self.assertEqual(result["cost_px"], 15*14)

    def test_partial_patch_measurement_stays_inside_the_tile(self):
        class AllPositive(TinyPredictor):
            def predict(self, features, scale):
                return (np.full(features.shape[:2], .9, np.float32),
                        np.ones(features.shape[:2], np.int32))

        image = np.full((128, 128, 3), 135, np.uint8)
        models = AllPositive()
        baseline = coarse_scan(image, detector="learned", models=models)["baseline"]
        result = inspect_tile(image, [10, 20, 61, 59], 1., baseline,
                              detector="learned", models=models)
        detection = result["detections"][0]
        self.assertEqual(detection["bbox"], [10, 20, 61, 59])
        self.assertEqual(detection["patch_count"], 4)
        self.assertLessEqual(detection["width_px"], 59)
        self.assertEqual(result["cost_px"], 61*59)

    def test_shallow_border_crossing_uses_the_touched_neighbour(self):
        class ShallowPredictor(TinyPredictor):
            def predict(self, features, scale):
                probability = np.full(features.shape[:2], .1, np.float32)
                for row in range(probability.shape[0]):
                    probability[row, row*2:row*2+2] = .9
                return probability, np.ones(features.shape[:2], np.int32)

        image = np.full((128, 256, 3), 135, np.uint8)
        models = ShallowPredictor()
        baseline = coarse_scan(image, grid=(2, 2), detector="learned", models=models)["baseline"]
        result = inspect_tile(image, [0, 0, 256, 128], 1.0, baseline,
                              detector="learned", models=models)
        bottom = [point for point in result["endpoints"] if point["y"] == 127]
        self.assertTrue(bottom)
        self.assertEqual((bottom[0]["dx"], bottom[0]["dy"]), (0, 1))
        self.assertGreater(abs(bottom[0]["tangent_dx"]), abs(bottom[0]["tangent_dy"]))


if __name__ == "__main__":
    unittest.main()
