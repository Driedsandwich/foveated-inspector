"""Small synthetic calibration-cost checks; no learned weights are required."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.coarse import coarse_scan
from inspector.features import feature_map, prepare_calibration

cv2.setNumThreads(2)


class Predictor:
    metadata = {"calibration": "sampled"}
    kind_names = {1: "crack"}

    def threshold(self, scale):
        return .6

    def predict(self, features, scale):
        return np.full(features.shape[:2], .1, np.float32), np.ones(features.shape[:2], np.int32)


class CalibrationTests(unittest.TestCase):
    def test_high_scale_feature_inputs_are_capped_before_policy(self):
        image = np.full((512, 512, 3), 130, np.uint8)
        calls = []

        def extract(working, stats=None):
            calls.append(working.shape[:2])
            return feature_map(working, stats)

        with patch("inspector.learned.feature_map", side_effect=extract):
            result = coarse_scan(image, detector="learned", models=Predictor(), grid=(8, 8))
        self.assertEqual(calls[0], (128, 128))
        high_cost = sum(height * width for height, width in calls[1:])
        self.assertLessEqual(high_cost, int(image.shape[0] * image.shape[1] * .02))
        self.assertEqual(high_cost, result["calibration_cost_px"])
        self.assertEqual(result["cost_px"], 128 ** 2 + high_cost)
        self.assertTrue(all(height < 128 and width < 128 for height, width in calls[1:]))
        details = result["baseline"]["calibration_details"]
        self.assertEqual(len(details["sample_regions"]), 16)
        self.assertEqual(details["prior_source"], "low_response_quarter_features")
        json.dumps(result, allow_nan=False)

    def test_shared_training_calibration_matches_inference(self):
        image = np.full((128, 128, 3), 130, np.uint8)
        stats, metadata = prepare_calibration(image, grid=(4, 4))
        result = coarse_scan(image, detector="learned", models=Predictor(), grid=(4, 4))
        self.assertEqual(stats, result["baseline"]["feature_stats"])
        self.assertEqual(metadata["sample_regions"], result["baseline"]["calibration_details"]["sample_regions"])

    def test_whole_is_an_explicit_costly_ablation(self):
        image = np.full((128, 128, 3), 130, np.uint8)
        result = coarse_scan(image, detector="learned", models=Predictor(), grid=(4, 4), calibration="whole")
        self.assertEqual(result["cost_px"], 128 ** 2 + 64 ** 2 + 32 ** 2)
        self.assertEqual(result["calibration_cost_px"], 128 ** 2 + 64 ** 2)

    def test_zero_cap_reuses_overview_without_high_resolution_calls(self):
        image = np.full((64, 64, 3), 130, np.uint8)
        calls = []

        def extract(working, stats=None):
            calls.append(working.shape[:2])
            return feature_map(working, stats)

        stats, details = prepare_calibration(image, cap_frac=0, extractor=extract)
        self.assertEqual(calls, [(16, 16)])
        self.assertEqual(details["additional_cost_px"], 0)
        self.assertEqual(stats["1"], stats["0.25"])


if __name__ == "__main__":
    unittest.main()
