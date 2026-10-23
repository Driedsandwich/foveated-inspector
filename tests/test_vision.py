"""Small synthetic checks for the overview and tile tools."""

import json
import unittest

import cv2
import numpy as np

from inspector.coarse import coarse_scan
from inspector.fine import inspect_tile

cv2.setNumThreads(2)


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.flat = np.full((512, 512, 3), 145, np.uint8)

    def test_flat_is_negative_and_json_safe(self):
        overview = coarse_scan(self.flat)
        result = inspect_tile(self.flat, [0, 0, 512, 512], 1.0, overview["baseline"])
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["cost_px"], 512 ** 2)
        self.assertEqual(max(map(max, overview["prior"])), 0.0)
        json.dumps(overview, allow_nan=False)
        json.dumps(result, allow_nan=False)

    def test_high_contrast_crack_and_border_endpoint(self):
        image = self.flat.copy()
        points = np.array([[255, 50], [247, 150], [265, 255], [250, 375], [257, 511]], np.int32)
        cv2.polylines(image, [points], False, (25, 25, 25), 3, cv2.LINE_AA)
        overview = coarse_scan(image)
        result = inspect_tile(image, [0, 0, 512, 512], 1.0, overview["baseline"])
        lines = [d for d in result["detections"] if d["kind_guess"] == "crack"]
        self.assertTrue(lines)
        self.assertGreater(max(d["length_px"] for d in lines), 250)
        self.assertTrue(any(e["dy"] == 1 for e in result["endpoints"]))
        self.assertGreater(result["score"], 0.7)
        json.dumps(result, allow_nan=False)

    def test_coordinates_and_rounding(self):
        overview = coarse_scan(self.flat, grid=(7, 9), work_width=511)
        self.assertEqual(sum(r[2] * r[3] for r in overview["cell_rects"]), 512 ** 2)
        result = inspect_tile(self.flat, [20, 30, 101, 99], 0.25, overview["baseline"])
        self.assertEqual(result["cost_px"], 25 * 24)
        with self.assertRaises(ValueError):
            inspect_tile(self.flat, [-1, 0, 20, 20], 1.0, {})
        with self.assertRaises(ValueError):
            inspect_tile(self.flat, [0, 0, 20, 20], 0.3, {})

    def test_clean_correlated_texture_has_no_confident_reports(self):
        rng = np.random.default_rng(140)
        noise = rng.normal(0, 14, (512, 512)).astype(np.float32)
        texture = cv2.GaussianBlur(noise, (0, 0), 0.7) + 145
        image = np.repeat(np.clip(texture, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        baseline = coarse_scan(image)["baseline"]
        result = inspect_tile(image, [0, 0, 512, 512], 1.0, baseline)
        self.assertFalse(any(d["score"] >= 0.75 for d in result["detections"]))

    def test_compact_branched_texture_is_not_a_long_crack(self):
        # Small crossing mineral-like marks have a long skeleton but no main
        # elongated axis. Skeleton length alone must not make them confident.
        image = np.full((512, 512, 3), 145, np.uint8)
        for y in (100, 200, 300, 400):
            for x in (100, 200, 300, 400):
                cv2.line(image, (x - 9, y), (x + 9, y), (35, 35, 35), 1)
                cv2.line(image, (x, y - 9), (x, y + 9), (35, 35, 35), 1)
        baseline = coarse_scan(image)["baseline"]
        result = inspect_tile(image, [0, 0, 512, 512], 1.0, baseline)
        self.assertFalse(any(d["kind_guess"] == "crack" and d["score"] >= 0.75
                             for d in result["detections"]))

    def test_solid_straight_crack_has_finite_width_and_border_endpoint(self):
        image = np.full((256, 256, 3), 145, np.uint8)
        image[60:256, 125:128] = 25
        overview = coarse_scan(image, work_width=128)
        result = inspect_tile(image, [0, 0, 256, 256], 1.0, overview["baseline"])
        lines = [d for d in result["detections"] if d["kind_guess"] == "crack"]
        self.assertTrue(lines)
        line = max(lines, key=lambda d: d["length_px"])
        self.assertGreater(line["length_px"], 180)
        self.assertLess(line["length_px"], 210)
        self.assertGreaterEqual(line["width_px"], 1)
        self.assertLess(line["width_px"], 6)
        self.assertGreater(line["score"], 0.75)
        self.assertGreaterEqual(line["patch_count"], 6)
        self.assertEqual(line["patch_count_source"], "occupied_analysis_grid_proxy")
        self.assertGreater(line["elongation"], 4)
        self.assertTrue(any(e["dy"] == 1 for e in result["endpoints"]))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
