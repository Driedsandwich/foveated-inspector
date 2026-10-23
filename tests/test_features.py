"""Small, deterministic checks for vectorized patch features."""

import copy
import json
import unittest

import cv2
import numpy as np

from inspector.features import FEATURE_NAMES, FEATURE_VERSION, _stats, feature_map

cv2.setNumThreads(2)


class FeatureTests(unittest.TestCase):
    def test_histogram_statistics_equal_exact_even_and_odd_medians(self):
        for values in (np.array([[0, 1, 4, 8]], np.uint8),
                       np.array([[0, 1, 1, 7, 9]], np.uint8)):
            stats = _stats(values)
            median = float(np.median(values))
            self.assertEqual(stats["median"], median)
            self.assertEqual(stats["mad"], float(np.median(np.abs(values.astype(np.float32) - median))))

    def test_partial_cells_and_serializable_calibration(self):
        image = np.full((35, 65, 3), 125, np.uint8)
        features, stats = feature_map(image)
        self.assertEqual(features.shape, (2, 3, 29))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(stats["feature_version"], FEATURE_VERSION)
        self.assertEqual(len(FEATURE_NAMES), 29)
        json.dumps(stats, allow_nan=False)
        np.testing.assert_allclose(features[:, :, FEATURE_NAMES.index("lab_l_mean")], 0)

    def test_reuse_preserves_surface_reference_and_does_not_mutate(self):
        reference = np.full((128, 128, 3), 100, np.uint8)
        _, stats = feature_map(reference)
        original = copy.deepcopy(stats)
        brighter = np.full((64, 64, 3), 180, np.uint8)
        shared, _ = feature_map(brighter, stats)
        local, _ = feature_map(brighter)
        index = FEATURE_NAMES.index("lab_l_mean")
        self.assertGreater(float(shared[:, :, index].mean()), 5)
        self.assertEqual(float(local[:, :, index].max()), 0)
        self.assertEqual(stats, original)

    def test_deterministic_and_schema_validation(self):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, (96, 96, 3), np.uint8)
        first, stats = feature_map(image)
        second, _ = feature_map(image, stats)
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            feature_map(image, {"feature_version": "old"})
        with self.assertRaises(ValueError):
            feature_map(np.zeros((32, 32), np.uint8))


if __name__ == "__main__":
    unittest.main()
