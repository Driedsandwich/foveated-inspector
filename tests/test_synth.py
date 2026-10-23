"""Small deterministic checks for procedural benchmark generation."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inspector.synth import (DEV_SURFACES, KINDS, REAL_CRACK_SURFACES,
                             TEST_SURFACES, inject_defects, make_benchmark)


cv2.setNumThreads(2)


class SynthTests(unittest.TestCase):
    def test_deterministic_and_exact_labels(self):
        background = np.full((512, 512, 3), 145, dtype=np.uint8)
        first = inject_defects(background, np.random.default_rng(7), 8, KINDS, .3)
        second = inject_defects(background, np.random.default_rng(7), 8, KINDS, .3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1].dtype, np.uint16)
        np.testing.assert_array_equal(first[1] != 0,
                                      np.any(first[0] != background, axis=2))
        self.assertEqual(set(np.unique(first[1])), set(range(9)))
        for defect in first[2]:
            selected = first[1] == defect["id"]
            self.assertEqual(defect["area_px"], int(selected.sum()))
            self.assertLess(defect["area_px"], background.shape[0] ** 2 * .015)
        small_count = sum(defect["area_px"] < 512 ** 2 * .003 for defect in first[2])
        self.assertGreater(small_count, len(first[2]) / 2)
        json.dumps(first[2])
        np.testing.assert_array_equal(background, 145)

    def test_every_kind_has_nonempty_metadata(self):
        background = np.full((512, 512, 3), 130, dtype=np.uint8)
        for kind in KINDS:
            with self.subTest(kind=kind):
                image, labels, defects = inject_defects(
                    background, np.random.default_rng(13), 1, [kind], .5)
                self.assertTrue(np.any(labels))
                self.assertEqual(defects[0]["kind"], kind)
                self.assertGreater(defects[0]["contrast"], 0)
                self.assertIn("difficulty", defects[0]["parameters"])
                if kind in ("crack", "scratch"):
                    self.assertGreater(defects[0]["length_px"], 0)
                self.assertTrue(np.any(image != background))

    def test_invalid_arguments(self):
        background = np.full((64, 64, 3), 145, dtype=np.uint8)
        for arguments in [(1, ["unknown"], .5), (1, KINDS, 1.1), (-1, KINDS, .5)]:
            with self.assertRaises(ValueError):
                inject_defects(background, np.random.default_rng(1), *arguments)

    def test_surface_split_and_manifest(self):
        self.assertEqual(len(DEV_SURFACES), 7)
        self.assertEqual(len(TEST_SURFACES), 7)
        self.assertFalse(DEV_SURFACES & TEST_SURFACES)
        self.assertFalse(REAL_CRACK_SURFACES & (DEV_SURFACES | TEST_SURFACES))
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            image = np.full((128, 128, 3), 150, dtype=np.uint8)
            paths = [root / "concrete.jpg", root / "rough_concrete.jpg"]
            for path in paths:
                cv2.imwrite(str(path), image)
            manifest = make_benchmark(paths, root / "out", 1, 41, size=128)
            self.assertEqual([item["split"] for item in manifest["images"]],
                             ["dev", "test"])
            saved = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(saved, manifest)
            for item in saved["images"]:
                labels = cv2.imread(str(root / "out" / item["labels"]),
                                    cv2.IMREAD_UNCHANGED)
                self.assertEqual(labels.dtype, np.uint16)
                self.assertGreater(int(labels.max()), 0)
            with self.assertRaises(FileExistsError):
                make_benchmark(paths, root / "out", 1, 41, size=128)
            with self.assertRaises(ValueError):
                make_benchmark([root / "cracked_concrete.jpg"], root / "bad", 1, 41)


if __name__ == "__main__":
    unittest.main()
