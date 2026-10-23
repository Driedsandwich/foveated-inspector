"""V2 checks use only small synthetic backgrounds and temporary local files."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inspector.synth import (KINDS, KINDS_V2, inject_defects, make_benchmark,
                             make_training_benchmark, merge_benchmark_shards)


cv2.setNumThreads(2)


class SynthV2Tests(unittest.TestCase):
    @staticmethod
    def background():
        rng = np.random.default_rng(8)
        gray = np.clip(rng.normal(130, 14, (512, 512)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_v2_determinism_and_actual_snr(self):
        background = self.background()
        first = inject_defects(background, np.random.default_rng(93), 8, version="v2")
        second = inject_defects(background, np.random.default_rng(93), 8, version="v2")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1].dtype, np.uint16)
        np.testing.assert_array_equal(first[1] > 0,
                                      np.any(first[0] != background, axis=2))
        before = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)
        after = cv2.cvtColor(first[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for defect in first[2]:
            selected = first[1] == defect["id"]
            x, y, w, h = defect["params"]["local_statistics_rect"]
            denominator = max(3., float(np.std(before[y:y+h, x:x+w], dtype=np.float64)))
            measured = float(np.mean(np.abs(after[selected] - before[selected]))) / denominator
            self.assertGreaterEqual(measured, 1.3)
            self.assertAlmostEqual(measured, defect["snr"], places=6)
            self.assertIn(defect["visible_at"], (.25, .5, 1.))
            self.assertGreaterEqual(defect["params"]["target_k"], 1.6)
            self.assertLessEqual(defect["params"]["target_k"], 4.)
        json.dumps(first[2], allow_nan=False)

    def test_each_v2_kind_and_v1_compatibility(self):
        for kind in KINDS_V2:
            with self.subTest(kind=kind):
                _, _, defects = inject_defects(self.background(), np.random.default_rng(1),
                                                1, [kind], difficulty=1., version="v2")
                self.assertEqual(defects[0]["kind"], kind)
                self.assertGreaterEqual(defects[0]["snr"], 1.3)
                if kind == "crack":
                    self.assertGreaterEqual(defects[0]["params"]["nominal_width_4096"], 1.5)
                if kind == "spall":
                    params = defects[0]["params"]
                    self.assertEqual(params["shadow_rim_side"], "upper_left")
                    self.assertGreaterEqual(params["texture_coarsening"], 3.)
                    self.assertLessEqual(params["texture_coarsening"], 5.)
                    self.assertNotIn("bright_rim_side", params)
        self.assertEqual(KINDS, ("crack", "scratch", "spall", "stain"))
        old = inject_defects(self.background(), np.random.default_rng(4), 2)
        explicit = inject_defects(self.background(), np.random.default_rng(4), 2, version="v1")
        np.testing.assert_array_equal(old[0], explicit[0])

    def test_efflorescence_can_follow_a_crack_downward(self):
        related = 0
        background = self.background()
        for seed in range(15):
            _, _, defects = inject_defects(background, np.random.default_rng(seed), 8,
                                            ["crack", "efflorescence"], version="v2")
            by_id = {item["id"]: item for item in defects}
            for defect in defects:
                relation = defect["params"]["placement_relation"]
                if relation["type"] == "below_crack":
                    crack = by_id[relation["defect_id"]]
                    self.assertEqual(crack["kind"], "crack")
                    self.assertGreaterEqual(defect["bbox"][1], crack["bbox"][1] + crack["bbox"][3])
                    related += 1
        self.assertGreater(related, 0)

    def test_training_namespace_never_includes_test_surface(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            dev = root / "concrete.jpg"
            test = root / "rough_concrete.jpg"
            for path in (dev, test):
                cv2.imwrite(str(path), self.background())
            evaluation = make_benchmark([dev, test], root / "eval", 1, 3, 512, version="v2")
            training = make_training_benchmark([dev], root / "train", 1, 3, 512)
            self.assertEqual(evaluation["version"], 2)
            self.assertEqual(training["images"][0]["split"], "train")
            self.assertNotEqual(evaluation["images"][0]["seed"], training["images"][0]["seed"])
            self.assertEqual({item["split"] for item in evaluation["images"]}, {"dev", "test"})
            self.assertTrue(all(3 <= len(item["defects"]) <= 8 for item in evaluation["images"]))
            with self.assertRaisesRegex(ValueError, "only use dev"):
                make_training_benchmark([test], root / "invalid", 1, 3, 512)

    def test_two_shards_merge_identically_to_unsharded_examples(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            source = root / "concrete.jpg"
            cv2.imwrite(str(source), np.full((128, 128, 3), 130, dtype=np.uint8))
            full = make_benchmark([source], root / "full", 2, 17, 128, version="v2")
            for shard in range(2):
                make_benchmark([source], root / "shards", 2, 17, 128, version="v2",
                                shard_index=shard, shard_count=2)
            directories = [root / "shards" / f"shard_{index:04d}_of_0002" for index in range(2)]
            merged = merge_benchmark_shards(reversed(directories), root / "merged")
            self.assertEqual(merged["n_images_total"], 2)
            self.assertEqual(merged["source_shard_count"], 2)
            for expected, actual in zip(full["images"], merged["images"]):
                self.assertEqual({key: value for key, value in expected.items()
                                  if key not in ("image", "labels")},
                                 {key: value for key, value in actual.items()
                                  if key not in ("image", "labels")})
                for key in ("image", "labels"):
                    self.assertEqual((root / "full" / expected[key]).read_bytes(),
                                     (root / "merged" / actual[key]).read_bytes())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                merge_benchmark_shards(directories[:1], root / "partial")
            with self.assertRaisesRegex(ValueError, "duplicate shard"):
                merge_benchmark_shards([directories[0], directories[0]], root / "duplicate")

    def test_shard_merge_validates_seed_and_training_surface(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            source = root / "concrete.jpg"
            cv2.imwrite(str(source), np.full((128, 128, 3), 130, dtype=np.uint8))
            make_training_benchmark([source], root / "train", 1, 17, 128)
            path = root / "train" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["images"][0]["seed"] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                merge_benchmark_shards([path], root / "bad_seed")
            manifest["surfaces"] = ["rough_concrete"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "only use dev"):
                merge_benchmark_shards([path], root / "bad_training")


if __name__ == "__main__":
    unittest.main()
