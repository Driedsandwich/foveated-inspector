"""Streaming and metadata-only shard merging on tiny synthetic photos."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inspector.synth import (assigned_sample_indices, make_benchmark,
                             make_training_benchmark, merge_benchmark_shards)


cv2.setNumThreads(2)


class SynthStreamingTests(unittest.TestCase):
    @staticmethod
    def write_surface(path):
        image = np.full((128, 128, 3), 135, dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise AssertionError("Could not prepare tiny test surface")

    def test_callback_preserves_bytes_seeds_and_manifest_while_releasing_files(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            surface = root / "concrete.jpg"
            self.write_surface(surface)
            reference = make_benchmark([surface], root / "reference", 2, 17, 128, version="v2")
            delivered = {}

            def upload_and_unlink(image_path, label_path, entry):
                self.assertTrue(image_path.is_file())
                self.assertTrue(label_path.is_file())
                self.assertEqual(image_path.name, entry["image"])
                self.assertEqual(label_path.name, entry["labels"])
                delivered[entry["global_index"]] = (image_path.read_bytes(), label_path.read_bytes())
                image_path.unlink()
                label_path.unlink()

            streamed = make_benchmark([surface], root / "stream", 2, 17, 128, version="v2",
                                       on_image_written=upload_and_unlink)
            self.assertEqual(reference, streamed)
            self.assertEqual(set(delivered), {0, 1})
            for entry in reference["images"]:
                expected = tuple((root / "reference" / entry[key]).read_bytes()
                                 for key in ("image", "labels"))
                self.assertEqual(delivered[entry["global_index"]], expected)
            self.assertFalse(list((root / "stream").glob("img_*")))
            self.assertEqual(json.loads((root / "stream/manifest.json").read_text()), streamed)
            self.assertEqual(len(streamed["generator_code_sha256"]), 64)
            self.assertEqual(streamed["numpy_version"], np.__version__)
            self.assertEqual(streamed["opencv_version"], cv2.__version__)

    def test_only_assigned_surfaces_need_to_exist(self):
        self.assertEqual(assigned_sample_indices(0, 3, 0, 2), [0, 2])
        self.assertEqual(assigned_sample_indices(1, 3, 0, 2), [1])
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            concrete, rough = root / "concrete.jpg", root / "rough_concrete.jpg"
            self.write_surface(concrete)
            shard = make_benchmark([rough, concrete], root / "shards", 1, 4, 128,
                                    shard_index=0, shard_count=2)
            self.assertEqual(shard["surfaces"], ["concrete", "rough_concrete"])
            self.assertEqual([entry["global_index"] for entry in shard["images"]], [0])
            with self.assertRaises(FileNotFoundError):
                make_benchmark([rough, concrete], root / "missing", 1, 4, 128,
                                shard_index=1, shard_count=2)
            with self.assertRaisesRegex(ValueError, "demo-only"):
                make_benchmark([concrete, root / "cracked_concrete.jpg"], root / "bad", 1, 4, 128,
                                shard_index=0, shard_count=2)
            empty = make_benchmark([root / "dirty_concrete.jpg"], root / "empty", 1, 4, 128,
                                    shard_index=1, shard_count=2)
            self.assertEqual(empty["images"], [])

    def test_training_forwards_callback_and_rejects_even_unassigned_test_backgrounds(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            concrete = root / "concrete.jpg"
            self.write_surface(concrete)
            entries = []

            def callback(image, labels, entry):
                entry["uploaded"] = True
                entries.append(entry)
                image.unlink()
                labels.unlink()

            result = make_training_benchmark([concrete, root / "dirty_concrete.jpg"], root / "train",
                                             1, 5, 128, shard_index=0, shard_count=2,
                                             on_image_written=callback)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["split"], "train")
            self.assertEqual(result["images"], entries)
            self.assertIs(result["images"][0], entries[0])
            self.assertTrue(result["images"][0]["uploaded"])
            with self.assertRaisesRegex(ValueError, "only use dev"):
                make_training_benchmark([concrete, root / "rough_concrete.jpg"], root / "invalid",
                                         1, 5, 128, shard_index=0, shard_count=2)

    def test_metadata_merge_needs_no_images_and_keeps_mirrored_relative_layout(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            concrete = root / "concrete.jpg"
            self.write_surface(concrete)

            def callback(image, labels, entry):
                image.unlink()
                labels.unlink()

            base = root / "shards"
            for index in range(2):
                make_benchmark([concrete], base, 2, 10, 128, shard_index=index,
                                shard_count=2, on_image_written=callback)
            paths = [base / f"shard_{index:04d}_of_0002/manifest.json" for index in range(2)]
            with self.assertRaisesRegex(ValueError, "Missing"):
                merge_benchmark_shards(paths, root / "verified")
            result = merge_benchmark_shards(paths, base, verify_images=False)
            for entry in result["images"]:
                index = entry["global_index"]
                self.assertEqual(entry["image"], f"shard_{index:04d}_of_0002/img_{index:04d}.jpg")
                self.assertFalse(Path(entry["image"]).is_absolute())
                self.assertNotIn("..", Path(entry["image"]).parts)
            self.assertEqual(len(result["images"]), 2)

    def test_metadata_merge_still_rejects_unsafe_paths_and_wrong_seeds(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            concrete = root / "concrete.jpg"
            self.write_surface(concrete)
            make_benchmark([concrete], root / "data", 1, 20, 128)
            path = root / "data/manifest.json"
            original = json.loads(path.read_text())
            for unsafe in ("../img_0000.jpg", "/img_0000.jpg", "C:/img_0000.jpg", "folder\\img_0000.jpg"):
                document = json.loads(json.dumps(original))
                document["images"][0]["image"] = unsafe
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    merge_benchmark_shards([path], root / "bad_path", verify_images=False)
            document = json.loads(json.dumps(original))
            document["images"][0]["seed"] += 1
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                merge_benchmark_shards([path], root / "bad_seed", verify_images=False)
            document = json.loads(json.dumps(original))
            document["images"][0]["image"] = "subfolder/img_0000.jpg"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                merge_benchmark_shards([path], root / "nested_path", verify_images=False)
            path.write_text(json.dumps(original))
            image_path = root / "data/img_0000.jpg"
            image_path.unlink()
            image_path.symlink_to(concrete.resolve())
            with self.assertRaisesRegex(ValueError, "symlink"):
                merge_benchmark_shards([path], root / "symlink", verify_images=False)


if __name__ == "__main__":
    unittest.main()
