"""Forest vote semantics, persistent inference, and training metric arithmetic."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.features import FEATURE_NAMES, FEATURE_VERSION
from inspector.model import DetectorModels, binary_probabilities, effective_threshold
from scripts.train_detector import (
    SAMPLE_VERSION, _canonical_json, _file_sha256, _forest, _sample_filename, extract_samples,
    _sample_identity, _seed, _write_sample_pack, fit_samples, load_merged_samples, load_samples,
    merge_array_records, merge_samples, patch_targets, planned_entries,
    select_samples, shard_entries, validation_metrics,
)

cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_effective_threshold_preserves_calibrated_floors(self):
        self.assertEqual(effective_threshold(.84, policy_floor=.55), .84)
        self.assertAlmostEqual(effective_threshold(.72, 1.1, .75), .825)
        self.assertEqual(effective_threshold(.99, 2), .99)
        self.assertEqual(effective_threshold(.8, 2), .97)
        for invalid in (0.99, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                effective_threshold(.8, invalid)

    def test_votes_and_save_load_match(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, (80, len(FEATURE_NAMES))).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int32)
        kinds = (x[:, 1] > 0).astype(np.int32) + 1
        binary = _forest(x, y, 1, 16, 5)
        kind = _forest(x, kinds, 2, 16, 5)
        probability = binary_probabilities(binary, x)
        votes = binary.getVotes(x, 0)
        class_one = int(np.flatnonzero(votes[0] == 1)[0])
        np.testing.assert_allclose(probability, votes[1:, class_one] / votes[1:].sum(axis=1))
        self.assertGreater(float(probability[y == 1].mean()), float(probability[y == 0].mean()))
        with tempfile.TemporaryDirectory(prefix="_test_models_", dir=ROOT) as temporary:
            path = Path(temporary)
            metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                        "kind_names": {"1": "crack", "2": "spall"}, "thresholds": {}, "models": {}}
            for scale in ("0.25", "0.5", "1"):
                filenames = {"binary": f"{scale}_binary.yml.gz", "kind": f"{scale}_kind.yml.gz"}
                binary.save(str(path / filenames["binary"]))
                kind.save(str(path / filenames["kind"]))
                metadata["thresholds"][scale] = 0.6
                metadata["models"][scale] = filenames
            (path / "metadata.json").write_text(json.dumps(metadata))
            loaded = DetectorModels(path)
            actual, predictions = loaded.predict(x.reshape(8, 10, -1), 0.5)
            np.testing.assert_array_equal(actual.ravel(), probability)
            self.assertEqual(predictions.shape, (8, 10))
            self.assertEqual(loaded.threshold(0.5), 0.6)
            thresholds = loaded.detector_thresholds
            self.assertEqual(thresholds, {"0.25": .6, "0.5": .6, "1": .6})
            thresholds["1"] = .1
            self.assertEqual(loaded.threshold(1.0), .6)
            self.assertEqual(loaded.kind_names, {1: "crack", 2: "spall"})
            with self.assertRaises(ValueError):
                loaded.predict(x.reshape(8, 10, -1), 0.3)

    def test_patch_targets_exact_small_and_partial_cells(self):
        labels = np.zeros((35, 65), np.uint16)
        labels[3, 4:7] = 1
        labels[33, 63:65] = 2
        binary, kind = patch_targets(labels, [{"id": 1, "kind": "crack"},
                                              {"id": 2, "kind": "spall"}],
                                      {"crack": 1, "spall": 2}, 3)
        self.assertEqual(binary.reshape(2, 3).tolist(), [[1, 0, 0], [0, 0, 0]])
        self.assertEqual(kind[0], 1)

    def test_auc_ties_and_precision_points(self):
        metrics = validation_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["precision_at_recall"]["0.8"]["precision"], 1.0)
        tied = validation_metrics(np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(tied["precision_at_recall"]["0.5"]["precision"], 0.5)

    def test_validation_weights_restore_background_prevalence(self):
        metrics = validation_metrics(np.array([1, 0]), np.array([0.5, 0.5]), np.array([1.0, 9.0]))
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["precision_at_recall"]["0.8"]["precision"], 0.1)
        self.assertEqual(metrics["weighted_negative"], 9.0)

    def test_tiny_extraction_never_opens_test_surface_and_does_not_fit(self):
        with tempfile.TemporaryDirectory(prefix="_test_extraction_", dir=ROOT) as temporary:
            root = Path(temporary)
            labels = np.zeros((96, 96), np.uint16)
            labels[10:80, 20:23] = 1
            labels[50:75, 60:85] = 2
            image = np.full((96, 96, 3), 128, np.uint8)
            image[labels == 1] = 35
            image[labels == 2] = 70
            cv2.imwrite(str(root / "source.png"), image)
            cv2.imwrite(str(root / "labels.png"), labels)
            defects = [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]
            entries = [{"image": name, "labels": "labels.png", "surface": surface,
                        "split": "train", "seed": seed, "defects": defects}
                       for name, surface, seed in (("source.png", "concrete", 1),
                                                   ("validation.png", "concrete_pavement", 2))]
            cv2.imwrite(str(root / "validation.png"), image)
            entries.append({"image": "must_not_be_opened.png", "surface": "rough_concrete",
                            "split": "test"})
            (root / "manifest.json").write_text(json.dumps({"images": entries}))
            original_read = cv2.imread

            def guarded_read(filename, *args):
                self.assertNotIn("must_not_be_opened", str(filename))
                return original_read(filename, *args)

            with patch("scripts.train_detector.cv2.imread", side_effect=guarded_read) as reader:
                with patch("scripts.train_detector._forest", side_effect=AssertionError("Extraction fitted a model")):
                    result = extract_samples(root, root / "samples", size=96, max_image_patches=8,
                                             holdout_surface="concrete_pavement")
            self.assertFalse(result["models_fitted"])
            self.assertEqual(result["n_images"], 2)
            self.assertEqual(reader.call_count, 4)
            samples, config = load_samples([Path(result["manifest"])])
            self.assertEqual(len(samples), 6)
            self.assertTrue(all(entry["split"] == "train" for entry in config["source_images"]))

    def test_two_shards_merge_identically_to_one_without_image_processing(self):
        defects = [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]
        manifest = {"images": [
            {"image": "fit_b.jpg", "surface": "concrete", "split": "train", "seed": 2, "defects": defects},
            {"image": "holdout_b.jpg", "surface": "concrete_pavement", "split": "train", "seed": 3, "defects": defects},
            {"image": "fit_a.jpg", "surface": "asphalt_floor", "split": "train", "seed": 1, "defects": defects},
            {"image": "holdout_a.jpg", "surface": "concrete_pavement", "split": "train", "seed": 4, "defects": defects},
            {"image": "must_never_be_opened.jpg", "surface": "rough_concrete", "split": "test"},
        ]}
        entries, kind_ids = planned_entries(manifest, "concrete_pavement")
        self.assertEqual(kind_ids, {"crack": 1, "spall": 2})
        self.assertEqual(len(entries), 4)

        def records_for(selected):
            records = []
            for entry in selected:
                for scale in ("0.25", "0.5", "1"):
                    # A 64-row feature fixture represents at most 256x256 pixels.
                    rng = np.random.default_rng(_seed(17, entry["image"], scale))
                    features = rng.normal(size=(64, len(FEATURE_NAMES))).astype(np.float32)
                    truth = np.zeros(64, np.int32)
                    truth[:8] = 1
                    kinds = np.zeros(64, np.int32)
                    kinds[:4], kinds[4:8] = 1, 2
                    arrays = select_samples(features, truth, kinds, entry["role"], 16,
                                            _seed(17, entry["image"], scale))
                    records.append({"image": entry["image"], "scale": scale,
                                    "role": entry["role"], "arrays": arrays})
            return records

        one = merge_array_records(records_for(shard_entries(entries, 0, 1)))
        parts = records_for(shard_entries(entries, 1, 2)) + records_for(shard_entries(entries, 0, 2))
        two = merge_array_records(parts)
        for key in one:
            for name in one[key]:
                np.testing.assert_array_equal(one[key][name], two[key][name])
        with self.assertRaises(ValueError):
            shard_entries(entries, 2, 2)

    def test_merge_and_npz_roundtrip_need_no_images_and_check_provenance(self):
        config = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                  "holdout_surface": "concrete_pavement", "kind_ids": {"crack": 1, "spall": 2},
                  "max_image_patches": 4,
                  "source_images": [
                      {"image": "fit.jpg", "surface": "concrete", "split": "train", "role": "fit", "seed": 1},
                      {"image": "held.jpg", "surface": "concrete_pavement", "split": "train", "role": "validation", "seed": 2},
                  ]}
        # Only tiny feature arrays are written; no source image exists or is read.
        with tempfile.TemporaryDirectory(prefix="_test_samples_", dir=ROOT) as temporary:
            path = Path(temporary)
            manifests = []
            for index in (0, 1):
                entry = config["source_images"][index]
                records = []
                for scale in ("0.25", "0.5", "1"):
                    filename = _sample_filename(entry["image"], scale)
                    arrays = {"x": np.zeros((4, len(FEATURE_NAMES)), np.float32),
                              "y": np.array([0, 0, 1, 1], np.int32),
                              "kind": np.array([0, 0, 1, 2], np.int32),
                              "weight": np.ones(4), "patch_index": np.arange(4, dtype=np.int32)}
                    digest = _write_sample_pack(path / filename, arrays, _sample_identity(entry, scale, config))
                    records.append({**entry, "scale": scale, "file": filename, "n_samples": 4,
                                    "sha256": digest, "n_source_patches": 4, "n_source_positive": 2})
                manifest_path = path / f"shard_{index:04d}_of_0002.json"
                # Canonical image ordering is fit.jpg then held.jpg.
                manifest_path.write_text(json.dumps({"sample_version": SAMPLE_VERSION, "config": config,
                                                     "shard_index": index, "shard_count": 2, "records": records}))
                manifests.append(manifest_path)
            expected, _ = load_samples(manifests)
            target = path / "merged.npz"
            summary = merge_samples(manifests, target)
            actual, loaded_config = load_merged_samples(target)
            self.assertFalse(summary["models_fitted"])
            self.assertEqual(summary["source_images_opened"], 0)
            self.assertEqual(config, loaded_config)
            for key in expected:
                for name in expected[key]:
                    np.testing.assert_array_equal(expected[key][name], actual[key][name])
            # The entire fitting stage is exercised with four feature rows per
            # role/scale and four trees, never with a benchmark image.
            model_output = path / "tiny_models"
            with patch("scripts.train_detector.cv2.imread", side_effect=AssertionError("Fit opened an image")):
                fit_result = fit_samples(target, model_output, trees=4, depth=3, max_fit_patches=8)
            self.assertEqual(fit_result["source_images_opened"], 0)
            loaded_model = DetectorModels(model_output)
            probability, kinds = loaded_model.predict(np.zeros((1, 1, len(FEATURE_NAMES)), np.float32), 1.0)
            self.assertEqual(probability.shape, (1, 1))
            self.assertEqual(kinds.shape, (1, 1))
            with self.assertRaises(FileExistsError):
                fit_samples(target, model_output, trees=4, depth=3, max_fit_patches=8)
            with self.assertRaises(ValueError):
                load_samples(manifests[:1])
            with self.assertRaises(FileExistsError):
                merge_samples(manifests, target)
            # A held-out NPZ cannot become fit data even if the caller updates
            # the outer digest after swapping same-shaped files on disk.
            original_manifest = manifests[0].read_text()
            fit_path = path / _sample_filename("fit.jpg", "0.25")
            held_path = path / _sample_filename("held.jpg", "0.25")
            original_bytes = fit_path.read_bytes()
            fit_path.write_bytes(held_path.read_bytes())
            swapped = json.loads(original_manifest)
            swapped["records"][0]["sha256"] = _file_sha256(fit_path)
            manifests[0].write_text(json.dumps(swapped))
            with self.assertRaises(ValueError):
                load_samples(manifests)
            fit_path.write_bytes(original_bytes)
            manifests[0].write_text(original_manifest)

            # Canonical array names prohibit a fit/validation alias, including
            # when a changed sidecar is embedded again with matching digests.
            sidecar_path = target.with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text())
            sidecar.pop("archive_sha256")
            sidecar.pop("metadata_sha256")
            sidecar["arrays"]["0.25:fit"]["x"] = "0.25_validation__x"
            embedded = _canonical_json(sidecar)
            with np.load(target, allow_pickle=False) as archive:
                contents = {name: archive[name].copy() for name in archive.files}
            contents["identity_json"] = np.array(embedded)
            np.savez(target, **contents)
            sidecar["metadata_sha256"] = hashlib.sha256(embedded.encode()).hexdigest()
            sidecar["archive_sha256"] = _file_sha256(target)
            sidecar_path.write_text(json.dumps(sidecar))
            with self.assertRaises(ValueError):
                load_merged_samples(target)
            tampered = json.loads(manifests[0].read_text())
            tampered["records"][0]["surface"] = "concrete_pavement"
            manifests[0].write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                load_samples(manifests)


if __name__ == "__main__":
    unittest.main()
