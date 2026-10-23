"""Image holdout and one-round hard-negative checks using tiny fixtures only."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from inspector.features import FEATURE_NAMES, FEATURE_VERSION, PATCH_SIZE
from scripts import train_detector as training


cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[1]


def manifest_fixture():
    defects = [{"id": 1, "kind": "crack"}, {"id": 2, "kind": "spall"}]
    images = [{"image": f"image_{index}.png", "labels": f"labels_{index}.png",
               "surface": "concrete", "seed": index, "split": "train", "defects": defects}
              for index in range(4)]
    images.append({"image": "never_open_test.png", "labels": "never_open_test_labels.png",
                   "surface": "rough_concrete", "split": "test"})
    return {"images": images}


def write_base_samples(root):
    entries, kinds = training.planned_entries(manifest_fixture())
    config = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
              "source_images": [training._source_record(entry) for entry in entries],
              "holdout_surface": None, "holdout_strategy": "image_quarter", "kind_ids": kinds,
              "max_image_patches": 4, "calibration": "sampled"}
    root.mkdir()
    records = []
    for entry in entries:
        for scale in ("0.25", "0.5", "1"):
            features = np.zeros((4, len(FEATURE_NAMES)), np.float32)
            features[:, 0] = [.1, .2, .8, .9]
            arrays = {"x": features, "y": np.array([0, 0, 1, 1], np.int32),
                      "kind": np.array([0, 0, 1, 2], np.int32), "weight": np.ones(4),
                      "patch_index": np.arange(4, dtype=np.int32)}
            name = training._sample_filename(entry["image"], scale)
            digest = training._write_sample_pack(root / name, arrays, training._sample_identity(entry, scale, config))
            records.append({**training._source_record(entry), "scale": scale, "file": name, "sha256": digest,
                            "n_samples": 4, "n_source_patches": 4, "n_source_positive": 2})
    manifest = root / "shard_0000_of_0001.json"
    manifest.write_text(json.dumps({"sample_version": training.SAMPLE_VERSION, "config": config,
                                    "shard_index": 0, "shard_count": 1, "records": records}))
    merged = root / "merged.npz"
    training.merge_samples([manifest], merged)
    samples, _ = training.load_merged_samples(merged)
    return merged, samples, config


def write_first_model(root, config, samples):
    root.mkdir()
    metadata = {"sample_config": config, "training_samples_sha256": training.training_samples_digest(samples),
                "hard_negative_rounds": 0, "models": {}, "thresholds": {},
                "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "kind_names": {"1": "crack", "2": "spall"}, "calibration": "sampled"}
    for scale in ("0.25", "0.5", "1"):
        names = {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
        metadata["models"][scale] = names
        metadata["thresholds"][scale] = .99
        for name in names.values():
            (root / name).write_bytes(b"fixture forest " + name.encode())
    (root / "metadata.json").write_text(json.dumps(metadata))


def write_hard_negatives(root, config, first, shard_count=2):
    root.mkdir()
    mining = training._mining_binding(first, config, .04)
    paths = []
    for shard in range(shard_count):
        records = []
        for entry in training.shard_entries(config["source_images"], shard, shard_count):
            if entry["role"] != "fit":
                continue
            for scale in ("0.25", "0.5", "1"):
                arrays = {"x": np.full((4, len(FEATURE_NAMES)), 99, np.float32),
                          "y": np.zeros(4, np.int32), "kind": np.zeros(4, np.int32),
                          "weight": np.ones(4), "patch_index": np.arange(4, dtype=np.int32),
                          "mining_score": np.array([.99, .98, .97, .96], np.float32)}
                name = training._sample_filename(entry["image"], scale)
                digest = training._write_sample_pack(root / name, arrays,
                    training._sample_identity(entry, scale, config, mining))
                records.append({**entry, "scale": scale, "file": name, "sha256": digest,
                                "n_samples": 4, "n_source_patches": 100, "n_source_positive": 0,
                                "n_empty_background": 100, "empty_background_only": True})
        path = root / f"shard_{shard:04d}_of_{shard_count:04d}.json"
        path.write_text(json.dumps({"sample_version": training.SAMPLE_VERSION, "config": config,
                                   "shard_index": shard, "shard_count": shard_count,
                                   "mining": mining, "records": records}))
        paths.append(path)
    return paths


class FakeForest:
    def save(self, filename):
        Path(filename).write_bytes(b"fixture refitted forest")


class TrainingV6Tests(unittest.TestCase):
    def test_source_manifest_binding_ignores_transport_but_retains_ground_truth_and_seed(self):
        first = manifest_fixture()
        first.update(s3_prefix="x132/datasets/base/", seed=20, source_catalog={"generator_code_sha256": "abc"})
        moved = json.loads(json.dumps(first))
        moved["s3_prefix"] = "x132/jobs/mining/"
        self.assertEqual(training.source_manifest_digest(first), training.source_manifest_digest(moved))
        moved["images"][0]["seed"] += 1
        self.assertNotEqual(training.source_manifest_digest(first), training.source_manifest_digest(moved))

    def test_default_holdout_keeps_fit_images_of_every_surface(self):
        manifest = manifest_fixture()
        second = [dict(entry, image="other_" + entry["image"], surface="asphalt_floor")
                  for entry in manifest["images"] if entry["split"] == "train"]
        manifest["images"].extend(second)
        entries, _ = training.planned_entries(manifest)
        for surface in ("concrete", "asphalt_floor"):
            selected = [entry for entry in entries if entry["surface"] == surface]
            self.assertEqual(sum(entry["role"] == "validation" for entry in selected), 1)
            self.assertEqual(sum(entry["role"] == "fit" for entry in selected), 3)
        self.assertFalse(any(entry["split"] == "test" for entry in entries))
        legacy, _ = training.planned_entries(manifest, "asphalt_floor")
        self.assertTrue(all(entry["role"] == "validation" for entry in legacy if entry["surface"] == "asphalt_floor"))
        with self.assertRaisesRegex(ValueError, "at least two"):
            training.planned_entries({"images": manifest_fixture()["images"][:1]})

    def test_thin_full_resolution_gt_is_excluded_even_when_nearest_resize_loses_it(self):
        labels = np.zeros((256, 256), np.uint16)
        labels[1, 1] = 1
        downscaled = cv2.resize(labels, (64, 64), interpolation=cv2.INTER_NEAREST)
        self.assertFalse(np.any(downscaled))
        empty = training.empty_background_patches(labels, (64, 64), (2, 2))
        self.assertEqual(empty.tolist(), [False, True, True, True])
        features = np.zeros((4, len(FEATURE_NAMES)), np.float32)
        chosen = training.select_hard_negatives(features, empty, np.array([1., .5, .8, .7]), .04, 4)
        self.assertEqual(chosen["patch_index"].tolist(), [2])
        self.assertFalse(np.any(chosen["y"]))

    def test_hard_negative_extraction_never_reads_validation_or_test_images(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            manifest = manifest_fixture()
            manifest["s3_prefix"] = "x132/datasets/initial/"
            image = np.full((256, 256, 3), 128, np.uint8)
            labels = np.zeros((256, 256), np.uint16)
            labels[1, 1] = 1
            labels[160:200, 160:200] = 2
            for entry in manifest["images"]:
                if entry["split"] == "train":
                    cv2.imwrite(str(root / entry["image"]), image)
                    cv2.imwrite(str(root / entry["labels"]), labels)
            (root / "manifest.json").write_text(json.dumps(manifest))

            def features(working, statistics):
                rows, cols = [(value + PATCH_SIZE - 1) // PATCH_SIZE for value in working.shape[:2]]
                result = np.zeros((rows, cols, len(FEATURE_NAMES)), np.float32)
                result[..., 0] = np.linspace(1, 0, rows * cols).reshape(rows, cols)
                return result, {}

            class PredictOnly:
                def __init__(self, directory):
                    pass

                def predict(self, values, scale):
                    return values[..., 0], np.zeros(values.shape[:2], np.int32)

            calibration = ({scale: {} for scale in ("0.25", "0.5", "1")}, 0)
            with (mock.patch.object(training, "feature_map", side_effect=features),
                  mock.patch.object(training, "prepare_calibration", return_value=calibration)):
                base = training.extract_samples(root, root / "base", size=128, max_image_patches=4)
                samples, config = training.load_samples([Path(base["manifest"])])
                write_first_model(root / "first", config, samples)
                manifest["s3_prefix"] = "x132/jobs/mining/"
                (root / "manifest.json").write_text(json.dumps(manifest))
                original_read = cv2.imread

                def guarded_read(filename, *args):
                    self.assertNotIn("never_open", str(filename))
                    self.assertNotIn("image_3.png", str(filename))
                    self.assertNotIn("labels_3.png", str(filename))
                    return original_read(filename, *args)

                with (mock.patch.object(training.cv2, "imread", side_effect=guarded_read) as reader,
                      mock.patch.object(training, "DetectorModels", PredictOnly),
                      mock.patch.object(training, "_forest", side_effect=AssertionError("Mining fitted a model"))):
                    mined = training.extract_samples(root, root / "mined", size=128, max_image_patches=4,
                                                      hard_negatives=True, model_dir=root / "first")
                self.assertEqual(reader.call_count, 6)
                self.assertEqual(mined["n_images"], 3)
                document = json.loads(Path(mined["manifest"]).read_text())
                self.assertEqual(len(document["config"]["source_images"]), 4)
                self.assertTrue(all(record["role"] == "fit" for record in document["records"]))
                for record in document["records"]:
                    with np.load(root / "mined" / record["file"], allow_pickle=False) as archive:
                        self.assertNotIn(0, archive["patch_index"].tolist())

    def test_refit_reserves_mined_negatives_recalculates_thresholds_and_refuses_second_round(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source, samples, config = write_base_samples(root / "base")
            write_first_model(root / "first", config, samples)
            manifests = write_hard_negatives(root / "mined", config, root / "first")
            with (mock.patch.object(training.cv2, "imread", side_effect=AssertionError("Fit opened an image")),
                  mock.patch.object(training, "_forest", return_value=FakeForest()) as fit,
                  mock.patch.object(training, "binary_probabilities", side_effect=lambda forest, x: x[:, 0])):
                result = training.fit_samples(source, root / "refit", trees=4, depth=3, max_fit_patches=8,
                    hard_negative_manifests=manifests, first_model_dir=root / "first")
            self.assertEqual(result["source_images_opened"], 0)
            binary_rows = fit.call_args_list[0].args[0]
            self.assertGreaterEqual(int(np.sum(binary_rows[:, 0] == 99)), 2)
            metadata = json.loads((root / "refit/metadata.json").read_text())
            self.assertEqual(metadata["hard_negative_rounds"], 1)
            self.assertTrue(all(value < .99 for value in metadata["thresholds"].values()))
            with self.assertRaisesRegex(ValueError, "only one round"):
                training.fit_samples(source, root / "again", trees=4, depth=3, max_fit_patches=8,
                    hard_negative_manifests=manifests, first_model_dir=root / "refit")

    def test_hard_negative_provenance_and_semantic_base_binding_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source, samples, config = write_base_samples(root / "base")
            write_first_model(root / "first", config, samples)
            manifests = write_hard_negatives(root / "mined", config, root / "first")
            _, mining, _ = training.load_hard_negative_samples(manifests, config, root / "first")
            reordered = {key: dict(reversed(list(data.items()))) for key, data in reversed(list(samples.items()))}
            self.assertEqual(training.training_samples_digest(samples), training.training_samples_digest(reordered))
            self.assertEqual(mining["training_samples_sha256"], training.training_samples_digest(samples))
            with self.assertRaisesRegex(ValueError, "every shard"):
                training.load_hard_negative_samples(manifests[:1], config, root / "first")
            altered = {key: {name: values.copy() for name, values in data.items()} for key, data in samples.items()}
            altered["1:fit"]["x"][0, 0] += 1
            with mock.patch.object(training, "load_merged_samples", return_value=(altered, config)):
                with self.assertRaisesRegex(ValueError, "exact base sample"):
                    training.fit_samples(source, root / "wrong_base", hard_negative_manifests=manifests,
                                         first_model_dir=root / "first")
            original = manifests[0].read_text()
            document = json.loads(original)
            document["records"][0]["image"] = "image_3.png"
            manifests[0].write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "held-out"):
                training.load_hard_negative_samples(manifests, config, root / "first")
            manifests[0].write_text(original)
            model = root / "first/scale_1_binary.yml.gz"
            model.write_bytes(model.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "binding"):
                training.load_hard_negative_samples(manifests, config, root / "first")


if __name__ == "__main__":
    unittest.main()
