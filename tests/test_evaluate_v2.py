"""Held-out summary isolation, visibility recall and image-bootstrap checks."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inspector.evaluate import aggregate_metrics, evaluate_result, write_summaries
from inspector.agent import InspectorConfig
from inspector.viz import trace_frames
from scripts.eval_real_patches import (SHARD_SCHEMA as REAL_SHARD_SCHEMA, evaluate_patches,
                                      main, merge_real_shards, select_patches)
from scripts.run_eval import (SHARD_SCHEMA as EVAL_SHARD_SCHEMA, merge_shards,
                             resized_truth, select_tuning_config, shard_entries, tuning_candidates)

cv2.setNumThreads(2)


class EvaluationV2Tests(unittest.TestCase):
    def metric(self, *, detector="classic", seed=0, image="a", found=1, total=2):
        return {"detector": detector, "method": "agent", "seed": seed, "image": image,
                "budget_frac": 0.1, "n_images": 1, "n_found": found, "n_defects": total,
                "false_positives": 0, "fine_cost_frac": 0.1, "coarse_cost_frac": 0.04,
                "calibration_cost_frac": 0.01, "total_cost_frac": 0.14,
                "fraction_pixels_processed": 0.14,
                "per_kind": {"crack": {"n_found": found, "n_defects": total, "false_positives": 0}},
                "per_visible_at": {"1.0": {"n_found": found, "n_defects": total}}}

    def test_detector_and_seed_are_never_mixed(self):
        records = [self.metric(), self.metric(detector="learned"), self.metric(seed=1)]
        summaries = aggregate_metrics(records)
        self.assertEqual(len(summaries), 3)
        self.assertEqual({(row["detector"], row["seed"]) for row in summaries},
                         {("classic", "0"), ("classic", "1"), ("learned", "0")})
        self.assertTrue(all(row["n_images"] == 1 for row in summaries))

    def test_visibility_counts_and_calibration_is_not_double_charged(self):
        labels = np.zeros((30, 30), np.uint16)
        labels[2:7, 2:7] = 1
        labels[20:25, 20:25] = 2
        defects = [{"id": 1, "kind": "crack", "visible_at": 1.0},
                   {"id": 2, "kind": "spall", "visible_at": 0.25}]
        result = {"reported_defects": [{"bbox": [2, 2, 5, 5], "scale": 1.0, "cell": 0}], "fine_cost_px": 90,
                  "coarse_cost_px": 180, "calibration_cost_px": 45}
        metrics = evaluate_result(result, labels, defects)
        self.assertEqual(metrics["per_visible_at"]["1.0"]["recall"], 1)
        self.assertEqual(metrics["per_visible_at"]["0.25"]["recall"], 0)
        self.assertAlmostEqual(metrics["total_cost_frac"], 0.3)
        self.assertAlmostEqual(metrics["calibration_cost_frac"], 0.05)
        result.update(elapsed_ms=10, preparation_elapsed_ms=30, preparation_included_in_elapsed=False)
        timing = evaluate_result(result, labels, defects)
        self.assertEqual(timing["wall_time_ms"], 10)
        self.assertEqual(timing["preparation_time_ms"], 30)
        self.assertEqual(timing["uncached_wall_estimate_ms"], 40)

    def test_cold_preparation_is_not_added_twice(self):
        labels = np.zeros((8, 8), np.uint16)
        cold = {"elapsed_ms": 40, "preparation_elapsed_ms": 30,
                "coarse": {"reused_preparation": False}}
        warm = {"elapsed_ms": 10, "preparation_elapsed_ms": 30,
                "coarse": {"reused_preparation": True}}
        self.assertEqual(evaluate_result(cold, labels, [])["uncached_wall_estimate_ms"], 40)
        self.assertEqual(evaluate_result(warm, labels, [])["uncached_wall_estimate_ms"], 40)
        self.assertEqual(evaluate_result({"elapsed_ms": 40, "preparation_elapsed_ms": 30}, labels, [])
                         ["uncached_wall_estimate_ms"], 40)

    def test_bootstrap_resamples_images_and_is_deterministic(self):
        records = [self.metric(image="a", found=0, total=10), self.metric(image="b", found=2, total=2)]
        one = aggregate_metrics(records)[0]
        two = aggregate_metrics(records)[0]
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["recall"], 2 / 12)
        self.assertEqual(one["recall_ci_low"], 0)
        self.assertEqual(one["recall_ci_high"], 1)
        self.assertEqual(one["bootstrap_replicates"], 2000)
        self.assertEqual(one["bootstrap_unit"], "image")
        duplicate = aggregate_metrics([records[0], records[0]])[0]
        self.assertEqual(duplicate["n_ci_images"], 1)

    def test_random_pool_tolerates_none_precision_when_a_seed_reports_nothing(self):
        records = []
        for seed in (0, 1, 2):
            row = self.metric(seed=seed, image="a", found=int(seed == 0), total=1)
            row["method"] = "random"
            row["precision"] = 1.0 if seed == 0 else None
            records.append(row)
        summaries = aggregate_metrics(records)
        self.assertTrue(any(item.get("seed") == "pooled" for item in summaries))

    def test_random_pool_averages_seeds_before_image_bootstrap(self):
        records = []
        for image in ("a", "b"):
            for seed in (0, 1, 2):
                row = self.metric(seed=seed, image=image, found=int(seed == 0), total=1)
                row["method"] = "random"
                records.append(row)
        summaries = aggregate_metrics(records)
        self.assertEqual(len(summaries), 4)
        pooled = next(row for row in summaries if row["seed"] == "pooled")
        self.assertEqual(pooled["n_images"], 2)
        self.assertEqual(pooled["n_ci_images"], 2)
        self.assertEqual(pooled["n_seeds"], 3)
        self.assertAlmostEqual(pooled["recall"], 1 / 3)
        self.assertAlmostEqual(pooled["recall_ci_low"], 1 / 3)
        self.assertAlmostEqual(pooled["recall_ci_high"], 1 / 3)

    def test_writers_expose_confidence_visibility_and_detector_legends(self):
        records = [self.metric(), self.metric(detector="learned")]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            write_summaries(records, tmp)
            out = Path(tmp)
            with (out / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("recall_ci_low", rows[0])
            self.assertIn("calibration_cost_frac", rows[0])
            self.assertTrue((out / "per_visibility.csv").exists())
            self.assertTrue((out / "recall_vs_total_cost.svg").exists())
            chart = (out / "recall_vs_budget.svg").read_text()
            self.assertIn("classic/agent", chart)
            self.assertIn("learned/agent", chart)

    def test_sampled_trace_frames_keep_original_step_numbers_and_endpoints(self):
        image = np.zeros((50, 50, 3), np.uint8)
        result = {"trace": [{"action": "inspect_tile", "cell": 0, "rect": [0, 0, 50, 50],
                             "scale": 1.0} for _ in range(20)]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            paths = trace_frames(image, result, tmp, frame_stride=3, max_frames=4)
            self.assertEqual(len(paths), 4)
            self.assertEqual(Path(paths[0]).name, "frame_0000.png")
            self.assertEqual(Path(paths[-1]).name, "frame_0020.png")
            self.assertEqual(len(result["trace"]), 20)


class RealPatchTests(unittest.TestCase):
    def test_missing_dataset_is_explicit_skip_without_model_import(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            missing = Path(tmp) / "does_not_exist"
            summary, predictions = evaluate_patches(missing, Path(tmp) / "missing_models")
            self.assertEqual(summary["status"], "skipped")
            self.assertEqual(predictions, [])
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = main(["--root", str(missing)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["reason"], "root_missing")

    def test_sampling_and_frozen_classifier(self):
        class FrozenDetector:
            def threshold(self, scale):
                self.scale = scale
                return 0.6

            def predict(self, features, scale):
                return features[:, :, 0], np.zeros(features.shape[:2], np.uint8)

        def extractor(image, stats=None):
            self.assertEqual(stats["feature_version"], "opencv_patch32_v2_1")
            self.assertIn("gray", stats)
            score = float(image.mean() / 255)
            return np.full((2, 2, 1), score, np.float32), {}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            for name, brightness in (("Positive", 240), ("Negative", 10)):
                (root / name).mkdir()
                for index in range(3):
                    cv2.imwrite(str(root / name / f"{index}.jpg"), np.full((32, 32, 3), brightness, np.uint8))
            self.assertEqual(select_patches(root, 2, 7), select_patches(root, 2, 7))
            summary, rows = evaluate_patches(root, root / "unused", per_class=2, seed=7,
                                             detector=FrozenDetector(), feature_extractor=extractor)
            self.assertEqual(summary["n_images"], 4)
            self.assertEqual(summary["accuracy"], 1)
            self.assertFalse(summary["trained_on_real_data"])
            self.assertEqual(len(rows), 4)
            json.dumps(summary, allow_nan=False)


class ShardMergeTests(unittest.TestCase):
    """JSON-only fixtures; these tests never decode or inspect an image."""

    def benchmark_outputs(self, root, count):
        evaluation = {"images": ["a.jpg", "b.jpg"], "detectors": ["classic"],
                      "methods": ["agent", "exhaustive"], "seed": 0,
                      "random_seeds": [0, 1, 2], "budgets": [0.1, 0.2],
                      "config": {}, "manifest_sha256": "fixture", "model_files": None}
        folders = []
        for index in range(count):
            folder = root / f"n{count}_{index}"
            folder.mkdir()
            images = shard_entries(evaluation["images"], index, count)
            rows = []
            for image in images:
                for method in evaluation["methods"]:
                    for budget in evaluation["budgets"]:
                        rows.append({"detector": "classic", "method": method, "seed": 0,
                                     "budget": budget, "image": image, "n_images": 1,
                                     "n_defects": 2, "n_found": 1, "false_positives": 0,
                                     "fine_cost_frac": budget, "per_kind": {}, "per_visible_at": {}})
            (folder / "results.json").write_text(json.dumps(rows))
            (folder / "run.json").write_text(json.dumps({"schema": EVAL_SHARD_SCHEMA,
                "evaluation": evaluation, "shard_count": count, "shard_index": index,
                "assigned_images": images}))
            folders.append(folder)
        return folders

    def test_two_evaluation_shards_equal_one_and_reject_missing_or_duplicate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            one = self.benchmark_outputs(root, 1)
            two = self.benchmark_outputs(root, 2)
            whole, _ = merge_shards(one)
            merged, meta = merge_shards(list(reversed(two)))
            self.assertEqual(merged, whole)
            self.assertEqual(aggregate_metrics(merged), aggregate_metrics(whole))
            self.assertFalse(meta["inference_rerun"])
            with self.assertRaises(ValueError):
                merge_shards(two[:1])
            with self.assertRaises(ValueError):
                merge_shards([two[0], two[0]])
            changed = json.loads((two[1] / "run.json").read_text())
            changed["evaluation"]["manifest_sha256"] = "different"
            (two[1] / "run.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                merge_shards(two)

    def real_outputs(self, root, count):
        selected = [{"image": "Positive/a.jpg", "label": 1}, {"image": "Negative/b.jpg", "label": 0}]
        evaluation = {"selected_images": selected, "threshold": 0.6, "model_files": {"fake": "fixed"}}
        folders = []
        for index in range(count):
            folder = root / f"real_n{count}_{index}"
            folder.mkdir()
            assigned = shard_entries(selected, index, count)
            rows = [{**item, "status": "ok", "score": 0.8 if item["label"] else 0.2,
                     "prediction": item["label"], "pixels_processed": 100, "n_feature_cells": 4}
                    for item in assigned]
            (folder / "predictions.json").write_text(json.dumps(rows))
            (folder / "run.json").write_text(json.dumps({"schema": REAL_SHARD_SCHEMA,
                "evaluation": evaluation, "shard_count": count, "shard_index": index,
                "assigned_images": [item["image"] for item in assigned], "wall_time_ms": 1}))
            folders.append(folder)
        return folders

    def test_real_patch_merge_matches_one_shard_without_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            one = self.real_outputs(root, 1)
            two = self.real_outputs(root, 2)
            whole, rows = merge_real_shards(one)
            merged, merged_rows = merge_real_shards(list(reversed(two)))
            self.assertEqual(merged_rows, rows)
            for key in ("recall", "specificity", "accuracy", "n_images", "pixels_processed"):
                self.assertEqual(merged[key], whole[key])
            with self.assertRaises(ValueError):
                merge_real_shards([two[0]])
            with self.assertRaises(ValueError):
                merge_real_shards([two[0], two[0]])

    def test_resized_visibility_uses_analysis_coordinates(self):
        labels = np.zeros((64, 64), np.uint16)
        labels[10:30, 10:30] = 1
        entry = {"image": "fixture.jpg", "labels": "fixture_labels.png", "defects": [
            {"id": 1, "kind": "crack", "visible_at": 0.25, "length_px": 20,
             "mean_width_px": 4, "params": {"nominal_width_px": 4}}]}
        with patch("scripts.run_eval.cv2.imread", return_value=labels):
            _, defects = resized_truth(Path("unused"), entry, (32, 32))
        self.assertEqual(defects[0]["source_visible_at"], 0.25)
        self.assertEqual(defects[0]["visible_at"], 0.5)
        self.assertEqual(defects[0]["length_px"], 10)
        self.assertEqual(defects[0]["params"]["nominal_width_px"], 2)
        self.assertEqual(entry["defects"][0]["params"]["nominal_width_px"], 4)

    def test_dev_tuning_shards_select_same_frozen_config_as_one_shard(self):
        candidates = tuning_candidates(InspectorConfig(), "learned")
        self.assertEqual(len(candidates), 24)
        self.assertEqual({item["config"]["threshold_multiplier"] for item in candidates}, {1.0, 1.05, 1.1})
        self.assertEqual({item["config"]["positive_threshold"] for item in candidates}, {0.6, 0.75, 0.85, 0.92})
        self.assertTrue(all(item["config"]["ambiguous_low"] <= 0.25 for item in candidates))
        evaluation = {"mode": "tuning", "split": "dev", "images": ["a.jpg", "b.jpg"],
                      "detectors": ["learned"], "methods": [item["method"] for item in candidates],
                      "candidates": candidates, "seed": 0, "random_seeds": [0, 1, 2],
                      "budgets": [0.2], "config": {}, "manifest_sha256": "dev-fixture"}
        rows = []
        for image in evaluation["images"]:
            for index, candidate in enumerate(candidates):
                rows.append({"detector": "learned", "method": candidate["method"], "seed": 0,
                             "budget": 0.2, "image": image, "n_images": 1, "n_defects": 1,
                             "n_found": int(index == 2), "recall": float(index == 2),
                             "false_positives": 1 if index == 2 else 0})
        expected, _ = select_tuning_config(rows, evaluation)
        self.assertEqual(expected["positive_threshold"], candidates[2]["config"]["positive_threshold"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folders = []
            for index in range(2):
                folder = Path(tmp) / str(index)
                folder.mkdir()
                images = shard_entries(evaluation["images"], index, 2)
                (folder / "run.json").write_text(json.dumps({"schema": EVAL_SHARD_SCHEMA,
                    "evaluation": evaluation, "shard_index": index, "shard_count": 2,
                    "assigned_images": images}))
                (folder / "results.json").write_text(json.dumps([row for row in rows if row["image"] in images]))
                folders.append(folder)
            merged, metadata = merge_shards(folders)
            selected, report = select_tuning_config(merged, evaluation)
            self.assertEqual(selected, expected)
            self.assertEqual(metadata["selected_method"], candidates[2]["method"])
            self.assertEqual(report["selection_split"], "dev")
        with self.assertRaises(ValueError):
            select_tuning_config(rows, {**evaluation, "split": "test"})
        with self.assertRaises(ValueError):
            select_tuning_config(rows[:-1], evaluation)


if __name__ == "__main__":
    unittest.main()
