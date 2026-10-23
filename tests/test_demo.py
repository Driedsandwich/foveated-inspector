"""Post-evaluation illustration selection and metadata-only demo merge."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.demo_trace import merge_demos, select_examples


class DemoTests(unittest.TestCase):
    def test_selection_uses_only_the_requested_detector_method_budget(self):
        records = [{"image": f"img_{index}.jpg", "method": "agent", "detector": "learned",
                    "budget": .2, "recall": recall} for index, recall in enumerate((.7, .1, .4))]
        records.append({"image": "other.jpg", "method": "random", "detector": "learned",
                        "budget": .2, "recall": 1.0})
        selected = select_examples(records, "learned", .2)
        self.assertEqual([row["image"] for row in selected], ["img_1.jpg", "img_2.jpg", "img_0.jpg"])
        with self.assertRaises(ValueError):
            select_examples(records, "classic", .2)

    def test_two_demo_shards_merge_without_images(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            inputs = [{"image": "a.jpg"}, {"image": "b.jpg"}]
            shards = []
            for index in range(2):
                shard = root / f"shard{index}"
                shard.mkdir()
                (shard / "trace.json").write_text(json.dumps({"image_ref": inputs[index]["image"]}))
                manifest = {"shard_index": index, "shard_count": 2, "inputs": inputs,
                            "settings": {"detector": "learned", "model_signature": {"fixture": "same-model"}},
                            "records": [{"global_index": index, "image": inputs[index]["image"],
                                         "trace": "trace.json", "directory": "."}]}
                (shard / "demo_manifest.json").write_text(json.dumps(manifest))
                shards.append(shard)
            merged = merge_demos(shards, root / "merged")
            self.assertEqual([row["global_index"] for row in merged["records"]], [0, 1])
            with self.assertRaises(ValueError):
                merge_demos(shards[:1], root / "incomplete")
            changed = json.loads((shards[1]/"demo_manifest.json").read_text())
            changed["settings"]["model_signature"] = {"fixture": "different-model"}
            (shards[1]/"demo_manifest.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                merge_demos(shards, root / "mixed-models")


if __name__ == "__main__":
    unittest.main()
