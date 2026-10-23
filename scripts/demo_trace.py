"""Shardable qualitative demos and post-evaluation trace illustrations.

Real-image inspection is an external-compute entry point, not a local Mac smoke
test. --merge only combines completed metadata; it never decodes images.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from inspector.agent import ActiveInspector, InspectorConfig
from inspector.tools import LocalBackend
from inspector.viz import overlay, trace_frames


def model_signature(model_dir: Path) -> dict:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    filenames = {"metadata.json"}
    for files in metadata["models"].values():
        filenames.update(files.values())
    result = {}
    for name in sorted(filenames):
        file = (model_dir / name).resolve()
        if not file.is_relative_to(model_dir.resolve()):
            raise ValueError("Model file must remain inside model_dir")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def select_examples(records: list[dict], detector: str, budget: float) -> list[dict]:
    """Choose worst/median/best illustrations after evaluation, never for tuning."""
    rows = [row for row in records if row.get("detector", "classic") == detector
            and row.get("method") == "agent" and row.get("recall") is not None
            and abs(float(row.get("budget_frac", row.get("budget", -1))) - budget) < 1e-9]
    by_image = {row["image"]: row for row in rows}
    ordered = sorted(by_image.values(), key=lambda row: (row["recall"], row["image"]))
    if not ordered:
        raise ValueError("No matching agent evaluation records for illustration selection")
    choices = [("worst", 0), ("median", len(ordered)//2), ("best", len(ordered)-1)]
    selected, seen = [], set()
    for label, index in choices:
        row = ordered[index]
        if row["image"] not in seen:
            selected.append({"label": label, "image": row["image"], "recall": row["recall"]})
            seen.add(row["image"])
    return selected


def merge_demos(paths: list[Path], out: Path) -> dict:
    """Combine complete nonoverlapping demo shards without opening image data."""
    manifests = []
    for path in paths:
        file = path / "demo_manifest.json" if path.is_dir() else path
        manifests.append((file, json.loads(file.read_text())))
    if not manifests:
        raise ValueError("No demo shards supplied")
    reference = manifests[0][1]
    if reference["settings"].get("detector") == "learned" and not reference["settings"].get("model_signature"):
        raise ValueError("Learned demo shards must identify their frozen model files")
    count = reference["shard_count"]
    if len(manifests) != count or {m["shard_index"] for _, m in manifests} != set(range(count)):
        raise ValueError("All demo shards must be supplied exactly once")
    records = []
    for file, manifest in manifests:
        if (manifest["settings"] != reference["settings"]
                or manifest["inputs"] != reference["inputs"]
                or manifest["shard_count"] != count):
            raise ValueError("Demo shard settings or inputs differ")
        for record in manifest["records"]:
            index = record["global_index"]
            if not 0 <= index < len(reference["inputs"]):
                raise ValueError("Invalid demo image index")
            if any(record.get(key) != value for key, value in reference["inputs"][index].items()):
                raise ValueError("Demo record does not match its planned input")
            if record["global_index"] % count != manifest["shard_index"]:
                raise ValueError("Demo image is assigned to the wrong shard")
            source = file.parent / record["trace"]
            if not source.is_file():
                raise ValueError(f"Missing completed trace: {source}")
            if json.loads(source.read_text()).get("image_ref") != record["image"]:
                raise ValueError("Completed demo trace belongs to a different image")
            import os
            records.append({**record, "trace": os.path.relpath(source.resolve(), out.resolve()),
                            "directory": os.path.relpath((file.parent / record["directory"]).resolve(),
                                                         out.resolve())})
    indices = [row["global_index"] for row in records]
    if sorted(indices) != list(range(len(reference["inputs"]))):
        raise ValueError("Demo shards contain duplicate or missing images")
    result = {"version": 2, "settings": reference["settings"], "inputs": reference["inputs"],
              "records": sorted(records, key=lambda row: row["global_index"]), "merged": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main(argv: list[str] | None = None) -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", type=Path)
    sources.add_argument("--images", type=Path, nargs="+")
    sources.add_argument("--select-from-eval", type=Path, help="results.json; choose worst/median/best agent images")
    sources.add_argument("--merge", type=Path, nargs="+", help="Completed demo shard folders or manifests")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--size", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--detector", choices=("classic", "learned"), default="classic")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bench", type=Path, help="Benchmark root for --select-from-eval")
    parser.add_argument("--frame-width", type=int, default=1024)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= shard-index < shard-count")
    if args.frame_width < 1 or args.frame_stride < 1 or (args.max_frames is not None and args.max_frames < 2):
        parser.error("frame dimensions/stride must be positive; max-frames must be >=2")
    if args.merge:
        merged = merge_demos(args.merge, args.out)
        print(json.dumps({"status": "merged", "n_images": len(merged["records"])}))
        return
    config = InspectorConfig(**json.loads(args.config.read_text())) if args.config else InspectorConfig()
    # Validate the budget before opening any model or image.
    ActiveInspector(None, args.budget, config)
    if args.select_from_eval:
        if args.bench is None:
            parser.error("--select-from-eval requires --bench")
        chosen = select_examples(json.loads(args.select_from_eval.read_text()), args.detector, args.budget)
        inputs = [{**row, "image": str(args.bench / row["image"])} for row in chosen]
    else:
        inputs = [{"image": str(path), "label": path.stem} for path in
                  ([args.image] if args.image else args.images)]
    output = args.out if args.shard_count == 1 else args.out / f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
    output.mkdir(parents=True, exist_ok=True)
    settings = {"budget": args.budget, "size": args.size, "detector": args.detector,
                "config": asdict(config), "frame_width": args.frame_width,
                "frame_stride": args.frame_stride, "max_frames": args.max_frames,
                "model_signature": model_signature(args.model_dir) if args.detector == "learned" else None}
    records = []
    indices = list(range(len(inputs)))[args.shard_index::args.shard_count]
    backend = LocalBackend(max_size=args.size, detector=args.detector, model_dir=args.model_dir) if indices else None
    for index in indices:
        source = inputs[index]
        image = source["image"]
        target = output if len(inputs) == 1 else output / f"image_{index:04d}_{source['label']}"
        target.mkdir(parents=True, exist_ok=True)
        result = ActiveInspector(backend, args.budget, config).run(image)
        result["image_ref"] = image
        (target / "trace.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        if not cv2.imwrite(str(target / "overlay.png"), overlay(backend.image(image), result, args.frame_width)):
            raise OSError("Could not write overlay")
        frames = trace_frames(backend.image(image), result, target / "frames", max_width=args.frame_width,
                              frame_stride=args.frame_stride, max_frames=args.max_frames)
        record = {"global_index": index, **source, "trace": str((target / "trace.json").relative_to(output)),
                  "directory": str(target.relative_to(output)), "n_frames": len(frames),
                  "budget_used_frac": result["budget_used_frac"], "n_tool_calls": result["n_tool_calls"]}
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    manifest = {"version": 2, "shard_index": args.shard_index, "shard_count": args.shard_count,
                "inputs": inputs, "settings": settings, "records": records}
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
