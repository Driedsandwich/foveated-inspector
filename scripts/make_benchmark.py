"""Generate a clean-background benchmark without network access."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from inspector.synth import (DEV_SURFACES, TEST_SURFACES, make_benchmark,
                             make_training_benchmark, merge_benchmark_shards)


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path,
                        default=Path(__file__).resolve().parents[2]/"data/polyhaven_4k")
    parser.add_argument("--per-surface", type=int, default=4,
                        help="Evaluation examples per selected clean surface (default: 4)")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based deterministic image shard index")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Total shards; >1 writes OUT/shard_IIII_of_NNNN")
    parser.add_argument("--merge", nargs="+", type=Path, metavar="MANIFEST_OR_DIR",
                        help="Merge complete shards to OUT/manifest.json without decoding images")
    parser.add_argument("--train-images", type=int, default=0,
                        help="Separate training examples PER selected DEV SURFACE; 0 disables")
    parser.add_argument("--train-out", type=Path,
                        help="Training destination (default: OUT/train)")
    parser.add_argument("--train-only", action="store_true",
                        help="Generate only the separate training set; requires --train-images")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--size", type=int, choices=(1024, 2048, 4096), default=4096)
    parser.add_argument("--surface-names", nargs="+", help="Optional explicit subset of clean stems")
    args = parser.parse_args()
    if args.merge:
        manifest = merge_benchmark_shards(args.merge, args.out)
        print(f"Merged manifest written to {args.out / 'manifest.json'} "
              f"({len(manifest['images'])} images; source images remain in shards)")
        return
    if args.per_surface < 1:
        parser.error("--per-surface must be positive")
    if args.train_images < 0:
        parser.error("--train-images must be nonnegative")
    if args.train_only and not args.train_images:
        parser.error("--train-only requires --train-images > 0")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require --shard-count >= 1 and 0 <= --shard-index < --shard-count")
    paths = sorted(args.surfaces.glob("*.jpg"))
    if args.surface_names:
        names = set(args.surface_names)
        paths = [p for p in paths if p.stem in names]
        missing = names - {p.stem for p in paths}
        if missing:
            parser.error(f"Missing surfaces: {sorted(missing)}")
    else:
        paths = [p for p in paths if p.stem in DEV_SURFACES | TEST_SURFACES]
    if not paths:
        parser.error("No JPEG surfaces found")
    train_paths = [path for path in paths if path.stem in DEV_SURFACES]
    if args.train_images and not train_paths:
        parser.error("Training requires at least one selected dev background")
    train_out = args.train_out or args.out / "train"
    if args.train_images and train_out.resolve() == args.out.resolve():
        parser.error("--train-out must differ from --out to isolate training and evaluation")
    if not args.train_only:
        manifest = make_benchmark(paths, args.out, args.per_surface, args.seed,
                                  size=args.size, version=args.version,
                                  shard_index=args.shard_index, shard_count=args.shard_count)
        eval_dir = (args.out if args.shard_count == 1 else
                    args.out / f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}")
        print(f"Benchmark written to {eval_dir / 'manifest.json'} "
              f"({len(manifest['images'])} images, {args.version})")
    if args.train_images:
        training = make_training_benchmark(train_paths, train_out, args.train_images,
                                            args.seed, args.size, version=args.version,
                                            shard_index=args.shard_index, shard_count=args.shard_count)
        train_dir = (train_out if args.shard_count == 1 else
                     train_out / f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}")
        print(f"Training written to {train_dir / 'manifest.json'} "
              f"({len(training['images'])} images; {args.train_images} per dev surface)")


if __name__ == "__main__":
    main()
