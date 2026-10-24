# Reproducing the results

Everything heavy runs on AWS Lambda; the laptop only plans, uploads small files and merges JSON. These are the exact steps used for the reported numbers (2026-10-24, region `us-east-1`).

## 0. Requirements

- Python 3.12 and `uv`; `pip install opencv-contrib-python-headless==5.0.0.93 boto3`
- AWS credentials in the standard SDK chain for an IAM user that can create Lambda, S3, DynamoDB, IAM roles, API Gateway and CloudWatch resources.
- New AWS accounts may cap Lambda memory at 3008 MB and account concurrency at 10; the scripts are sized for that.
- Surface photos from Poly Haven (CC0), 4K diffuse JPEGs, in `data/polyhaven_4k/<asset>.jpg`. Development: asphalt_floor, brushed_concrete, concrete, concrete_floor_01, concrete_pavement, dirty_concrete, granular_concrete. Test: brushed_concrete_2, concrete_floor_02, concrete_slab_wall_02, gravel_concrete, grooved_concrete_driveway, ribbed_concrete_wall, rough_concrete. Demo only (real cracks, no ground truth): cracked_concrete, concrete_wall_005, worn_concrete_floor.

## 1. Offline checks

```sh
python -m unittest discover -s tests          # synthetic inputs only
bash aws/build_layer.sh --arch all --glibc 2.28   # Lambda layers from Linux wheels (no Docker)
```

## 2. Private benchmark worker and data

```sh
python aws/deploy.py --bench-only --execute --region us-east-1
python aws/upload_surfaces.py --surfaces data/polyhaven_4k --execute
python aws/run_bench.py --job generate --job-id v4-current --per-surface 4 --train-per-surface 4 \
  --seed 20261024 --size 4096 --version v2 --shards 28 --out artifacts/benchmark_v4 --merge --execute
```

## 3. Detector training with hard negatives (development surfaces only)

```sh
python aws/run_bench.py --job train_samples --benchmark-s3 x132/benchmarks/v4-current --job-id v6-train \
  --shards 7 --size 4096 --seed 20261024 --max-image-patches 1024 --calibration sampled --out artifacts/v6-train --execute
python aws/run_bench.py --job fit_samples --job-id v6-first-fit --shards 1 --sample-manifests <7 shard manifests of v6-train> \
  --seed 20261024 --out artifacts/v6-first-fit --execute
python aws/run_bench.py --job hard_negatives --benchmark-s3 x132/benchmarks/v4-current --job-id v6-hard --shards 7 \
  --size 4096 --seed 20261024 --max-image-patches 1024 --calibration sampled \
  --model-dir artifacts/v6-first-fit/shard_0000/models --hard-negative-fraction 0.04 --out artifacts/v6-hard --execute
python aws/run_bench.py --job fit_samples --job-id v6-refit --shards 1 --sample-manifests <v6-train manifests> \
  --hard-negatives --hard-negative-manifests <7 shard manifests of v6-hard> \
  --model-dir artifacts/v6-first-fit/shard_0000/models --seed 20261024 --out artifacts/v6-refit --execute
```

## 4. Policy selection on development data, evaluation on test surfaces

```sh
python aws/run_bench.py --job tune --benchmark-s3 x132/benchmarks/v4-current --split dev --detectors learned \
  --model-dir artifacts/v6-refit/shard_0000/models --job-id v8-tune --shards 14 --size 4096 --budgets 0.4 \
  --out artifacts/v8-tune --merge --execute
python aws/run_bench.py --job eval --benchmark-s3 x132/benchmarks/v4-current --split test \
  --model-dir artifacts/v6-refit/shard_0000/models --config artifacts/v8-tune/merged/config.json \
  --job-id v8-test-all --shards 14 --size 4096 --budgets 0.05 0.10 0.20 0.40 --detectors learned classic \
  --methods agent exhaustive random raster coarse_only --save-traces --out artifacts/v8-test-all --merge --execute
```

Summaries: `artifacts/v8-test-all/merged/summary.csv`, per-surface tables in `surface_summary.md`, traces under each shard's `traces/`.

## 5. Public demo and teardown

```sh
X132_POLICY_CONFIG_FILE=artifacts/v8-tune/merged/config.json python aws/deploy.py --execute --region us-east-1 \
  --model-dir artifacts/v6-refit/shard_0000/models --samples-manifest aws/samples.example.json --api-rate 1 --api-burst 2
python aws/run_bench.py --job arm-benchmark --image-ref s3://<bucket>/x132/samples/cracked-concrete.jpg \
  --detectors classic learned --tile 0 0 256 256 --tile 1536 1536 256 256 --tile 3072 3072 256 256 \
  --job-id v5-arm --out artifacts/arm-vs-x86 --execute
python aws/teardown.py --execute   # deletes resources recorded in aws/state.json after per-type ownership checks; versioned buckets are left for manual review
```

## 6. Independent real-patch check

```sh
python scripts/eval_real_patches.py --root <Positive/ and Negative/ folders from Özgenel 2019> \
  --model-dir artifacts/v6-refit/shard_0000/models --per-class 1000 --seed 20261024 --out artifacts/real_patches
```
