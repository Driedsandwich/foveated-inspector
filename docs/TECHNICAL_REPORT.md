# Foveated Inspector — technical report

Budget-aware active visual inspection with OpenCV 5 on AWS. Like the fovea of the eye, it looks at full resolution only where it needs to.
OpenCV AI Competition 2026 entry (Agentic Vision path). Code: https://github.com/Driedsandwich/foveated-inspector · Live demo (throttled): https://s041j0ehzk.execute-api.us-east-1.amazonaws.com

**Summary.** An agent inspects a 4096 × 4096 concrete photo under a fixed budget of full-resolution analysis: it glances at an overview at ¼ of the width and height, chooses which 256 × 256 cells to examine more closely, reports a defect only when full resolution confirms it, and queues cells that stay uncertain at full resolution for a human. On held-out surfaces with synthetic defects, the agent's flags are right 42.9 % of the time at a 40 % budget, against 8.3 % for random cells given the same budget; at a 5 % budget the agent's flags are right 87.5 % of the time. Its flagged cells, however, cover only 19–30 % of the defects; the closer looks mainly buy precision, since reporting the overview's candidates directly covers slightly more defects (34.8 %) but only 20.4 % of those flags are right. The budget counts the closer looks; the ¼ overview and calibration add 7.8 % of the image's pixels (Section 6.1). The test surfaces were never used for training or tuning, but we did look at test results of two withdrawn versions (Section 7). The detector was trained on synthetic defects and does not yet transfer to real crack photos (52.9 % accuracy on 2,000 real patches, half of them cracks, i.e. close to chance); this is the main limitation. All results below were measured on 2026-10-24: evaluation, training and timing on AWS Lambda, and the real-patch check (Section 6.3) on a laptop. The evaluation code, traces and summaries are in the repository.

## 1. Problem and users

Inspection photographs of concrete (drone surveys of bridge decks, line-scan images of walls and pavements) are large, and most of the surface is defect-free. Running a full-resolution detector over every pixel is slow and expensive, and on busy textures it also produces a flood of false alarms that someone has to review. Inspectors and asset owners need to know **where to look**, and they need the system to be honest about what it did not examine.

The question this project measures: *with a fixed budget of full-resolution analysis, can an agent decide where to look well enough that its flags are worth a human's attention?*

## 2. What the system does

The agent runs a perception → decision → action loop:

1. **Coarse scan** of the whole image at ¼ resolution (OpenCV 5) → a belief map over a 16 × 16 grid of cells (each cell is 256 × 256 pixels at full resolution); the belief of a cell is the coarse model's highest patch probability in it (the share of the forest's trees that vote "defect") (patches and models are described in Section 3).
2. **Choose the next look**: a pending full-resolution confirmation of a candidate goes first; otherwise the agent scores each cell at its next zoom level (¼, ½, full) by the cell's current defect belief, multiplied by a weight for a cell a crack is heading into or for an ambiguous cell worth a closer look, divided by the full-resolution pixels that look would cost, and takes the highest score.
3. **Inspect the tile** with OpenCV 5 and `cv2.ml` random forests (one per scale) → candidate defects with kind, length and width.
4. **The evidence decides the next action**: escalate the same cell to a higher zoom when ambiguous; follow a crack whose endpoint touches the cell border into the neighbouring cell; **report a defect only when the full-resolution evidence overlaps the coarse candidate that triggered the look** (or is a thin linear structure); send an unresolved cell to a **human-review queue**.
5. **Stop** when the pixel budget is spent or no remaining cell is likely enough.

Every step (reason, observation, belief before/after, pixel cost) is written to a JSON trace file that the web UI replays. The policy is deterministic and rule-based; no generative model is involved.

## 3. OpenCV 5 implementation

- **Features** (whole-tile filtering and block pooling over 32 × 32 patches, 29 values per patch): morphological black-hat and top-hat at several kernel sizes, Hessian ridge strength from second-order Sobel, multi-orientation Gabor energy via `getGaborKernel` + `filter2D`, local standard deviation, Canny edge density, and Lab ΔL\*/Δa\*/Δb\*/chroma against the surface median. Every response is normalised by robust statistics of the same surface (median and median absolute deviation, MAD), estimated from the coarse overview plus a small stratified sample of calibration tiles (≤ 2 % of the image), so the normalisation itself does not require a full-resolution pass.
- **Models**: `cv2.ml.RTrees` per scale (a binary defect/background forest and a second forest for the defect kind: crack, spall, efflorescence, rust stain), trained only on development surfaces (7 of the 14 surfaces, Section 5). We chose small CPU forests over a deep detector so that a tool call on a 256 × 256 tile runs in tens of milliseconds on a CPU-only Lambda (Section 4); we did not compare against a deep detector. Hard-negative mining: after a first fit, the highest-scoring background patches of the development training images are added as negatives and the model is refitted once. Validation uses an image-level holdout.
- **Geometry**: `connectedComponentsWithStats`, `minAreaRect` (elongation), `PCACompute` (direction for crack following), `distanceTransform` (width), `ximgproc.thinning` (length).
- OpenCV 5.0.0 (`opencv-contrib-python-headless==5.0.0.93`) is used both locally and on Lambda (manylinux_2_28 wheels in a Lambda layer).

## 4. AWS deployment

![architecture](architecture.svg)

- **Public path**: API Gateway HTTP API with per-route throttling (1 request/s, burst 2) → `x132-web` (arm64, 256 MB, no OpenCV; input validation, per-IP and daily caps before any image is read) → `x132-inspect` (arm64 Graviton, 3008 MB, OpenCV 5 layer, models from S3).
- **Storage**: a private Amazon S3 bucket (object storage) (surfaces, benchmark, models, traces, overlays; raw uploads expire after 1 day, and the validated copy that is inspected and its results after 14 days), DynamoDB (inspections, human reviews, rate limits that expire automatically via a time-to-live attribute), CloudWatch (logs kept 7 days, invocation alarm).
- **All heavy computation on Lambda**: benchmark generation, feature extraction, training, hard-negative mining, tuning and evaluation run in a sharded `x132-bench` function (only the small real-patch check in Section 6.3 ran on a laptop); for these jobs the laptop only plans, fans out and merges small JSON files. Example: generating the 84-image 4096 px benchmark (28 training images and 28 development images, which are different images of the same 7 development surfaces, and 28 test images) took 43 s with 28 shards.
- **Responsible operation**: AWS Free-plan account (per AWS's documentation, the Free account plan ensures no charges are incurred; "Choosing a plan", AWS Billing User Guide, https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html, accessed 2026-10-23), execution roles scoped to the project's own resources, no public bucket, no Lambda Function URL on the public path, and `teardown.py` deletes the resources recorded in its state file after an ownership check suited to each resource type (it refuses to delete versioned buckets automatically). The account's Lambda concurrency limit is 10, so reserved concurrency could not be used; throttling and daily caps are applied at API Gateway and in the cheap front function instead.

**Graviton vs x86_64** (same OpenCV 5 tool, same three 256 × 256 tiles, three warm runs each): median execution 38.9 ms vs 36.7 ms; median billed duration 39 ms vs 37 ms — about the same. We did not measure cost. The sample is small.

## 5. Evaluation design

- **Backgrounds**: 14 photographed concrete surfaces from Poly Haven (CC0), 4096 × 4096. Development surfaces (7) and test surfaces (7) are disjoint. We used synthetic defects on real photographs because we had no licensed high-resolution real defect images with pixel-level ground truth within the competition window. The public crack patches used in Section 6.3 (Özgenel, 2019) are small classification patches without pixel-level ground truth, so we use them only as a transfer check.
- **Defects**: crack, spall, efflorescence and rust stain, injected with exact pixel ground truth. Each defect's contrast is calibrated against its local texture: the signal-to-noise ratio (SNR, mean intensity change on the defect divided by the local texture's standard deviation) is at least 1.3 (achieved minimum 1.42, median 2.63). The coarsest scale at which each defect is geometrically visible is recorded: of the 306 defects in the 56 evaluation images (28 development, 28 test), 251 are visible at ¼, 53 only from ½, 2 only at full resolution.
- **Data use**: training, hard-negative mining and policy tuning use development surfaces only. The test split (28 images, 155 defects) is used for the reported numbers.
- **Matching** (pooled over all defects in the test set): a report finds a defect if the intersection-over-union (IoU) of its box with the defect's box is ≥ 0.1, or if it covers ≥ 30 % of the defect's pixels while its area is ≤ 9 × the defect box. Anything else is a false alarm.
- **Cell-level view** (what the agent decides is *where to look*): a cell is *flagged* when it holds a confirmed report; a defect counts as found if one flagged cell contains ≥ 30 % of its pixels, and *defect-cell recall* is the share of defects found this way; we also report the fraction of cells flagged and the fraction of flagged cells that contain a defect ("flag precision").
- **Baselines**: random cells (3 random seeds, averaged per image) and raster order use the same full-resolution budget as the agent. Coarse-only (¼ candidates reported directly, no full-resolution look) and exhaustive full-resolution analysis are reference points at very different costs, so their pixel cost is shown next to their results.

## 6. Results

### 6.1 Test surfaces (28 images, 155 defects)

Policy configuration, selected on development data only by the rule "highest mean per-image box recall with ≤ 10 false alarms per image, otherwise the best recall with ≤ 50" (Section 6.2): the first closer look at a cell is at ½ resolution, and the detection threshold is 0.75 × 1.05 = 0.7875 at every scale (the tuning grid multiplies each candidate threshold by 1.00, 1.05 or 1.10; the result is never below the per-scale threshold fixed when the detector was validated). A defect is still reported only after the full-resolution check in Section 2.

| Method | Full-res budget | Defect-cell recall | Cells flagged | Flag precision | Box recall | Box precision | False alarms / image | Pixels analysed* |
|---|---|---|---|---|---|---|---|---|
| Active agent | 5 % | 19.4 % | 0.6 % | **87.5 %** | 16.1 % | 58.7 % | 0.7 | 12.8 % |
| Active agent | 40 % | 30.3 % | 1.8 % | **42.9 %** | 23.2 % | 24.7 % | 4.4 | 47.8 % |
| Coarse only (¼) | none | 34.8 % | 6.7 % | 20.4 % | 33.5 % | 15.8 % | 14.5 | 7.8 % |
| Random cells | 40 % | 29.7 % | 12.5 % | 8.3 % | 20.2 % | 2.7 % | 45.6 | 47.7 % |
| Raster order | 40 % | 25.8 % | 12.9 % | 7.7 % | 14.2 % | 1.8 % | 48.5 | 47.7 % |
| Exhaustive full-res | all cells | 70.3 % | 31.9 % | 8.6 % | 45.8 % | 2.7 % | 116.8 | 107.8 % |

\* Full-resolution-equivalent pixels, including the ¼ overview (6.25 %) and calibration tiles, as a fraction of the image; this is why exhaustive analysis exceeds 100 %. Box recall and precision pool all 155 defects; random cells average three seeds per image.

**Reading.** At the same 40 % budget the agent's defect-cell recall is about the same as random cells' (30.3 % vs 29.7 %) while flagging about seven times fewer cells (1.8 % vs 12.5 %), so a flag from the agent is about five times as likely to be right (42.9 % vs 8.3 %). At a 5 % budget its flags are right 87.5 % of the time. The cost is recall: with this conservative configuration the agent misses most defects, and exhaustive full-resolution analysis has the highest defect-cell recall (70.3 %) but produces 116.8 false alarms per image. Coarse-only reporting, which is the agent's own first step, has a slightly higher defect-cell recall (34.8 %) with fewer pixels (7.8 %), but only 20.4 % of its flags are right; the agent's full-resolution looks buy precision, not recall. On development surfaces the selected configuration reached 50.6 % defect-cell recall, so recall drops on the unseen test textures; a likely cause is the texture shift between surfaces, which we did not analyse further.

**Hand-crafted detector under the same policy.** We also evaluated the classic OpenCV detector (hand-written OpenCV rules on morphology, Gabor and Lab colour responses, with thresholds set from each image's own statistics; no learned model) with the same policy configuration, which was tuned for the learned detector only. At a 40 % budget it reaches 33.5 % defect-cell recall and 55.2 % flag precision, higher than the learned detector (30.3 % and 42.9 %), but its box precision is lower (17.4 % vs 24.7 %) and it raises more false alarms (6.3 vs 4.4 per image); at a 5 % budget it is lower on both cell measures (14.8 % and 86.0 % vs 19.4 % and 87.5 %). On these test surfaces the learned detector does not clearly beat the rule-based one.

### 6.2 Operating points on development data

Budget 40 %; the selection rule is the one stated in Section 6.1.

| Initial zoom | Threshold | Box recall (mean per image) | False alarms / image | Defect-cell recall | Cells flagged | Flag precision |
|---|---|---|---|---|---|---|
| ¼ | 0.60 | 67.3 % | 82.6 | 70.9 % | 22.5 % | 13.5 % |
| ½ | 0.75 | 57.7 % | 21.0 | 61.7 % | 6.3 % | 48.7 % |
| ½ | 0.75 × 1.05 (selected) | 43.7 % | 5.6 | 50.6 % | 2.6 % | 68.8 % |

These points show the trade-off a deployment would choose from: more recall costs many more flags for a human to review.

### 6.3 Real crack patches

Independent check, no training on real data: on 1,000 crack and 1,000 non-crack 227 × 227 patches from Özgenel (2019), the full-resolution model (a patch counts as defective if its highest patch score reaches the detector's validated full-resolution threshold) labels 98.0 % of crack patches and also 92.2 % of non-crack patches as defective (accuracy 52.9 %, close to the 50 % of guessing on this balanced set). The detector trained on synthetic defects does not transfer to real crack photographs at the patch level. This is the main limitation of the current model.

## 7. What went wrong and how we caught it

Two intermediate results were withdrawn: the first looked much better than the numbers above, and the second came from a degenerate configuration.

1. An early rule for merging overlapping reports into one box chained nearby reports along cracks into giant boxes. The agent's report boxes covered a median 77 % of each image, and the original matching rule did not penalise large boxes, so box recall under that legacy rule looked like 94.8 % at a 40 % budget. We found it by looking at the overlay of the live demo, measured the box-size distribution, and replaced both the merge rule (bounded pairs, no chaining) and the matching rule (IoU / bounded coverage, above). The legacy columns are kept in the summaries for transparency.
2. A first selection threshold (≤ 5 false alarms per image) picked a degenerate configuration with low recall; we changed the threshold using the development trade-off table only. The configuration behind the reported test numbers was selected on development data. However, we looked at test results of the two withdrawn versions during development, so the test set is not untouched, and a fresh test set has not yet been generated.

## 8. Limitations

- Defects in the benchmark are synthetic on real photographed backgrounds; the real-patch check shows the detector does not yet transfer to real cracks.
- Recall at useful precision is low; the active policy is limited by the detector's discrimination on busy textures.
- The live demo samples are real cracked surfaces without ground truth. In a recorded demo run on the cracked-concrete sample (20 % budget; an earlier run gave the same counts) the agent reported 1 defect confirmed at full resolution, listed 4 coarse candidates and queued 39 cells for human review.
- Timing results are from warm Lambda invocations and small samples. One 4096 × 4096 inspection through the public API with the reported configuration took 3.7 s of compute (7.0 s round trip, 20 % budget, one recorded run).

## 9. Responsible use

This is a triage aid, not a safety certification. The trace shows which parts of an image were never examined at full resolution, cells still uncertain after full-resolution inspection are routed to a human, and the reported numbers include the failure cases above. Our benchmark and demo samples contain no people or personal data; photos uploaded to the demo and their results are set to expire after 14 days (S3 lifecycle rules) (the raw upload after 1 day), and users should not upload images of people.

## 10. AI disclosure

The code and this report were written with AI coding assistants (OpenAI Codex and Anthropic Claude Code) under human direction. All AWS runs and reported numbers were checked against the stored run artifacts. The agent in the product is a rule-based policy and does not call a generative model.

## 11. Credits and licences

- Surface photos: Poly Haven, CC0 — *Powered by Poly Haven*.
- Real crack patches: Ç. F. Özgenel, "Concrete Crack Images for Classification", Mendeley Data, V2 (2019), doi:10.17632/5y9wdsg2zt.2, CC BY 4.0.
- OpenCV 5 (Apache 2.0).
