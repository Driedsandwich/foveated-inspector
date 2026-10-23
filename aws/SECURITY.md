# Operating conditions for the public demo (v5)

The public entry point is an API Gateway HTTP API. Rate limiting and OpenCV image processing are split into separate Lambda functions. The account information below was measured by the user on 2026-10-24. This conversation did not connect to AWS. The price list was not checked online either, and the unit prices are assumptions for calculation.

## Measurements and public scope

The account concurrency limit was 10, and the unreserved amount was also 10. A minimum of 10 unreserved slots is required, so an operation that reserves 2 slots for web fails. No quota increase will be requested. Reserved concurrency is disabled by default. On the HTTP API `$default` stage, the default applied to each route is set to 1 request per second with a burst of 2. These can be changed with `--api-rate` and `--api-burst`. This is not a setting that limits the total of the 6 routes to 1 request per second; the configured total is 6 requests per second with a burst of 12. Service-side throttling is best-effort and is not a billing cap.

There are 6 public routes. This is a demo without authentication. CORS is limited to the API origin, GET/POST, Content-Type, and two kinds of access-token headers.

```text
GET  /
GET  /api/samples
POST /api/inspect
POST /api/upload-url
POST /api/review
GET  /api/inspection/{id}
```

`x132-web` has 256 MB and no OpenCV layer. It checks the body size, format, and budget, and the DynamoDB usage limits, before accessing any image. The rejection path has no import of cv2/NumPy/agent and no image fetch. Latency has not been measured, and there is no guarantee that it always finishes in a few ms, even during a network failure.

Only permitted inspections are called synchronously into the private `x132-inspect`. The inspection function is arm64, 3008 MB, 25 seconds; web is 30 seconds, and the API integration wait is also 30 seconds. The agent normally stops starting new tool calls within 18 seconds and saves partial results. A single OpenCV operation cannot be interrupted, so a cold start or a heavy photo can cause a timeout or 503. The tool is 30 seconds, and the private bench is 900 seconds. `cv2.setNumThreads(2)` is used at every image-processing entry point.

Lambda public invocation permission is limited to the SourceArn and SourceAccount of the owned API. No public URL is created for inspect/tool/bench. If an owned web function from an older deployment has a Function URL, it is changed to AWS_IAM and the known public permission is removed. Even if the URL entry in state is missing because of a mid-run failure, the owned function is checked and handled.

Only when the reserved slots will be used in the future, specify `--reserve-web-concurrency --web-concurrency 2`. If 10 unreserved slots cannot be kept after the change, an explicit message is shown and the reservation is skipped; no limit increase is made. In the account measured this time, the reservation is skipped.

## Input, usage limits, and storage

- Only API Gateway's `requestContext.http.sourceIp` is used for the IP. Forwarded headers and IPs in the body are not trusted.
- Inspection, upload-URL issuance, and sample-list retrieval each have a separate counter, with 20 per hour per IP and 200 per UTC day overall. It is not 200 per day summed across all APIs. Review is 100 per hour per IP. DynamoDB conditional updates are used, and if the limit cannot be verified the request is stopped with 503. At fixed-window boundaries, a short burst, or up to twice the daily limit in permitted requests over 24 consecutive hours, is possible.
- The JSON body is up to 16 KiB, the budget is a finite number from 0.02 to 0.40, and the detector is only classic/learned. An arbitrary S3 path or external URL cannot be specified.
- Uploads are JPEG/PNG, 8 MiB or less, and 8192 px or less on each side. Conditions on size and type are attached to the presigned POST. The S3 size, type, and image header are checked, and the bytes read with the ETag specified are fixed to a separate input key. The private inspect also checks the dimensions after decoding.
- Only photos in the admin `samples/` are allowed up to 32 MiB. The 8 MiB limit for public uploads is kept.
- Review and re-retrieval require the access token returned at creation, and only its hash is stored. Review targets are limited to the queue, and a note is up to 500 characters. The model is not trained automatically.
- A GET presigned URL is valid for 15 minutes, a POST for 5 minutes, and an upload permission for 10 minutes. Anyone who knows the URL can access S3 directly within the expiry period. The number of times a URL can be reused is not limited.
- Uploads have a 1-day S3 lifecycle, fixed inputs and results have a 14-day S3 lifecycle, and DynamoDB records have a 14-day TTL. Deletion is not immediate. Raw IP values are not stored; SHA-256 is used for the limit key.
- Authentication uses only the standard SDK chain. Events, images, tokens, and credentials are not written to logs. `X132_DEBUG_ERRORS=1` is kept as an optional deployment diagnostic and masks 12-digit numbers in service errors. Internal errors are not returned in public responses.

## Permissions and operations

The execution roles for web, inspect, tool, and bench are separate. The only thing web can call is inspect. Inspect reads the project's model, samples, and fixed inputs, and stores results and inspection records. The model is checked against an allow-list of 7 files, their sizes, and SHA-256, and is pinned to a version with a content hash.

State version 1 keeps the existing bucket, function, and role names, and adds the API and inspect. The memory of every function is 3008 MB or less. The bench's `X132_BENCH_MEMORY_MB` is also checked against the limit. The 4096 MB of ephemeral storage is separate from memory capacity.

The CloudWatch alarm on the number of web invocations defaults to 60 per minute. The alarm is not an automatic cutoff or a cost cap, and no external notification target has been added. The operator monitors the remaining credits and the invocation volume. The planned end date is 2026-11-09, and no automatic deletion is scheduled.

## Daily cost under a sustained attack

The following is a **calculation assumption** based on the standard us-east-1 unit prices. It is not a value obtained from the actual price list. Free tier, credits, discounts, and tax are not subtracted.

| Item | Assumed unit price |
|---|---:|
| HTTP API | $1.00 / 1 million requests |
| Lambda requests | $0.20 / 1 million |
| Lambda arm64 compute | $0.0000133334 / GB-second |
| DynamoDB provisioned capacity | write $0.00065/WCU-hour, read $0.00013/RCU-hour |
| S3 PUT/POST | $0.005 / 1,000 requests |
| S3 GET | $0.0004 / 1,000 requests |
| S3 Standard storage | $0.023 / GB-month |
| Outbound transfer | $0.09 / GB |
| CloudWatch Logs ingestion | $0.50 / GB |

This is a scenario in which the limits work as configured on all 6 routes, each at 1 request per second plus an initial burst of 2, so `N=6×(86,400+2)=518,412` requests/day reach Lambda. The calculation deliberately assumes long processing times: every web invocation uses 30 seconds, and the 200 permitted inspections each use 25 seconds.

```text
API                           = 518,412 / 1,000,000 × $1                   = $0.518412
web                           = 518,412 × 0.25 GB × 30 s × $0.0000133334   = $51.841459
inspect                       = 200 × (3008/1024) GB × 25 s × $0.0000133334 = $0.195834
Lambda requests               = (518,412 + 200) / 1,000,000 × $0.20        = $0.103722
DynamoDB (3 tables x 5 capacity each) = 24 × 15 × ($0.00065 + $0.00013)    = $0.280800
Subtotal                                                                   ≈ $52.94 / UTC day
```

This value is not reduced by the concurrency limit of 10, so it is an overestimate in that respect too. If a rejected web invocation took 100 ms, the web compute portion would be about $0.172805 and the subtotal about $1.27/day. The 100 ms is not a measurement. Initialization time, S3, logs, storage and transfer, and access that does not go through the API are not included.

**An absolute worst case for the total bill cannot be guaranteed as a finite amount by this configuration alone.** The limits are best-effort. Under the conservative assumption that every received request is billable, sending 1,000 requests per second costs `86,400 × 1,000 / 1,000,000 × $1 = $86.40/day` for the API portion alone. How 429 and similar responses are actually billed needs separate confirmation.

GET/POST to S3 presigned URLs does not go through the Gateway. Fetching a 32 MiB photo 100,000 times is `32 × 1024² × 100,000 / 10⁹ = 3,355.4432 GB`, which is about $301.99 at the assumed transfer unit price. GET charges are separate. Write charges from reusing a POST also cannot be limited by the number of new permissions alone. Therefore $52.94 is not called a daily cap. The operating boundary is checking the remaining credits of the Free plan and the stop procedure below.

## Emergency stop (run by the user)

The following has not been run in this conversation. It reads the API ID from the local state.

```sh
X132_API_ID="$(../.venv/bin/python -B -c 'import json; from pathlib import Path; print(json.loads(Path("aws/state.json").read_text())["http_api"]["id"])')"
aws apigatewayv2 update-stage --api-id "$X132_API_ID" --stage-name '$default' --default-route-settings '{"ThrottlingRateLimit":0,"ThrottlingBurstLimit":0}' --region us-east-1
```

Changing the limits to zero is not treated as a guarantee of an immediate, absolute cutoff. To remove the API, use the following. Even after deletion, running Lambda invocations and issued S3 presigned URLs do not stop immediately. Redeploying recreates the API.

```sh
aws apigatewayv2 delete-api --api-id "$X132_API_ID" --region us-east-1
../.venv/bin/python -B aws/teardown.py --dry-run
../.venv/bin/python -B aws/teardown.py --execute
```

The old procedure of setting reserved concurrency to 0 is not used as a stop method in this account. Keep the state. Deletion targets only the owned resources that were recorded.

The changed API, CORS, permissions, cold starts, and the memory needed for 8192 px have not been verified on AWS. The pixel cost does not represent price or network time. This demo does not guarantee the safety of structures.
