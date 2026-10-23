"""Private, admitted inspection worker. Heavy libraries load after grant checks.

There is no public function URL. The web role invokes this worker with a short
TTL, DynamoDB-bound admission payload; client requests cannot choose arbitrary
images, buckets, functions or execution parameters.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time

from aws.image_validation import ValidationError
from aws.security import ServiceUnavailable, digest, inspection_options

MAX_WORK_MS = 18000
ID = re.compile(r"^[a-f0-9]{32}$")


def validate_event(event: dict, prefix: str) -> dict:
    required = {"inspection_id", "image_key", "source", "budget_frac", "detector", "time_limit_ms"}
    if not isinstance(event, dict) or not required <= event.keys() or set(event) - required - {"sample_id"}:
        raise ValidationError("Invalid private inspection event.")
    ident, key = event["inspection_id"], event["image_key"]
    if not isinstance(ident, str) or not ID.fullmatch(ident) or not isinstance(key, str):
        raise ValidationError("Invalid private image identity.")
    inspection_options(event)
    limit = event["time_limit_ms"]
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1000 <= limit <= MAX_WORK_MS:
        raise ValidationError("Invalid private time allowance.")
    if event["source"] == "sample":
        if (not key.startswith(prefix + "samples/") or len(key) > 1024 or "\\" in key
                or any(part in ("", ".", "..") for part in key.split("/"))
                or any(ord(c) < 32 for c in key)
                or not isinstance(event.get("sample_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", event["sample_id"])):
            raise ValidationError("Invalid admitted sample.")
    elif event["source"] == "upload":
        if "sample_id" in event or not re.fullmatch(re.escape(prefix) + "inputs/" + ident + r"\.(jpg|png)", key):
            raise ValidationError("Invalid frozen upload identity.")
    else:
        raise ValidationError("Invalid private image source.")
    return dict(event)


def _admission(ddb, table: str, event: dict, clock) -> None:
    result = ddb.get_item(TableName=table, Key={"id": {"S": "admission:" + event["inspection_id"]}}, ConsistentRead=True)
    item = result.get("Item")
    if not item or int(item.get("expires_at", {}).get("N", "0")) <= clock():
        raise ValidationError("Inspection admission is missing or expired.")
    if json.loads(item["payload"]["S"]) != event:
        raise ValidationError("Inspection admission does not match its image and options.")


def _remaining(context) -> int:
    return int(context.get_remaining_time_in_millis()) if context is not None else 25000


def run_inspection(event: dict, context, *, s3, ddb, bucket: str, prefix: str,
                   backend_factory=None, inspector_factory=None, clock=time.time,
                   started=None) -> dict:
    """Called only after the standard-library identity/grant/time checks."""
    import cv2
    cv2.setNumThreads(2)
    from inspector.agent import ActiveInspector
    from inspector.viz import overlay
    detector, budget = event["detector"], float(event["budget_frac"])
    if backend_factory is None:
        from aws.runtime import get_backend
        backend = get_backend(bucket=bucket, prefix=prefix, detector=detector, s3_client=s3,
                              calibration="sampled", remote_tools=False)
    else:
        backend = backend_factory(detector=detector, remote_tools=False)
    spent = 0 if started is None else int((time.monotonic() - started) * 1000)
    run_limit = min(event["time_limit_ms"] - spent, _remaining(context) - 5000)
    if run_limit < 1000:
        raise ServiceUnavailable("Insufficient worker time remains.")
    factory = inspector_factory or ActiveInspector
    image_ref = f"s3://{bucket}/{event['image_key']}"
    policy = os.environ.get("X132_POLICY_CONFIG")
    if policy:
        from inspector.agent import InspectorConfig
        inspector = factory(backend, budget, InspectorConfig(**json.loads(policy)))
    else:
        inspector = factory(backend, budget)
    result = inspector.run(image_ref, time_limit_ms=run_limit)
    if _remaining(context) < 2500:
        raise ServiceUnavailable("Insufficient worker time remains to save an inspection.")
    area = result["coarse"]["image_shape"][0] * result["coarse"]["image_shape"][1]
    partial = bool(str(result.get("stop_reason", "")).lower().startswith("time limit") or result.get("partial", False))
    summary = {key: result.get(key) for key in ("budget_used_frac", "fine_cost_px", "coarse_cost_px",
               "calibration_cost_px", "n_tool_calls", "elapsed_ms", "stop_reason")}
    summary.update({"n_detections": len(result["reported_defects"]), "detector": detector,
                    "n_candidates": len(result.get("coarse_candidates", [])),
                    "reporting_version": result.get("reporting_version", "legacy"),
                    "calibration_cost_frac": result.get("calibration_cost_px", 0) / area,
                    "total_cost_frac": (result["fine_cost_px"] + result["coarse_cost_px"]) / area,
                    "partial": partial})
    ident = event["inspection_id"]
    keys = {name: prefix + f"outputs/{ident}/{name}" for name in ("trace.json", "overlay.png", "preview.png")}
    s3.put_object(Bucket=bucket, Key=keys["trace.json"], Body=json.dumps(result, allow_nan=False).encode(), ContentType="application/json")
    image = backend.image(image_ref)
    rendered = overlay(image, result)
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise ServiceUnavailable("Could not encode inspection overlay.")
    s3.put_object(Bucket=bucket, Key=keys["overlay.png"], Body=encoded.tobytes(), ContentType="image/png")
    ratio = min(1.0, 1024 / max(image.shape[:2]))
    preview = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise ServiceUnavailable("Could not encode inspection preview.")
    s3.put_object(Bucket=bucket, Key=keys["preview.png"], Body=encoded.tobytes(), ContentType="image/png")
    token = secrets.token_urlsafe(24)
    record = {"inspection_id": ident, "token_hash": digest(token), "summary": summary,
              "reported_defects": result["reported_defects"],
              "coarse_candidates": result.get("coarse_candidates", []),
              "human_review_queue": result["human_review_queue"], "image_key": event["image_key"],
              "trace_key": keys["trace.json"], "overlay_key": keys["overlay.png"], "preview_key": keys["preview.png"]}
    ddb.put_item(TableName=os.environ.get("X132_INSPECTIONS_TABLE", "x132-inspections"),
                 Item={"id": {"S": ident}, "expires_at": {"N": str(int(clock()) + 14 * 86400)},
                       "payload": {"S": json.dumps(record, allow_nan=False)}})

    def url(key):
        return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900)

    return {"inspection_id": ident, "summary": summary, "human_review_queue": result["human_review_queue"],
            "reported_defects": result["reported_defects"], "coarse_candidates": result.get("coarse_candidates", []),
            "trace_url": url(record["trace_key"]), "overlay_url": url(record["overlay_key"]),
            "preview_url": url(record["preview_key"]), "image_url": url(record["image_key"]), "review_token": token}


def handler(event, context=None, *, s3_client=None, ddb_client=None, backend_factory=None,
            inspector_factory=None, bucket=None, prefix=None, clock=time.time):
    started = time.monotonic()
    try:
        prefix = prefix or os.environ.get("X132_PREFIX", "x132/")
        validated = validate_event(event, prefix)
        if _remaining(context) < 8000:
            raise ServiceUnavailable("Insufficient worker time remains.")
        if ddb_client is None:
            import boto3
            ddb_client = boto3.client("dynamodb")
        _admission(ddb_client, os.environ.get("X132_INSPECTIONS_TABLE", "x132-inspections"), validated, clock)
        bucket = bucket or os.environ.get("X132_BUCKET")
        if not bucket:
            raise ServiceUnavailable("Inspection storage is unavailable.")
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        result = run_inspection(validated, context, s3=s3_client, ddb=ddb_client, bucket=bucket, prefix=prefix,
                                backend_factory=backend_factory, inspector_factory=inspector_factory, clock=clock,
                                started=started)
        return {"ok": True, "result": result}
    except (ValidationError, KeyError, TypeError, ValueError, OverflowError):
        return {"ok": False, "status": 400, "error": {"code": "invalid_admission", "message": "Inspection admission is invalid."}}
    except Exception:
        return {"ok": False, "status": 503, "error": {"code": "inspection_unavailable", "message": "The private inspection could not complete."}}


lambda_handler = handler
