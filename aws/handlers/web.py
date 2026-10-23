"""Light public admission/router. No OpenCV, NumPy or inspector imports here.

The default execution path invokes only the private x132-inspect Lambda after
all input, per-IP and global UTC-day gates. Explicit test factories can use an
in-process adapter, which imports the worker only after the same admission.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import time
import uuid

from aws.image_validation import MAX_BYTES, ValidationError, validate_image_header
from aws.security import (RateLimiter, RateLimitExceeded, ServiceUnavailable,
                          digest, inspection_options, parse_body)

_APP = None
ID = re.compile(r"^[a-f0-9]{32}$")
MAX_WORK_MS = 18000


def response(status: int, body, *, content_type="application/json") -> dict:
    return {"statusCode": status, "headers": {
        "Content-Type": content_type, "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' https: blob: data:; "
            "connect-src 'self' https:; frame-ancestors 'none'; base-uri 'none'"},
        "body": body if isinstance(body, str) else json.dumps(body, allow_nan=False)}


def request_options(event: dict, prefix="x132/") -> dict | None:
    """Pure preflight, also used before creating SDK clients in handler()."""
    if not isinstance(event, dict):
        raise ValidationError("Invalid HTTP request.")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    if not isinstance(method, str) or not isinstance(path, str):
        raise ValidationError("Invalid HTTP method or path.")
    routes = {("GET", "/api/samples"), ("POST", "/api/upload-url"),
              ("POST", "/api/inspect"), ("POST", "/api/review")}
    lookup = method == "GET" and isinstance(path, str) and path.startswith("/api/inspection/")
    if (method, path) not in routes and not lookup:
        return None
    ip = event.get("requestContext", {}).get("http", {}).get("sourceIp")
    if not isinstance(ip, str) or len(ip) > 128:
        raise ValidationError("Missing or invalid source address.")
    try:
        ipaddress.ip_address(ip)
    except (ValueError, TypeError):
        raise ValidationError("Missing or invalid source address.") from None
    raw_headers = event.get("headers") or {}
    if not isinstance(raw_headers, dict):
        raise ValidationError("Invalid request headers.")
    headers = {str(key).lower(): value for key, value in raw_headers.items()}
    result = {"method": method, "path": path, "ip": ip, "headers": headers}
    if method == "GET":
        if event.get("body"):
            parse_body(event, set())
        if lookup and not ID.fullmatch(path.rsplit("/", 1)[-1]):
            raise ValidationError("Invalid inspection id.")
        return result
    if path == "/api/upload-url":
        body = parse_body(event, {"content_type", "size_bytes", "filename"}, {"content_type", "size_bytes"})
        mime, size = body["content_type"], body["size_bytes"]
        if (mime not in ("image/jpeg", "image/png") or isinstance(size, bool) or not isinstance(size, int)
                or not 0 < size <= MAX_BYTES or ("filename" in body and
                    (not isinstance(body["filename"], str) or len(body["filename"]) > 255))):
            raise ValidationError("Upload must be JPEG/PNG and at most 8 MB.")
    elif path == "/api/inspect":
        body = parse_body(event, {"sample_id", "upload_key", "budget_frac", "detector"})
        budget, detector = inspection_options(body)
        if ("sample_id" in body) == ("upload_key" in body):
            raise ValidationError("Choose exactly one sample_id or upload_key.")
        if "sample_id" in body:
            if not isinstance(body["sample_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", body["sample_id"]):
                raise ValidationError("Invalid sample id.")
        else:
            key, token = body["upload_key"], headers.get("x-upload-token")
            if not isinstance(key, str) or not re.fullmatch(re.escape(prefix) + r"uploads/[a-f0-9]{32}\.(jpg|png)", key):
                raise ValidationError("Invalid upload reference.")
            if not isinstance(token, str) or not 16 <= len(token) <= 256:
                raise ValidationError("A valid upload access token is required.")
        result.update(budget=budget, detector=detector)
    else:
        body = parse_body(event, {"inspection_id", "cell", "decision", "note"}, {"inspection_id", "cell", "decision"})
        if not isinstance(body["inspection_id"], str) or not ID.fullmatch(body["inspection_id"]):
            raise ValidationError("Invalid inspection id.")
        if (isinstance(body["cell"], bool) or not isinstance(body["cell"], int) or body["cell"] < 0
                or body["decision"] not in ("approve", "reject")
                or not isinstance(body.get("note", ""), str) or len(body.get("note", "")) > 500):
            raise ValidationError("Invalid review decision or note.")
    result["body"] = body
    return result


class WebApplication:
    def __init__(self, s3_client, ddb_client, backend_factory=None, *, bucket: str,
                 prefix="x132/", samples=None, clock=time.time, inspector_factory=None,
                 lambda_client=None, inspect_function=None, heavy_invoker=None):
        self.s3, self.ddb, self.factory = s3_client, ddb_client, backend_factory
        self.bucket, self.prefix, self.clock = bucket, prefix, clock
        self.inspector_factory = inspector_factory
        self.lambda_client, self.heavy_invoker = lambda_client, heavy_invoker
        self._managed_lambda = False
        self.inspect_function = inspect_function or os.environ.get("X132_INSPECT_FUNCTION", "x132-inspect")
        if self.inspect_function != "x132-inspect":
            raise ValidationError("The private inspection function is not configured correctly.")
        self.samples = samples
        self.rate = RateLimiter(ddb_client, os.getenv("X132_RATELIMIT_TABLE", "x132-ratelimit"), clock)
        self.inspections = os.getenv("X132_INSPECTIONS_TABLE", "x132-inspections")
        self.reviews = os.getenv("X132_REVIEWS_TABLE", "x132-reviews")

    def _put(self, table: str, key: str, value: dict, lifetime=14 * 86400):
        self.ddb.put_item(TableName=table, Item={"id": {"S": key},
            "expires_at": {"N": str(int(self.clock()) + lifetime)},
            "payload": {"S": json.dumps(value, allow_nan=False)}})

    def _get(self, table: str, key: str) -> dict:
        result = self.ddb.get_item(TableName=table, Key={"id": {"S": key}}, ConsistentRead=True)
        item = result.get("Item")
        if not item or int(item.get("expires_at", {}).get("N", "0")) <= self.clock():
            raise ValidationError("Record is missing or expired.")
        return json.loads(item["payload"]["S"])

    def _read(self, key: str, cap: int, **kw) -> bytes:
        result = self.s3.get_object(Bucket=self.bucket, Key=key, **kw)
        stream = result["Body"]
        try:
            data = stream.read(cap + 1)
        finally:
            if hasattr(stream, "close"):
                stream.close()
        if len(data) > cap:
            raise ValidationError("Stored object exceeds the allowed size.")
        return data

    def _key(self, key, *, sample=False):
        root = self.prefix + ("samples/" if sample else "")
        if (not isinstance(key, str) or not key.startswith(root) or len(key) > 1024 or "\\" in key
                or any(part in ("", ".", "..") for part in key.split("/")) or any(ord(c) < 32 for c in key)):
            raise ValidationError("Invalid stored object reference.")
        return key

    def _samples(self) -> list[dict]:
        if self.samples is None:
            self.samples = json.loads(self._read(self.prefix + "samples/manifest.json", 256 * 1024))
        values = self.samples.get("samples", []) if isinstance(self.samples, dict) else self.samples
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise ValidationError("Prepared samples are unavailable.")
        ids = set()
        for sample in values:
            if (not isinstance(sample, dict) or not isinstance(sample.get("id"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sample["id"]) or sample["id"] in ids):
                raise ValidationError("Prepared sample metadata is invalid.")
            ids.add(sample["id"])
            self._key(sample.get("key"), sample=True)
            if sample.get("thumbnail_key"):
                self._key(sample["thumbnail_key"], sample=True)
        return values

    def _url(self, key: str) -> str:
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": self._key(key)}, ExpiresIn=900)

    @staticmethod
    def _authorize(record: dict, headers: dict, name="x-inspection-token"):
        token = headers.get(name, "")
        if not isinstance(token, str) or not hmac.compare_digest(digest(token), record.get("token_hash", "!")):
            raise ValidationError("A valid access token is required.")

    def _public(self, record: dict) -> dict:
        return {"inspection_id": record["inspection_id"], "summary": record["summary"],
                "reported_defects": record.get("reported_defects", []),
                "coarse_candidates": record.get("coarse_candidates", []),
                "human_review_queue": record["human_review_queue"], "trace_url": self._url(record["trace_key"]),
                "overlay_url": self._url(record["overlay_key"]), "preview_url": self._url(record["preview_key"]),
                "image_url": self._url(record["image_key"])}

    @staticmethod
    def _remaining(context):
        return int(context.get_remaining_time_in_millis()) if context is not None else 30000

    def _invoke(self, payload, context):
        if self.heavy_invoker is not None:
            envelope = self.heavy_invoker(payload)
        elif self.factory is not None:
            # Explicit offline adapter; never selected by the public handler.
            from aws.handlers.inspect import handler as inspect_handler
            envelope = inspect_handler(payload, context, s3_client=self.s3, ddb_client=self.ddb,
                                       backend_factory=self.factory, inspector_factory=self.inspector_factory,
                                       bucket=self.bucket, prefix=self.prefix, clock=self.clock)
        else:
            remaining = self._remaining(context)
            if remaining < 5000:
                raise ServiceUnavailable("Insufficient time remains to invoke inspection.")
            if self.lambda_client is None or self._managed_lambda:
                import boto3
                from botocore.config import Config
                if self._managed_lambda and hasattr(self.lambda_client, "close"):
                    self.lambda_client.close()
                timeout = min(26.0, remaining / 1000 - 1)
                self.lambda_client = boto3.client("lambda", config=Config(connect_timeout=2, read_timeout=timeout,
                                                                          retries={"total_max_attempts": 1}))
                self._managed_lambda = True
            result = self.lambda_client.invoke(FunctionName="x132-inspect", InvocationType="RequestResponse",
                                                Payload=json.dumps(payload, allow_nan=False).encode())
            stream = result["Payload"]
            try:
                raw = stream.read(512 * 1024 + 1)
            finally:
                if hasattr(stream, "close"):
                    stream.close()
            if result.get("FunctionError") or len(raw) > 512 * 1024:
                raise ServiceUnavailable("Inspection timed out or could not complete. Please try a prepared sample later.")
            envelope = json.loads(raw)
        if not isinstance(envelope, dict) or envelope.get("ok") is not True or not isinstance(envelope.get("result"), dict):
            raise ServiceUnavailable("Inspection timed out or could not complete. Please try a prepared sample later.")
        return envelope["result"]

    def dispatch(self, event: dict, context=None) -> dict:
        options = request_options(event, self.prefix)
        if options is None:
            return response(404, {"error": "Route not found."})
        method, path, ip, headers = (options[name] for name in ("method", "path", "ip", "headers"))
        if method == "GET" and path == "/api/samples":
            self.rate.admit(ip, "samples")
            return response(200, {"samples": [{**{k: s[k] for k in ("id", "title", "width", "height") if k in s},
                "image_url": self._url(s["key"]), "thumbnail_url": self._url(s.get("thumbnail_key") or s["key"])}
                for s in self._samples()]})
        if method == "POST" and path == "/api/upload-url":
            body = options["body"]
            self.rate.admit(ip, "upload")
            mime = body["content_type"]
            ident, token = uuid.uuid4().hex, secrets.token_urlsafe(24)
            key = self.prefix + "uploads/" + ident + (".jpg" if mime == "image/jpeg" else ".png")
            self._put(self.inspections, "upload:" + ident, {"key": key, "token_hash": digest(token), "ip_hash": digest(ip), "mime": mime}, 600)
            post = self.s3.generate_presigned_post(Bucket=self.bucket, Key=key,
                Fields={"Content-Type": mime}, Conditions=[{"Content-Type": mime}, ["content-length-range", 1, MAX_BYTES]], ExpiresIn=300)
            return response(200, {"upload_key": key, "upload_token": token, "post": post})
        if method == "POST" and path == "/api/inspect":
            body = options["body"]
            self.rate.admit(ip, "inspect")
            if self._remaining(context) < 12000:
                raise ServiceUnavailable("Insufficient time remains to start inspection.")
            ident = uuid.uuid4().hex
            if "sample_id" in body:
                matches = [sample for sample in self._samples() if sample["id"] == body["sample_id"]]
                if len(matches) != 1:
                    raise ValidationError("Unknown sample.")
                image_key = self._key(matches[0]["key"], sample=True)
            else:
                key = body["upload_key"]
                grant = self._get(self.inspections, "upload:" + Path(key).stem)
                self._authorize(grant, headers, "x-upload-token")
                if grant["key"] != key or grant["ip_hash"] != digest(ip):
                    raise ValidationError("Upload belongs to a different request.")
                head = self.s3.head_object(Bucket=self.bucket, Key=key)
                if not 0 < head["ContentLength"] <= MAX_BYTES or head.get("ContentType") != grant["mime"]:
                    raise ValidationError("Stored upload size or content type is invalid.")
                raw = self._read(key, MAX_BYTES, IfMatch=head["ETag"])
                _, _, actual_mime = validate_image_header(raw)
                if actual_mime != grant["mime"]:
                    raise ValidationError("Image format does not match its content type.")
                image_key = self.prefix + "inputs/" + ident + Path(key).suffix
                self.s3.put_object(Bucket=self.bucket, Key=image_key, Body=raw, ContentType=actual_mime)
                del raw
            remaining = self._remaining(context)
            allowance = min(MAX_WORK_MS, remaining - 8000)
            if allowance < 4000:
                raise ServiceUnavailable("Insufficient time remains after preparing the image.")
            payload = {"inspection_id": ident, "image_key": image_key, "source": "sample" if "sample_id" in body else "upload",
                       "budget_frac": options["budget"], "detector": options["detector"], "time_limit_ms": int(allowance)}
            if "sample_id" in body:
                payload["sample_id"] = body["sample_id"]
            self._put(self.inspections, "admission:" + ident, payload, 60)
            return response(200, self._invoke(payload, context))
        if method == "GET" and path.startswith("/api/inspection/"):
            record = self._get(self.inspections, path.rsplit("/", 1)[-1])
            self._authorize(record, headers)
            return response(200, self._public(record))
        if method == "POST" and path == "/api/review":
            body = options["body"]
            record = self._get(self.inspections, body["inspection_id"])
            self._authorize(record, headers)
            if body["cell"] not in [item["cell"] for item in record["human_review_queue"]]:
                raise ValidationError("Cell is not in the review queue.")
            self.rate.consume(ip, "review", limit=100)
            self._put(self.reviews, f"{body['inspection_id']}:{body['cell']}", {**body, "note": body.get("note", ""), "updated_at": int(self.clock())})
            return response(200, {"saved": True, "inspection_id": body["inspection_id"], "cell": body["cell"], "decision": body["decision"]})
        return response(404, {"error": "Route not found."})


def handler(event, context=None):
    global _APP
    try:
        if event.get("requestContext", {}).get("http", {}).get("method", "GET") == "GET" and event.get("rawPath", "/") == "/":
            return response(200, (Path(__file__).resolve().parents[1] / "static/index.html").read_text(), content_type="text/html; charset=utf-8")
        if request_options(event, os.environ.get("X132_PREFIX", "x132/")) is None:
            return response(404, {"error": "Route not found."})
        if _APP is None:
            import boto3
            from botocore.config import Config
            config = Config(connect_timeout=2, read_timeout=3, retries={"total_max_attempts": 1})
            _APP = WebApplication(boto3.client("s3", config=config), boto3.client("dynamodb", config=config),
                                  bucket=os.environ["X132_BUCKET"], prefix=os.environ.get("X132_PREFIX", "x132/"))
        return _APP.dispatch(event, context)
    except ValidationError as exc:
        return response(400, {"error": str(exc)})
    except RateLimitExceeded as exc:
        return response(429, {"error": str(exc)})
    except ServiceUnavailable as exc:
        return response(503, {"error": str(exc)})
    except Exception:
        return response(503, {"error": "Inspection service unavailable. Please try a prepared sample later."})


lambda_handler = handler
