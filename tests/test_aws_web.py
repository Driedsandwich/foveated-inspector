"""Public API contract on <=64px synthetic data; never creates SDK clients."""

import io
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aws.handlers import web
from aws.image_validation import ValidationError, validate_image_bytes, validate_image_header
from aws.security import RateLimiter, RateLimitExceeded, ServiceUnavailable, inspection_options, parse_body

cv2.setNumThreads(2)


class FakeError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class FakeDDB:
    def __init__(self):
        self.items, self.counts = {}, {}

    def put_item(self, TableName, Item):
        self.items[TableName, Item["id"]["S"]] = Item

    def get_item(self, TableName, Key, **kw):
        return {"Item": self.items.get((TableName, Key["id"]["S"]))}

    def update_item(self, TableName, Key, ExpressionAttributeValues, **kw):
        key = (TableName, Key["id"]["S"])
        count = self.counts.get(key, 0)
        if count >= int(ExpressionAttributeValues[":limit"]["N"]):
            raise FakeError("ConditionalCheckFailedException")
        self.counts[key] = count+1


class FakeS3:
    def __init__(self):
        image = np.full((64, 64, 3), 128, np.uint8)
        self.objects = {"x132/samples/flat.png": cv2.imencode(".png", image)[1].tobytes()}
        self.puts = []

    def get_object(self, Bucket, Key, **kw):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body
        self.puts.append(Key)

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key]), "ContentType": "image/png", "ETag": '"same"'}

    def generate_presigned_url(self, action, Params, ExpiresIn):
        return "https://example.invalid/"+Params["Key"]

    def generate_presigned_post(self, **kw):
        self.post = kw
        return {"url": "https://example.invalid", "fields": {"key": kw["Key"], **kw["Fields"]}}


def event(path, body=None, headers=None, ip="192.0.2.1"):
    return {"rawPath": path, "requestContext": {"http": {
        "method": "GET" if body is None else "POST", "sourceIp": ip}},
        "body": json.dumps(body) if body is not None else None, "headers": headers or {}}


class SecurityTests(unittest.TestCase):
    def test_rate_limit_atomic_windows_and_fail_closed(self):
        client = FakeDDB()
        limiter = RateLimiter(client, clock=lambda: 100)
        for _ in range(20):
            limiter.consume("192.0.2.1", "inspect")
        with self.assertRaises(RateLimitExceeded):
            limiter.consume("192.0.2.1", "inspect")
        RateLimiter(client, clock=lambda: 3700).consume("192.0.2.1", "inspect")
        with patch.object(client, "update_item", side_effect=RuntimeError("no database")):
            with self.assertRaises(ServiceUnavailable):
                limiter.consume("192.0.2.2", "inspect")

    def test_json_and_numeric_validation(self):
        for budget in (True, float("nan"), .01, .41, "0.1"):
            with self.assertRaises(ValidationError):
                inspection_options({"budget_frac": budget})
        with self.assertRaises(ValidationError):
            parse_body({"body": '{"budget_frac":NaN}'}, {"budget_frac"})
        with self.assertRaises(ValidationError):
            parse_body({"body": '"a"'}, set())
        with self.assertRaises(ValidationError):
            parse_body({"body": "x"*23000}, set())

    def test_header_limits_before_decode(self):
        image = np.full((32, 64, 3), 100, np.uint8)
        for suffix in (".jpg", ".png"):
            raw = cv2.imencode(suffix, image)[1].tobytes()
            self.assertEqual(validate_image_header(raw)[:2], (64, 32))
            self.assertEqual(validate_image_bytes(raw).shape, (32, 64, 3))
        raw = bytearray(cv2.imencode(".png", image)[1].tobytes())
        raw[16:20] = (8193).to_bytes(4, "big")
        with patch("cv2.imdecode") as decode:
            with self.assertRaises(ValidationError):
                validate_image_bytes(bytes(raw))
            decode.assert_not_called()
        with self.assertRaises(ValidationError):
            validate_image_header(b"GIF89a")


class WebTests(unittest.TestCase):
    def setUp(self):
        self.s3, self.ddb = FakeS3(), FakeDDB()
        from inspector.aws_backend import AwsBackend
        self.app = web.WebApplication(self.s3, self.ddb,
            lambda **kw: AwsBackend("test-bucket", s3_client=self.s3, **kw),
            bucket="test-bucket", samples=[{"id": "flat", "title": "Flat synthetic surface",
                "key": "x132/samples/flat.png", "width": 64, "height": 64}], clock=lambda: 1000)

    def call(self, value):
        return json.loads(self.app.dispatch(value)["body"])

    def test_private_inspect_uses_dev_policy_from_environment(self):
        from inspector.agent import ActiveInspector
        captured = []

        def factory(backend, budget, config):
            captured.append(config)
            return ActiveInspector(backend, budget, config)

        self.app.inspector_factory = factory
        policy = {"grid": [2, 2], "work_width": 64, "initial_scale": .25,
                  "positive_threshold": .6, "detection_threshold": .6, "threshold_multiplier": 1.05}
        with patch.dict("os.environ", {"X132_POLICY_CONFIG": json.dumps(policy)}):
            result = self.call(event("/api/inspect", {"sample_id": "flat", "budget_frac": .05, "detector": "classic"}))
        self.assertIn("inspection_id", result)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].initial_scale, .25)
        self.assertEqual(captured[0].threshold_multiplier, 1.05)
        self.assertEqual(captured[0].positive_threshold, .6)

    def test_sample_inspect_and_private_review_contract(self):
        listing = self.call(event("/api/samples"))
        self.assertEqual(listing["samples"][0]["id"], "flat")
        self.assertNotIn("key", listing["samples"][0])
        result = self.call(event("/api/inspect", {"sample_id": "flat", "budget_frac": .05, "detector": "classic"}))
        self.assertEqual(result["summary"]["detector"], "classic")
        self.assertIn("trace_url", result)
        self.assertIn("overlay_url", result)
        self.assertIn("preview_url", result)
        self.assertIn("calibration_cost_frac", result["summary"])
        self.assertIn("reported_defects", result)
        self.assertIn("coarse_candidates", result)
        self.assertEqual(result["summary"]["n_candidates"], len(result["coarse_candidates"]))
        ident = result["inspection_id"]
        with self.assertRaises(ValidationError):
            self.call(event("/api/inspection/"+ident))
        auth = {"X-Inspection-Token": result["review_token"]}
        fetched = self.call(event("/api/inspection/"+ident, headers=auth))
        self.assertEqual(fetched["inspection_id"], ident)
        self.assertEqual(fetched["reported_defects"], result["reported_defects"])
        self.assertEqual(fetched["coarse_candidates"], result["coarse_candidates"])
        record = self.app._get(self.app.inspections, ident)
        self.assertNotIn(result["review_token"], json.dumps(record))
        record["human_review_queue"] = [{"cell": 2, "reason": "fixture"}]
        self.app._put(self.app.inspections, ident, record)
        review = self.call(event("/api/review", {"inspection_id": ident, "cell": 2,
            "decision": "approve", "note": "confirmed"}, auth))
        self.assertTrue(review["saved"])
        with self.assertRaises(ValidationError):
            self.call(event("/api/review", {"inspection_id": ident, "cell": 3, "decision": "reject"}, auth))
        self.assertTrue(any(key.endswith("trace.json") for key in self.s3.puts))

    def test_upload_is_scoped_and_frozen(self):
        grant = self.call(event("/api/upload-url", {"content_type": "image/png", "size_bytes": 300}))
        self.assertIn(["content-length-range", 1, 8*1024*1024], self.s3.post["Conditions"])
        self.s3.objects[grant["upload_key"]] = self.s3.objects["x132/samples/flat.png"]
        with self.assertRaises(ValidationError):
            self.call(event("/api/inspect", {"upload_key": grant["upload_key"]}))
        result = self.call(event("/api/inspect", {"upload_key": grant["upload_key"]},
                                {"X-Upload-Token": grant["upload_token"]}))
        self.assertIn("/inputs/", result["image_url"])

    def test_public_routes_and_sanitized_failure(self):
        html = web.handler(event("/"), None)
        self.assertEqual(html["statusCode"], 200)
        with patch.object(web, "_APP", self.app):
            result = web.handler(event("/api/inspect", {"sample_id": "unknown"}), None)
            self.assertEqual(result["statusCode"], 400)
            with patch.object(self.app, "dispatch", side_effect=RuntimeError("secret marker")):
                result = web.handler(event("/api/samples"), None)
                self.assertEqual(result["statusCode"], 503)
                self.assertNotIn("secret marker", result["body"])


if __name__ == "__main__":
    unittest.main()
