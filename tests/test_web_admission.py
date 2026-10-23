"""Same-process import guards and fake admission tests; no heavy compute or SDK."""

import builtins
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from aws.handlers import web
from aws.security import digest


class Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class DDB:
    def __init__(self):
        self.items, self.counts = {}, {}

    def update_item(self, TableName, Key, ExpressionAttributeValues, **kwargs):
        key = (TableName, Key["id"]["S"])
        count = self.counts.get(key, 0)
        if count >= int(ExpressionAttributeValues[":limit"]["N"]):
            raise Error("ConditionalCheckFailedException")
        self.counts[key] = count + 1

    def put_item(self, TableName, Item):
        self.items[TableName, Item["id"]["S"]] = Item

    def get_item(self, TableName, Key, **kwargs):
        return {"Item": self.items.get((TableName, Key["id"]["S"]))}


class S3:
    def __init__(self):
        self.reads, self.posts = [], []

    def get_object(self, Bucket, Key, **kwargs):
        self.reads.append(Key)
        if Key != "x132/samples/manifest.json":
            raise AssertionError("public admission must not read image bodies")
        data = json.dumps({"samples": [{"id": "flat", "title": "Synthetic fixture", "key": "x132/samples/flat.png"}]}).encode()
        return {"Body": io.BytesIO(data)}

    def head_object(self, **kwargs):
        raise AssertionError("rejected admission must not access image metadata")

    def generate_presigned_url(self, action, Params, ExpiresIn):
        return "https://example.invalid/" + Params["Key"]

    def generate_presigned_post(self, **kwargs):
        self.posts.append(kwargs)
        return {"url": "https://example.invalid", "fields": {}}


class Heavy:
    def __init__(self):
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        return {"ok": True, "result": {"inspection_id": payload["inspection_id"], "summary": {"partial": True},
                "human_review_queue": [], "trace_url": "https://example.invalid/trace", "overlay_url": "https://example.invalid/overlay",
                "preview_url": "https://example.invalid/preview", "image_url": "https://example.invalid/image", "review_token": "fake-token"}}


def event(body=None, path="/api/inspect", ip="192.0.2.1", method=None):
    return {"rawPath": path, "requestContext": {"http": {"method": method or ("POST" if body is not None else "GET"), "sourceIp": ip}},
            "headers": {}, "body": json.dumps(body) if body is not None else None}


def import_guard():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("cv2", "numpy", "inspector"):
            raise AssertionError("heavy import before admission: " + name)
        return original(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=guarded)


def fresh_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        s3, ddb, heavy = S3(), DDB(), Heavy()
        app = web.WebApplication(s3, ddb, bucket="test-bucket", clock=lambda: 1000, heavy_invoker=heavy)
        return app, s3, ddb, heavy

    def call(self, app, request, context=None):
        with patch.object(web, "_APP", app):
            return web.handler(request, context)

    def test_fresh_public_module_static_and_rejected_requests_import_no_heavy_library(self):
        with import_guard():
            module = fresh_module(Path(web.__file__), "_admission_public_web")
            self.assertEqual(module.handler(event(path="/", method="GET"))["statusCode"], 200)
            self.assertEqual(module.handler(event(path="/not-a-route", method="GET"))["statusCode"], 404)
            bad = event({"sample_id": "flat", "budget_frac": 1e300})
            self.assertEqual(module.handler(bad)["statusCode"], 400)
            invalid = event({"sample_id": "flat"})
            invalid["body"] = "x" * 23000
            self.assertEqual(module.handler(invalid)["statusCode"], 400)
            self.assertIsNone(module._APP)
            validation = fresh_module(Path(web.__file__).parents[1] / "image_validation.py", "_admission_header_validation")
            with self.assertRaises(validation.ValidationError):
                validation.validate_image_header(b"invalid-header")

    def test_bad_body_budget_and_source_shapes_never_read_s3_or_invoke_worker(self):
        app, s3, ddb, heavy = self.fixture()
        invalid = [
            {"sample_id": ["flat"]}, {"sample_id": "../flat"},
            {"sample_id": "flat", "upload_key": "x132/uploads/" + "a" * 32 + ".jpg"},
            {"upload_key": "s3://other-bucket/image.jpg"}, {"sample_id": "flat", "budget_frac": True},
            {"sample_id": "flat", "budget_frac": .41}, {"sample_id": "flat", "detector": "shell"},
        ]
        with import_guard():
            for body in invalid:
                self.assertEqual(self.call(app, event(body))["statusCode"], 400)
        self.assertEqual((s3.reads, heavy.calls, ddb.counts), ([], [], {}))

    def test_twenty_ip_admissions_then_429_with_no_more_metadata_or_worker_calls(self):
        app, s3, _, heavy = self.fixture()
        with import_guard():
            for _ in range(20):
                self.assertEqual(self.call(app, event({"sample_id": "flat"}))["statusCode"], 200)
            before = list(s3.reads)
            self.assertEqual(self.call(app, event({"sample_id": "flat"}))["statusCode"], 429)
        self.assertEqual(len(heavy.calls), 20)
        self.assertEqual(s3.reads, before)

    def test_global_daily_inspect_cap_and_upload_cap_run_before_s3_access(self):
        for action, path, body in (("inspect", "/api/inspect", {"sample_id": "flat"}),
                                   ("upload", "/api/upload-url", {"content_type": "image/png", "size_bytes": 100})):
            app, s3, ddb, heavy = self.fixture()
            key = f"daily-{action}:{digest('global')}:0"
            ddb.counts[app.rate.table, key] = 200
            with import_guard():
                result = self.call(app, event(body, path=path))
            self.assertEqual(result["statusCode"], 429)
            self.assertEqual((s3.reads, s3.posts, heavy.calls), ([], [], []))

    def test_utc_daily_window_and_fail_closed_rate_database(self):
        app, s3, ddb, heavy = self.fixture()
        ddb.counts[app.rate.table, f"daily-inspect:{digest('global')}:0"] = 200
        app.rate.clock = lambda: 86400
        with import_guard():
            self.assertEqual(self.call(app, event({"sample_id": "flat"}))["statusCode"], 200)
        self.assertEqual(len(heavy.calls), 1)
        app, s3, ddb, heavy = self.fixture()
        with patch.object(ddb, "update_item", side_effect=RuntimeError("private-database-error")), import_guard():
            result = self.call(app, event({"sample_id": "flat"}))
        self.assertEqual(result["statusCode"], 503)
        self.assertEqual((s3.reads, heavy.calls), ([], []))

    def test_default_transport_invokes_only_private_function_after_bound_grant(self):
        app, s3, ddb, heavy = self.fixture()

        class Client:
            def __init__(self):
                self.calls = []

            def invoke(self, **request):
                self.calls.append(request)
                payload = json.loads(request["Payload"])
                return {"Payload": io.BytesIO(json.dumps(heavy(payload)).encode())}

        client = Client()
        app.heavy_invoker, app.lambda_client = None, client
        with import_guard():
            result = self.call(app, event({"sample_id": "flat", "budget_frac": .05}))
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(client.calls[0]["FunctionName"], "x132-inspect")
        payload = heavy.calls[0]
        grant = app._get(app.inspections, "admission:" + payload["inspection_id"])
        self.assertEqual(grant, payload)
        self.assertLessEqual(payload["time_limit_ms"], 18000)

    def test_preparation_time_guard_returns_503_without_heavy_invocation(self):
        app, _, _, heavy = self.fixture()

        class Context:
            def __init__(self):
                self.remaining = iter((30000, 11000))

            def get_remaining_time_in_millis(self):
                return next(self.remaining)

        with import_guard():
            result = self.call(app, event({"sample_id": "flat"}), Context())
        self.assertEqual(result["statusCode"], 503)
        self.assertEqual(heavy.calls, [])

    def test_private_timeout_is_honest_503_without_error_payload_or_retry(self):
        app, _, _, _ = self.fixture()

        class Client:
            def __init__(self):
                self.calls = 0

            def invoke(self, **kwargs):
                self.calls += 1
                return {"FunctionError": "Unhandled", "Payload": io.BytesIO(b'{"errorMessage":"private-token"}')}

        client = Client()
        app.heavy_invoker, app.lambda_client = None, client
        with import_guard():
            result = self.call(app, event({"sample_id": "flat"}))
        self.assertEqual(result["statusCode"], 503)
        self.assertNotIn("private-token", result["body"])
        self.assertEqual(client.calls, 1)

    def test_private_worker_rejects_unbound_or_changed_event_before_heavy_import(self):
        with import_guard():
            module = fresh_module(Path(web.__file__).with_name("inspect.py"), "_admission_private_inspect")
            payload = {"inspection_id": "a" * 32, "image_key": "x132/samples/flat.png", "source": "sample",
                       "sample_id": "flat", "budget_frac": .1, "detector": "classic", "time_limit_ms": 18000}
            ddb = DDB()
            result = module.handler(payload, ddb_client=ddb, bucket="test-bucket", clock=lambda: 1000)
            self.assertFalse(result["ok"])
            ddb.put_item(TableName="x132-inspections", Item={"id": {"S": "admission:" + "a" * 32},
                         "expires_at": {"N": "1060"}, "payload": {"S": json.dumps(payload)}})
            changed = {**payload, "image_key": "x132/samples/other.png"}
            result = module.handler(changed, ddb_client=ddb, bucket="test-bucket", clock=lambda: 1000)
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], 400)


if __name__ == "__main__":
    unittest.main()
