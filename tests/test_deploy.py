"""Metadata-only infrastructure checks; no boto3, credentials or network needed."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aws import deploy, teardown


class FakePaginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, **kwargs):
        self.client.list_arguments = kwargs
        yield {"Contents": [{"Key": "x132/outputs/a.json"}, {"Key": "x132/uploads/b.jpg"}]}


class FakeS3:
    def __init__(self, foreign=False):
        self.foreign = foreign
        self.deleted = []
        self.list_arguments = None

    def get_bucket_versioning(self, **kwargs):
        return {}

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self)

    def delete_objects(self, **kwargs):
        self.deleted.extend(item["Key"] for item in kwargs["Delete"]["Objects"])
        return {}

    def list_objects_v2(self, **kwargs):
        return {"Contents": [{"Key": "other-project/keep.txt"}]} if self.foreign else {}


class FakeAwsError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeWaiter:
    def wait(self, **kwargs):
        return None


class FakeApiGateway:
    def __init__(self):
        self.calls = []
        self.apis = {}
        self.integrations = {}
        self.routes = {}
        self.stages = {}

    def __getattr__(self, operation):
        def call(**arguments):
            self.calls.append((operation, arguments))
            api_id = arguments.get("ApiId")
            if operation == "get_apis":
                return {"Items": list(self.apis.values())}
            if operation == "get_api":
                if api_id not in self.apis:
                    raise FakeAwsError("NotFoundException")
                return dict(self.apis[api_id])
            if operation == "create_api":
                api_id = "api" + str(len(self.apis) + 1)
                detail = {**arguments, "ApiId": api_id,
                          "ApiEndpoint": f"https://{api_id}.execute-api.us-east-1.amazonaws.com"}
                self.apis[api_id] = detail
                return dict(detail)
            if operation == "update_api":
                self.apis[api_id].update(arguments)
                return dict(self.apis[api_id])
            if operation == "delete_api":
                del self.apis[api_id]
                return {}
            stores = {"get_integrations": self.integrations, "get_routes": self.routes, "get_stages": self.stages}
            if operation in stores:
                return {"Items": [item for item in stores[operation].values() if item["ApiId"] == api_id]}
            if operation == "create_integration":
                identifier = "i" + str(len(self.integrations) + 1)
                self.integrations[identifier] = {**arguments, "IntegrationId": identifier}
                return dict(self.integrations[identifier])
            if operation == "update_integration":
                self.integrations[arguments["IntegrationId"]].update(arguments)
                return dict(self.integrations[arguments["IntegrationId"]])
            if operation == "create_route":
                identifier = "r" + str(len(self.routes) + 1)
                self.routes[identifier] = {**arguments, "RouteId": identifier}
                return dict(self.routes[identifier])
            if operation == "update_route":
                self.routes[arguments["RouteId"]].update(arguments)
                return dict(self.routes[arguments["RouteId"]])
            if operation in {"create_stage", "update_stage"}:
                key = (api_id, arguments["StageName"])
                self.stages[key] = {**self.stages.get(key, {}), **arguments}
                return dict(self.stages[key])
            raise AssertionError(f"Unexpected API Gateway fake operation: {operation}")
        return call


class FakeDeploymentClient:
    """Small stateful API fake, deliberately containing no authentication code."""
    def __init__(self, service):
        self.service = service
        self.calls = []
        self.resources = {}
        self.tags = {}
        self.permissions = {}
        self.objects = {}
        self.url = None
        self.layer_count = 0

    def __getattr__(self, operation):
        def call(**arguments):
            self.calls.append((operation, arguments))
            if operation == "get_caller_identity":
                return {"Account": "123456789012"}
            if operation == "get_waiter":
                return FakeWaiter()
            if operation == "head_bucket":
                if not self.resources.get("bucket"):
                    raise FakeAwsError("404")
                return {}
            if operation == "create_bucket":
                self.resources["bucket"] = arguments["Bucket"]
                return {}
            if operation == "put_bucket_tagging":
                self.tags["bucket"] = arguments["Tagging"]["TagSet"]
                return {}
            if operation == "get_bucket_tagging":
                return {"TagSet": self.tags["bucket"]}
            if operation == "get_bucket_versioning":
                return {}
            if operation == "head_object":
                if arguments["Key"] not in self.objects:
                    raise FakeAwsError("404")
                return self.objects[arguments["Key"]]
            if operation == "put_object":
                self.objects[arguments["Key"]] = {"Metadata": arguments.get("Metadata", {})}
                return {}
            if operation == "describe_table":
                name = arguments["TableName"]
                if name not in self.resources:
                    raise FakeAwsError("ResourceNotFoundException")
                return {"Table": self.resources[name]}
            if operation == "create_table":
                name = arguments["TableName"]
                arn = f"arn:aws:dynamodb:us-east-1:123456789012:table/{name}"
                self.resources[name] = {"TableArn": arn, "KeySchema": arguments["KeySchema"]}
                self.tags[arn] = arguments["Tags"]
                return {}
            if operation == "describe_time_to_live":
                return {"TimeToLiveDescription": {"TimeToLiveStatus": "ENABLED", "AttributeName": "expires_at"}}
            if operation == "get_role":
                if arguments["RoleName"] not in self.resources:
                    raise FakeAwsError("NoSuchEntity")
                return {"Role": self.resources[arguments["RoleName"]]}
            if operation == "create_role":
                name = arguments["RoleName"]
                self.resources[name] = {"Arn": f"arn:aws:iam::123456789012:role/{name}"}
                self.tags[name] = arguments["Tags"]
                return {"Role": self.resources[name]}
            if operation == "list_role_tags":
                return {"Tags": self.tags[arguments["RoleName"]]}
            if operation == "create_log_group":
                name = arguments["logGroupName"]
                arn = f"arn:aws:logs:us-east-1:123456789012:log-group:{name}"
                if arn in self.tags:
                    raise FakeAwsError("ResourceAlreadyExistsException")
                self.tags[arn] = arguments["tags"]
                return {}
            if operation in {"list_tags_for_resource", "list_tags_of_resource"}:
                arn = arguments.get("ResourceArn", arguments.get("resourceArn", arguments.get("ResourceARN")))
                return {"tags" if self.service == "logs" else "Tags": self.tags[arn]}
            if operation == "publish_layer_version":
                self.layer_count += 1
                name = arguments["LayerName"]
                arn = f"arn:aws:lambda:us-east-1:123456789012:layer:{name}:{self.layer_count}"
                self.resources[arn] = True
                return {"LayerVersionArn": arn, "Version": self.layer_count}
            if operation == "get_layer_version_by_arn":
                if arguments["Arn"] not in self.resources:
                    raise FakeAwsError("ResourceNotFoundException")
                return {}
            if operation == "get_function":
                name = arguments["FunctionName"]
                if name not in self.resources:
                    raise FakeAwsError("ResourceNotFoundException")
                return {"Configuration": self.resources[name]}
            if operation == "create_function":
                name = arguments["FunctionName"]
                arn = f"arn:aws:lambda:us-east-1:123456789012:function:{name}"
                self.resources[name] = {"FunctionArn": arn, "Architectures": arguments["Architectures"]}
                self.tags[arn] = arguments["Tags"]
                return {}
            if operation == "list_tags":
                return {"Tags": self.tags[arguments["Resource"]]}
            if operation == "get_function_url_config":
                if not self.url:
                    raise FakeAwsError("ResourceNotFoundException")
                return {"FunctionUrl": self.url}
            if operation == "create_function_url_config":
                self.url = "https://example.lambda-url.us-east-1.on.aws/"
                return {"FunctionUrl": self.url}
            if operation == "remove_permission":
                if arguments["StatementId"] not in self.permissions:
                    raise FakeAwsError("ResourceNotFoundException")
                del self.permissions[arguments["StatementId"]]
                return {}
            if operation == "add_permission":
                self.permissions[arguments["StatementId"]] = arguments
                return {}
            if operation == "describe_alarms":
                return {"MetricAlarms": [{"AlarmName": "x132-web-invocations"}]
                        if self.resources.get("alarm") else []}
            if operation == "put_metric_alarm":
                self.resources["alarm"] = True
                self.tags["arn:aws:cloudwatch:us-east-1:123456789012:alarm:x132-web-invocations"] = arguments["Tags"]
                return {}
            if operation in {"put_public_access_block", "put_bucket_encryption", "put_bucket_cors",
                             "put_bucket_lifecycle_configuration", "update_time_to_live", "put_role_policy",
                             "update_assume_role_policy", "put_retention_policy", "update_function_code",
                             "update_function_configuration", "put_function_concurrency", "update_function_url_config"}:
                return {}
            raise AssertionError(f"Unexpected fake API: {self.service}.{operation}")
        # Waiters accept their name positionally, unlike most boto3 operations.
        return (lambda name: FakeWaiter()) if operation == "get_waiter" else call


class FakeSession:
    def __init__(self):
        self.clients = {}

    def client(self, name, **kwargs):
        if name not in self.clients:
            self.clients[name] = FakeApiGateway() if name == "apigatewayv2" else FakeDeploymentClient(name)
        return self.clients[name]


class DeployTests(unittest.TestCase):
    def test_bench_bootstrap_creates_no_public_functions_and_preserves_model_keys(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            directory = Path(temporary)
            arm, x86, code = [directory / filename for filename in ("arm.zip", "x86.zip", "code.zip")]
            for path in (arm, x86, code):
                path.write_bytes(b"small fake artifact: " + path.name.encode())
            config = replace(deploy.DeployConfig(), state=directory / "state.json",
                             layer_arm64=arm, layer_x86=x86, bench_only=True)
            check = {"problems": [], "model_files": [], "sample_uploads": [],
                     "public_samples": {}, "code_files": []}
            session = FakeSession()
            with (mock.patch.object(deploy, "preflight", return_value=check),
                  mock.patch.object(deploy, "package_code", return_value=code)):
                result = deploy.execute_deploy(config, sdk_session=session)
            self.assertEqual(result["deployment_mode"], "bench_only")
            calls = session.clients["lambda"].calls
            creates = [args["FunctionName"] for operation, args in calls if operation == "create_function"]
            self.assertEqual(creates, ["x132-bench"])
            settings = next(args for operation, args in calls if operation == "create_function")
            self.assertEqual(settings["EphemeralStorage"]["Size"], 4096)
            self.assertEqual(settings["Timeout"], 900)
            self.assertFalse(any(operation == "create_function_url_config" for operation, _ in calls))
            self.assertNotIn("apigatewayv2", session.clients)
            self.assertNotIn("inspect", json.loads(config.state.read_text())["functions"])
            self.assertNotIn("x132/models/manifest.json", session.clients["s3"].objects)
            self.assertNotIn("x132/samples/manifest.json", session.clients["s3"].objects)

    def test_stateful_fake_deployment_is_idempotent_and_gateway_permission_is_scoped(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            directory = Path(temporary)
            arm, x86, code = [directory / filename for filename in ("arm.zip", "x86.zip", "code.zip")]
            for path in (arm, x86, code):
                path.write_bytes(b"small fake artifact: " + path.name.encode())
            config = replace(deploy.DeployConfig(), state=directory / "state.json",
                             layer_arm64=arm, layer_x86=x86)
            models = [directory / "metadata.json", *(directory / f"scale_{scale}_{kind}.yml.gz"
                      for scale in ("0.25", "0.5", "1") for kind in ("binary", "kind"))]
            for model in models:
                model.write_bytes(b"fake model " + model.name.encode())
            check = {"problems": [], "model_files": models, "sample_uploads": [],
                     "public_samples": {"version": 1, "samples": []}, "code_files": []}
            session = FakeSession()
            policy_file = directory / "policy.json"
            policy = {"initial_scale": .25, "positive_threshold": .6, "detection_threshold": .6}
            policy_file.write_text(json.dumps(policy))
            with (mock.patch.object(deploy, "preflight", return_value=check),
                  mock.patch.object(deploy, "package_code", return_value=code),
                  mock.patch.dict("os.environ", {"X132_POLICY_CONFIG_FILE": str(policy_file)}),
                  mock.patch.object(deploy.importlib, "import_module", side_effect=AssertionError("SDK"))):
                first = deploy.execute_deploy(config, sdk_session=session)
                second = deploy.execute_deploy(config, sdk_session=session)
            self.assertEqual(first["owned_resource_count"], second["owned_resource_count"])
            self.assertEqual(session.clients["lambda"].layer_count, 2)
            creates = [item for item in session.clients["lambda"].calls if item[0] == "create_function"]
            self.assertEqual(len(creates), 5)
            state = json.loads(config.state.read_text())
            self.assertEqual(state["model_prefix"], "x132/models/" + state["model_bundle_sha256"] + "/")
            self.assertEqual(len(state["model_bundle_sha256"]), 64)
            for _, arguments in creates:
                if arguments["FunctionName"] == "x132-bench":
                    self.assertEqual(arguments["EphemeralStorage"]["Size"], 4096)
                    self.assertEqual(arguments["Timeout"], 900)
                else:
                    self.assertEqual(arguments["EphemeralStorage"]["Size"], 1024)
                environment = arguments["Environment"]["Variables"]
                self.assertEqual(environment["X132_MODEL_SOURCE"], "s3")
                self.assertEqual(environment["X132_MODEL_PREFIX"], state["model_prefix"])
                self.assertEqual(environment["X132_INSPECT_FUNCTION"], "x132-inspect")
                self.assertEqual(json.loads(environment["X132_POLICY_CONFIG"]), policy)
                if arguments["FunctionName"] == "x132-web":
                    self.assertEqual(arguments["MemorySize"], 256)
                    self.assertEqual(arguments["Layers"], [])
                    self.assertEqual(arguments["Timeout"], 30)
                if arguments["FunctionName"] == "x132-inspect":
                    self.assertEqual(arguments["MemorySize"], 3008)
                    self.assertEqual(arguments["Timeout"], 25)
                    self.assertEqual(len(arguments["Layers"]), 1)
            self.assertIn(state["model_prefix"] + "manifest.json", session.clients["s3"].objects)
            permissions = session.clients["lambda"].permissions
            self.assertEqual(set(permissions), {"x132-http-api"})
            self.assertEqual(permissions["x132-http-api"]["Principal"], "apigateway.amazonaws.com")
            self.assertEqual(permissions["x132-http-api"]["SourceArn"],
                             "arn:aws:execute-api:us-east-1:123456789012:api1/*/*/*")
            self.assertEqual(set(session.clients), {"s3", "iam", "lambda", "dynamodb", "logs", "cloudwatch", "sts", "apigatewayv2"})
            self.assertFalse(any(operation == "put_function_concurrency" for operation, _ in session.clients["lambda"].calls))
            self.assertFalse(any(operation == "create_function_url_config" for operation, _ in session.clients["lambda"].calls))
            gateway = session.clients["apigatewayv2"]
            self.assertEqual(len(gateway.apis), 1)
            self.assertEqual(len(gateway.integrations), 1)
            self.assertEqual(len(gateway.routes), 6)
            self.assertEqual(state["http_api"]["endpoint"], first["api_endpoint"])

    def test_dry_run_has_no_sdk_import_state_write_or_aws_call(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            root = Path(temporary)
            state = root / "state.json"
            output = io.StringIO()
            with (mock.patch.object(deploy.importlib, "import_module", side_effect=AssertionError("SDK imported")),
                  mock.patch.object(deploy, "save_state", side_effect=AssertionError("state written")),
                  redirect_stdout(output)):
                status = deploy.main(["--dry-run", "--state", str(state),
                                      "--model-dir", str(root / "missing_models"),
                                      "--samples-manifest", str(root / "missing_samples.json"),
                                      "--layer-arm64", str(root / "missing_arm.zip"),
                                      "--layer-x86", str(root / "missing_x86.zip")])
            self.assertEqual(status, 0)
            plan = json.loads(output.getvalue())
            self.assertFalse(plan["will_contact_aws"])
            self.assertFalse(plan["will_write_state"])
            self.assertTrue(plan["prerequisite_problems"])
            self.assertFalse(state.exists())
            self.assertEqual(plan["functions"]["bench"]["timeout"], 900)
            self.assertEqual(plan["functions"]["bench"]["ephemeral_storage_mb"], 4096)
            self.assertEqual(plan["functions"]["tool_x86"]["architecture"], "x86_64")
            self.assertEqual(plan["functions"]["web"]["memory_mb"], 256)
            self.assertFalse(plan["functions"]["web"]["uses_layer"])
            self.assertEqual(plan["functions"]["inspect"]["memory_mb"], 3008)
            self.assertIsNone(plan["web_reserved_concurrency"])
            operations = {operation for group in plan["planned_calls"] for operation in group["operations"]}
            self.assertIn("CreateBucket", operations)
            self.assertIn("PutPublicAccessBlock", operations)
            self.assertIn("AddPermission", operations)
            self.assertIn("CreateApi", operations)
            self.assertNotIn("CreateFunctionUrlConfig", operations)
            self.assertNotIn("PutFunctionConcurrency", operations)

    def test_bench_only_plan_has_no_public_api_calls(self):
        config = replace(deploy.DeployConfig(), bench_only=True)
        operations = {operation for group in deploy.planned_api_calls(config) for operation in group["operations"]}
        self.assertIn("CreateFunction", operations)
        self.assertNotIn("CreateFunctionUrlConfig", operations)
        self.assertNotIn("AddPermission", operations)
        self.assertNotIn("PutMetricAlarm", operations)

    def test_real_preflight_failure_never_constructs_sdk_or_state(self):
        with (mock.patch.object(deploy, "preflight", return_value={"problems": ["models missing"]}),
              mock.patch.object(deploy.importlib, "import_module") as sdk,
              mock.patch.object(deploy, "save_state") as save):
            with self.assertRaisesRegex(ValueError, "models missing"):
                deploy.execute_deploy(deploy.DeployConfig())
            sdk.assert_not_called()
            save.assert_not_called()

    def test_runtime_policies_are_project_scoped_and_separate(self):
        for role in ("tool", "web", "bench", "inspect"):
            policy = deploy.role_policy(role, "x132-example", "us-east-1", "123456789012")
            for statement in policy["Statement"]:
                self.assertNotIn("*", statement["Action"])
                self.assertNotIn("*", statement["Resource"])
            encoded = json.dumps(policy)
            if role == "tool":
                self.assertNotIn("dynamodb:", encoded)
                self.assertNotIn("s3:PutObject", encoded)
                self.assertNotIn("lambda:InvokeFunction", encoded)
            if role == "web":
                self.assertIn("x132/inputs/*", encoded)
                self.assertIn("dynamodb:UpdateItem", encoded)
                self.assertNotIn("x132-tool-x86", encoded)
                self.assertNotIn("s3:ListBucket", encoded)
                invocation = [item for item in policy["Statement"] if "lambda:InvokeFunction" in item["Action"]]
                self.assertEqual(invocation[0]["Resource"], ["arn:aws:lambda:us-east-1:123456789012:function:x132-inspect"])
            if role == "inspect":
                self.assertIn("dynamodb:GetItem", encoded)
                self.assertIn("dynamodb:PutItem", encoded)
                self.assertNotIn("dynamodb:UpdateItem", encoded)
                self.assertNotIn("x132-reviews", encoded)
                self.assertNotIn("x132-ratelimit", encoded)
                self.assertNotIn("x132/uploads/", encoded)
                self.assertNotIn("lambda:InvokeFunction", encoded)

    def test_ownership_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            deploy.assert_owned({"x132:project": "x132", "x132:deployment": "somebody-else"},
                                {"suffix": "unit-test"}, "resource")

    def test_samples_public_manifest_does_not_expose_local_paths(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            root = Path(temporary)
            (root / "sample.jpg").write_bytes(b"test-only-placeholder")
            source = root / "samples.json"
            source.write_text(json.dumps({"samples": [{"id": "surface", "title": "Concrete",
                "key": "x132/samples/surface.jpg", "local_path": "sample.jpg",
                "width": 4096, "height": 4096, "private_note": "never publish"}]}))
            uploads, public = deploy._samples(source)
            self.assertEqual(len(uploads), 1)
            self.assertNotIn("local_path", json.dumps(public))
            self.assertNotIn("private_note", json.dumps(public))

    def test_model_bundle_requires_six_distinct_files(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            directory = Path(temporary)
            models = {}
            for scale in ("0.25", "0.5", "1"):
                models[scale] = {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                for filename in models[scale].values():
                    (directory / filename).write_bytes(b"test-only-placeholder")
            (directory / "metadata.json").write_text(json.dumps({"models": models, "calibration": "sampled"}))
            self.assertEqual(len(deploy._model_files(directory)), 7)
            models["1"]["kind"] = "../outside.yml"
            (directory / "metadata.json").write_text(json.dumps({"models": models, "calibration": "sampled"}))
            with self.assertRaises(ValueError):
                deploy._model_files(directory)

    def test_code_selection_excludes_state_infrastructure_and_artifacts(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            root = Path(temporary)
            for filename in ("aws/state.json", "aws/deploy.py", "aws/runtime.py", "aws/private_local.py",
                             "aws/static/index.html", "aws/handlers/web.py", "inspector/agent.py",
                             "inspector/__pycache__/old.py", "scripts/eval.py"):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder")
            selected = {path.relative_to(root).as_posix() for path in deploy.code_files(root)}
            self.assertEqual(selected, {"aws/runtime.py", "aws/static/index.html", "aws/handlers/web.py",
                                        "inspector/agent.py", "scripts/eval.py"})

    def test_teardown_dry_run_no_sdk_or_writes(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            state = Path(temporary) / "state.json"
            with mock.patch.object(teardown.importlib, "import_module", side_effect=AssertionError("SDK")):
                plan = teardown.teardown_plan(state)
            self.assertEqual(plan["resources"], [])
            self.assertFalse(plan["will_contact_aws"])
            self.assertFalse(state.exists())

    def test_bucket_deletion_preserves_foreign_prefix(self):
        s3 = FakeS3(foreign=True)
        with self.assertRaisesRegex(RuntimeError, "retained objects"):
            teardown.delete_project_objects(s3, "x132-example")
        self.assertEqual(s3.list_arguments["Prefix"], "x132/")
        self.assertEqual(s3.deleted, ["x132/outputs/a.json", "x132/uploads/b.jpg"])
        self.assertNotIn("other-project/keep.txt", s3.deleted)

    def test_teardown_rejects_unrecorded_resource_names_before_sdk(self):
        with self.assertRaises(ValueError):
            teardown._validate_resource({"type": "function", "name": "unrelated-function"},
                                        {"bucket": "x132-example"})


if __name__ == "__main__":
    unittest.main()
