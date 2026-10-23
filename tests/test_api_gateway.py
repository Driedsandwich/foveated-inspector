"""Offline HTTP API, legacy URL and concurrency guard regression checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aws import deploy, teardown
from test_deploy import FakeApiGateway, FakeDeploymentClient, FakeSession


def owned_state():
    return {"version": 1, "project": "x132", "region": "us-east-1", "bucket": "x132-unit",
            "prefix": "x132/", "suffix": "unit", "resources": [],
            "functions": {key: value for key, value in deploy.FUNCTIONS.items() if key != "inspect"},
            "tables": dict(deploy.TABLES)}


def legacy_resources():
    region, account = "us-east-1", "123456789012"
    items = [{"type": "bucket", "name": "x132-unit"}]
    items += [{"type": "table", "name": name} for name in deploy.TABLES.values()]
    items += [{"type": "role", "name": f"x132-{role}-role"} for role in ("tool", "web", "bench")]
    for key in ("tool_arm64", "tool_x86", "web", "bench"):
        name = "/aws/lambda/" + deploy.FUNCTIONS[key]
        items.append({"type": "log_group", "name": name,
                      "arn": f"arn:aws:logs:{region}:{account}:log-group:{name}"})
    for version, arch in enumerate(("arm64", "x86_64"), 1):
        name = f"x132-opencv-{arch}"
        items.append({"type": "layer", "name": f"{name}:{version}", "layer_name": name,
                      "version": version, "arn": f"arn:aws:lambda:{region}:{account}:layer:{name}:{version}"})
    items.append({"type": "function", "name": "x132-bench"})
    return items


class ApiGatewayTests(unittest.TestCase):
    def test_api_is_idempotent_throttled_and_permission_is_scoped(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            config = replace(deploy.DeployConfig(), state=Path(temporary) / "state.json")
            state, api, lam = owned_state(), FakeApiGateway(), FakeDeploymentClient("lambda")
            first = deploy.configure_http_api(api, lam, state, config, "123456789012")
            second = deploy.configure_http_api(api, lam, state, config, "123456789012")
            self.assertEqual(first, second)
            self.assertEqual(len(api.apis), 1)
            self.assertEqual(len(api.integrations), 1)
            self.assertEqual({route["RouteKey"] for route in api.routes.values()}, set(deploy.HTTP_ROUTES))
            self.assertEqual(len(api.routes), 6)
            integration = next(iter(api.integrations.values()))
            self.assertEqual(integration["IntegrationType"], "AWS_PROXY")
            self.assertEqual(integration["PayloadFormatVersion"], "2.0")
            self.assertEqual(integration["TimeoutInMillis"], 30000)
            stage = api.stages[("api1", "$default")]
            self.assertTrue(stage["AutoDeploy"])
            self.assertEqual(stage["DefaultRouteSettings"], {"ThrottlingRateLimit": 1.0, "ThrottlingBurstLimit": 2})
            cors = api.apis["api1"]["CorsConfiguration"]
            self.assertEqual(cors["AllowOrigins"], [first])
            self.assertFalse(cors["AllowCredentials"])
            self.assertEqual(set(cors["AllowHeaders"]), set(deploy.TOKEN_HEADERS))
            permission = lam.permissions["x132-http-api"]
            self.assertEqual(permission["Principal"], "apigateway.amazonaws.com")
            self.assertEqual(permission["SourceArn"], "arn:aws:execute-api:us-east-1:123456789012:api1/*/*/*")
            saved = json.loads(config.state.read_text())
            self.assertEqual(saved["version"], 1)
            self.assertEqual(saved["http_api"]["id"], "api1")
            self.assertEqual(saved["http_api"]["stage"], "$default")
            self.assertEqual(saved["resources"], [{"type": "http_api", "name": "x132-public", "id": "api1",
                                                     "arn": "arn:aws:apigateway:us-east-1::/apis/api1"}])

    def test_created_api_is_recorded_before_failing_integration_and_can_resume(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            config = replace(deploy.DeployConfig(), state=Path(temporary) / "state.json")
            state, api, lam = owned_state(), FakeApiGateway(), FakeDeploymentClient("lambda")
            with mock.patch.object(api, "create_integration", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    deploy.configure_http_api(api, lam, state, config, "123456789012")
            saved = json.loads(config.state.read_text())
            self.assertEqual(saved["http_api"]["id"], "api1")
            self.assertEqual(saved["resources"][0]["type"], "http_api")
            deploy.configure_http_api(api, lam, saved, config, "123456789012")
            self.assertEqual(len(api.apis), 1)
            self.assertEqual(len(api.routes), 6)

    def test_foreign_api_ownership_is_never_adopted(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            config = replace(deploy.DeployConfig(), state=Path(temporary) / "state.json")
            api = FakeApiGateway()
            api.create_api(Name="x132-public", ProtocolType="HTTP", Tags={"x132:deployment": "foreign"})
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                deploy.configure_http_api(api, FakeDeploymentClient("lambda"), owned_state(), config, "123456789012")
            self.assertFalse(any(operation == "update_api" for operation, _ in api.calls))
            self.assertFalse(config.state.exists())

    def test_partial_legacy_deployment_without_url_field_is_closed(self):
        state, lam = owned_state(), FakeDeploymentClient("lambda")
        state["resources"] = [{"type": "function", "name": "x132-web"}]
        arn = "arn:aws:lambda:us-east-1:123456789012:function:x132-web"
        lam.resources["x132-web"] = {"FunctionArn": arn}
        lam.tags[arn] = deploy.ownership_tags(state)
        lam.url = "https://legacy.example.lambda-url.us-east-1.on.aws/"
        lam.permissions = {"x132-public-url": {}, "x132-public-via-url": {}, "keep-unrelated": {}}
        self.assertNotIn("function_url", state)
        self.assertTrue(deploy.disable_legacy_public_url(lam, state))
        updates = [args for operation, args in lam.calls if operation == "update_function_url_config"]
        self.assertEqual(updates, [{"FunctionName": "x132-web", "AuthType": "AWS_IAM"}])
        self.assertEqual(set(lam.permissions), {"keep-unrelated"})
        self.assertTrue(state["legacy_function_url_disabled"])

    def test_legacy_url_is_not_touched_for_untracked_or_foreign_function(self):
        state, lam = owned_state(), FakeDeploymentClient("lambda")
        self.assertFalse(deploy.disable_legacy_public_url(lam, state))
        self.assertEqual(lam.calls, [])
        state["resources"] = [{"type": "function", "name": "x132-web"}]
        arn = "arn:aws:lambda:us-east-1:123456789012:function:x132-web"
        lam.resources["x132-web"] = {"FunctionArn": arn}
        lam.tags[arn] = {"x132:project": "x132", "x132:deployment": "foreign"}
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            deploy.disable_legacy_public_url(lam, state)
        self.assertFalse(any(operation == "get_function_url_config" for operation, _ in lam.calls))

    def test_concurrency_is_off_by_default_and_obeys_exact_headroom_guard(self):
        lam = mock.Mock()
        self.assertFalse(deploy.maybe_reserve_web_concurrency(lam, deploy.DeployConfig())["requested"])
        self.assertEqual(lam.mock_calls, [])
        config = replace(deploy.DeployConfig(), reserve_web_concurrency=True, web_concurrency=2)
        lam.get_account_settings.return_value = {"AccountLimit": {"UnreservedConcurrentExecutions": 10}}
        lam.get_function_concurrency.return_value = {}
        skipped = deploy.maybe_reserve_web_concurrency(lam, config)
        self.assertFalse(skipped["applied"])
        self.assertIn("ten unreserved", skipped["reason"])
        lam.put_function_concurrency.assert_not_called()
        lam.get_account_settings.return_value = {"AccountLimit": {"UnreservedConcurrentExecutions": 12}}
        self.assertTrue(deploy.maybe_reserve_web_concurrency(lam, config)["applied"])
        lam.put_function_concurrency.assert_called_once_with(FunctionName="x132-web", ReservedConcurrentExecutions=2)

    def test_memory_above_new_account_limit_stops_before_sdk(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            root = Path(temporary)
            config = replace(deploy.DeployConfig(), state=root / "state.json", model_dir=root / "models",
                             samples_manifest=root / "samples.json", layer_arm64=root / "arm.zip", layer_x86=root / "x86.zip")
            with mock.patch.object(deploy, "BENCH_MEMORY_MB", 5000):
                check = deploy.preflight(config)
            self.assertTrue(any("128..3008" in problem for problem in check["problems"]))
            with (mock.patch.object(deploy, "preflight", return_value=check),
                  mock.patch.object(deploy.importlib, "import_module") as sdk):
                with self.assertRaises(ValueError):
                    deploy.execute_deploy(config)
                sdk.assert_not_called()

    def test_legacy_fourteen_resources_remain_compatible_with_bench_only(self):
        with tempfile.TemporaryDirectory(dir=deploy.APP_ROOT) as temporary:
            root = Path(temporary)
            state = owned_state()
            state["resources"] = legacy_resources()
            self.assertEqual(len(state["resources"]), 14)
            for resource in state["resources"]:
                teardown._validate_resource(resource, state)
            path = root / "state.json"
            path.write_text(json.dumps(state))
            artifacts = [root / name for name in ("arm.zip", "x86.zip", "code.zip")]
            for artifact in artifacts:
                artifact.write_bytes(artifact.name.encode())
            config = replace(deploy.DeployConfig(), state=path, bench_only=True,
                             layer_arm64=artifacts[0], layer_x86=artifacts[1])
            check = {"problems": [], "model_files": [], "sample_uploads": [], "public_samples": {}, "code_files": []}
            session = FakeSession()
            with (mock.patch.object(deploy, "preflight", return_value=check),
                  mock.patch.object(deploy, "package_code", return_value=artifacts[2])):
                deploy.execute_deploy(config, sdk_session=session)
            updated = json.loads(path.read_text())
            self.assertEqual(len(updated["resources"]), 14)
            self.assertEqual(updated["functions"], state["functions"])
            self.assertNotIn("http_api", updated)
            self.assertNotIn("apigatewayv2", session.clients)
            self.assertFalse(any(item["name"] == "x132-inspect" for item in updated["resources"]))

    def test_teardown_removes_api_before_functions(self):
        state = owned_state()
        api = {"type": "http_api", "name": "x132-public", "id": "api1",
               "arn": "arn:aws:apigateway:us-east-1::/apis/api1"}
        state["http_api"] = {"id": "api1"}
        state["resources"] = [{"type": "function", "name": "x132-web"}, api]
        removed = []
        with (mock.patch.object(teardown, "load_state", return_value=state),
              mock.patch.object(teardown, "save_state"),
              mock.patch.object(teardown, "_delete", side_effect=lambda resource, *_: removed.append(resource["type"]))):
            result = teardown.execute_teardown(deploy.APP_ROOT / "unused-test-state.json", sdk_session=FakeSession())
        self.assertEqual(removed, ["http_api", "function"])
        self.assertEqual(result["remaining"], 0)
        self.assertNotIn("http_api", state)


if __name__ == "__main__":
    unittest.main()
