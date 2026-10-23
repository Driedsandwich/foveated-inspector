"""Remove only resources recorded and owned by this deployment's state.

--dry-run reads local state only. --execute is deliberately required for any
AWS call or deletion. Foreign S3 keys and foreign ownership tags stop deletion;
no other bucket, prefix, role, function or table is cleaned up opportunistically.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
from aws.deploy import (FUNCTIONS, HTTP_API_NAME, PREFIX, TABLES, assert_owned, error_code,
                        load_state, save_state, _missing)


def teardown_plan(path: Path) -> dict:
    state = load_state(path)
    return {"dry_run": True, "will_contact_aws": False, "will_write_state": False,
            "region": state["region"] if state else None,
            "resources": [{"type": item["type"], "name": item["name"]}
                          for item in state.get("resources", [])] if state else [],
            "s3_delete_prefix": PREFIX,
            "guards": ["matching project and deployment ownership tags",
                       "state-recorded resources only", "foreign bucket objects preserved",
                       "versioned buckets require separate manual review"]}


def delete_project_objects(s3, bucket: str) -> None:
    """Delete this prefix only, then refuse bucket removal if any key remains."""
    if s3.get_bucket_versioning(Bucket=bucket).get("Status") in {"Enabled", "Suspended"}:
        raise RuntimeError("Refusing automatic deletion of a bucket with version history")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if any(not item["Key"].startswith(PREFIX) for item in keys):
            raise RuntimeError("S3 returned a key outside the owned prefix")
        for offset in range(0, len(keys), 1000):
            response = s3.delete_objects(Bucket=bucket,
                                         Delete={"Objects": keys[offset:offset + 1000], "Quiet": True})
            if response.get("Errors"):
                raise RuntimeError("Some project S3 objects could not be removed")
    if s3.list_objects_v2(Bucket=bucket, MaxKeys=1).get("Contents"):
        raise RuntimeError("Bucket contains retained objects; it will not be deleted")


def _validate_resource(resource: dict, state: dict) -> None:
    kind, name = resource["type"], resource["name"]
    allowed = {"function": set(FUNCTIONS.values()), "table": set(TABLES.values()),
               "role": {"x132-tool-role", "x132-web-role", "x132-bench-role", "x132-inspect-role"},
               "log_group": {"/aws/lambda/" + function for function in FUNCTIONS.values()},
               "alarm": {"x132-web-invocations"}, "bucket": {state["bucket"]}}
    if kind == "http_api":
        api_id = resource.get("id", "")
        if (name != HTTP_API_NAME or not isinstance(api_id, str)
                or not re.fullmatch(r"[a-z0-9]{1,32}", api_id)
                or resource.get("arn") != f"arn:aws:apigateway:{state['region']}::/apis/{api_id}"
                or state.get("http_api", {}).get("id", api_id) != api_id):
            raise ValueError("Invalid recorded HTTP API ownership")
    elif kind == "layer":
        layer = resource.get("layer_name", "")
        version = resource.get("version")
        arn = resource.get("arn", "")
        if (layer not in {"x132-opencv-arm64", "x132-opencv-x86_64"}
                or type(version) is not int or version < 1
                or name != f"{layer}:{version}"
                or not re.fullmatch(r"arn:aws:lambda:" + re.escape(state["region"])
                                    + r":[0-9]{12}:layer:" + re.escape(layer) + ":" + str(version), arn)):
            raise ValueError("Invalid recorded layer ownership")
    elif kind not in allowed or name not in allowed[kind]:
        raise ValueError("State contains a resource outside the fixed project names")


def _delete(resource: dict, state: dict, clients: dict) -> None:
    kind, name = resource["type"], resource["name"]
    if kind == "http_api":
        client = clients["apigatewayv2"]
        detail = client.get_api(ApiId=resource["id"])
        if detail.get("Name") != HTTP_API_NAME or detail.get("ProtocolType") != "HTTP":
            raise RuntimeError("Recorded HTTP API has an incompatible name or protocol")
        assert_owned(detail.get("Tags", {}), state, name)
        client.delete_api(ApiId=resource["id"])
    elif kind == "function":
        client = clients["lambda"]
        arn = client.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
        assert_owned(client.list_tags(Resource=arn).get("Tags", {}), state, name)
        if name == FUNCTIONS["web"]:
            try:
                client.delete_function_url_config(FunctionName=name)
            except Exception as error:
                if not _missing(error):
                    raise
        client.delete_function(FunctionName=name)
    elif kind == "table":
        client = clients["dynamodb"]
        arn = client.describe_table(TableName=name)["Table"]["TableArn"]
        tags = client.list_tags_of_resource(ResourceArn=arn).get("Tags", [])
        assert_owned({item["Key"]: item["Value"] for item in tags}, state, name)
        client.delete_table(TableName=name)
    elif kind == "role":
        client = clients["iam"]
        tags = client.list_role_tags(RoleName=name).get("Tags", [])
        assert_owned({item["Key"]: item["Value"] for item in tags}, state, name)
        try:
            client.delete_role_policy(RoleName=name, PolicyName="x132-runtime")
        except Exception as error:
            if not _missing(error):
                raise
        # Do not detach policies another person attached to this role.
        client.delete_role(RoleName=name)
    elif kind == "log_group":
        client = clients["logs"]
        assert_owned(client.list_tags_for_resource(resourceArn=resource["arn"]).get("tags", {}), state, name)
        client.delete_log_group(logGroupName=name)
    elif kind == "alarm":
        client = clients["cloudwatch"]
        tags = client.list_tags_for_resource(ResourceARN=resource["arn"]).get("Tags", [])
        assert_owned({item["Key"]: item["Value"] for item in tags}, state, name)
        client.delete_alarms(AlarmNames=[name])
    elif kind == "layer":
        clients["lambda"].delete_layer_version(LayerName=resource["layer_name"], VersionNumber=resource["version"])
    elif kind == "bucket":
        client = clients["s3"]
        tags = client.get_bucket_tagging(Bucket=name).get("TagSet", [])
        assert_owned({item["Key"]: item["Value"] for item in tags}, state, name)
        delete_project_objects(client, name)
        client.delete_bucket(Bucket=name)


def execute_teardown(path: Path, sdk_session=None) -> dict:
    state = load_state(path)
    if not state or not state["resources"]:
        return {"removed": 0, "remaining": 0, "status": "nothing recorded"}
    # Validate the entire requested set before making any AWS call.
    for resource in state["resources"]:
        _validate_resource(resource, state)
    if sdk_session is None:
        sdk_session = importlib.import_module("boto3").Session(region_name=state["region"])
    services = ["s3", "iam", "lambda", "dynamodb", "logs", "cloudwatch"]
    if any(resource["type"] == "http_api" for resource in state["resources"]):
        services.append("apigatewayv2")
    clients = {name: sdk_session.client(name, region_name=state["region"]) for name in services}
    order = {"http_api": -1, "alarm": 0, "function": 1, "layer": 2, "log_group": 3,
             "table": 4, "role": 5, "bucket": 6}
    resources = sorted(state["resources"], key=lambda resource: order[resource["type"]])
    removed = 0
    for resource in resources:
        try:
            _delete(resource, state, clients)
        except Exception as error:
            if not _missing(error):
                raise
        state["resources"].remove(resource)
        if resource["type"] == "http_api":
            state.pop("http_api", None)
        if resource["type"] == "function" and resource["name"] == FUNCTIONS["web"]:
            state.pop("function_url", None)
        save_state(path, state)
        removed += 1
    state["status"] = "torn_down"
    save_state(path, state)
    return {"removed": removed, "remaining": len(state["resources"]), "status": "torn_down"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--state", type=Path, default=APP_ROOT / "aws/state.json")
    arguments = parser.parse_args(argv)
    try:
        result = teardown_plan(arguments.state) if arguments.dry_run else execute_teardown(arguments.state)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        detail = str(error) if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)) else error_code(error)
        print(f"Teardown stopped ({type(error).__name__}): {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
