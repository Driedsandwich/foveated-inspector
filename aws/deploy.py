"""Explicit, state-owned AWS deployment. --dry-run never imports an AWS SDK.

This module has not been validated against a live account. Account eligibility,
credits, IAM permissions and Lambda concurrency quotas must be checked by the
operator; a plan does not promise free execution or reserve a spending limit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import time
import zipfile

# New AWS accounts cap Lambda memory at 3008 MB (measured 2026-10-24: ValidationException above 3008).
BENCH_MEMORY_MB = int(os.environ.get("X132_BENCH_MEMORY_MB", "3008"))


APP_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "x132/"
PROJECT = "x132"
MAX_UNZIPPED_BYTES = 250 * 1024 * 1024
MAX_MODELS_BYTES = 32 * 1024 * 1024
FUNCTIONS = {"tool_arm64": "x132-tool-arm64", "tool_x86": "x132-tool-x86",
             "web": "x132-web", "bench": "x132-bench", "inspect": "x132-inspect"}
TABLES = {"inspections": "x132-inspections", "reviews": "x132-reviews",
          "ratelimit": "x132-ratelimit"}
HANDLERS = {"tool_arm64": "aws.handlers.tool.handler",
            "tool_x86": "aws.handlers.tool.handler", "web": "aws.handlers.web.handler",
            "bench": "aws.handlers.bench.handler", "inspect": "aws.handlers.inspect.handler"}
HTTP_API_NAME = "x132-public"
HTTP_ROUTES = ("GET /", "GET /api/samples", "POST /api/upload-url", "POST /api/inspect",
               "GET /api/inspection/{id}", "POST /api/review")
TOKEN_HEADERS = ["Content-Type", "X-Inspection-Token", "X-Upload-Token"]


@dataclass(frozen=True)
class DeployConfig:
    region: str = "us-east-1"
    state: Path = APP_ROOT / "aws/state.json"
    samples_manifest: Path = APP_ROOT / "aws/samples.example.json"
    model_dir: Path = APP_ROOT / "models"
    layer_arm64: Path = APP_ROOT / "build/layer.zip"
    layer_x86: Path = APP_ROOT / "build/layer_x86_64.zip"
    web_concurrency: int = 2
    invocation_alarm_threshold: int = 60
    bench_only: bool = False
    reserve_web_concurrency: bool = False
    api_rate: float = 1.0
    api_burst: int = 2


def function_limits(name: str) -> dict:
    return {"architecture": "x86_64" if name == "tool_x86" else "arm64",
            "timeout": 900 if name == "bench" else 25 if name == "inspect" else 30,
            "memory_mb": BENCH_MEMORY_MB if name == "bench" else 3008 if name == "inspect"
            else 256 if name == "web" else 2048,
            "ephemeral_storage_mb": 4096 if name == "bench" else 1024,
            "uses_layer": name != "web"}


def state_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_relative_to(APP_ROOT):
        raise ValueError("State and generated deployment files must remain inside app/")
    return result


def load_state(path: str | Path) -> dict | None:
    location = state_path(path)
    if not location.exists():
        return None
    state = json.loads(location.read_text(encoding="utf-8"))
    if (state.get("version") != 1 or state.get("project") != PROJECT
            or state.get("prefix") != PREFIX or not state.get("suffix")
            or not isinstance(state.get("resources"), list)):
        raise ValueError("State is not an owned x132 deployment")
    return state


def save_state(path: str | Path, state: dict) -> None:
    location = state_path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    temporary = location.with_suffix(location.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(location)


def error_code(error: Exception) -> str:
    return str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))


def _missing(error: Exception) -> bool:
    return error_code(error) in {"404", "NoSuchBucket", "NoSuchEntity", "ResourceNotFoundException",
                                 "ResourceNotFound", "NoSuchKey", "NotFoundException"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(directory: Path) -> list[Path]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Missing trained models/metadata.json; deployment will not invent models")
    if metadata_path.stat().st_size > 256 * 1024:
        raise ValueError("Model metadata exceeds 256 KiB")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("calibration") != "sampled":
        raise ValueError("Production model metadata must use calibration='sampled'")
    expected = {scale: {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                for scale in ("0.25", "0.5", "1")}
    if metadata.get("models") != expected:
        raise ValueError("Production models must use the seven canonical runtime filenames")
    names = []
    for scale in ("0.25", "0.5", "1"):
        for kind in ("binary", "kind"):
            name = metadata.get("models", {}).get(scale, {}).get(kind)
            if not isinstance(name, str) or Path(name).name != name or not name:
                raise ValueError("Model metadata must name six local, distinct forest files")
            names.append(name)
    if len(set(names)) != 6:
        raise ValueError("The three scales must reference six distinct forest files")
    files = [metadata_path, *(directory / name for name in names)]
    if any(not path.is_file() for path in files):
        raise ValueError("One or more trained forest files are missing")
    if any(not 0 < path.stat().st_size <= 8 * 1024 * 1024 for path in files):
        raise ValueError("Each production model file must be nonempty and at most 8 MiB")
    if sum(path.stat().st_size for path in files) > MAX_MODELS_BYTES:
        raise ValueError("The metadata and six model files exceed the 32 MiB upload limit")
    return files


def _samples(path: Path) -> tuple[list[tuple[Path, str]], dict]:
    if not path.is_file():
        raise ValueError("Missing prepared samples manifest")
    document = json.loads(path.read_text(encoding="utf-8"))
    samples = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(samples, list) or not samples:
        raise ValueError("Samples manifest requires a nonempty samples list")
    uploads, public, identifiers, keys = [], [], set(), set()
    for sample in samples:
        identifier = sample.get("id", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier):
            raise ValueError("Invalid sample id")
        if identifier in identifiers:
            raise ValueError("Duplicate sample id")
        identifiers.add(identifier)
        source = path.parent / sample.get("local_path", "")
        key = sample.get("key", "")
        if (not isinstance(key, str) or not key.startswith(PREFIX + "samples/")
                or ".." in key.split("/") or key in keys
                or Path(key).suffix.lower() not in (".jpg", ".jpeg", ".png")
                or not source.is_file()):
            raise ValueError("Each sample needs a unique project sample key and existing local image")
        keys.add(key)
        uploads.append((source, key))
        record = {field: sample[field] for field in ("id", "title", "key", "width", "height")
                  if field in sample}
        thumbnail = sample.get("thumbnail_path")
        thumbnail_key = sample.get("thumbnail_key")
        if thumbnail and thumbnail_key:
            thumbnail_path = path.parent / thumbnail
            if (not thumbnail_path.is_file() or not isinstance(thumbnail_key, str)
                    or not thumbnail_key.startswith(PREFIX + "samples/")
                    or ".." in thumbnail_key.split("/") or thumbnail_key in keys):
                raise ValueError("Invalid prepared thumbnail path or key")
            keys.add(thumbnail_key)
            uploads.append((thumbnail_path, thumbnail_key))
            record["thumbnail_key"] = thumbnail_key
        # No image processing here: an absent prepared thumbnail uses the image.
        public.append(record)
    return uploads, {"version": 1, "samples": public}


def code_files(root: Path = APP_ROOT) -> list[Path]:
    files = []
    excluded = {"deploy.py", "teardown.py", "run_bench.py"}
    for directory in ("inspector", "aws", "scripts"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if ("__pycache__" in relative.parts
                    or any(part in {"artifacts", "build", ".cache", "state"} for part in relative.parts)):
                continue
            if directory == "aws" and path.name in excluded:
                continue
            if directory == "aws" and path.suffix == ".py":
                runtime_files = {"__init__.py", "runtime.py", "image_validation.py", "security.py"}
                if not (relative.parts[1:2] == ("handlers",)
                        or len(relative.parts) == 2 and path.name in runtime_files):
                    continue
            if path.suffix == ".py" or (relative.parts[:2] == ("aws", "static")
                                         and path.suffix in {".html", ".js", ".css", ".svg"}):
                files.append(path)
    return sorted(files)


def preflight(config: DeployConfig) -> dict:
    """Read local prerequisite metadata only; never build, decode images or use AWS."""
    problems = []
    warnings = []
    state_path(config.state)
    if config.web_concurrency < 1 or config.invocation_alarm_threshold < 1:
        problems.append("Concurrency and alarm threshold must be positive")
    if (not math.isfinite(config.api_rate) or config.api_rate <= 0
            or type(config.api_burst) is not int or config.api_burst < 1):
        problems.append("API rate must be finite and positive; burst must be a positive integer")
    if any(type(function_limits(name)["memory_mb"]) is not int
           or not 128 <= function_limits(name)["memory_mb"] <= 3008 for name in FUNCTIONS):
        problems.append("All Lambda memory settings, including X132_BENCH_MEMORY_MB, must be 128..3008 MiB")
    try:
        models = _model_files(config.model_dir)
    except (ValueError, OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
        models = []
        if config.bench_only:
            warnings.append("Bench bootstrap omits unavailable or incompatible model files; existing S3 models are preserved")
        else:
            problems.append(str(error))
    try:
        sample_uploads, public_samples = _samples(config.samples_manifest)
    except (ValueError, OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
        sample_uploads, public_samples = [], {}
        if config.bench_only and (not config.samples_manifest.exists()
                                  or "nonempty samples list" in str(error)):
            warnings.append("Bench bootstrap omits missing or empty sample catalog")
        else:
            problems.append(str(error))
    sources = code_files()
    code_bytes = sum(path.stat().st_size for path in sources)
    required = ["aws/handlers/tool.py", "aws/handlers/web.py", "aws/handlers/bench.py", "aws/handlers/inspect.py",
                "aws/runtime.py", "aws/image_validation.py", "aws/security.py",
                "aws/__init__.py", "aws/handlers/__init__.py", "aws/static/index.html"]
    if config.bench_only:
        required = [name for name in required if name not in
                    {"aws/handlers/tool.py", "aws/handlers/web.py", "aws/handlers/inspect.py", "aws/static/index.html"}]
    for filename in required:
        if not (APP_ROOT / filename).is_file():
            problems.append(f"Missing Lambda source {filename}")
    layer_bytes = {}
    for architecture, path in (("arm64", config.layer_arm64), ("x86_64", config.layer_x86)):
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len({member.filename for member in members}) != len(members):
                    raise ValueError("Layer has duplicate archive paths")
                if any(member.filename.startswith("/") or ".." in Path(member.filename).parts
                       for member in members):
                    raise ValueError("Layer has unsafe archive paths")
                size = sum(member.file_size for member in members)
                if not any(member.filename.startswith("python/cv2/") for member in members):
                    raise ValueError("Layer is missing python/cv2")
                if not any(member.filename.startswith("python/numpy/") for member in members):
                    raise ValueError("Layer is missing python/numpy")
                if size + code_bytes >= MAX_UNZIPPED_BYTES:
                    raise ValueError("Combined uncompressed code and layer must be below 250 MiB")
                layer_manifest = archive.getinfo("layer_manifest.json")
                if layer_manifest.file_size > 4096:
                    raise ValueError("Layer build manifest is too large")
                build = json.loads(archive.read(layer_manifest))
                if build.get("architecture") != architecture or build.get("python") != "3.12":
                    raise ValueError("Layer architecture or Python version does not match its target")
                layer_bytes[architecture] = size
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            problems.append(f"{architecture} layer prerequisite: {error}")
    return {"problems": problems, "warnings": warnings, "model_files": models, "sample_uploads": sample_uploads,
            "public_samples": public_samples, "code_files": sources,
            "code_bytes": code_bytes, "layer_bytes": layer_bytes}


def planned_api_calls(config: DeployConfig) -> list[dict]:
    """Describe conditional live calls without constructing an SDK object."""
    functions = [FUNCTIONS["bench"]] if config.bench_only else list(FUNCTIONS.values())
    calls = [
        {"service": "sts", "operations": ["GetCallerIdentity"],
         "condition": "execute only; account id is used for ARN scopes and never printed"},
        {"service": "s3", "operations": ["HeadBucket", "CreateBucket", "GetBucketTagging", "PutBucketTagging",
            "GetBucketVersioning", "PutPublicAccessBlock", "PutBucketEncryption", "PutBucketLifecycleConfiguration",
            "HeadObject", "PutObject"], "scope": "one state-owned bucket; x132/ objects only",
         "condition": "create missing owned bucket; upload available prepared files and deployment archives"},
        {"service": "iam", "operations": ["GetRole", "ListRoleTags", "CreateRole", "UpdateAssumeRolePolicy",
            "PutRolePolicy"], "scope": (["x132-bench-role"] if config.bench_only else
            ["x132-tool-role", "x132-web-role", "x132-bench-role", "x132-inspect-role"]),
         "condition": "create or update only matching deployment ownership tags"},
        {"service": "dynamodb", "operations": ["DescribeTable", "ListTagsOfResource", "CreateTable",
            "DescribeTimeToLive", "UpdateTimeToLive"], "scope": list(TABLES.values()),
         "condition": "create if missing; enable expires_at TTL if not already enabled"},
        {"service": "logs", "operations": ["CreateLogGroup", "ListTagsForResource", "PutRetentionPolicy"],
         "scope": ["/aws/lambda/" + name for name in functions],
         "condition": "create or verify ownership; retain logs for seven days"},
        {"service": "lambda", "operations": ["GetLayerVersionByArn", "PublishLayerVersion"],
         "scope": ["x132-opencv-arm64", "x132-opencv-x86_64"],
         "condition": "reuse state-recorded matching layer hash or publish one new version"},
        {"service": "lambda", "operations": ["GetFunction", "ListTags", "CreateFunction", "UpdateFunctionCode",
            "UpdateFunctionConfiguration", "GetFunctionConfiguration"], "scope": functions,
         "condition": "create or update matching owned functions; wait for active/updated state"},
    ]
    if not config.bench_only:
        calls.extend([
            {"service": "apigatewayv2", "operations": ["GetApis", "GetApi", "CreateApi", "UpdateApi",
                "GetIntegrations", "CreateIntegration", "UpdateIntegration", "GetRoutes", "CreateRoute",
                "UpdateRoute", "GetStages", "CreateStage", "UpdateStage"],
             "scope": [HTTP_API_NAME], "condition": "owned HTTP API, six explicit routes, throttled $default stage"},
            {"service": "lambda", "operations": ["GetFunctionUrlConfig", "UpdateFunctionUrlConfig", "RemovePermission", "AddPermission"],
             "scope": [FUNCTIONS["web"]],
             "condition": "disable a state-tracked legacy public URL; permit only API Gateway service with API-scoped SourceArn"},
            {"service": "s3", "operations": ["PutBucketCors"],
             "scope": "owned bucket; exact HTTP API origin only"},
            {"service": "cloudwatch", "operations": ["DescribeAlarms", "ListTagsForResource", "PutMetricAlarm"],
             "scope": ["x132-web-invocations"], "condition": "owned alarm only; no external actions"},
        ])
        if config.reserve_web_concurrency:
            calls.append({"service": "lambda", "operations": ["GetAccountSettings", "GetFunctionConcurrency",
                "PutFunctionConcurrency"], "scope": [FUNCTIONS["web"]],
                "condition": "explicit opt-in only; skip if fewer than ten unreserved executions would remain"})
    return calls


def deployment_plan(config: DeployConfig) -> dict:
    state = load_state(config.state)
    if state and state["region"] != config.region:
        raise ValueError("State region differs; use its original region or a different state file")
    check = preflight(config)
    selected_functions = {"bench": FUNCTIONS["bench"]} if config.bench_only else FUNCTIONS
    return {"dry_run": True, "region": config.region,
            "planned_calls": planned_api_calls(config),
            "services": ["S3", "IAM", "Lambda", "DynamoDB", "CloudWatchLogs", "CloudWatch", "STS"]
            + ([] if config.bench_only else ["API Gateway HTTP API"]),
            "bucket": state["bucket"] if state else "x132-<new-random-suffix>", "prefix": PREFIX,
            "model_prefix": (state.get("model_prefix", PREFIX + "models/") if config.bench_only and state
                             else PREFIX + "models/" if config.bench_only
                             else PREFIX + "models/<content-bundle-sha256>/"),
            "functions": {name: {"name": function, **function_limits(name)}
                          for name, function in selected_functions.items()},
            "bench_only": config.bench_only,
            "omitted_functions": [name for key, name in FUNCTIONS.items() if key not in selected_functions],
            "tables": {name: {"name": table, "hash_key": "id", "ttl": "expires_at",
                              "billing": "PROVISIONED", "read_units": 5, "write_units": 5}
                       for name, table in TABLES.items()},
            "public_entrypoint": ("No public endpoint created; existing web deployment is left unchanged"
                                  if config.bench_only else "throttled HTTP API only; no public Lambda Function URL"),
            "http_api": None if config.bench_only else {"name": HTTP_API_NAME, "routes": list(HTTP_ROUTES),
                "stage": "$default", "auto_deploy": True, "payload_format": "2.0", "timeout_ms": 30000,
                "rate_per_second": config.api_rate, "burst": config.api_burst,
                "throttle_scope": "per-route default; not a shared API-wide limit",
                "aggregate_rate_estimate_rps": len(HTTP_ROUTES) * config.api_rate,
                "aggregate_burst_estimate": len(HTTP_ROUTES) * config.api_burst,
                "throttle_is_best_effort": True,
                "cors_origin": "exact API endpoint origin", "allow_headers": TOKEN_HEADERS},
            "web_reserved_concurrency": config.web_concurrency if config.reserve_web_concurrency else None,
            "reserve_web_concurrency": config.reserve_web_concurrency,
            "alarm": {"metric": "AWS/Lambda Invocations", "period_seconds": 60,
                      "threshold": config.invocation_alarm_threshold, "external_actions": []},
            "log_retention_days": 7, "s3_expiration_days": {"uploads": 1, "inputs": 14, "outputs": 14},
            "model_files": [path.name for path in check["model_files"]],
            "sample_count": len(check["public_samples"].get("samples", [])),
            "code_uncompressed_bytes": check["code_bytes"], "layer_uncompressed_bytes": check["layer_bytes"],
            "prerequisite_problems": check["problems"],
            "prerequisite_warnings": check["warnings"],
            "not_verified": ["Free plan and credits", "account quotas and permissions",
                             "Lambda native-wheel compatibility", "live IAM and HTTP API behavior"],
            "will_write_state": False, "will_contact_aws": False}


def ownership_tags(state: dict) -> dict[str, str]:
    return {"x132:project": PROJECT, "x132:deployment": state["suffix"]}


def assert_owned(tags: dict, state: dict, resource: str) -> None:
    if any(tags.get(key) != value for key, value in ownership_tags(state).items()):
        raise RuntimeError(f"Refusing to modify a resource without matching ownership tags: {resource}")


def _record(state: dict, path: Path, kind: str, name: str, **extra) -> None:
    resource = {"type": kind, "name": name, **extra}
    existing = next((item for item in state["resources"]
                     if item["type"] == kind and item["name"] == name), None)
    if existing:
        existing.update(resource)
    else:
        state["resources"].append(resource)
    save_state(path, state)


def role_policy(role: str, bucket: str, region: str, account: str) -> dict:
    """Resource-restricted runtime policies, separate from operator permissions."""
    object_root = f"arn:aws:s3:::{bucket}/{PREFIX}"
    functions = [f"arn:aws:lambda:{region}:{account}:function:{name}" for name in FUNCTIONS.values()]
    if role not in {"tool", "web", "bench", "inspect"}:
        raise ValueError("Unknown runtime role")
    log_names = ([FUNCTIONS[role]] if role in {"web", "bench", "inspect"}
                 else [FUNCTIONS["tool_arm64"], FUNCTIONS["tool_x86"]])
    statements = [{"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                   "Resource": [f"arn:aws:logs:{region}:{account}:log-group:/aws/lambda/{name}:*"
                                for name in log_names]}]
    read_prefixes = ["samples", "models", "inputs"] if role == "inspect" else ["samples", "models", "uploads", "inputs"]
    if role in ("web", "bench", "inspect"):
        read_prefixes.append("outputs")
    statements.append({"Effect": "Allow", "Action": ["s3:GetObject"],
                       "Resource": [object_root + prefix + "/*" for prefix in read_prefixes]})
    if role == "web":
        statements.extend([
            {"Effect": "Allow", "Action": ["s3:PutObject"],
             "Resource": [object_root + "uploads/*", object_root + "inputs/*", object_root + "outputs/*"]},
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"],
             "Resource": [f"arn:aws:lambda:{region}:{account}:function:{FUNCTIONS['inspect']}"]},
            {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
             "Resource": [f"arn:aws:dynamodb:{region}:{account}:table/{name}" for name in TABLES.values()]},
        ])
    elif role == "inspect":
        statements.extend([
            {"Effect": "Allow", "Action": ["s3:PutObject"], "Resource": [object_root + "outputs/*"]},
            {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem"],
             "Resource": [f"arn:aws:dynamodb:{region}:{account}:table/{TABLES['inspections']}"]},
        ])
    elif role == "bench":
        statements.extend([
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
             "Resource": [object_root + "*"]},
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket}"],
             "Condition": {"StringLike": {"s3:prefix": [PREFIX, PREFIX + "*"]}}},
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": functions},
        ])
    return {"Version": "2012-10-17", "Statement": statements}


def package_code(paths: list[Path], destination: Path) -> Path:
    state_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in paths:
            entry = zipfile.ZipInfo(source.relative_to(APP_ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, source.read_bytes())
    return destination


def _upload(s3, bucket: str, key: str, path: Path, content_type: str = "application/octet-stream") -> str:
    digest = _sha256(path)
    try:
        previous = s3.head_object(Bucket=bucket, Key=key)
        if previous.get("Metadata", {}).get("sha256") == digest:
            return digest
    except Exception as error:
        if not _missing(error):
            raise
    with path.open("rb") as handle:
        s3.put_object(Bucket=bucket, Key=key, Body=handle, ContentType=content_type,
                      ServerSideEncryption="AES256", Metadata={"sha256": digest})
    return digest


def _put_json(s3, bucket: str, key: str, value: dict) -> None:
    content = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=content, ContentType="application/json",
                  ServerSideEncryption="AES256")


def _api_items(client, operation: str, api_id: str | None = None) -> list[dict]:
    request = {"MaxResults": "100"}
    if api_id is not None:
        request["ApiId"] = api_id
    items = []
    while True:
        page = getattr(client, operation)(**request)
        items.extend(page.get("Items", []))
        token = page.get("NextToken")
        if not token:
            return items
        request["NextToken"] = token


def configure_http_api(client, lam, state: dict, config: DeployConfig, account: str) -> str:
    """Create or repair only the state-owned, throttled HTTP API."""
    detail = None
    previous = state.get("http_api", {})
    if previous.get("id"):
        if not re.fullmatch(r"[a-z0-9]{1,32}", previous["id"]):
            raise ValueError("Invalid recorded HTTP API id")
        try:
            detail = client.get_api(ApiId=previous["id"])
        except Exception as error:
            if not _missing(error):
                raise
    if detail is None:
        matches = [item for item in _api_items(client, "get_apis") if item.get("Name") == HTTP_API_NAME]
        if len(matches) > 1:
            raise RuntimeError("Multiple HTTP APIs use the project name; resolve ownership before deployment")
        if matches:
            detail = client.get_api(ApiId=matches[0]["ApiId"])
        else:
            detail = client.create_api(Name=HTTP_API_NAME, ProtocolType="HTTP",
                                       DisableExecuteApiEndpoint=False, Tags=ownership_tags(state))
            # Persist the provider id immediately, before routes or integration
            # setup, so an interrupted deployment can be resumed or removed.
            api_id = detail["ApiId"]
            state["http_api"] = {"id": api_id, "endpoint": detail.get("ApiEndpoint")}
            _record(state, config.state, "http_api", HTTP_API_NAME, id=api_id,
                     arn=f"arn:aws:apigateway:{config.region}::/apis/{api_id}")
    if detail.get("Name") != HTTP_API_NAME or detail.get("ProtocolType") != "HTTP":
        raise RuntimeError("Recorded HTTP API has an incompatible name or protocol")
    assert_owned(detail.get("Tags", {}), state, HTTP_API_NAME)
    api_id = detail["ApiId"]
    endpoint = detail.get("ApiEndpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        raise RuntimeError("HTTP API did not return an HTTPS endpoint")
    origin = endpoint.rstrip("/")
    current = state.setdefault("http_api", {})
    current.update({"id": api_id, "endpoint": origin})
    _record(state, config.state, "http_api", HTTP_API_NAME, id=api_id,
             arn=f"arn:aws:apigateway:{config.region}::/apis/{api_id}")
    client.update_api(ApiId=api_id, DisableExecuteApiEndpoint=False, CorsConfiguration={
        "AllowOrigins": [origin], "AllowMethods": ["GET", "POST"],
        "AllowHeaders": TOKEN_HEADERS, "AllowCredentials": False, "MaxAge": 300})
    web_arn = f"arn:aws:lambda:{config.region}:{account}:function:{FUNCTIONS['web']}"
    integrations = _api_items(client, "get_integrations", api_id)
    matches = [item for item in integrations if item.get("IntegrationId") == current.get("integration_id")]
    if not matches:
        matches = [item for item in integrations if item.get("IntegrationUri") == web_arn]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous web integrations in the owned HTTP API")
    integration_settings = {"ApiId": api_id, "IntegrationType": "AWS_PROXY", "IntegrationUri": web_arn,
                            "IntegrationMethod": "POST", "PayloadFormatVersion": "2.0", "TimeoutInMillis": 30000}
    if matches:
        integration_id = matches[0]["IntegrationId"]
        client.update_integration(IntegrationId=integration_id, **integration_settings)
    else:
        integration_id = client.create_integration(**integration_settings)["IntegrationId"]
    current["integration_id"] = integration_id
    save_state(config.state, state)
    routes = _api_items(client, "get_routes", api_id)
    if any(route.get("RouteKey") not in HTTP_ROUTES for route in routes):
        raise RuntimeError("Owned HTTP API contains an unexpected route; review it before continuing")
    by_key = {route["RouteKey"]: route for route in routes}
    current["routes"] = {}
    for route_key in HTTP_ROUTES:
        settings = {"ApiId": api_id, "RouteKey": route_key,
                    "Target": "integrations/" + integration_id, "AuthorizationType": "NONE"}
        if route_key in by_key:
            route_id = by_key[route_key]["RouteId"]
            client.update_route(RouteId=route_id, **settings)
        else:
            route_id = client.create_route(**settings)["RouteId"]
        current["routes"][route_key] = route_id
        save_state(config.state, state)
    stages = _api_items(client, "get_stages", api_id)
    if any(stage.get("StageName") != "$default" for stage in stages):
        raise RuntimeError("Owned HTTP API contains an unexpected stage; review its throttle settings")
    stage_settings = {"ApiId": api_id, "StageName": "$default", "AutoDeploy": True,
                      "DefaultRouteSettings": {"ThrottlingRateLimit": float(config.api_rate),
                                               "ThrottlingBurstLimit": config.api_burst}}
    if stages:
        client.update_stage(**stage_settings)
    else:
        client.create_stage(**stage_settings, Tags=ownership_tags(state))
    current.update({"stage": "$default", "rate_per_second": config.api_rate, "burst": config.api_burst,
                    "throttle_scope": "per_route"})
    save_state(config.state, state)
    try:
        lam.remove_permission(FunctionName=FUNCTIONS["web"], StatementId="x132-http-api")
    except Exception as error:
        if not _missing(error):
            raise
    lam.add_permission(FunctionName=FUNCTIONS["web"], StatementId="x132-http-api",
                        Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
                        SourceAccount=account,
                        SourceArn=f"arn:aws:execute-api:{config.region}:{account}:{api_id}/*/*/*")
    return origin


def disable_legacy_public_url(lam, state: dict) -> bool:
    tracked = any(item.get("type") == "function" and item.get("name") == FUNCTIONS["web"]
                  for item in state.get("resources", []))
    if not tracked:
        return False
    try:
        function = lam.get_function(FunctionName=FUNCTIONS["web"])["Configuration"]
        assert_owned(lam.list_tags(Resource=function["FunctionArn"]).get("Tags", {}),
                     state, FUNCTIONS["web"])
        legacy = lam.get_function_url_config(FunctionName=FUNCTIONS["web"])
    except Exception as error:
        if not _missing(error):
            raise
        return False
    try:
        lam.update_function_url_config(FunctionName=FUNCTIONS["web"], AuthType="AWS_IAM")
    except Exception as error:
        if not _missing(error):
            raise
    for statement in ("x132-public-url", "x132-public-via-url"):
        try:
            lam.remove_permission(FunctionName=FUNCTIONS["web"], StatementId=statement)
        except Exception as error:
            if not _missing(error):
                raise
    state["legacy_function_url_disabled"] = True
    if legacy.get("FunctionUrl"):
        state.setdefault("function_url", legacy["FunctionUrl"])
    return True


def maybe_reserve_web_concurrency(lam, config: DeployConfig) -> dict:
    if not config.reserve_web_concurrency:
        return {"requested": False, "applied": False, "reason": "Reservation disabled by default"}
    limits = lam.get_account_settings().get("AccountLimit", {})
    current = lam.get_function_concurrency(FunctionName=FUNCTIONS["web"]).get("ReservedConcurrentExecutions", 0)
    available = limits.get("UnreservedConcurrentExecutions")
    if type(available) is not int or type(current) is not int:
        return {"requested": True, "applied": False, "reason": "Account concurrency headroom is unavailable"}
    if available - (config.web_concurrency - current) < 10:
        return {"requested": True, "applied": False,
                "reason": "Reservation skipped: fewer than ten unreserved executions would remain"}
    lam.put_function_concurrency(FunctionName=FUNCTIONS["web"],
                                  ReservedConcurrentExecutions=config.web_concurrency)
    return {"requested": True, "applied": True, "reserved": config.web_concurrency}


def execute_deploy(config: DeployConfig, sdk_session=None) -> dict:
    """Run only after explicit --execute; injectable SDK session supports offline tests."""
    check = preflight(config)
    if check["problems"]:
        raise ValueError("Deployment prerequisites failed: " + "; ".join(check["problems"]))
    state = load_state(config.state)
    if state and state["region"] != config.region:
        raise ValueError("State region differs from requested region")
    if sdk_session is None:
        sdk_session = importlib.import_module("boto3").Session(region_name=config.region)
    service_names = ["s3", "iam", "lambda", "dynamodb", "logs", "cloudwatch", "sts"]
    if not config.bench_only:
        service_names.append("apigatewayv2")
    clients = {service: sdk_session.client(service, region_name=config.region) for service in service_names}
    account = clients["sts"].get_caller_identity()["Account"]
    if state is None:
        suffix = secrets.token_hex(6)
        state = {"version": 1, "project": PROJECT, "region": config.region,
                 "bucket": f"x132-{suffix}", "prefix": PREFIX, "suffix": suffix,
                 "functions": {key: name for key, name in FUNCTIONS.items()
                               if not config.bench_only or key != "inspect"},
                 "tables": dict(TABLES), "resources": []}
        save_state(config.state, state)
    if not config.bench_only:
        state.setdefault("functions", {}).update(FUNCTIONS)
    tags = ownership_tags(state)
    tag_list = [{"Key": key, "Value": value} for key, value in tags.items()]
    bucket, region = state["bucket"], config.region
    s3, iam, lam, ddb, logs, cw = [clients[name] for name in
                                  ("s3", "iam", "lambda", "dynamodb", "logs", "cloudwatch")]
    try:
        s3.head_bucket(Bucket=bucket)
        bucket_tags = {item["Key"]: item["Value"] for item in
                       s3.get_bucket_tagging(Bucket=bucket).get("TagSet", [])}
        assert_owned(bucket_tags, state, bucket)
    except Exception as error:
        if not _missing(error):
            raise
        request = {"Bucket": bucket}
        if region != "us-east-1":
            request["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**request)
        _record(state, config.state, "bucket", bucket)
        s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": tag_list})
    _record(state, config.state, "bucket", bucket)
    if s3.get_bucket_versioning(Bucket=bucket).get("Status") == "Enabled":
        raise RuntimeError("Owned bucket unexpectedly has versioning enabled; inspect it manually")
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(Bucket=bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": [
        {"ID": "x132-expire-uploads", "Filter": {"Prefix": PREFIX + "uploads/"},
         "Status": "Enabled", "Expiration": {"Days": 1}},
        {"ID": "x132-expire-outputs", "Filter": {"Prefix": PREFIX + "outputs/"},
         "Status": "Enabled", "Expiration": {"Days": 14}},
        {"ID": "x132-expire-inputs", "Filter": {"Prefix": PREFIX + "inputs/"},
         "Status": "Enabled", "Expiration": {"Days": 14}},
    ]})
    for source, key in check["sample_uploads"]:
        media_type = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
        _upload(s3, bucket, key, source, media_type)
    if check["public_samples"].get("samples"):
        _put_json(s3, bucket, PREFIX + "samples/manifest.json", check["public_samples"])
    model_prefix = state.get("model_prefix", PREFIX + "models/")
    if check["model_files"]:
        entries = [{"name": source.name, "sha256": _sha256(source), "size_bytes": source.stat().st_size}
                   for source in sorted(check["model_files"], key=lambda source: source.name)]
        bundle_sha = hashlib.sha256(json.dumps(entries, sort_keys=True,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
        model_prefix = f"{PREFIX}models/{bundle_sha}/"
        for source in check["model_files"]:
            _upload(s3, bucket, model_prefix + source.name, source)
        _put_json(s3, bucket, model_prefix + "manifest.json", {"files": entries})
        state["model_prefix"] = model_prefix
        state["model_bundle_sha256"] = bundle_sha
    for table in TABLES.values():
        try:
            detail = ddb.describe_table(TableName=table)["Table"]
            current = ddb.list_tags_of_resource(ResourceArn=detail["TableArn"]).get("Tags", [])
            assert_owned({item["Key"]: item["Value"] for item in current}, state, table)
            if detail.get("KeySchema") != [{"AttributeName": "id", "KeyType": "HASH"}]:
                raise RuntimeError(f"Owned table has an incompatible key schema: {table}")
            if detail.get("BillingModeSummary", {}).get("BillingMode", "PROVISIONED") != "PROVISIONED":
                raise RuntimeError(f"Owned table has an unexpected billing mode: {table}")
        except Exception as error:
            if not _missing(error):
                raise
            ddb.create_table(TableName=table, AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
                             KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                             BillingMode="PROVISIONED", ProvisionedThroughput={"ReadCapacityUnits": 5,
                                                                             "WriteCapacityUnits": 5}, Tags=tag_list)
            _record(state, config.state, "table", table)
            ddb.get_waiter("table_exists").wait(TableName=table, WaiterConfig={"Delay": 2, "MaxAttempts": 60})
        _record(state, config.state, "table", table)
        ttl = ddb.describe_time_to_live(TableName=table).get("TimeToLiveDescription", {})
        if ttl.get("TimeToLiveStatus") not in {"ENABLED", "ENABLING"}:
            ddb.update_time_to_live(TableName=table, TimeToLiveSpecification={"Enabled": True,
                                                                                 "AttributeName": "expires_at"})
        elif ttl.get("AttributeName") != "expires_at":
            raise RuntimeError(f"Owned table uses another TTL attribute: {table}")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal":
             {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    role_arns = {}
    for role in (("bench",) if config.bench_only else ("tool", "web", "bench", "inspect")):
        name = f"x132-{role}-role"
        try:
            detail = iam.get_role(RoleName=name)["Role"]
            current = iam.list_role_tags(RoleName=name).get("Tags", [])
            assert_owned({item["Key"]: item["Value"] for item in current}, state, name)
            iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust))
        except Exception as error:
            if not _missing(error):
                raise
            detail = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust), Tags=tag_list)["Role"]
            _record(state, config.state, "role", name)
        role_arns[role] = detail["Arn"]
        iam.put_role_policy(RoleName=name, PolicyName="x132-runtime", PolicyDocument=json.dumps(
            role_policy(role, bucket, region, account)))
        _record(state, config.state, "role", name)
    selected_functions = {"bench": FUNCTIONS["bench"]} if config.bench_only else FUNCTIONS
    for function in selected_functions.values():
        group = "/aws/lambda/" + function
        group_arn = f"arn:aws:logs:{region}:{account}:log-group:{group}"
        try:
            logs.create_log_group(logGroupName=group, tags=tags)
            _record(state, config.state, "log_group", group, arn=group_arn)
        except Exception as error:
            if error_code(error) != "ResourceAlreadyExistsException":
                raise
            assert_owned(logs.list_tags_for_resource(resourceArn=group_arn).get("tags", {}), state, group)
        logs.put_retention_policy(logGroupName=group, retentionInDays=7)
        _record(state, config.state, "log_group", group, arn=group_arn)
    layer_arns = {}
    state.setdefault("layers", {})
    for architecture, source in (("arm64", config.layer_arm64), ("x86_64", config.layer_x86)):
        digest = _sha256(source)
        previous = state["layers"].get(architecture, {})
        if previous.get("sha256") == digest:
            try:
                lam.get_layer_version_by_arn(Arn=previous["arn"])
                layer_arns[architecture] = previous["arn"]
                continue
            except Exception as error:
                if not _missing(error):
                    raise
        key = f"{PREFIX}deployment/layers/{architecture}-{digest}.zip"
        _upload(s3, bucket, key, source)
        layer = lam.publish_layer_version(LayerName=f"x132-opencv-{architecture}",
                    Content={"S3Bucket": bucket, "S3Key": key}, CompatibleRuntimes=["python3.12"],
                    CompatibleArchitectures=[architecture], Description="x132 numpy and headless OpenCV")
        layer_arns[architecture] = layer["LayerVersionArn"]
        state["layers"][architecture] = {"sha256": digest, "arn": layer["LayerVersionArn"]}
        _record(state, config.state, "layer", f"x132-opencv-{architecture}:{layer['Version']}",
                 layer_name=f"x132-opencv-{architecture}",
                 arn=layer["LayerVersionArn"], version=layer["Version"])
    code = package_code(check["code_files"], APP_ROOT / "build/function.zip")
    code_digest = _sha256(code)
    code_key = PREFIX + "deployment/function-" + code_digest + ".zip"
    _upload(s3, bucket, code_key, code)
    environment = {"X132_BUCKET": bucket, "X132_PREFIX": PREFIX,
                   "X132_TOOL_FUNCTION": FUNCTIONS["tool_arm64"], "X132_MODEL_PREFIX": model_prefix,
                   "X132_INSPECT_FUNCTION": FUNCTIONS["inspect"],
                   "X132_SAMPLES_KEY": PREFIX + "samples/manifest.json", "X132_MODEL_DIR": "/opt/models",
                   "X132_MODEL_SOURCE": "s3", "X132_CALIBRATION": "sampled",
                   "X132_INSPECTIONS_TABLE": TABLES["inspections"], "X132_REVIEWS_TABLE": TABLES["reviews"],
                   "X132_RATELIMIT_TABLE": TABLES["ratelimit"]}
    policy_file = os.environ.get("X132_POLICY_CONFIG_FILE")
    if policy_file:
        # Dev-tuned policy (e.g. artifacts/<tune>/merged/config.json) for the public inspect path.
        environment["X132_POLICY_CONFIG"] = json.dumps(json.loads(Path(policy_file).read_text()), separators=(",", ":"))
    selected_functions = {"bench": FUNCTIONS["bench"]} if config.bench_only else FUNCTIONS
    for key, function in selected_functions.items():
        limits = function_limits(key)
        architecture = limits["architecture"]
        role = "tool" if key.startswith("tool") else key
        settings = {"Runtime": "python3.12", "Role": role_arns[role], "Handler": HANDLERS[key],
                    "Timeout": limits["timeout"], "MemorySize": limits["memory_mb"],
                    "Environment": {"Variables": environment},
                    "Layers": [layer_arns[architecture]] if limits["uses_layer"] else [],
                    "EphemeralStorage": {"Size": limits["ephemeral_storage_mb"]}}
        try:
            detail = lam.get_function(FunctionName=function)
            assert_owned(lam.list_tags(Resource=detail["Configuration"]["FunctionArn"]).get("Tags", {}),
                         state, function)
            if detail["Configuration"].get("Architectures") != [architecture]:
                raise RuntimeError(f"Owned function architecture differs: {function}")
            lam.update_function_code(FunctionName=function, S3Bucket=bucket, S3Key=code_key)
            lam.get_waiter("function_updated_v2").wait(FunctionName=function)
            lam.update_function_configuration(FunctionName=function, **settings)
        except Exception as error:
            if not _missing(error):
                raise
            for attempt in range(6):
                try:
                    lam.create_function(FunctionName=function, Code={"S3Bucket": bucket, "S3Key": code_key},
                                        Architectures=[architecture], Tags=tags, **settings)
                    _record(state, config.state, "function", function)
                    lam.get_waiter("function_active_v2").wait(FunctionName=function)
                    break
                except Exception as role_error:
                    if error_code(role_error) != "InvalidParameterValueException" or attempt == 5:
                        raise
                    time.sleep(2)
        lam.get_waiter("function_updated_v2").wait(FunctionName=function)
        _record(state, config.state, "function", function)
    if config.bench_only:
        state["deployment_mode"] = "bench_only"
        state["code_sha256"] = code_digest
        save_state(config.state, state)
        return {"region": region, "function_url": state.get("function_url"),
                "state": str(config.state), "deployment_mode": "bench_only",
                "owned_resource_count": len(state["resources"]),
                "live_validation": "deployment API results only; no benchmark or inference run"}
    state["web_concurrency"] = maybe_reserve_web_concurrency(lam, config)
    disable_legacy_public_url(lam, state)
    origin = configure_http_api(clients["apigatewayv2"], lam, state, config, account)
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [{
        "AllowedOrigins": [origin], "AllowedMethods": ["GET", "POST", "HEAD"],
        "AllowedHeaders": ["content-type", "x-amz-*"], "ExposeHeaders": ["ETag"], "MaxAgeSeconds": 300}]})
    alarm_name = "x132-web-invocations"
    alarm_arn = f"arn:aws:cloudwatch:{region}:{account}:alarm:{alarm_name}"
    alarms = cw.describe_alarms(AlarmNames=[alarm_name]).get("MetricAlarms", [])
    if alarms:
        current = cw.list_tags_for_resource(ResourceARN=alarm_arn).get("Tags", [])
        assert_owned({item["Key"]: item["Value"] for item in current}, state, alarm_name)
    cw.put_metric_alarm(AlarmName=alarm_name, Namespace="AWS/Lambda", MetricName="Invocations",
                         Dimensions=[{"Name": "FunctionName", "Value": FUNCTIONS["web"]}],
                         Statistic="Sum", Period=60, EvaluationPeriods=1, DatapointsToAlarm=1,
                         Threshold=config.invocation_alarm_threshold, ComparisonOperator="GreaterThanThreshold",
                         TreatMissingData="notBreaching", ActionsEnabled=False, Tags=tag_list)
    _record(state, config.state, "alarm", alarm_name, arn=alarm_arn)
    state["deployment_mode"] = "full"
    state["code_sha256"] = code_digest
    save_state(config.state, state)
    return {"region": region, "api_endpoint": origin, "state": str(config.state),
            "web_concurrency": state["web_concurrency"],
            "owned_resource_count": len(state["resources"]), "live_validation": "deployment API results only"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--state", type=Path, default=APP_ROOT / "aws/state.json")
    parser.add_argument("--samples-manifest", type=Path, default=APP_ROOT / "aws/samples.example.json")
    parser.add_argument("--model-dir", type=Path, default=APP_ROOT / "models")
    parser.add_argument("--layer-arm64", type=Path, default=APP_ROOT / "build/layer.zip")
    parser.add_argument("--layer-x86", type=Path, default=APP_ROOT / "build/layer_x86_64.zip")
    parser.add_argument("--web-concurrency", type=int, default=2)
    parser.add_argument("--reserve-web-concurrency", action="store_true",
                        help="Opt in to reservation only when account headroom leaves ten unreserved executions")
    parser.add_argument("--api-rate", type=float, default=1.0, help="HTTP API requests per second")
    parser.add_argument("--api-burst", type=int, default=2, help="HTTP API burst allowance")
    parser.add_argument("--invocation-alarm-threshold", type=int, default=60)
    parser.add_argument("--bench-only", action="store_true",
                        help="Bootstrap the private bench function without trained models or a public web URL")
    arguments = parser.parse_args(argv)
    config = DeployConfig(region=arguments.region, state=arguments.state,
                          samples_manifest=arguments.samples_manifest, model_dir=arguments.model_dir,
                          layer_arm64=arguments.layer_arm64, layer_x86=arguments.layer_x86,
                          web_concurrency=arguments.web_concurrency,
                          invocation_alarm_threshold=arguments.invocation_alarm_threshold,
                          bench_only=arguments.bench_only,
                          reserve_web_concurrency=arguments.reserve_web_concurrency,
                          api_rate=arguments.api_rate, api_burst=arguments.api_burst)
    try:
        result = deployment_plan(config) if arguments.dry_run else execute_deploy(config)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        # Do not print SDK responses, identities, credentials or request bodies.
        detail = str(error) if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)) else error_code(error)
        if os.environ.get("X132_DEBUG_ERRORS") == "1":
            # Opt-in: show the service message with account-like numbers masked.
            message = getattr(error, "response", {}).get("Error", {}).get("Message", "")
            detail = f"{detail}: {re.sub(r'[0-9]{12}', '<account>', message)[:400]}"
        print(f"Deployment stopped ({type(error).__name__}): {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
