"""Input contracts and fail-closed DynamoDB rate limits, without boto3 imports."""

import base64
import hashlib
import json
import math
import time

from .image_validation import ValidationError


class RateLimitExceeded(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_body(event: dict, allowed: set[str], required: set[str] = frozenset()) -> dict:
    body = event.get("body") or "{}"
    if not isinstance(body, str) or len(body) > 22000:
        raise ValidationError("Request body is too large.")
    try:
        raw = base64.b64decode(body, validate=True) if event.get("isBase64Encoded") else body.encode()
        if len(raw) > 16384:
            raise ValidationError("Request body is too large.")
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("Body must be a JSON object.") from exc
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValidationError("Unexpected or missing request fields.")
    return value


def inspection_options(body: dict) -> tuple[float, str]:
    budget = body.get("budget_frac", .1)
    detector = body.get("detector", "classic")
    if (isinstance(budget, bool) or not isinstance(budget, (int, float)) or
            not math.isfinite(budget) or not .02 <= budget <= .4):
        raise ValidationError("budget_frac must be between 0.02 and 0.40.")
    if detector not in ("classic", "learned"):
        raise ValidationError("detector must be classic or learned.")
    return float(budget), detector


class RateLimiter:
    def __init__(self, client, table: str = "x132-ratelimit", clock=time.time):
        self.client, self.table, self.clock = client, table, clock

    def admit(self, ip: str, action: str) -> None:
        """Gate before S3 work: 20/IP/hour and 200/action/UTC calendar day.

        Epoch-aligned 86400-second windows start at 00:00 UTC. Separate atomic
        counters fail closed; a rejected global request can still consume its
        conservative per-IP allowance, but cannot enter the heavy worker.
        """
        self.consume(ip, action, limit=20, period=3600)
        self.consume("global", "daily-" + action, limit=200, period=86400)

    def consume(self, ip: str, action: str, limit: int = 20, period: int = 3600) -> None:
        now = int(self.clock())
        window = now // period
        key = f"{action}:{digest(ip)}:{window}"
        try:
            self.client.update_item(
                TableName=self.table, Key={"id": {"S": key}},
                UpdateExpression="SET expires_at = :ttl ADD used :one",
                ConditionExpression="attribute_not_exists(used) OR used < :limit",
                ExpressionAttributeValues={":ttl": {"N": str((window+1)*period+3600)},
                                           ":one": {"N": "1"}, ":limit": {"N": str(limit)}})
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise RateLimitExceeded("Request limit reached; try again later.") from exc
            raise ServiceUnavailable("Rate limiting is temporarily unavailable.") from exc
