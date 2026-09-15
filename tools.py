"""Service-scoped live telemetry tools."""

import json
import os
import subprocess
from typing import Any

import requests
from langchain_core.tools import tool

MAX_LOGS = 50
MAX_TRACES = 20
MAX_SPANS = 50


def _docker_status(container: str) -> str:
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}|"
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"Unable to inspect container: {error}"

    if result.returncode != 0:
        return f"Unable to inspect container: {result.stderr.strip()}"

    state, _, health = result.stdout.strip().partition("|")
    return f"container_state={state}\nhealth_check={health or 'unknown'}"


def _opensearch_url() -> str:
    configured_url = os.getenv("OPENSEARCH_URL")
    if configured_url:
        return configured_url.rstrip("/")

    try:
        result = subprocess.run(
            ["docker", "port", "opensearch", "9200/tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            published_port = result.stdout.splitlines()[0].rsplit(":", 1)[-1]
            if published_port.isdigit():
                return f"http://localhost:{published_port}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "http://localhost:9200"


def _severity_filter(level: str) -> dict[str, Any] | None:
    ranges = {
        "trace": {"gte": 1, "lt": 5},
        "debug": {"gte": 5, "lt": 9},
        "info": {"gte": 9, "lt": 13},
        "warn": {"gte": 13, "lt": 17},
        "error": {"gte": 17, "lt": 21},
        "fatal": {"gte": 21},
    }
    if level.lower() in {"all", "*"}:
        return None
    bounds = ranges.get(level.lower())
    if bounds is None:
        raise ValueError(f"Unsupported log level: {level}")
    return {"range": {"severity.number": bounds}}


def _compact(value: Any, limit: int = 1_000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return " ".join(text.split())[:limit]


def _get_service_logs(service: str, level: str, limit: int) -> str:
    try:
        severity_filter = _severity_filter(level)
    except ValueError as error:
        return str(error)

    filters: list[dict[str, Any]] = [
        {"term": {"resource.service.name.keyword": service}},
        {"range": {"@timestamp": {"gte": "now-10m"}}},
    ]
    if severity_filter:
        filters.append(severity_filter)

    query = {
        "size": max(1, min(limit, MAX_LOGS)),
        "_source": [
            "@timestamp",
            "severity",
            "body",
            "traceId",
            "spanId",
        ],
        "sort": [{"@timestamp": {"order": "desc"}}],
        "query": {"bool": {"filter": filters}},
    }

    try:
        response = requests.post(
            f"{_opensearch_url()}/{os.getenv('OPENSEARCH_INDEX', 'otel-logs-*')}/_search",
            json=query,
            timeout=10,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
    except (requests.RequestException, ValueError) as error:
        return f"Unable to query OpenSearch: {error}"

    output = []
    for hit in hits:
        source = hit.get("_source", {})
        body = _compact(source.get("body", ""))
        if "feature_flag" in body.lower():
            continue
        severity = source.get("severity", {}).get("text", "UNKNOWN")
        parts = [
            f"timestamp={source.get('@timestamp', 'unknown')}",
            f"level={severity}",
            f"body={body}",
        ]
        if source.get("traceId"):
            parts.append(f"trace_id={source['traceId']}")
        if source.get("spanId"):
            parts.append(f"span_id={source['spanId']}")
        output.append(" | ".join(parts))

    return "\n".join(output) or "No recent matching local logs found."


def _local_spans(trace: dict[str, Any], service: str) -> list[dict[str, Any]]:
    processes = trace.get("processes", {})
    return [
        span
        for span in trace.get("spans", [])
        if processes.get(span.get("processID"), {}).get("serviceName") == service
    ]


def _span_to_text(span: dict[str, Any]) -> str:
    tags = {
        tag["key"]: tag.get("value")
        for tag in span.get("tags", [])
        if not tag["key"].startswith("feature_flag")
    }
    interesting_tags = [
        "error",
        "otel.status_code",
        "otel.status_description",
        "rpc.grpc.status_code",
        "grpc.status_code",
        "grpc.error_message",
        "error.type",
        "error.message",
        "http.response.status_code",
        "server.address",
        "server.port",
    ]
    parts = [
        f"operation={_compact(span.get('operationName', 'unknown'), 200)}",
        f"duration_us={span.get('duration', 'unknown')}",
    ]
    parts.extend(
        f"{key}={_compact(tags[key], 500)}" for key in interesting_tags if key in tags
    )

    for log in span.get("logs", []):
        fields = {
            field["key"]: field.get("value")
            for field in log.get("fields", [])
            if not field["key"].startswith("feature_flag")
        }
        if str(fields.get("event", "")).startswith("feature_flag"):
            continue
        event_parts = [
            f"{key}={_compact(fields[key], 500)}"
            for key in ("event", "exception.type", "exception.message")
            if key in fields
        ]
        if event_parts:
            parts.append(f"event=({', '.join(event_parts)})")

    return " | ".join(parts)


def _get_service_traces(service: str, limit: int) -> str:
    try:
        response = requests.get(
            f"{os.getenv('JAEGER_URL', 'http://localhost:8080/jaeger/ui').rstrip('/')}/api/traces",
            params={
                "service": service,
                "limit": max(1, min(limit, MAX_TRACES)),
                "lookback": os.getenv("TELEMETRY_LOOKBACK", "10m"),
            },
            timeout=10,
        )
        response.raise_for_status()
        traces = response.json().get("data", [])
    except (requests.RequestException, ValueError) as error:
        return f"Unable to query Jaeger: {error}"

    output = []
    for trace in traces:
        for span in _local_spans(trace, service):
            output.append(
                f"trace_id={trace.get('traceID', 'unknown')} | {_span_to_text(span)}"
            )

    return "\n".join(output[:MAX_SPANS]) or "No recent local traces found."


@tool
def get_checkout_status() -> str:
    """Inspect only the Checkout container's runtime and health status."""
    return _docker_status("checkout")


@tool
def get_checkout_logs(level: str = "error", limit: int = 50) -> str:
    """Inspect only recent Checkout logs from OpenSearch."""
    return _get_service_logs("checkout", level, limit)


@tool
def get_checkout_traces(limit: int = 20) -> str:
    """Inspect only recent spans generated by Checkout in Jaeger."""
    return _get_service_traces("checkout", limit)


@tool
def get_payment_status() -> str:
    """Inspect only the Payment container's runtime and health status."""
    return _docker_status("payment")


@tool
def get_payment_logs(level: str = "error", limit: int = 50) -> str:
    """Inspect only recent Payment logs from OpenSearch."""
    return _get_service_logs("payment", level, limit)


@tool
def get_payment_traces(limit: int = 20) -> str:
    """Inspect only recent spans generated by Payment in Jaeger."""
    return _get_service_traces("payment", limit)


@tool
def get_shipping_status() -> str:
    """Inspect only the Shipping container's runtime and health status."""
    return _docker_status("shipping")


@tool
def get_shipping_logs(level: str = "error", limit: int = 50) -> str:
    """Inspect only recent Shipping logs from OpenSearch."""
    return _get_service_logs("shipping", level, limit)


@tool
def get_shipping_traces(limit: int = 20) -> str:
    """Inspect only recent spans generated by Shipping in Jaeger."""
    return _get_service_traces("shipping", limit)


CHECKOUT_TOOLS = [get_checkout_status, get_checkout_logs, get_checkout_traces]
PAYMENT_TOOLS = [get_payment_status, get_payment_logs, get_payment_traces]
SHIPPING_TOOLS = [get_shipping_status, get_shipping_logs, get_shipping_traces]
