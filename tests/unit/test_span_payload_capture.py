import json
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest

from agnt5._capture import SpanPayloadPolicy, content_capture_mode
from agnt5.context import _current_context, set_current_context
from agnt5.function import FunctionRegistry, function
from agnt5.workflow import WorkflowContext, WorkflowEntity
from agnt5.workflow_activation import execute_function_callable


@pytest.fixture
def span_sink(monkeypatch):
    spans = []

    @contextmanager
    def create_span(_name, _kind, _context, attributes):
        class Span:
            def set_attribute(self, key, value):
                attributes[key] = value

        spans.append(attributes)
        yield Span()

    monkeypatch.setattr("agnt5.tracing.create_span", create_span)
    return spans


@pytest.fixture(autouse=True)
def capture_environment(monkeypatch):
    for name in (
        "AGNT5_CAPTURE",
        "AGNT5_CAPTURE_CONTENT_MODE",
        "AGNT5_LLM_CAPTURE_CONTENT",
        "AGNT5_SPAN_ATTRIBUTE_VALUE_MAX_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


async def run_function_path(path, context, payload):
    @function(name="capture_boundary")
    async def echo(ctx, payload: dict) -> dict:
        return payload

    try:
        if path == "legacy":
            return await context.step(echo, payload=payload)
        return await execute_function_callable(
            context, "capture_boundary", "echo", "echo", "step-cid", (), {"payload": payload}
        )
    finally:
        FunctionRegistry.clear()


@pytest.mark.parametrize("path", ["legacy", "activation"])
@pytest.mark.parametrize(
    "setting",
    [
        None,
        *[
            ("AGNT5_CAPTURE_CONTENT_MODE", value)
            for value in ("metadata-only", "invalid", "false", "off", "0", "no")
        ],
        *[("AGNT5_LLM_CAPTURE_CONTENT", value) for value in ("false", "off", "0", "no", "invalid")],
        *[("AGNT5_CAPTURE", value) for value in ("false", "off", "0", "no")],
    ],
)
async def test_workflow_payloads_are_private_by_default(path, setting, span_sink, monkeypatch):
    calls = []
    if setting is not None:
        if setting[0] == "AGNT5_CAPTURE":
            monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "full")
        monkeypatch.setenv(*setting)
    context = WorkflowContext(
        WorkflowEntity("run-1"), "run-1", span_payload_policy=SpanPayloadPolicy(calls.append)
    )
    context.emit_async = AsyncMock()
    context._emit_checkpoint_async = AsyncMock()
    payload = {"bot_token": "synthetic-bot-secret", "message": "private message"}

    assert await run_function_path(path, context, payload) == payload
    assert len(span_sink) == 1
    assert "input.data" not in span_sink[0]
    assert "output.data" not in span_sink[0]
    assert span_sink[0]["run_id"] == "run-1"
    assert "synthetic-bot-secret" not in json.dumps(span_sink)
    assert calls == []


@pytest.mark.parametrize("path", ["legacy", "activation"])
@pytest.mark.parametrize("mode", ["full", "redacted"])
async def test_workflow_opt_in_redacts_copies_without_changing_checkpoints(
    path, mode, span_sink, monkeypatch
):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", mode)
    calls = []

    def redact(value):
        calls.append(value)
        if "kwargs" in value:
            value = value["kwargs"]["payload"]
        value["message"] = "application redacted"
        value["new_token"] = "synthetic-hook-secret"
        return value

    entity = WorkflowEntity("run-1")
    context = WorkflowContext(entity, "run-1", span_payload_policy=SpanPayloadPolicy(redact))
    context.emit_async = AsyncMock()
    context._emit_checkpoint_async = AsyncMock()
    payload = {
        "bot_token": "synthetic-bot-secret",
        "message": "private message",
        "nested": [{"Authorization": "synthetic-header-secret"}],
        "body": json.dumps({"credentials": {"password": "synthetic-password"}}),
    }
    before = json.dumps(payload)

    result = await run_function_path(path, context, payload)

    assert result is payload
    assert json.dumps(payload) == before
    assert len(calls) == 2
    for direction in ("input", "output"):
        captured = json.loads(span_sink[0][f"{direction}.data"])
        assert captured["bot_token"] == "[REDACTED]"
        assert captured["nested"][0]["Authorization"] == "[REDACTED]"
        assert captured["new_token"] == "[REDACTED]"
        assert captured["message"] == "application redacted"
    exported = json.dumps(span_sink)
    for secret in (
        "synthetic-bot-secret",
        "synthetic-header-secret",
        "synthetic-password",
        "synthetic-hook-secret",
        "private message",
    ):
        assert secret not in exported
    completed = (
        (context.emit_async if path == "legacy" else context._emit_checkpoint_async)
        .await_args_list[-1]
        .args[0]
    )
    if path == "legacy":
        assert completed.output_data["result"] == payload
        assert entity.get_completed_step(entity._step_events[0]["step_name"]) == payload
    else:
        assert completed.output_data == payload


@pytest.mark.parametrize("path", ["legacy", "activation"])
@pytest.mark.parametrize("failure", ["raise", "omit", "unsupported"])
async def test_redactor_failure_never_falls_back_to_raw_payload(
    path, failure, span_sink, monkeypatch, caplog
):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "full")

    class Unsupported:
        def __repr__(self):
            raise AssertionError("capture must not call repr")

    def redact(_value):
        if failure == "raise":
            raise RuntimeError("synthetic-hook-secret")
        return Unsupported() if failure == "unsupported" else None

    context = WorkflowContext(
        WorkflowEntity("run-1"), "run-1", span_payload_policy=SpanPayloadPolicy(redact)
    )
    context.emit_async = AsyncMock()
    context._emit_checkpoint_async = AsyncMock()
    payload = {"message": "private message"}

    assert await run_function_path(path, context, payload) == payload
    assert "input.data" not in span_sink[0]
    assert "output.data" not in span_sink[0]
    assert "synthetic-hook-secret" not in caplog.text


@pytest.mark.parametrize("limit", ["0", "-1", "invalid", "1", "32", "8192"])
def test_span_capture_enforces_utf8_byte_cap(monkeypatch, limit):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "full")
    monkeypatch.setenv("AGNT5_SPAN_ATTRIBUTE_VALUE_MAX_BYTES", limit)
    captured = SpanPayloadPolicy().capture(
        json.dumps({"message": "😀" * 8192}, ensure_ascii=False).encode()
    )
    if limit in {"0", "-1", "invalid"}:
        assert captured is None
    else:
        preview, truncated = captured
        assert len(preview.encode()) <= int(limit)
        assert truncated is True
        assert "\ufffd" not in preview


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"plain text", '"plain text"'),
        (b'{"bot_token":"synthetic-secret"', '"[unparseable payload omitted]"'),
        (b"\xff", None),
        (b'{"nested":{"bot_token":"synthetic-secret"}}', '{"nested": {"bot_token": "[REDACTED]"}}'),
    ],
)
def test_native_byte_policy_handles_non_json_and_invalid_payloads(monkeypatch, payload, expected):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "full")
    captured = SpanPayloadPolicy().capture(payload)
    assert (captured[0] if captured is not None else None) == expected


def test_nested_workflow_inherits_parent_policy():
    parent = WorkflowContext(WorkflowEntity("parent"), "parent")
    token = set_current_context(parent)
    try:
        child = WorkflowContext(WorkflowEntity("child"), "child")
        assert child._span_payload_policy is parent._span_payload_policy
    finally:
        _current_context.reset(token)


def test_explicit_content_mode_wins_over_legacy_boolean(monkeypatch):
    monkeypatch.setenv("AGNT5_LLM_CAPTURE_CONTENT", "true")
    assert content_capture_mode() == "full"
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "invalid")
    assert content_capture_mode() == "metadata-only"
    monkeypatch.setenv("AGNT5_LLM_CAPTURE_CONTENT", "off")
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", " ReDaCtEd ")
    assert content_capture_mode() == "redacted"


def test_worker_registers_its_policy_at_native_boundary(fake_native_core, monkeypatch):
    from agnt5 import Worker

    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "full")
    calls = []

    def redact(value):
        calls.append(value)
        return {"message": "application redacted"}

    worker = Worker(service_name="capture-test", span_payload_redactor=redact)
    capture = worker._rust_worker.span_payload_capture
    original = b'{"message":"private message","bot_token":"synthetic-secret"}'
    assert capture(original) == ('{"message": "application redacted"}', False)
    assert calls == [{"message": "private message", "bot_token": "[REDACTED]"}]
    assert original == b'{"message":"private message","bot_token":"synthetic-secret"}'
