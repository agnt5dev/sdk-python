"""Journal capture for the OpenAI Agents SDK.

Rides the public tracing API: ``enable()`` registers a ``TracingProcessor``
via ``add_trace_processor`` — no patching. Span types map onto the canonical
taxonomy: agent spans → ``agent.*``, generation/response spans → ``lm.*``,
function spans → ``tool_call.*``. Other span types (handoff, guardrail, ...)
emit nothing but stay transparent for parenting, so a generation span nested
under a handoff still parents to the right agent span.

While a response/generation span is active, the raw ``openai`` client
capture suppresses itself (see ``suppresses_client_capture``) — the same
underlying model call would otherwise be journaled twice. Client calls made
from user tool code (function spans) are still captured by the client patch.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any, Optional

from .._ids import generate_cid
from ..agent.events import (
    AgentCompleted,
    AgentFailed,
    AgentStarted,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from ._common import (
    ambient_context,
    build_lm_completed,
    build_lm_failed,
    build_lm_started,
    capture_error_message,
    capture_payload,
    content_capture_enabled,
    observed_metadata,
    safe_emit,
    supported_package_version,
)

logger = logging.getLogger(__name__)

SOURCE = "openai_agents"
PROVIDER = "openai"

_enabled = False
_processor: Optional["CaptureProcessor"] = None


class _UnavailableSpanData:
    pass


class _TracingProcessorFallback:
    pass


add_trace_processor: Any = None
get_current_span: Any = None
TracingProcessor: Any = _TracingProcessorFallback
AgentSpanData: Any = _UnavailableSpanData
FunctionSpanData: Any = _UnavailableSpanData
GenerationSpanData: Any = _UnavailableSpanData
ResponseSpanData: Any = _UnavailableSpanData

try:
    from agents.tracing import add_trace_processor, get_current_span
    from agents.tracing.processor_interface import TracingProcessor
    from agents.tracing.span_data import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        ResponseSpanData,
    )

    _agents_available = True
except ImportError:
    TracingProcessor = object  # type: ignore[assignment,misc]
    _agents_available = False


def enable() -> bool:
    """Register the capture trace processor. Idempotent; False when absent."""
    global _enabled, _processor
    if _enabled:
        return True
    if not _agents_available:
        logger.debug("openai-agents not installed; capture disabled")
        return False
    if not supported_package_version("openai-agents", minimum=(0, 3, 0), max_major_exclusive=1):
        return False
    if _processor is None:
        try:
            processor = CaptureProcessor()
            add_trace_processor(processor)
            _processor = processor
        except Exception:
            logger.warning("openai_agents capture registration failed", exc_info=True)
            return False
    _enabled = True
    logger.debug("openai_agents capture enabled")
    return True


def disable() -> None:
    """Deactivate capture (the registered processor goes inert).

    The Agents SDK has no public remove-processor API; the processor checks
    the module flag and no-ops when disabled.
    """
    global _enabled
    _enabled = False


def suppresses_client_capture() -> bool:
    """True when the SDK's own model call is in flight (dedupe guard).

    Only response/generation spans suppress the raw client capture; client
    calls from user tool code run under function spans and stay captured.
    """
    if not _enabled or not _agents_available or _processor is None:
        return False
    try:
        span = get_current_span()
        return span is not None and _processor.owns_span(span)
    except Exception:
        return False


def _never_raise_processor(method: Any) -> Any:
    """Tracing processors are observational and must never fail an agent run."""

    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception:
            logger.debug("openai_agents capture processor failed", exc_info=True)
            return None

    return guarded


class _SpanState:
    __slots__ = ("ctx", "cid", "parent_cid", "start", "kind", "request_name", "provider")

    def __init__(
        self,
        ctx: Any,
        cid: str,
        parent_cid: str,
        kind: str,
        request_name: str = "",
        provider: str = PROVIDER,
    ) -> None:
        self.ctx = ctx
        self.cid = cid
        self.parent_cid = parent_cid
        self.start = time.monotonic()
        self.kind = kind
        self.request_name = request_name
        self.provider = provider


class CaptureProcessor(TracingProcessor):  # type: ignore[misc]
    """Translates Agents SDK spans into canonical journal events."""

    def __init__(self) -> None:
        self._spans: dict[tuple[str, str], _SpanState] = {}
        self._lock = threading.RLock()

    def owns_span(self, span: Any) -> bool:
        try:
            key = (span.trace_id, span.span_id)
        except Exception:
            return False
        with self._lock:
            state = self._spans.get(key)
            return state is not None and state.kind == "lm"

    # -- processor interface -------------------------------------------------

    @_never_raise_processor
    def on_trace_start(self, trace: Any) -> None:
        return None

    @_never_raise_processor
    def on_trace_end(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            with self._lock:
                for key in [k for k in self._spans if k[0] == trace_id]:
                    self._spans.pop(key, None)
        return None

    @_never_raise_processor
    def on_span_start(self, span: Any) -> None:
        if not _enabled:
            return None
        ctx = ambient_context()
        if ctx is None:
            return None
        parent_cid = self._parent_cid(span, ctx)
        data = span.span_data
        key = (span.trace_id, span.span_id)
        cid = generate_cid()

        if isinstance(data, AgentSpanData):
            with self._lock:
                self._spans[key] = _SpanState(ctx, cid, parent_cid, "agent")
            safe_emit(
                ctx,
                AgentStarted(
                    name=data.name,
                    correlation_id=cid,
                    parent_correlation_id=parent_cid,
                    tool_names=list(itertools.islice(iter(data.tools or []), 128)),
                    metadata=observed_metadata(SOURCE, data.name),
                ),
            )
        elif isinstance(data, (GenerationSpanData, ResponseSpanData)):
            # Response spans learn their model only at end; start under the
            # provider name and resolve on completion.
            provider, request_name = _provider_and_name(getattr(data, "model", None))
            with self._lock:
                self._spans[key] = _SpanState(ctx, cid, parent_cid, "lm", request_name, provider)
            input_data = (
                capture_payload(_lm_input_data(data)) if content_capture_enabled() else None
            )
            safe_emit(
                ctx,
                build_lm_started(
                    source=SOURCE,
                    name=request_name,
                    model=request_name,
                    provider=provider,
                    correlation_id=cid,
                    parent_correlation_id=parent_cid,
                    input_data=input_data,
                ),
            )
        elif isinstance(data, FunctionSpanData):
            with self._lock:
                self._spans[key] = _SpanState(ctx, cid, parent_cid, "tool", data.name)
            safe_emit(
                ctx,
                ToolCallStarted(
                    name=data.name,
                    correlation_id=cid,
                    parent_correlation_id=parent_cid,
                    tool_name=data.name,
                    tool_call_id=cid,
                    input_data=(capture_payload({"input": data.input})),
                    metadata=observed_metadata(SOURCE, data.name),
                ),
            )
        else:
            # Transparent passthrough: children parent through to our parent.
            with self._lock:
                self._spans[key] = _SpanState(ctx, parent_cid, parent_cid, "passthrough")
        return None

    @_never_raise_processor
    def on_span_end(self, span: Any) -> None:
        with self._lock:
            state = self._spans.pop((span.trace_id, span.span_id), None)
        if state is None or state.kind == "passthrough" or not _enabled:
            return None
        duration_ms = int((time.monotonic() - state.start) * 1000)
        error = getattr(span, "error", None)
        data = span.span_data
        if state.kind == "agent":
            self._end_agent(state, data, duration_ms, error)
        elif state.kind == "lm":
            self._end_lm(state, data, duration_ms, error)
        elif state.kind == "tool":
            self._end_tool(state, data, duration_ms, error)
        return None

    @_never_raise_processor
    def shutdown(self) -> None:
        return None

    @_never_raise_processor
    def force_flush(self) -> None:
        return None

    # -- emit helpers --------------------------------------------------------

    def _end_agent(self, state: _SpanState, data: Any, duration_ms: int, error: Any) -> None:
        name = getattr(data, "name", "") or ""
        if error:
            safe_emit(
                state.ctx,
                AgentFailed(
                    name=name,
                    correlation_id=state.cid,
                    parent_correlation_id=state.parent_cid,
                    error_code="SpanError",
                    error_message=_error_message(error),
                    duration_ms=duration_ms,
                    metadata=observed_metadata(SOURCE, name),
                ),
            )
            return
        safe_emit(
            state.ctx,
            AgentCompleted(
                name=name,
                correlation_id=state.cid,
                parent_correlation_id=state.parent_cid,
                duration_ms=duration_ms,
                metadata=observed_metadata(SOURCE, name),
            ),
        )

    def _end_lm(self, state: _SpanState, data: Any, duration_ms: int, error: Any) -> None:
        if error:
            safe_emit(
                state.ctx,
                build_lm_failed(
                    source=SOURCE,
                    name=state.request_name,
                    model=state.request_name,
                    provider=state.provider,
                    correlation_id=state.cid,
                    parent_correlation_id=state.parent_cid,
                    duration_ms=duration_ms,
                    error=RuntimeError(_error_message(error)),
                ),
            )
            return
        fields = _lm_completed_fields(data, state.request_name, state.provider)
        name = fields.pop("name")
        provider = fields.pop("_provider", state.provider)
        response_error = fields.pop("_error", None)
        if response_error is not None:
            safe_emit(
                state.ctx,
                build_lm_failed(
                    source=SOURCE,
                    name=name,
                    model=name,
                    provider=provider,
                    correlation_id=state.cid,
                    parent_correlation_id=state.parent_cid,
                    duration_ms=duration_ms,
                    error=response_error,
                ),
            )
            return
        safe_emit(
            state.ctx,
            build_lm_completed(
                source=SOURCE,
                name=name,
                model=name,
                provider=provider,
                correlation_id=state.cid,
                parent_correlation_id=state.parent_cid,
                duration_ms=duration_ms,
                **fields,
            ),
        )

    def _end_tool(self, state: _SpanState, data: Any, duration_ms: int, error: Any) -> None:
        name = state.request_name
        if error:
            safe_emit(
                state.ctx,
                ToolCallFailed(
                    name=name,
                    correlation_id=state.cid,
                    parent_correlation_id=state.parent_cid,
                    tool_name=name,
                    tool_call_id=state.cid,
                    error_code="SpanError",
                    error_message=_error_message(error),
                    duration_ms=duration_ms,
                    metadata=observed_metadata(SOURCE, name),
                ),
            )
            return
        safe_emit(
            state.ctx,
            ToolCallCompleted(
                name=name,
                correlation_id=state.cid,
                parent_correlation_id=state.parent_cid,
                tool_name=name,
                tool_call_id=state.cid,
                duration_ms=duration_ms,
                output_data=capture_payload({"result": getattr(data, "output", None)}),
                metadata=observed_metadata(SOURCE, name),
            ),
        )

    def _parent_cid(self, span: Any, ctx: Any) -> str:
        parent_id = getattr(span, "parent_id", None)
        if parent_id:
            with self._lock:
                state = self._spans.get((span.trace_id, parent_id))
            if state is not None:
                return state.cid
        return ctx.correlation_id


_KNOWN_MODEL_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "mistral",
        "meta",
        "ollama",
        "openai",
        "openrouter",
        "together_ai",
        "vertex_ai",
    }
)
_OPENAI_MODEL_PREFIXES = (
    "chatgpt-",
    "codex-",
    "computer-use-",
    "ft:",
    "gpt-",
    "o1",
    "o3",
    "o4",
    "text-",
)


def _provider_and_name(model: Any) -> tuple[str, str]:
    model_text = str(model or "").strip()
    if "/" in model_text:
        prefix, remainder = model_text.split("/", 1)
        provider = prefix.lower().replace("-", "_")
        if provider in _KNOWN_MODEL_PROVIDERS and remainder:
            return provider, model_text
    lower = model_text.lower()
    for provider, prefixes in (
        ("anthropic", ("claude",)),
        ("google", ("gemini",)),
        ("mistral", ("mistral", "codestral", "ministral")),
        ("cohere", ("command",)),
        ("deepseek", ("deepseek",)),
        ("meta", ("llama",)),
    ):
        if lower.startswith(prefixes):
            return provider, f"{provider}/{model_text}"
    if not model_text or lower.startswith(_OPENAI_MODEL_PREFIXES):
        return PROVIDER, f"{PROVIDER}/{model_text}" if model_text else PROVIDER
    return "openai-compatible", f"openai-compatible/{model_text}"


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return capture_error_message(error.get("message") or error)
    return capture_error_message(getattr(error, "message", None) or error)


def _usage_int(usage: Any, *names: str) -> int:
    if not usage:
        return 0
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _lm_input_data(data: Any) -> Optional[dict[str, Any]]:
    try:
        value = getattr(data, "input", None)
        config = getattr(data, "model_config", None) or {}
        return {
            "system_prompt": None,
            "messages": value,
            "temperature": config.get("temperature") if isinstance(config, dict) else None,
            "max_tokens": config.get("max_tokens") if isinstance(config, dict) else None,
            "tools_count": 0,
        }
    except Exception:
        return None


def _lm_completed_fields(data: Any, request_name: str, provider: str) -> dict[str, Any]:
    response = getattr(data, "response", None)
    if response is not None:
        # ResponseSpanData carries a full openai Response — reuse the raw
        # client capture's extraction so both paths stay in lockstep.
        from .openai import _responses_completed_fields

        resolved_provider, _ = _provider_and_name(getattr(response, "model", None))
        fields = _responses_completed_fields(response, request_name, resolved_provider)
        fields["_provider"] = resolved_provider
        return fields

    usage = getattr(data, "usage", None)
    model = getattr(data, "model", None)
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    output_data: Optional[dict[str, Any]] = None
    if content_capture_enabled():
        output_data = capture_payload({"output": getattr(data, "output", None), "tool_calls": None})
    resolved_provider, resolved_name = _provider_and_name(model)
    return {
        "name": resolved_name if model else request_name,
        "_provider": resolved_provider if model else provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _usage_int(usage, "total_tokens") or input_tokens + output_tokens,
        "cached_tokens": _usage_int(usage, "cached_tokens"),
        "finish_reason": None,
        "output_data": output_data,
    }
