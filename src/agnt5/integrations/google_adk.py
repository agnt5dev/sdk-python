"""Journal capture for Google ADK agents.

Capture rides ADK's public plugin API (``BasePlugin`` callbacks); only the
attachment is a patch — ``enable()`` wraps ``Runner.__init__`` to register
the plugin post-init when absent. Named ``google_adk`` everywhere: bare
``adk`` collides with the AGNT5-internal Agent Development Toolkit in
sdk-core.

Callbacks observe and never override: every hook returns None so ADK
proceeds normally. ADK skip semantics mean ``after_*`` may never fire for a
span (e.g. another plugin short-circuits an agent); unpaired ``.started``
events are an accepted consequence.
"""

from __future__ import annotations

import functools
import itertools
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    capture_text_limit,
    content_capture_enabled,
    observed_metadata,
    safe_emit_async,
    supported_package_version,
)

logger = logging.getLogger(__name__)

SOURCE = "google_adk"
PROVIDER = "google"
PLUGIN_NAME = "agnt5_capture"

_patched = False
_original_runner_init: Optional[Callable[..., None]] = None
_runner_wrapper: Optional[Callable[..., None]] = None


class _BasePluginFallback:
    def __init__(self, **_: Any) -> None:
        pass


BasePlugin: Any = _BasePluginFallback

try:
    from google.adk.plugins.base_plugin import BasePlugin

    _adk_available = True
except ImportError:
    _adk_available = False


def _never_raise_callback(method: Callable[..., Any]) -> Callable[..., Any]:
    """ADK propagates plugin exceptions into the user run; capture never may."""

    @functools.wraps(method)
    async def guarded(*args: Any, **kwargs: Any) -> Any:
        if not _patched:
            return None
        try:
            return await method(*args, **kwargs)
        except Exception:
            logger.debug("google_adk capture callback failed", exc_info=True)
            return None

    return guarded


@dataclass
class _SpanState:
    cid: str
    start: float
    parent_cid: str
    name: str = ""
    provider: str = PROVIDER
    call_id: str = ""


def enable() -> bool:
    """Attach the capture plugin to every Runner. Idempotent; False when ADK is absent."""
    global _patched, _original_runner_init, _runner_wrapper
    if _patched:
        return True
    if not _adk_available:
        logger.debug("google-adk not installed; capture disabled")
        return False
    if not supported_package_version("google-adk", minimum=(1, 7, 0), max_major_exclusive=3):
        return False

    try:
        from google.adk.runners import Runner
    except ImportError:
        logger.debug("google-adk Runner API unavailable; capture disabled")
        return False

    original_runner_init = Runner.__init__
    _original_runner_init = original_runner_init

    @functools.wraps(original_runner_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_runner_init(self, *args, **kwargs)
        if not _patched:
            return
        # Post-init registration covers every construction path (plugins
        # kwarg, App, InMemoryRunner) without the deprecated plugins kwarg.
        try:
            manager = getattr(self, "plugin_manager", None)
            if manager is not None and not any(
                getattr(p, "name", None) == PLUGIN_NAME for p in manager.plugins
            ):
                manager.register_plugin(CapturePlugin())
        except Exception:
            logger.debug("capture plugin injection failed", exc_info=True)

    _runner_wrapper = patched_init
    Runner.__init__ = patched_init  # type: ignore[method-assign]
    _patched = True
    logger.debug("google_adk capture enabled")
    return True


def disable() -> None:
    """Restore Runner.__init__ (opt-out and test isolation)."""
    global _patched, _original_runner_init, _runner_wrapper
    if not _patched:
        return
    from google.adk.runners import Runner

    if Runner.__init__ is _runner_wrapper and _original_runner_init is not None:
        Runner.__init__ = _original_runner_init  # type: ignore[method-assign]
    _original_runner_init = None
    _runner_wrapper = None
    _patched = False


class CapturePlugin(BasePlugin):  # type: ignore[misc]
    """Translates ADK plugin callbacks into canonical journal events.

    Span state is keyed by invocation so concurrent invocations (and
    parallel sub-agents, via agent_name / function_call_id) don't collide.
    """

    def __init__(self) -> None:
        if _adk_available:
            super().__init__(name=PLUGIN_NAME)
        # A ContextVar isolates sibling ParallelAgent tasks while inheriting the
        # parent stack into each child task.
        self._agent_stack: ContextVar[tuple[str, ...]] = ContextVar(
            f"agnt5_google_adk_agent_stack_{id(self)}", default=()
        )
        self._agent_spans: dict[tuple[str, str], _SpanState] = {}
        self._model_spans: dict[tuple[str, str], list[_SpanState]] = {}
        self._tool_spans: dict[tuple[str, str], _SpanState] = {}

    # -- agent lifecycle ----------------------------------------------------

    @_never_raise_callback
    async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        invocation_id = callback_context.invocation_id
        stack = self._agent_stack.get()
        cid = generate_cid()
        parent = stack[-1] if stack else ctx.correlation_id
        self._agent_spans[(invocation_id, callback_context.agent_name)] = _SpanState(
            cid=cid,
            start=time.monotonic(),
            parent_cid=parent,
            name=callback_context.agent_name,
        )
        self._agent_stack.set((*stack, cid))
        await safe_emit_async(
            ctx,
            AgentStarted(
                name=callback_context.agent_name,
                correlation_id=cid,
                parent_correlation_id=parent,
                agent_model=str(getattr(agent, "model", "") or ""),
                metadata=observed_metadata(SOURCE, callback_context.agent_name),
            ),
        )
        return None

    @_never_raise_callback
    async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        span = self._agent_spans.pop(
            (callback_context.invocation_id, callback_context.agent_name), None
        )
        if span is None:
            return None
        stack = self._agent_stack.get()
        if stack and stack[-1] == span.cid:
            self._agent_stack.set(stack[:-1])
        await safe_emit_async(
            ctx,
            AgentCompleted(
                name=callback_context.agent_name,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                duration_ms=int((time.monotonic() - span.start) * 1000),
                metadata=observed_metadata(SOURCE, callback_context.agent_name),
            ),
        )
        return None

    @_never_raise_callback
    async def on_agent_error_callback(
        self, *, agent: Any, callback_context: Any, error: Exception
    ) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        span = self._agent_spans.pop(
            (callback_context.invocation_id, callback_context.agent_name), None
        )
        if span is None:
            parent = self._parent_for(ctx)
            span = _SpanState(
                cid=generate_cid(),
                start=time.monotonic(),
                parent_cid=parent,
                name=callback_context.agent_name,
            )
        stack = self._agent_stack.get()
        if stack and stack[-1] == span.cid:
            self._agent_stack.set(stack[:-1])
        await safe_emit_async(
            ctx,
            AgentFailed(
                name=callback_context.agent_name,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                error_code=type(error).__name__,
                error_message=capture_error_message(error),
                duration_ms=int((time.monotonic() - span.start) * 1000),
                metadata=observed_metadata(SOURCE, callback_context.agent_name),
            ),
        )
        return None

    # -- model lifecycle ----------------------------------------------------

    @_never_raise_callback
    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        cid = generate_cid()
        provider, request_name = _provider_and_name(getattr(llm_request, "model", None))
        parent = self._parent_for(ctx)
        key = (callback_context.invocation_id, callback_context.agent_name)
        self._model_spans.setdefault(key, []).append(
            _SpanState(
                cid=cid,
                start=time.monotonic(),
                parent_cid=parent,
                name=request_name,
                provider=provider,
            )
        )
        await safe_emit_async(
            ctx,
            build_lm_started(
                source=SOURCE,
                name=request_name,
                model=request_name,
                provider=provider,
                correlation_id=cid,
                parent_correlation_id=parent,
                input_data=(
                    capture_payload(_request_input_data(llm_request))
                    if content_capture_enabled()
                    else None
                ),
            ),
        )
        return None

    @_never_raise_callback
    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        # Streaming/live emits partial responses; only the final one closes the span.
        if getattr(llm_response, "partial", None):
            return None
        key = (callback_context.invocation_id, callback_context.agent_name)
        spans = self._model_spans.get(key)
        if not spans:
            return None
        span = spans.pop()
        if not spans:
            self._model_spans.pop(key, None)
        model_version = getattr(llm_response, "model_version", None)
        _, name = (
            _provider_and_name(model_version, default_provider=span.provider)
            if model_version
            else (span.provider, span.name)
        )
        usage = getattr(llm_response, "usage_metadata", None)
        finish_reason = getattr(llm_response, "finish_reason", None)

        output_data: Optional[dict[str, Any]] = None
        if content_capture_enabled():
            output_data = capture_payload(
                {
                    "output": _response_text(llm_response),
                    "tool_calls": _response_tool_calls(llm_response),
                }
            )

        await safe_emit_async(
            ctx,
            build_lm_completed(
                source=SOURCE,
                name=name,
                model=name,
                provider=span.provider,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                duration_ms=int((time.monotonic() - span.start) * 1000),
                input_tokens=_usage_int(usage, "prompt_token_count"),
                output_tokens=_usage_int(usage, "candidates_token_count"),
                total_tokens=_usage_int(usage, "total_token_count"),
                cached_tokens=_usage_int(usage, "cached_content_token_count"),
                finish_reason=getattr(finish_reason, "name", None) if finish_reason else None,
                output_data=output_data,
            ),
        )
        return None

    @_never_raise_callback
    async def on_model_error_callback(
        self, *, callback_context: Any, llm_request: Any, error: Exception
    ) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        key = (callback_context.invocation_id, callback_context.agent_name)
        spans = self._model_spans.get(key)
        if spans:
            span = spans.pop()
            if not spans:
                self._model_spans.pop(key, None)
        else:
            provider, request_name = _provider_and_name(getattr(llm_request, "model", None))
            span = _SpanState(
                cid=generate_cid(),
                start=time.monotonic(),
                parent_cid=self._parent_for(ctx),
                name=request_name,
                provider=provider,
            )
        await safe_emit_async(
            ctx,
            build_lm_failed(
                source=SOURCE,
                name=span.name,
                model=span.name,
                provider=span.provider,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                duration_ms=int((time.monotonic() - span.start) * 1000),
                error=error,
            ),
        )
        return None

    # -- tool lifecycle -----------------------------------------------------

    @_never_raise_callback
    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        cid = generate_cid()
        tool_name = getattr(tool, "name", "") or ""
        call_id = str(getattr(tool_context, "function_call_id", None) or cid)
        key = (tool_context.invocation_id, _tool_context_key(tool_context))
        parent = self._parent_for(ctx)
        self._tool_spans[key] = _SpanState(
            cid=cid,
            start=time.monotonic(),
            parent_cid=parent,
            name=tool_name,
            call_id=call_id,
        )
        await safe_emit_async(
            ctx,
            ToolCallStarted(
                name=tool_name,
                correlation_id=cid,
                parent_correlation_id=parent,
                tool_name=tool_name,
                tool_call_id=call_id,
                input_data=capture_payload(dict(tool_args)),
                metadata=observed_metadata(SOURCE, tool_name),
            ),
        )
        return None

    @_never_raise_callback
    async def after_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, result: dict[str, Any]
    ) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        key = (tool_context.invocation_id, _tool_context_key(tool_context))
        span = self._tool_spans.pop(key, None)
        if span is None:
            return None
        await safe_emit_async(
            ctx,
            ToolCallCompleted(
                name=span.name,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                tool_name=span.name,
                tool_call_id=span.call_id,
                duration_ms=int((time.monotonic() - span.start) * 1000),
                output_data=capture_payload({"result": result}),
                metadata=observed_metadata(SOURCE, span.name),
            ),
        )
        return None

    @_never_raise_callback
    async def on_tool_error_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, error: Exception
    ) -> None:
        ctx = ambient_context()
        if ctx is None:
            return None
        tool_name = getattr(tool, "name", "") or ""
        key = (tool_context.invocation_id, _tool_context_key(tool_context))
        span = self._tool_spans.pop(key, None)
        if span is None:
            cid = generate_cid()
            call_id = str(getattr(tool_context, "function_call_id", None) or cid)
            span = _SpanState(
                cid=cid,
                start=time.monotonic(),
                parent_cid=self._parent_for(ctx),
                name=tool_name,
                call_id=call_id,
            )
        await safe_emit_async(
            ctx,
            ToolCallFailed(
                name=span.name,
                correlation_id=span.cid,
                parent_correlation_id=span.parent_cid,
                tool_name=span.name,
                tool_call_id=span.call_id,
                error_code=type(error).__name__,
                error_message=capture_error_message(error),
                duration_ms=int((time.monotonic() - span.start) * 1000),
                metadata=observed_metadata(SOURCE, span.name),
            ),
        )
        return None

    # -- run lifecycle (state hygiene) --------------------------------------

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        try:
            self._purge(getattr(invocation_context, "invocation_id", None))
        except Exception:
            logger.debug("google_adk capture cleanup failed", exc_info=True)
        return None

    async def on_run_error_callback(self, *, invocation_context: Any, error: Exception) -> None:
        try:
            self._purge(getattr(invocation_context, "invocation_id", None))
        except Exception:
            logger.debug("google_adk capture cleanup failed", exc_info=True)
        return None

    # -- helpers ------------------------------------------------------------

    def _parent_for(self, ctx: Any, exclude: Optional[str] = None) -> str:
        stack = self._agent_stack.get()
        for cid in reversed(stack):
            if cid != exclude:
                return cid
        return ctx.correlation_id

    def _purge(self, invocation_id: Optional[str]) -> None:
        if not invocation_id:
            return
        self._agent_stack.set(())
        for spans in (self._agent_spans, self._model_spans, self._tool_spans):
            for key in [k for k in spans if k[0] == invocation_id]:
                spans.pop(key, None)


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
        "ollama",
        "openai",
        "openrouter",
        "together_ai",
        "vertex_ai",
    }
)


def _provider_and_name(model: Any, *, default_provider: str = PROVIDER) -> tuple[str, str]:
    text = str(model or "").strip()
    if not text:
        return default_provider, default_provider
    if "/" in text:
        provider = text.split("/", 1)[0].lower().replace("-", "_")
        if provider in _KNOWN_MODEL_PROVIDERS:
            return provider, text
    return default_provider, f"{default_provider}/{text}"


def _usage_int(usage: Any, attr: str) -> int:
    if usage is None:
        return 0
    try:
        return int(getattr(usage, attr, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _tool_context_key(tool_context: Any) -> str:
    call_id = getattr(tool_context, "function_call_id", None)
    if call_id:
        return f"call:{call_id}"
    # ADK reuses the same ToolContext object for before/after/error callbacks,
    # even when a synthetic/custom invocation omitted its function-call ID.
    return f"context:{id(tool_context)}"


def _request_input_data(llm_request: Any) -> Optional[dict[str, Any]]:
    try:
        config = getattr(llm_request, "config", None)
        system_instruction = getattr(config, "system_instruction", None)
        messages = [
            content.model_dump(exclude_none=True)
            for content in itertools.islice(iter(getattr(llm_request, "contents", None) or []), 128)
        ]
        return {
            "system_prompt": system_instruction if isinstance(system_instruction, str) else None,
            "messages": messages,
            "temperature": getattr(config, "temperature", None),
            "max_tokens": getattr(config, "max_output_tokens", None),
            "tools_count": len(getattr(llm_request, "tools_dict", None) or {}),
        }
    except Exception:
        logger.debug("llm_request serialization failed", exc_info=True)
        return None


def _response_text(llm_response: Any) -> Optional[str]:
    try:
        parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
        captured: list[str] = []
        captured_length = 0
        for part in itertools.islice(iter(parts), 128):
            text = getattr(part, "text", None)
            if not text:
                continue
            remaining = capture_text_limit() - captured_length
            if remaining <= 0:
                break
            addition = str(text)[:remaining]
            captured.append(addition)
            captured_length += len(addition)
        text = "".join(captured)
        return text or None
    except Exception:
        return None


def _response_tool_calls(llm_response: Any) -> Optional[list[dict[str, Any]]]:
    try:
        parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
        calls = [
            {
                "id": getattr(fc, "id", None),
                "name": getattr(fc, "name", None),
                "arguments": getattr(fc, "args", None),
            }
            for fc in itertools.islice((getattr(p, "function_call", None) for p in parts), 128)
            if fc is not None
        ]
        return calls or None
    except Exception:
        return None
