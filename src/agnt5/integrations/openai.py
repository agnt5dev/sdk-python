"""Journal capture for the raw ``openai`` Python SDK.

Patches the create methods of chat completions, the Responses API, and
embeddings (sync + async) so calls made by user code inside AGNT5 components
emit canonical ``lm.*`` events. Plain setattr wrapping, originals stored for
restore; activates only when the ``openai`` package is importable.

Streaming calls return a transparent proxy that accumulates bounded chunks
and emits ``lm.completed`` only on exhaustion. Early close emits ``lm.failed``
so partial output is never recorded as a successful generation (usage appears
only when the caller sets ``stream_options={"include_usage": True}``).
"""

from __future__ import annotations

import functools
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from ._common import (
    ambient_context,
    build_lm_completed,
    build_lm_failed,
    build_lm_started,
    capture_payload,
    capture_text_limit,
    content_capture_enabled,
    new_capture_span,
    safe_emit,
    safe_emit_async,
    supported_package_version,
)

logger = logging.getLogger(__name__)

SOURCE = "openai"
PROVIDER = "openai"

_patched = False
_originals: dict[str, Callable[..., Any]] = {}
_wrappers: dict[str, Callable[..., Any]] = {}


@dataclass(frozen=True)
class _Surface:
    """Per-API extraction spec: request/response shapes differ per surface."""

    key: str
    input_data: Callable[[dict[str, Any]], dict[str, Any]]
    completed: Callable[[Any, str, str], dict[str, Any]]
    stream_state: Optional[Callable[[], Any]] = None
    accumulate: Optional[Callable[[Any, Any], None]] = None
    stream_fields: Optional[Callable[[Any, str, str], dict[str, Any]]] = None
    stream_terminal: Optional[Callable[[Any], bool]] = None


def enable() -> bool:
    """Patch the openai client. Idempotent; False when openai is absent."""
    global _patched
    if _patched:
        return True
    if not supported_package_version("openai", minimum=(1, 66, 0), max_major_exclusive=3):
        return False
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
        from openai.resources.embeddings import AsyncEmbeddings, Embeddings
    except ImportError:
        logger.debug("openai not installed; capture disabled")
        return False

    try:
        _patch("chat", Completions, AsyncCompletions, _CHAT)
        _patch("embeddings", Embeddings, AsyncEmbeddings, _EMBEDDINGS)
        try:
            from openai.resources.responses import AsyncResponses, Responses

            _patch("responses", Responses, AsyncResponses, _RESPONSES)
        except ImportError:
            logger.debug("openai Responses API unavailable; skipping")
    except Exception:
        logger.warning("openai capture patching failed; rolling back", exc_info=True)
        _restore_patches()
        return False

    _patched = True
    logger.debug("openai capture enabled")
    return True


def disable() -> None:
    """Restore the original methods (opt-out and test isolation)."""
    global _patched
    _patched = False
    _restore_patches()


def _capture_classes() -> dict[str, tuple[type, type]]:
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings

    classes: dict[str, tuple[type, type]] = {
        "chat": (Completions, AsyncCompletions),
        "embeddings": (Embeddings, AsyncEmbeddings),
    }
    try:
        from openai.resources.responses import AsyncResponses, Responses

        classes["responses"] = (Responses, AsyncResponses)
    except ImportError:
        pass
    return classes


def _restore_patches() -> None:
    try:
        classes = _capture_classes()
    except ImportError:
        classes = {}
    for key, (sync_cls, async_cls) in classes.items():
        for suffix, cls in (("sync", sync_cls), ("async", async_cls)):
            patch_key = f"{key}_{suffix}"
            original = _originals.get(patch_key)
            wrapper = _wrappers.get(patch_key)
            if original is not None and wrapper is not None and cls.create is wrapper:
                cls.create = original  # type: ignore[method-assign]
    _originals.clear()
    _wrappers.clear()


def _patch(key: str, sync_cls: type, async_cls: type, surface: _Surface) -> None:
    _originals[f"{key}_sync"] = sync_cls.create
    _originals[f"{key}_async"] = async_cls.create
    sync_wrapper = _wrap_sync(sync_cls.create, surface)
    async_wrapper = _wrap_async(async_cls.create, surface)
    _wrappers[f"{key}_sync"] = sync_wrapper
    _wrappers[f"{key}_async"] = async_wrapper
    sync_cls.create = sync_wrapper  # type: ignore[method-assign]
    async_cls.create = async_wrapper  # type: ignore[method-assign]


# =============================================================================
# Wrappers
# =============================================================================


def _suppressed_by_agents_capture() -> bool:
    # The OpenAI Agents SDK capture journals its own model calls; don't
    # journal the same underlying client call twice.
    try:
        from . import openai_agents

        return openai_agents.suppresses_client_capture()
    except Exception:
        return False


def _wrap_sync(original: Callable[..., Any], surface: _Surface) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ambient_context()
        streaming = bool(kwargs.get("stream"))
        if (
            not _patched
            or ctx is None
            or (streaming and surface.stream_state is None)
            or _suppressed_by_agents_capture()
        ):
            return original(self, *args, **kwargs)

        cid, parent = new_capture_span(ctx)
        provider = _provider_for_resource(self)
        request_name = _prefixed(kwargs.get("model"), provider)
        capture_kwargs = _snapshot_capture_kwargs(surface, kwargs)
        try:
            safe_emit(
                ctx,
                _started_event(surface, request_name, provider, cid, parent, capture_kwargs),
            )
        except Exception:
            logger.debug("openai started-event capture failed", exc_info=True)
        start = time.monotonic()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            try:
                safe_emit(ctx, _failed_event(request_name, provider, cid, parent, start, exc))
            except Exception:
                logger.debug("openai failed-event capture failed", exc_info=True)
            raise
        if streaming:
            return _CaptureStream(
                response, ctx, surface, request_name, provider, cid, parent, start
            )
        try:
            safe_emit(
                ctx,
                _completed_event(
                    surface.completed(response, request_name, provider),
                    provider,
                    cid,
                    parent,
                    start,
                ),
            )
        except Exception:
            logger.debug("openai completed-event capture failed", exc_info=True)
        return response

    return wrapper


def _wrap_async(original: Callable[..., Any], surface: _Surface) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ambient_context()
        streaming = bool(kwargs.get("stream"))
        if (
            not _patched
            or ctx is None
            or (streaming and surface.stream_state is None)
            or _suppressed_by_agents_capture()
        ):
            return await original(self, *args, **kwargs)

        cid, parent = new_capture_span(ctx)
        provider = _provider_for_resource(self)
        request_name = _prefixed(kwargs.get("model"), provider)
        capture_kwargs = _snapshot_capture_kwargs(surface, kwargs)
        try:
            await safe_emit_async(
                ctx,
                _started_event(surface, request_name, provider, cid, parent, capture_kwargs),
            )
        except Exception:
            logger.debug("openai started-event capture failed", exc_info=True)
        start = time.monotonic()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            try:
                await safe_emit_async(
                    ctx, _failed_event(request_name, provider, cid, parent, start, exc)
                )
            except Exception:
                logger.debug("openai failed-event capture failed", exc_info=True)
            raise
        if streaming:
            return _CaptureAsyncStream(
                response, ctx, surface, request_name, provider, cid, parent, start
            )
        try:
            await safe_emit_async(
                ctx,
                _completed_event(
                    surface.completed(response, request_name, provider),
                    provider,
                    cid,
                    parent,
                    start,
                ),
            )
        except Exception:
            logger.debug("openai completed-event capture failed", exc_info=True)
        return response

    return wrapper


def _snapshot_capture_kwargs(surface: _Surface, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot iterable inputs without consuming what the OpenAI call receives."""
    capture_kwargs = dict(request_kwargs)
    keys = ("messages", "tools") if surface.key == "chat" else ()
    if surface.key == "responses":
        keys = ("tools",)
    for key in keys:
        value = request_kwargs.get(key)
        if value is None or isinstance(value, (str, bytes, dict)):
            continue
        try:
            if isinstance(value, (list, tuple)):
                capture_kwargs[key] = value[:128]
            else:
                capture_iter, request_iter = itertools.tee(iter(value))
                request_kwargs[key] = request_iter
                capture_kwargs[key] = list(itertools.islice(capture_iter, 128))
        except Exception:
            logger.debug("unable to snapshot OpenAI iterable %s", key, exc_info=True)
            capture_kwargs.pop(key, None)
    return capture_kwargs


def _provider_for_resource(resource: Any) -> str:
    """Avoid pricing arbitrary OpenAI-compatible endpoints as OpenAI."""
    try:
        client = getattr(resource, "_client", None)
        base_url = getattr(client, "base_url", None) or getattr(client, "_base_url", None)
        host = (urlparse(str(base_url)).hostname or "").lower()
        if not host or host == "api.openai.com" or host.endswith(".openai.azure.com"):
            return PROVIDER
        return "openai-compatible"
    except Exception:
        return PROVIDER


# =============================================================================
# Streaming proxies
# =============================================================================


class _StreamCaptureMixin:
    """Chunk accumulation + one-shot event construction shared by both proxies."""

    def _init_capture(
        self,
        inner: Any,
        ctx: Any,
        surface: _Surface,
        request_name: str,
        provider: str,
        cid: str,
        parent: str,
        start: float,
    ) -> None:
        self._inner = inner
        self._ctx = ctx
        self._surface = surface
        self._request_name = request_name
        self._provider = provider
        self._cid = cid
        self._parent = parent
        self._start = start
        self._state = surface.stream_state()  # type: ignore[misc]
        self._finished = False

    def _accumulate(self, chunk: Any) -> None:
        try:
            self._surface.accumulate(self._state, chunk)  # type: ignore[misc]
        except Exception:
            logger.debug("stream accumulate failed", exc_info=True)

    def _completed(self) -> Optional[Any]:
        if self._finished:
            return None
        self._finished = True
        try:
            fields = self._surface.stream_fields(  # type: ignore[misc]
                self._state, self._request_name, self._provider
            )
            return _completed_event(fields, self._provider, self._cid, self._parent, self._start)
        except Exception as exc:
            logger.debug("stream completion capture failed", exc_info=True)
            try:
                return _failed_event(
                    self._request_name,
                    self._provider,
                    self._cid,
                    self._parent,
                    self._start,
                    exc,
                )
            except Exception:
                logger.debug("stream fallback failure capture failed", exc_info=True)
                return None

    def _failed(self, error: BaseException) -> Optional[Any]:
        if self._finished:
            return None
        self._finished = True
        try:
            return _failed_event(
                self._request_name,
                self._provider,
                self._cid,
                self._parent,
                self._start,
                error,
            )
        except Exception:
            logger.debug("stream failure capture failed", exc_info=True)
            return None

    def _closed(self) -> Optional[Any]:
        """Complete only when the stream exposed a provider terminal marker."""
        if self._finished:
            return None
        try:
            terminal = self._surface.stream_terminal
            if terminal is not None and terminal(self._state):
                return self._completed()
        except Exception:
            logger.debug("stream terminal-state detection failed", exc_info=True)
        return self._failed(RuntimeError("OpenAI stream closed before exhaustion"))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class _CaptureStream(_StreamCaptureMixin):
    def __init__(self, inner, ctx, surface, request_name, provider, cid, parent, start):
        self._init_capture(inner, ctx, surface, request_name, provider, cid, parent, start)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self._inner.__next__()
        except StopIteration:
            event = self._completed()
            if event is not None:
                safe_emit(self._ctx, event)
            raise
        except Exception as exc:
            event = self._failed(exc)
            if event is not None:
                safe_emit(self._ctx, event)
            raise
        self._accumulate(chunk)
        return chunk

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            result = self._inner.__exit__(exc_type, exc, tb)
        except Exception as close_error:
            event = self._failed(close_error)
            if event is not None:
                safe_emit(self._ctx, event)
            raise
        event = self._failed(exc) if exc is not None else self._closed()
        if event is not None:
            safe_emit(self._ctx, event)
        return result

    def close(self):
        try:
            self._inner.close()
        except Exception as exc:
            event = self._failed(exc)
            if event is not None:
                safe_emit(self._ctx, event)
            raise
        event = self._closed()
        if event is not None:
            safe_emit(self._ctx, event)


class _CaptureAsyncStream(_StreamCaptureMixin):
    def __init__(self, inner, ctx, surface, request_name, provider, cid, parent, start):
        self._init_capture(inner, ctx, surface, request_name, provider, cid, parent, start)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._inner.__anext__()
        except StopAsyncIteration:
            event = self._completed()
            if event is not None:
                await safe_emit_async(self._ctx, event)
            raise
        except Exception as exc:
            event = self._failed(exc)
            if event is not None:
                await safe_emit_async(self._ctx, event)
            raise
        self._accumulate(chunk)
        return chunk

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            result = await self._inner.__aexit__(exc_type, exc, tb)
        except Exception as close_error:
            event = self._failed(close_error)
            if event is not None:
                await safe_emit_async(self._ctx, event)
            raise
        event = self._failed(exc) if exc is not None else self._closed()
        if event is not None:
            await safe_emit_async(self._ctx, event)
        return result

    async def close(self):
        try:
            await self._inner.close()
        except Exception as exc:
            event = self._failed(exc)
            if event is not None:
                await safe_emit_async(self._ctx, event)
            raise
        event = self._closed()
        if event is not None:
            await safe_emit_async(self._ctx, event)


# =============================================================================
# Event construction
# =============================================================================


def _prefixed(model: Optional[str], provider: str = PROVIDER) -> str:
    # Provider-prefixed model string, matching the native LM client's naming
    # (e.g. "openai/gpt-4o-mini") so server-side cost enrichment applies.
    if not model:
        return provider
    model_text = str(model)
    if model_text.startswith(f"{provider}/"):
        return model_text
    return f"{provider}/{model_text}"


def _token(usage: Any, attr: str) -> int:
    if usage is None:
        return 0
    try:
        return int(getattr(usage, attr, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _started_event(
    surface: _Surface,
    name: str,
    provider: str,
    cid: str,
    parent: str,
    kwargs: dict[str, Any],
) -> Any:
    input_data = capture_payload(surface.input_data(kwargs)) if content_capture_enabled() else None
    return build_lm_started(
        source=SOURCE,
        name=name,
        model=name,
        provider=provider,
        correlation_id=cid,
        parent_correlation_id=parent,
        input_data=input_data,
    )


def _completed_event(
    fields: dict[str, Any], provider: str, cid: str, parent: str, start: float
) -> Any:
    fields = dict(fields)
    name = fields.pop("name")
    error = fields.pop("_error", None)
    if error is not None:
        return build_lm_failed(
            source=SOURCE,
            name=name,
            model=name,
            provider=provider,
            correlation_id=cid,
            parent_correlation_id=parent,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=error,
        )
    return build_lm_completed(
        source=SOURCE,
        name=name,
        model=name,
        provider=provider,
        correlation_id=cid,
        parent_correlation_id=parent,
        duration_ms=int((time.monotonic() - start) * 1000),
        **fields,
    )


def _failed_event(
    name: str,
    provider: str,
    cid: str,
    parent: str,
    start: float,
    error: BaseException,
) -> Any:
    return build_lm_failed(
        source=SOURCE,
        name=name,
        model=name,
        provider=provider,
        correlation_id=cid,
        parent_correlation_id=parent,
        duration_ms=int((time.monotonic() - start) * 1000),
        error=error,
    )


# =============================================================================
# Chat completions surface
# =============================================================================


def _chat_input_data(kwargs: dict[str, Any]) -> dict[str, Any]:
    system_prompt, messages = _split_messages(kwargs.get("messages"))
    return {
        "system_prompt": system_prompt,
        "messages": messages,
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        "tools_count": len(kwargs.get("tools") or []),
    }


def _chat_completed_fields(response: Any, request_name: str, provider: str) -> dict[str, Any]:
    # Prefer the resolved model on the response over the request alias.
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)

    output_data: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        finish_reason = getattr(first, "finish_reason", None)
        if content_capture_enabled():
            message = getattr(first, "message", None)
            output_data = capture_payload(
                {
                    "output": getattr(message, "content", None),
                    "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
                }
            )

    return {
        "name": _prefixed(model, provider) if model else request_name,
        "input_tokens": _token(usage, "prompt_tokens"),
        "output_tokens": _token(usage, "completion_tokens"),
        "total_tokens": _token(usage, "total_tokens"),
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
        "finish_reason": finish_reason,
        "output_data": output_data,
    }


class _ChatStreamState:
    __slots__ = (
        "model",
        "finish_reason",
        "usage",
        "parts",
        "parts_length",
        "tool_arguments_length",
        "tool_calls",
    )

    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.usage: Any = None
        self.parts: list[str] = []
        self.parts_length = 0
        self.tool_arguments_length = 0
        self.tool_calls: dict[int, dict[str, Any]] = {}


def _chat_accumulate(state: _ChatStreamState, chunk: Any) -> None:
    if getattr(chunk, "model", None):
        state.model = chunk.model
    if getattr(chunk, "usage", None):
        state.usage = chunk.usage
    choices = getattr(chunk, "choices", None)
    if not choices:
        return
    first = choices[0]
    if getattr(first, "finish_reason", None):
        state.finish_reason = first.finish_reason
    delta = getattr(first, "delta", None)
    if delta is None:
        return
    content = getattr(delta, "content", None)
    if content and content_capture_enabled():
        remaining = capture_text_limit() - state.parts_length - state.tool_arguments_length
        if remaining > 0:
            part = str(content)[:remaining]
            state.parts.append(part)
            state.parts_length += len(part)
    for tc in getattr(delta, "tool_calls", None) or []:
        index = getattr(tc, "index", 0) or 0
        entry = state.tool_calls.get(index)
        if entry is None:
            if len(state.tool_calls) >= 128:
                continue
            entry = {"id": None, "name": None, "arguments": ""}
            state.tool_calls[index] = entry
        if getattr(tc, "id", None):
            entry["id"] = tc.id
        function = getattr(tc, "function", None)
        if function is not None:
            if getattr(function, "name", None):
                entry["name"] = function.name
            if getattr(function, "arguments", None):
                remaining = capture_text_limit() - state.parts_length - state.tool_arguments_length
                if remaining > 0:
                    addition = str(function.arguments)[:remaining]
                    entry["arguments"] += addition
                    state.tool_arguments_length += len(addition)


def _chat_stream_fields(
    state: _ChatStreamState, request_name: str, provider: str
) -> dict[str, Any]:
    usage = state.usage
    details = getattr(usage, "prompt_tokens_details", None)
    output_data: Optional[dict[str, Any]] = None
    if content_capture_enabled():
        tool_calls = [state.tool_calls[i] for i in sorted(state.tool_calls)] or None
        output_data = capture_payload(
            {"output": "".join(state.parts) or None, "tool_calls": tool_calls}
        )
    return {
        "name": _prefixed(state.model, provider) if state.model else request_name,
        "input_tokens": _token(usage, "prompt_tokens"),
        "output_tokens": _token(usage, "completion_tokens"),
        "total_tokens": _token(usage, "total_tokens"),
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
        "finish_reason": state.finish_reason,
        "output_data": output_data,
    }


_CHAT = _Surface(
    key="chat",
    input_data=_chat_input_data,
    completed=_chat_completed_fields,
    stream_state=_ChatStreamState,
    accumulate=_chat_accumulate,
    stream_fields=_chat_stream_fields,
    stream_terminal=lambda state: state.finish_reason is not None,
)


# =============================================================================
# Responses API surface
# =============================================================================


def _responses_input_data(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "system_prompt": kwargs.get("instructions"),
        "messages": kwargs.get("input"),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_output_tokens"),
        "tools_count": len(kwargs.get("tools") or []),
    }


def _responses_completed_fields(response: Any, request_name: str, provider: str) -> dict[str, Any]:
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)

    output_data: Optional[dict[str, Any]] = None
    if content_capture_enabled():
        output_data = capture_payload(
            {
                "output": getattr(response, "output_text", None) or None,
                "tool_calls": _serialize_response_tool_calls(getattr(response, "output", None)),
            }
        )

    status = str(getattr(response, "status", "") or "").lower()
    response_error: Optional[BaseException] = None
    if status and status != "completed":
        error = getattr(response, "error", None)
        message = (
            error.get("message") if isinstance(error, dict) else getattr(error, "message", None)
        ) or f"response status is {status}"
        response_error = RuntimeError(str(message))

    return {
        "name": _prefixed(model, provider) if model else request_name,
        "input_tokens": _token(usage, "input_tokens"),
        "output_tokens": _token(usage, "output_tokens"),
        "total_tokens": _token(usage, "total_tokens"),
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
        "finish_reason": getattr(response, "status", None),
        "output_data": output_data,
        "_error": response_error,
    }


def _responses_stream_state() -> dict[str, Any]:
    return {"response": None, "error": None}


def _responses_accumulate(state: dict[str, Any], event: Any) -> None:
    # The terminal stream event carries the complete Response object.
    event_type = getattr(event, "type", None)
    if event_type in {"response.completed", "response.failed", "response.incomplete"}:
        state["response"] = event.response
    elif event_type == "error":
        message = getattr(event, "message", None) or "Responses API stream error"
        state["error"] = RuntimeError(str(message))


def _responses_stream_fields(
    state: dict[str, Any], request_name: str, provider: str
) -> dict[str, Any]:
    if state["response"] is not None:
        return _responses_completed_fields(state["response"], request_name, provider)
    return {
        "name": request_name,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "finish_reason": None,
        "output_data": None,
        "_error": state["error"]
        or RuntimeError("Responses API stream ended without a terminal response"),
    }


_RESPONSES = _Surface(
    key="responses",
    input_data=_responses_input_data,
    completed=_responses_completed_fields,
    stream_state=_responses_stream_state,
    accumulate=_responses_accumulate,
    stream_fields=_responses_stream_fields,
    stream_terminal=lambda state: state["response"] is not None or state["error"] is not None,
)


# =============================================================================
# Embeddings surface
# =============================================================================


def _embeddings_input_data(kwargs: dict[str, Any]) -> dict[str, Any]:
    value = kwargs.get("input")
    return {
        "input": value,
        "input_count": len(value) if isinstance(value, (list, tuple)) else 1,
    }


def _embeddings_completed_fields(response: Any, request_name: str, provider: str) -> dict[str, Any]:
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    data = getattr(response, "data", None)
    return {
        "name": _prefixed(model, provider) if model else request_name,
        "input_tokens": _token(usage, "prompt_tokens"),
        "output_tokens": 0,
        "total_tokens": _token(usage, "total_tokens"),
        "cached_tokens": 0,
        "finish_reason": None,
        "output_data": {"embeddings_count": len(data) if data else 0},
    }


_EMBEDDINGS = _Surface(
    key="embeddings",
    input_data=_embeddings_input_data,
    completed=_embeddings_completed_fields,
)


# =============================================================================
# Serialization helpers
# =============================================================================


def _split_messages(messages: Any) -> tuple[Optional[str], list[Any]]:
    """Mirror the native LM client's shape: system prompt separate from turns."""
    system_prompt: Optional[str] = None
    turns: list[Any] = []
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role in ("system", "developer") and system_prompt is None:
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            system_prompt = content if isinstance(content, str) else None
            continue
        turns.append(message if isinstance(message, dict) else _message_to_dict(message))
    return system_prompt, turns


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(exclude_none=True)
        except Exception:
            pass
    return {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
    }


def _serialize_tool_calls(tool_calls: Any) -> Optional[list[dict[str, Any]]]:
    if not tool_calls:
        return None
    serialized = []
    for call in itertools.islice(iter(tool_calls), 128):
        function = getattr(call, "function", None)
        serialized.append(
            {
                "id": getattr(call, "id", None),
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None),
            }
        )
    return serialized


def _serialize_response_tool_calls(output: Any) -> Optional[list[dict[str, Any]]]:
    if not output:
        return None
    calls = [
        {
            "id": getattr(item, "call_id", None),
            "name": getattr(item, "name", None),
            "arguments": getattr(item, "arguments", None),
        }
        for item in itertools.islice(iter(output), 128)
        if getattr(item, "type", None) == "function_call"
    ]
    return calls or None
