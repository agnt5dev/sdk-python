"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import contextvars
import json
import logging
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Optional,
)

from ._telemetry import ContextLogger, get_execution_logger
from .events import Event, EventEmitter, EventEnvelope

if TYPE_CHECKING:
    from .memoization import MemoizationManager
    from .sandbox import Sandbox


# Task-local storage (NOT global) - each asyncio task gets its own copy
_current_context: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar(
    "_current_context", default=None
)
_current_memo_namespace: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_memo_namespace", default=None
)


def _memo_namespace_part(value: str) -> str:
    """Return a compact, stable namespace segment for memo step keys."""
    safe = []
    for char in str(value):
        if char.isalnum() or char in ("_", "-"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")[:80] or "scope"


class _CorrelationFilter(logging.Filter):
    """Inject correlation IDs (run_id, trace_id, span_id) into log records."""

    def __init__(self, runtime_context: Any) -> None:
        super().__init__()
        self.runtime_context = runtime_context

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.runtime_context.run_id
        if self.runtime_context.trace_id:
            record.trace_id = self.runtime_context.trace_id
        if self.runtime_context.span_id:
            record.span_id = self.runtime_context.span_id
        return True


@dataclass
class LLMRuntimeOptions:
    """Request-scoped LLM execution overrides supplied by the runtime."""

    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (self.model, self.temperature, self.max_tokens, self.top_p)
        )


@dataclass
class RuntimeContext:
    """Runtime-provided execution options available to user code."""

    llm: LLMRuntimeOptions = field(default_factory=LLMRuntimeOptions)
    prompts: dict[str, LLMRuntimeOptions] = field(default_factory=dict)


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _llm_options_from_mapping(data: dict[str, Any]) -> LLMRuntimeOptions:
    return LLMRuntimeOptions(
        model=str(data["model"]).strip() if data.get("model") else None,
        temperature=_parse_float(data.get("temperature")),
        max_tokens=_parse_int(
            data.get("max_tokens")
            if data.get("max_tokens") is not None
            else data.get("max_output_tokens")
        ),
        top_p=_parse_float(data.get("top_p")),
    )


def runtime_context_from_metadata(metadata: Optional[dict[str, Any]]) -> RuntimeContext:
    """Build public runtime options from dispatch metadata."""
    runtime = RuntimeContext()
    if not metadata:
        return runtime

    llm_data: dict[str, Any] = {}
    raw_llm = metadata.get("agnt5.llm")
    if isinstance(raw_llm, str) and raw_llm.strip():
        try:
            parsed = json.loads(raw_llm)
            if isinstance(parsed, dict):
                llm_data.update(parsed)
        except json.JSONDecodeError:
            pass
    elif isinstance(raw_llm, dict):
        llm_data.update(raw_llm)

    flat_keys = {
        "model": "agnt5.llm.model",
        "temperature": "agnt5.llm.temperature",
        "max_tokens": "agnt5.llm.max_tokens",
        "max_output_tokens": "agnt5.llm.max_output_tokens",
        "top_p": "agnt5.llm.top_p",
    }
    for target, key in flat_keys.items():
        if key in metadata:
            llm_data[target] = metadata[key]

    runtime.llm = _llm_options_from_mapping(llm_data)

    raw_prompts = metadata.get("agnt5.prompts")
    prompts_data: dict[str, Any] = {}
    if isinstance(raw_prompts, str) and raw_prompts.strip():
        try:
            parsed = json.loads(raw_prompts)
            if isinstance(parsed, dict):
                prompts_data.update(parsed)
        except json.JSONDecodeError:
            pass
    elif isinstance(raw_prompts, dict):
        prompts_data.update(raw_prompts)

    for prompt_id, prompt_data in prompts_data.items():
        if not isinstance(prompt_data, dict):
            continue
        llm_override = prompt_data.get("llm", prompt_data)
        if isinstance(llm_override, dict):
            runtime.prompts[str(prompt_id)] = _llm_options_from_mapping(llm_override)
    return runtime


class Context:
    """Base context providing logging, event emission, and execution metadata.

    Extended by FunctionContext, WorkflowContext, and AgentContext.
    """

    def __init__(
        self,
        run_id: str,
        correlation_id: str,
        parent_correlation_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        session_id: Optional[str] = None,
        enable_memoization: bool = False,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
        trace_metadata: Optional[dict[str, str]] = None,
        memo_namespace: Optional[str] = None,
    ) -> None:
        self._run_id = run_id
        self._attempt = attempt
        self._runtime_context = runtime_context
        self._session_id = session_id
        self._is_streaming = is_streaming
        self._worker = worker
        self._trace_metadata = trace_metadata
        self.runtime = runtime_context_from_metadata(trace_metadata)

        # Correlation tracking for event hierarchy (required, never null)
        self._correlation_id: str = correlation_id
        self._parent_correlation_id: str = parent_correlation_id
        self._component_name: Optional[str] = None
        self._memo_namespace = memo_namespace or ""
        self._memo_child_sequences: dict[str, int] = {}

        self._emitter: Optional[EventEmitter] = None
        self._sandbox: Optional["Sandbox"] = None

        if enable_memoization:
            from .memoization import MemoizationManager

            self._memo: Optional["MemoizationManager"] = MemoizationManager(self)
        else:
            self._memo = None

        base_logger = get_execution_logger()
        logger_extra: dict[str, Any] = {
            "run_id": run_id,
            "_agnt5_context_ref": weakref.ref(self),
        }
        if runtime_context:
            if runtime_context.trace_id:
                logger_extra["trace_id"] = runtime_context.trace_id
            if runtime_context.span_id:
                logger_extra["span_id"] = runtime_context.span_id
        self._logger = ContextLogger(base_logger, logger_extra)

    @property
    def run_id(self) -> str:
        """Unique execution identifier."""
        return self._run_id

    @property
    def attempt(self) -> int:
        """Current retry attempt (0-indexed)."""
        return self._attempt

    @property
    def metadata(self) -> dict[str, str]:
        """Runtime dispatch metadata for this invocation."""
        return dict(self._trace_metadata or {})

    @property
    def logger(self) -> ContextLogger:
        """Logger with correlation IDs. Supports keyword args as log attributes."""
        return self._logger

    @property
    def sandbox(self) -> Optional["Sandbox"]:
        """Sandbox for code execution and workspace file operations.

        Available when a sandbox is configured on the worker or passed explicitly.
        Returns None if no sandbox is configured.
        """
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: Optional["Sandbox"]) -> None:
        self._sandbox = value

    @property
    def session_id(self) -> Optional[str]:
        """Session identifier for multi-turn conversations."""
        return self._session_id

    @property
    def correlation_id(self) -> str:
        """Current correlation ID for event hierarchy."""
        return self._correlation_id

    @correlation_id.setter
    def correlation_id(self, value: str) -> None:
        """Set the current correlation ID."""
        self._correlation_id = value

    @property
    def parent_correlation_id(self) -> str:
        """Parent correlation ID for event hierarchy."""
        return self._parent_correlation_id

    @parent_correlation_id.setter
    def parent_correlation_id(self, value: str) -> None:
        """Set the parent correlation ID."""
        self._parent_correlation_id = value

    @property
    def component_name(self) -> Optional[str]:
        """Component name for events."""
        return self._component_name

    @component_name.setter
    def component_name(self, value: str) -> None:
        """Set the component name."""
        self._component_name = value

    def _get_emitter(self) -> EventEmitter:
        """Get or create the event emitter (lazy initialization)."""
        if self._emitter is None:
            # Pass trace metadata, execution authority, and project/deployment
            # IDs as base_metadata so every checkpoint event carries them back
            # to the engine. The runtime validates the lease fields before it
            # accepts worker-authored lifecycle mutations.
            # The current engine cache key is still (tenant_id, run_id), where
            # tenant_id is a legacy alias for project identity on worker/runtime
            # paths. Events must therefore preserve the same value stamped on
            # run.queued during the migration window.
            trace_base = {}
            if self._trace_metadata:
                for key in (
                    "traceparent",
                    "tracestate",
                    "experiment_id",
                    "tenant_id",
                    "deployment_id",
                    "attempt",
                    "max_attempts",
                    "component_name",
                    "component_type",
                    "dispatch_mode",
                    "worker_id",
                    "worker_session_id",
                    "lease_id",
                    "lease_attempt",
                ):
                    if key in self._trace_metadata:
                        trace_base[key] = self._trace_metadata[key]
            self._emitter = EventEmitter(
                run_id=self._run_id,
                base_metadata=trace_base or None,
            )
            if self._worker is not None:
                self._emitter.set_worker(self._worker)
                logging.getLogger(__name__).debug(
                    f"[Context._get_emitter] EventEmitter created with worker for run_id={self._run_id}"
                )
            else:
                logging.getLogger(__name__).warning(
                    f"[Context._get_emitter] EventEmitter created WITHOUT worker for run_id={self._run_id}"
                )
        return self._emitter

    def emit(self, event: Event) -> EventEnvelope:
        """Emit a typed event.

        The event already contains correlation_id and parent_correlation_id.
        """
        emitter = self._get_emitter()
        return emitter.emit(event)

    async def emit_async(self, event: Event) -> EventEnvelope:
        """Emit a typed event asynchronously.

        Checkpoint gRPC runs on the tokio runtime natively, avoiding thread pool overhead.
        """
        emitter = self._get_emitter()
        return await emitter.emit_async(event)

    async def emit_batch_async(self, events: list) -> None:
        """Emit multiple events in a single AppendBatch RPC.

        Reduces gRPC overhead by batching non-terminal events (e.g., started events).
        """
        emitter = self._get_emitter()
        await emitter.emit_batch_async(events)

    @contextmanager
    def as_parent(self) -> Generator[None, None, None]:
        """Set this context's correlation_id as parent for nested component events."""
        old_parent = self._parent_correlation_id
        self._parent_correlation_id = self._correlation_id
        try:
            yield
        finally:
            self._parent_correlation_id = old_parent

    def current_memo_namespace(self) -> str:
        """Return the active memo namespace for this task."""
        override = _current_memo_namespace.get()
        if override and (
            not self._memo_namespace
            or override == self._memo_namespace
            or override.startswith(f"{self._memo_namespace}.")
        ):
            return override
        return self._memo_namespace or override or ""

    def allocate_memo_child_scope(self, kind: str, name: str) -> str:
        """
        Allocate a deterministic child namespace below the current memo scope.

        The sequence number keeps repeated sibling calls from sharing keys
        while remaining stable for a replay that follows the same execution
        order.
        """
        base = self.current_memo_namespace()
        kind_part = _memo_namespace_part(kind)
        name_part = _memo_namespace_part(name)
        counter_key = f"{base}\0{kind_part}\0{name_part}"
        seq = self._memo_child_sequences.get(counter_key, 0)
        self._memo_child_sequences[counter_key] = seq + 1
        child = f"{kind_part}.{name_part}.{seq}"
        return f"{base}.{child}" if base else child

    @contextmanager
    def memo_child_scope(self, kind: str, name: str) -> Generator[str, None, None]:
        """Temporarily set a deterministic child memo namespace for this task."""
        namespace = self.allocate_memo_child_scope(kind, name)
        token = _current_memo_namespace.set(namespace)
        try:
            yield namespace
        finally:
            _current_memo_namespace.reset(token)

    def get_event_context(self) -> dict[str, str]:
        """Get correlation_id and parent_correlation_id for event hierarchy."""
        return {
            "correlation_id": self._correlation_id,
            "parent_correlation_id": self._parent_correlation_id,
        }

    def get_event_metadata(self) -> dict[str, str]:
        """Get metadata fields for events. Subclasses can override."""
        meta: dict[str, str] = {}
        if self._component_name:
            meta["name"] = self._component_name
        return meta

    def set_as_parent(self, correlation_id: str) -> str:
        """Set correlation_id as parent. Returns previous parent for restoration."""
        original_parent = self._parent_correlation_id
        self._correlation_id = correlation_id
        self._parent_correlation_id = correlation_id
        return original_parent

    def restore_parent(self, original_parent: str) -> None:
        """Restore the parent correlation ID to a previous value."""
        self._parent_correlation_id = original_parent


def get_current_context() -> Optional[Context]:
    """Get the current execution context from task-local storage."""
    return _current_context.get()


def set_current_context(ctx: Context) -> contextvars.Token:
    """Set the current context. Returns token for reset via _current_context.reset(token)."""
    return _current_context.set(ctx)
