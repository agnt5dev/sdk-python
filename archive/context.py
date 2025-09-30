"""Execution context exposed to durable Python functions."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import partial
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from .function import ComponentExecutionResult, StateTransitionPayload
    from .entity import EntityProxy

from .durable import RetryPolicy
from .llm import ChatMessage, LlmClient, LlmResponse, Tool, LlmError, LlmStreamChunk
from ._compat import _rust_available

if _rust_available:
    try:
        from ._core import Context as _CoreContext
    except Exception:  # pragma: no cover - extension load failure
        _CoreContext = None  # type: ignore
else:  # pragma: no cover - Rust core unavailable
    _CoreContext = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _ensure_json_serialisable(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(
            "Step results must be JSON-serialisable for checkpoint persistence. "
            f"Got value of type '{type(value).__name__}' with error: {exc}."
        ) from exc
    return value


def _normalise_key(step_name: str, key: Optional[str]) -> str:
    return key or step_name


def _normalise_headers(headers: Any) -> Dict[str, str]:
    if not headers:
        return {}
    if isinstance(headers, Mapping):
        return {str(key).lower(): str(value) for key, value in headers.items()}
    if isinstance(headers, Iterable):
        normalised: Dict[str, str] = {}
        for item in headers:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                key, value = item
                normalised[str(key).lower()] = str(value)
        return normalised
    return {}


def _mapping_from_metadata(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Iterable):
        mapping: Dict[str, Any] = {}
        for item in raw:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                key, value = item
                mapping[str(key)] = value
        return mapping
    return {}


@dataclass
class StepCheckpoint:
    name: str
    key: Optional[str]
    status: str
    attempt: int
    result: Any
    updated_at: datetime

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StepCheckpoint":
        name = payload.get("name")
        if not name:
            raise ValueError("Checkpoint payload missing 'name'")
        key = payload.get("key")
        status = (payload.get("status") or "SUCCEEDED").upper()
        attempt = int(payload.get("attempt", 1))
        raw_result = payload.get("result")
        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                result = raw_result
        else:
            result = raw_result
        timestamp_raw = payload.get("updated_at")
        if isinstance(timestamp_raw, str):
            try:
                updated_at = datetime.fromisoformat(timestamp_raw)
            except ValueError:
                updated_at = datetime.now(timezone.utc)
        else:
            updated_at = datetime.now(timezone.utc)
        return cls(
            name=name, key=key, status=status, attempt=attempt, result=result, updated_at=updated_at
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "status": self.status,
            "attempt": self.attempt,
            "result": self.result,
            "updated_at": self.updated_at.replace(tzinfo=timezone.utc).isoformat(),
        }


class StepCheckpointStore:
    """Maintains step checkpoints for a single invocation."""

    def __init__(self, checkpoints: Sequence[StepCheckpoint]):
        self._completed: Dict[Tuple[str, str], StepCheckpoint] = {}
        self._new: Dict[Tuple[str, str], StepCheckpoint] = {}

        for checkpoint in checkpoints:
            key = (checkpoint.name, _normalise_key(checkpoint.name, checkpoint.key))
            if checkpoint.status == "SUCCEEDED":
                self._completed[key] = checkpoint

    def get_completed(self, name: str, key: Optional[str]) -> Optional[StepCheckpoint]:
        lookup_key = (name, _normalise_key(name, key))
        if lookup_key in self._new:
            return self._new[lookup_key]
        return self._completed.get(lookup_key)

    def record_success(
        self, name: str, key: Optional[str], result: Any, attempt: int
    ) -> StepCheckpoint:
        checkpoint = StepCheckpoint(
            name=name,
            key=key,
            status="SUCCEEDED",
            attempt=attempt,
            result=result,
            updated_at=datetime.now(timezone.utc),
        )
        lookup_key = (name, _normalise_key(name, key))
        self._new[lookup_key] = checkpoint
        return checkpoint

    def export_new(self) -> List[Dict[str, Any]]:
        if not self._new:
            return []
        return [checkpoint.to_payload() for checkpoint in self._new.values()]


class MetricsClient:
    """Simple metrics facade that integrates with OpenTelemetry when available."""

    def __init__(self, attributes: Dict[str, Any]):
        self._attributes = attributes
        try:
            from opentelemetry.metrics import get_meter  # type: ignore

            self._meter = get_meter("agnt5.sdk")
        except Exception:  # pragma: no cover - only triggered without OTEL installed
            self._meter = None
        self._counters: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}

    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        attributes = {**self._attributes, **labels}
        if self._meter:
            if name not in self._counters:
                self._counters[name] = self._meter.create_counter(name)
            self._counters[name].add(value, attributes=attributes)
        else:
            logger.debug("Metric increment %s=%s %s", name, value, attributes)

    def observe(self, name: str, value: float, **labels: Any) -> None:
        attributes = {**self._attributes, **labels}
        if self._meter:
            if name not in self._histograms:
                self._histograms[name] = self._meter.create_histogram(name)
            self._histograms[name].record(value, attributes=attributes)
        else:
            logger.debug("Metric observe %s=%s %s", name, value, attributes)


class SpanContext:
    """Tracing helper exposing a context manager for child spans."""

    def __init__(self, attributes: Dict[str, Any]):
        self._attributes = attributes
        try:
            from opentelemetry import trace  # type: ignore

            self._tracer = trace.get_tracer("agnt5.sdk")
        except Exception:  # pragma: no cover - only triggered without OTEL installed
            self._tracer = None

    @contextmanager
    def start(self, name: str, **attributes: Any):
        merged = {**self._attributes, **attributes}
        if self._tracer:
            from opentelemetry.trace import Status, StatusCode  # type: ignore

            with self._tracer.start_as_current_span(name, attributes=merged) as span:
                try:
                    yield span
                except Exception as exc:  # pragma: no cover - user exception path
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        else:
            yield None


class Context:
    """Runtime context provided to durable function executions."""

    def __init__(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt: int,
        invocation_id: str,
        service_name: str,
        component_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        step_checkpoints: Optional[Sequence[Dict[str, Any]]] = None,
        llm_client: Optional[LlmClient] = None,
        parent_run_id: Optional[str] = None,
        component_type: Optional[str] = None,
        object_id: Optional[str] = None,
        method_name: Optional[str] = None,
        state_snapshot: Optional[bytes] = None,
        journal_position: Optional[int] = None,
    ) -> None:
        self.run_id = run_id
        self.step_id = step_id
        self.attempt = attempt
        self.parent_run_id = parent_run_id or (metadata or {}).get("parent_run_id")
        self._invocation_id = invocation_id
        self._service_name = service_name
        self._component_name = component_name
        self._metadata: Dict[str, Any] = dict(metadata or {})
        self._llm_client = llm_client
        self.component_type = component_type or str(
            self._metadata.get("component_type", "function")
        )
        self.object_id = object_id or self._metadata.get("object_id")
        self.method_name = method_name or self._metadata.get("method_name")
        self.journal_position = (
            journal_position
            if journal_position is not None
            else self._metadata.get("journal_position")
        )
        self.state_version = self._metadata.get("state_version")
        self._tenant_id = str(
            self._metadata.get("tenant_id") or os.getenv("AGNT5_TENANT_ID", "default")
        )
        session_candidate = (
            self._metadata.get("session_id") or self._metadata.get("run_id") or run_id
        )
        self._session_id = str(session_candidate) if session_candidate else None

        self._core_ctx = None
        if _CoreContext is not None:
            try:
                if self._session_id:
                    self._core_ctx = _CoreContext.with_runtime(
                        self._tenant_id,
                        self._session_id,
                        run_id,
                        step_id or "step",
                        attempt,
                        invocation_id,
                    )
                else:
                    # Fallback to placeholder handle when session metadata is missing
                    self._core_ctx = _CoreContext()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to initialise core runtime context: %s", exc)
                try:
                    self._core_ctx = _CoreContext()
                except Exception:
                    self._core_ctx = None
        else:
            self._core_ctx = None

        parsed = self._parse_checkpoints(step_checkpoints)
        self._checkpoints = StepCheckpointStore(parsed)
        self._sleep_invocations = 0
        self._spawn_counter = 0
        self._pending_spawns: list[SpawnHandle] = []

        self._logger = logging.LoggerAdapter(
            logging.getLogger("agnt5.durable"),
            {
                "run_id": self.run_id,
                "step_id": self.step_id,
                "attempt": self.attempt,
                "component": self._component_name,
            },
        )

        metric_attributes = {
            "run_id": self.run_id,
            "component": self._component_name,
            "service": self._service_name,
        }
        self._metrics = MetricsClient(metric_attributes)
        self._span = SpanContext(metric_attributes)
        self._headers = MappingProxyType(_normalise_headers(self._metadata.get("headers")))
        self._resources = ResourcesClient(_mapping_from_metadata(self._metadata.get("resources")))
        self._eval = EvalClient(self)
        self._secrets = SecretsClient(_mapping_from_metadata(self._metadata.get("secrets")))
        self._config = ConfigClient(_mapping_from_metadata(self._metadata.get("config")))
        self._tools_client = ToolClient(self)
        self._llm_facade = _ContextLlmFacade(self)
        self._memory = MemoryClient(
            state_snapshot=state_snapshot or self._metadata.get("state_snapshot"),
            object_id=self.object_id,
            journal_position=self.journal_position,
        )
        self._agent_client = AgentClient(self)
        self._signal_client = SignalClient(self)
        self._timer_client = TimerClient(self)
        self._human_client = HumanClient(self)

    @staticmethod
    def _parse_checkpoints(raw: Optional[Sequence[Dict[str, Any]]]) -> List[StepCheckpoint]:
        checkpoints: List[StepCheckpoint] = []
        if not raw:
            return checkpoints

        for item in raw:
            try:
                checkpoints.append(StepCheckpoint.from_payload(item))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to parse step checkpoint %s: %s", item, exc)
        return checkpoints

    def log(self) -> logging.Logger:
        return self._logger  # type: ignore[return-value]

    def metrics(self) -> MetricsClient:
        return self._metrics

    def trace_span(self) -> SpanContext:
        return self._span

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep duration must be non-negative")
        if seconds == 0:
            return

        # Ensure stable timer identity across retries within the same step
        self._sleep_invocations += 1
        timer_key = f"{self.step_id or 'step'}:sleep:{self._sleep_invocations}"

        checkpoint = self._checkpoints.get_completed("__sleep__", timer_key)
        if checkpoint:
            self._logger.debug("Skipping sleep '%s'; checkpoint already recorded", timer_key)
            return

        if not self.run_id:
            self._logger.debug(
                "No run_id available; falling back to asyncio.sleep for '%s'", timer_key
            )
            await asyncio.sleep(seconds)
            return

        delay = timedelta(seconds=float(seconds))
        dedupe_id = f"{self.run_id}:{timer_key}"

        try:
            await self.timer.schedule(
                timer_key,
                run_id=self.run_id,
                waiting_step=self.step_id,
                delay=delay,
                dedupe_id=dedupe_id,
            )
        except Exception as exc:
            self._logger.warning(
                "Durable sleep scheduling failed for '%s'; falling back to asyncio.sleep: %s",
                timer_key,
                exc,
            )
            await asyncio.sleep(seconds)
            return

        await self._wait_for_timer_completion(timer_key, delay)
        self._checkpoints.record_success(
            "__sleep__", timer_key, {"seconds": seconds}, attempt=self.attempt
        )

    async def step(
        self,
        name: str,
        fn: Callable[[], Awaitable[T] | T],
        *,
        key: Optional[str] = None,
        retry: Optional[RetryPolicy] = None,
    ) -> T:
        lookup_key = _normalise_key(name, key)
        cached = self._checkpoints.get_completed(name, lookup_key if key else None)
        if cached:
            self._logger.debug(
                "Returning cached result for step '%s' (key=%s)",
                name,
                lookup_key,
            )
            return cached.result  # type: ignore[return-value]

        if retry:
            self._logger.debug(
                "Overriding step retry policy for '%s': %s",
                name,
                retry.to_dict(),
            )

        self._logger.debug("Executing step '%s' (key=%s)", name, lookup_key)

        result = fn()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            result = await result  # type: ignore[assignment]

        materialised = _ensure_json_serialisable(result)
        self._checkpoints.record_success(name, key, materialised, attempt=self.attempt)
        return materialised  # type: ignore[return-value]

    def spawn(
        self,
        handler: Any,
        /,
        *args: Any,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> SpawnHandle:
        """Spawn a child durable invocation and return a handle for later joining."""

        loop = asyncio.get_running_loop()
        spawn_key = self._allocate_spawn_key(key)
        child_invocation_id = str(uuid.uuid4())

        async def runner() -> Any:
            return await self._run_spawn(handler, args, kwargs, spawn_key, child_invocation_id)

        task = loop.create_task(runner())
        handle = SpawnHandle(spawn_key, child_invocation_id, task, self)
        self._pending_spawns.append(handle)
        return handle

    async def join(self, handles: Optional[Sequence[SpawnHandle]] = None) -> List[Any]:
        """Await completion of the provided spawn handles (or all outstanding)."""

        if handles is None:
            handles = list(self._pending_spawns)
        results: List[Any] = []
        for handle in handles:
            results.append(await handle.result())
        return results

    async def gather(self, *handles: SpawnHandle) -> List[Any]:
        """Convenience alias for :meth:`join`."""

        if not handles:
            return await self.join()
        return await self.join(list(handles))

    async def parallel(self, *handles: SpawnHandle) -> List[Any]:
        """Await multiple spawn handles in parallel (alias of :meth:`gather`)."""

        return await self.gather(*handles)

    async def send_to(
        self,
        target: str,
        message: Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, str]] = None,
        priority: Optional[int] = None,
        message_type: str = "coordination",
    ) -> str:
        """Send a message to another durable participant within the run."""

        if not target:
            raise ValueError("target must be provided")
        if not self.run_id:
            raise RuntimeError("send_to requires a run context")

        _ensure_json_serialisable(message)

        transport = _GatewayTransport(self)

        metadata_map: Dict[str, str] = {}
        if metadata:
            for key, value in metadata.items():
                metadata_map[str(key)] = str(value)

        if self._invocation_id:
            metadata_map.setdefault("from_invocation_id", str(self._invocation_id))
        if self.step_id:
            metadata_map.setdefault("from_step", self.step_id)

        body: Dict[str, Any] = {
            "target": target,
            "message": message,
            "message_type": message_type,
            "metadata": metadata_map,
        }

        if priority is not None and priority > 0:
            body["priority"] = int(priority)

        response = await asyncio.to_thread(
            transport.request,
            "POST",
            f"/runs/{urllib.parse.quote(self.run_id, safe='')}/messages",
            body,
        )

        message_id = response.get("message_id") or response.get("messageId")
        if not message_id:
            raise RuntimeError("gateway did not return a message_id for send_to")

        return str(message_id)

    def subscribe(
        self,
        target: str,
        *,
        poll_interval: float = 1.0,
        limit: int = 50,
    ) -> AsyncIterator[InboundMessage]:
        """Yield messages for ``target`` as they become available."""

        if not target:
            raise ValueError("target must be provided")
        if not self.run_id:
            raise RuntimeError("subscribe requires a run context")

        async def iterator() -> AsyncIterator[InboundMessage]:
            transport = _GatewayTransport(self)
            after: Optional[int] = None

            while True:
                query: Dict[str, Any] = {
                    "target": target,
                    "status": "pending",
                    "limit": str(max(1, min(limit, 200))),
                }
                if after:
                    query["after"] = str(after)

                path = f"/runs/{urllib.parse.quote(self.run_id, safe='')}/messages?{urllib.parse.urlencode(query)}"

                try:
                    response = await asyncio.to_thread(transport.request, "GET", path, None)
                except Exception as exc:  # pragma: no cover - network errors
                    self._logger.debug("message poll failed: %s", exc)
                    await asyncio.sleep(poll_interval)
                    continue

                items = response.get("messages") or []
                if not items:
                    await asyncio.sleep(poll_interval)
                    continue

                for item in items:
                    message_id = str(item.get("message_id") or item.get("messageId"))
                    payload = item.get("payload")
                    metadata_raw = item.get("metadata") or {}
                    metadata_map: Dict[str, str] = {}
                    if isinstance(metadata_raw, Mapping):
                        for key, value in metadata_raw.items():
                            metadata_map[str(key)] = str(value)

                    created_at = 0
                    if ts := item.get("created_at"):
                        try:
                            created_at = int(ts)
                        except (TypeError, ValueError):
                            created_at = 0

                    status = str(item.get("status") or "pending")

                    inbound = InboundMessage(
                        message_id=message_id,
                        target=target,
                        payload=payload,
                        metadata=metadata_map,
                        created_at=created_at,
                        status=status,
                        _context=self,
                    )

                    if created_at > 0 and (after is None or created_at > after):
                        after = created_at

                    yield inbound

        return iterator()

    async def _ack_message(
        self,
        message_id: str,
        *,
        target: str,
        acked_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        if not message_id:
            raise ValueError("message_id must be provided for ack")
        if not self.run_id:
            raise RuntimeError("ack requires a run context")

        transport = _GatewayTransport(self)
        body: Dict[str, Any] = {"channel": target}
        if acked_by:
            body["acked_by"] = acked_by
        if reason:
            body["reason"] = reason

        await asyncio.to_thread(
            transport.request,
            "POST",
            f"/runs/{urllib.parse.quote(self.run_id, safe='')}/messages/{urllib.parse.quote(message_id, safe='')}/ack",
            body,
        )

    @property
    def llm(self) -> "_ContextLlmFacade":
        return self._llm_facade

    @property
    def tools(self) -> "ToolClient":
        return self._tools_client

    @property
    def resources(self) -> "ResourcesClient":
        return self._resources

    @property
    def eval(self) -> "EvalClient":
        return self._eval

    @property
    def memory(self) -> "MemoryClient":
        return self._memory

    @property
    def agent(self) -> "AgentClient":
        return self._agent_client

    @property
    def human(self) -> "HumanClient":
        return self._human_client

    def entity(self, entity_type: str, entity_key: str) -> "EntityProxy":
        """Create a proxy for invoking entity methods.

        Args:
            entity_type: The entity type name (e.g., "ConversationAgent")
            entity_key: The unique entity instance key (e.g., "conv-123")

        Returns:
            EntityProxy for method invocation

        Example:
            ```python
            tracker = ctx.entity("NewsTracker", "main")
            is_new = await tracker.mark_seen(article_id)
            ```
        """
        from .entity import EntityProxy

        return EntityProxy(entity_type, entity_key, self)

    def _ensure_core_ctx(self) -> Any:
        if self._core_ctx is None:
            raise RuntimeError(
                "Durable runtime controls unavailable; ensure the AGNT5 runtime service client is configured."
            )
        return self._core_ctx

    @property
    def runtime(self) -> Any:
        return self._ensure_core_ctx().runtime()

    @property
    def signals(self) -> Any:
        return self._ensure_core_ctx().signals()

    @property
    def timers(self) -> Any:
        return self._ensure_core_ctx().timers()

    @property
    def tasks(self) -> Any:
        return self._ensure_core_ctx().tasks()

    @property
    def signal(self) -> "SignalClient":
        return self._signal_client

    @property
    def timer(self) -> "TimerClient":
        return self._timer_client

    def headers(self) -> Mapping[str, str]:
        return self._headers

    def secrets(self) -> "SecretsClient":
        return self._secrets

    def config(self) -> "ConfigClient":
        return self._config

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._metadata))

    async def llm_complete(
        self,
        *,
        prompt: str,
        model: str = "gpt-4o-mini",
        schema: Optional[Dict[str, Any]] = None,
    ) -> LlmResponse:
        client = self._ensure_llm_client()
        start = time.perf_counter()
        try:
            response = await client.complete(
                prompt=prompt, model=model, schema=schema, context=self
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics = self.metrics()
            metrics.increment("llm.request.count", model=model, mode="complete", status="success")
            metrics.observe("llm.latency.ms", duration_ms, model=model, mode="complete")
            if response.usage.total_tokens:
                metrics.observe(
                    "llm.tokens.total", response.usage.total_tokens, model=model, mode="complete"
                )
            return response
        except LlmError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics = self.metrics()
            metrics.increment("llm.request.count", model=model, mode="complete", status="error")
            metrics.observe("llm.latency.ms", duration_ms, model=model, mode="complete")
            self._logger.error("llm_complete failed: %s", exc)
            raise

    async def llm_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str = "gpt-4o-mini",
        tools: Optional[Sequence[Tool]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[LlmStreamChunk], Union[Awaitable[None], None]]] = None,
    ) -> LlmResponse:
        client = self._ensure_llm_client()
        start = time.perf_counter()
        try:
            response = await client.chat(
                messages=messages,
                model=model,
                tools=tools,
                context=self,
                stream=stream,
                on_chunk=on_chunk,
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics = self.metrics()
            mode = "chat-stream" if stream else "chat"
            metrics.increment("llm.request.count", model=model, mode=mode, status="success")
            metrics.observe("llm.latency.ms", duration_ms, model=model, mode=mode)
            if response.usage.total_tokens:
                metrics.observe(
                    "llm.tokens.total", response.usage.total_tokens, model=model, mode=mode
                )
            return response
        except LlmError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics = self.metrics()
            mode = "chat-stream" if stream else "chat"
            metrics.increment("llm.request.count", model=model, mode=mode, status="error")
            metrics.observe("llm.latency.ms", duration_ms, model=model, mode=mode)
            self._logger.error("llm_chat failed: %s", exc)
            raise

    def export_new_checkpoints(self) -> List[Dict[str, Any]]:
        return self._checkpoints.export_new()

    def _ensure_llm_client(self) -> LlmClient:
        if self._llm_client is not None:
            return self._llm_client
        client = LlmClient.from_environment()
        if client is None:
            raise RuntimeError(
                "LLM client not configured. Set AGNT5_LLM_PROXY_URL or provide a custom LlmClient."
            )
        self._llm_client = client
        return client

    async def _run_spawn(
        self,
        handler: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        spawn_key: str,
        child_invocation_id: str,
    ) -> Any:
        async def invoke_child() -> Any:
            return await self._execute_spawn(handler, args, kwargs, spawn_key, child_invocation_id)

        step_name = f"spawn:{spawn_key}"
        return await self.step(step_name, invoke_child, key=spawn_key)

    async def _execute_spawn(
        self,
        handler: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        spawn_key: str,
        child_invocation_id: str,
    ) -> Any:
        if self._can_spawn_via_gateway():
            try:
                return await self._spawn_via_gateway(
                    handler, args, kwargs, spawn_key, child_invocation_id
                )
            except Exception as exc:
                self._logger.warning(
                    "Spawn via gateway failed; falling back to in-process execution: %s",
                    exc,
                )

        return await self._spawn_local(handler, args, kwargs, spawn_key, child_invocation_id)

    def _can_spawn_via_gateway(self) -> bool:
        if not self.run_id:
            return False
        tenant_id = str(self._metadata.get("tenant_id") or "").strip()
        if not tenant_id:
            return False
        return True

    async def _spawn_local(
        self,
        handler: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        spawn_key: str,
        child_invocation_id: str,
    ) -> Any:
        handler_name, handler_callable = self._resolve_spawn_handler(handler)
        payload = self._encode_spawn_payload(handler_callable, args, kwargs)

        metadata = {
            "run_id": self.run_id,
            "parent_invocation_id": self._invocation_id,
            "spawn_key": spawn_key,
            "attempt": str(self.attempt),
        }
        if self.step_id:
            metadata["parent_step_id"] = self.step_id

        context_dict = {
            "invocation_id": child_invocation_id,
            "service_name": self._service_name,
            "handler_name": handler_name,
            "metadata": metadata,
        }

        return await asyncio.to_thread(
            self._execute_component_sync,
            handler_name,
            payload,
            context_dict,
        )

    async def _spawn_via_gateway(
        self,
        handler: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        spawn_key: str,
        child_invocation_id: str,
    ) -> Any:
        service_name = str(self._service_name or self._metadata.get("service_name") or "").strip()
        if not service_name:
            raise RuntimeError("Service name unavailable for spawn request")

        handler_name, handler_callable = self._resolve_spawn_handler(handler)
        payload_bytes = self._encode_spawn_payload(handler_callable, args, kwargs)

        if payload_bytes:
            try:
                input_payload = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                self._logger.debug("spawn payload decode failed; sending raw string: %s", exc)
                input_payload = payload_bytes.decode("utf-8", errors="ignore")
        else:
            input_payload = {}

        metadata: Dict[str, str] = {
            "run_id": self.run_id or "",
            "parent_invocation_id": str(self._invocation_id or ""),
            "spawn_key": spawn_key,
            "attempt": str(self.attempt),
        }
        if self.step_id:
            metadata["parent_step_id"] = self.step_id
        if self.object_id:
            metadata["parent_object_id"] = self.object_id
        correlation_id = str(self._metadata.get("correlation_id") or "").strip()
        if correlation_id:
            metadata.setdefault("correlation_id", correlation_id)

        transport = _GatewayTransport(self)

        body: Dict[str, Any] = {
            "service_name": service_name,
            "handler_name": handler_name,
            "input_data": input_payload,
            "spawn_key": spawn_key,
            "metadata": metadata,
            "child_invocation_id": child_invocation_id,
            "parent_invocation_id": str(self._invocation_id or ""),
        }

        if self.object_id:
            body["parent_object_id"] = self.object_id

        trace_id = str(self._metadata.get("trace_id") or "").strip()
        if trace_id:
            body["trace_id"] = trace_id

        response = await asyncio.to_thread(
            transport.request,
            "POST",
            f"/runs/{urllib.parse.quote(self.run_id, safe='')}/spawns",
            body,
        )

        returned_child = str(response.get("child_invocation_id") or "")
        if returned_child and returned_child != child_invocation_id:
            self._logger.debug(
                "Spawn response child invocation id mismatch (expected %s, got %s)",
                child_invocation_id,
                returned_child,
            )

        return await self._wait_for_spawn_completion(child_invocation_id, spawn_key)

    async def _wait_for_spawn_completion(self, invocation_id: str, spawn_key: str) -> Any:
        transport = _GatewayTransport(self)
        delay = 0.5

        status_path = f"/status/{urllib.parse.quote(invocation_id, safe='')}"
        output_path = f"/output/{urllib.parse.quote(invocation_id, safe='')}"

        while True:
            try:
                status_response = await asyncio.to_thread(transport.request, "GET", status_path)
            except RuntimeError as exc:
                if "404" in str(exc):  # Spawn not visible yet; retry shortly
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 5.0)
                    continue
                raise

            status = str(status_response.get("status") or "").lower()

            if status in {"completed", "failed", "cancelled"}:
                if status == "completed":
                    output_response = await asyncio.to_thread(transport.request, "GET", output_path)
                    return output_response.get("output")

                error_message = (
                    status_response.get("errorMessage") or f"spawn {invocation_id} {status}"
                )
                raise RuntimeError(
                    f"Spawned invocation '{invocation_id}' (key={spawn_key}) failed: {error_message}"
                )

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)

    def _resolve_spawn_handler(self, handler: Any) -> tuple[str, Callable[..., Any]]:
        definition = getattr(handler, "_agnt5_durable", None)
        if definition is not None:
            return definition.name, definition.handler

        if isinstance(handler, str):
            from .durable.registry import get_registry  # Local import to avoid cycle

            registry = get_registry()
            definition = registry.get(handler)
            if not definition:
                raise ValueError(f"Durable function '{handler}' is not registered")
            return definition.name, definition.handler

        if callable(handler):
            name = getattr(handler, "_agnt5_handler_name", getattr(handler, "__name__", None))
            if not name:
                raise ValueError("Unable to determine handler name for spawned function")
            return str(name), handler

        raise TypeError("spawn expects a durable function or registered handler name")

    def _encode_spawn_payload(
        self,
        handler: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> bytes:
        if not args and not kwargs:
            payload: Any = {}
        elif kwargs and not args:
            payload = dict(kwargs)
        elif args and not kwargs:
            payload = args[0] if len(args) == 1 else list(args)
        else:
            mapped = self._map_args_to_parameters(handler, args, kwargs)
            if mapped is None:
                payload = list(args)
            else:
                payload = mapped

        _ensure_json_serialisable(payload)
        return json.dumps(payload, default=_json_default).encode("utf-8")

    def _map_args_to_parameters(
        self,
        handler: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return None

        params = [param for param in signature.parameters.values()]
        if not params:
            return None

        # Skip context parameter
        params = params[1:]
        mapping: Dict[str, Any] = {}

        for index, arg in enumerate(args):
            if index >= len(params):
                return None
            param = params[index]
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return None
            mapping[param.name] = arg

        for key, value in kwargs.items():
            mapping[key] = value

        return mapping

    def _execute_component_sync(
        self,
        handler_name: str,
        input_data: bytes,
        context: Dict[str, Any],
    ) -> Any:
        from .function import execute_component  # Local import to avoid cycle

        execution = execute_component(
            name=handler_name,
            input_data=input_data,
            context=context,
        )
        output = execution.output_data
        return self._decode_spawn_output(output)

    def _decode_spawn_output(self, output: bytes) -> Any:
        if not output:
            return None
        try:
            return json.loads(output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                return output.decode("utf-8")
            except UnicodeDecodeError:  # pragma: no cover - binary payloads
                return output

    def _allocate_spawn_key(self, explicit: Optional[str]) -> str:
        self._spawn_counter += 1
        if explicit:
            return str(explicit)
        base = self.step_id or "step"
        return f"{base}-spawn-{self._spawn_counter}"

    def _remove_spawn_handle(self, handle: SpawnHandle) -> None:
        try:
            self._pending_spawns.remove(handle)
        except ValueError:  # pragma: no cover - already removed
            pass

    async def _wait_for_timer_completion(self, timer_key: str, delay: timedelta) -> None:
        transport = _GatewayTransport(self)
        total_seconds = max(delay.total_seconds(), 0.0)
        poll_interval = min(5.0, max(0.2, total_seconds / 10.0 if total_seconds else 0.2))
        grace = max(5.0, total_seconds * 0.25)
        deadline = time.monotonic() + total_seconds + grace

        # Poll the gateway for timer completion; fall back to local sleep if unavailable.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._logger.warning(
                    "Timer '%s' still pending after %.2fs grace; continuing execution",
                    timer_key,
                    total_seconds,
                )
                return
            try:
                waits = await asyncio.to_thread(
                    transport.request,
                    "GET",
                    f"/runs/{self.run_id}/waits",
                    None,
                )
            except Exception as exc:
                self._logger.debug(
                    "Failed to query pending waits for timer '%s'; falling back to local sleep: %s",
                    timer_key,
                    exc,
                )
                await asyncio.sleep(max(0.0, remaining))
                return

            timers = waits.get("timers") or []
            pending = False
            for timer in timers:
                if timer.get("timer_key") != timer_key:
                    continue
                waiting_step = timer.get("waiting_step")
                if waiting_step and waiting_step != self.step_id:
                    continue
                pending = True
                break
            if not pending:
                return

            await asyncio.sleep(min(poll_interval, max(0.0, remaining)))


@dataclass
class MemoryStateTransition:
    operation: str
    key: str
    old_value: Optional[bytes]
    new_value: Optional[bytes]
    timestamp_ms: int


@dataclass
class MemoryStateUpdate:
    new_state: bytes
    transitions: list[MemoryStateTransition]
    output_data: bytes = b""


class MemoryClient:
    """Durable memory helper providing CRUD operations with transition tracking."""

    def __init__(
        self,
        *,
        state_snapshot: Optional[bytes],
        object_id: Optional[str],
        journal_position: Optional[int],
    ) -> None:
        self._object_id = object_id
        self._journal_position = journal_position
        self._state = self._load_snapshot(state_snapshot)
        self._transitions: list[MemoryStateTransition] = []
        self._changed = False

    async def get(self, key: str, default: Any = None) -> Any:
        value = self._state.get(key, default)
        return copy.deepcopy(value)

    async def set(self, key: str, value: Any) -> None:
        prepared = self._prepare_value(value)
        previous = copy.deepcopy(self._state.get(key)) if key in self._state else None
        self._state[key] = copy.deepcopy(prepared)
        self._record_transition("set", key, previous, prepared)

    async def delete(self, key: str) -> None:
        if key not in self._state:
            return
        previous = copy.deepcopy(self._state.pop(key))
        self._record_transition("delete", key, previous, None)

    async def append(self, key: str, value: Any, *, limit: Optional[int] = None) -> None:
        existing = self._state.get(key)
        if existing is None:
            existing_list: list[Any] = []
        elif isinstance(existing, list):
            existing_list = copy.deepcopy(existing)
        else:
            raise TypeError(f"Memory key '{key}' is not a list; cannot append")

        prepared_value = self._prepare_value(value)
        existing_list.append(copy.deepcopy(prepared_value))
        if limit is not None and limit > 0 and len(existing_list) > limit:
            existing_list = existing_list[-limit:]

        previous = copy.deepcopy(self._state.get(key)) if key in self._state else None
        self._state[key] = existing_list
        self._record_transition("append", key, previous, existing_list)

    def export_state_update(self) -> Optional[MemoryStateUpdate]:
        if not self._changed:
            return None
        try:
            snapshot_bytes = json.dumps(self._state).encode("utf-8")
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.error("Failed to encode memory snapshot: %s", exc)
            return None

        return MemoryStateUpdate(
            new_state=snapshot_bytes,
            transitions=list(self._transitions),
        )

    @staticmethod
    def _prepare_value(value: Any) -> Any:
        _ensure_json_serialisable(value)
        # Normalize through JSON to avoid non-serializable types lingering
        try:
            return json.loads(json.dumps(value))
        except (TypeError, ValueError):  # pragma: no cover - should be guarded earlier
            return value

    @staticmethod
    def _encode_value(value: Any) -> Optional[bytes]:
        if value is None:
            return None
        try:
            return json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.warning("Failed to encode memory value for transition: %s", value)
            return None

    def _load_snapshot(self, snapshot: Optional[bytes]) -> Dict[str, Any]:
        if not snapshot:
            return {}
        try:
            if isinstance(snapshot, (bytes, bytearray, memoryview)):
                text = bytes(snapshot).decode("utf-8")
                return json.loads(text)
            if isinstance(snapshot, str):
                return json.loads(snapshot)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to decode state snapshot for object %s: %s", self._object_id, exc
            )
        return {}

    def _record_transition(self, operation: str, key: str, old_value: Any, new_value: Any) -> None:
        timestamp_ms = int(time.time() * 1000)
        transition = MemoryStateTransition(
            operation=operation,
            key=key,
            old_value=self._encode_value(old_value),
            new_value=self._encode_value(new_value),
            timestamp_ms=timestamp_ms,
        )
        self._transitions.append(transition)
        self._changed = True


@dataclass
class AgentCallResult:
    """Result returned from ``ctx.agent.call`` executions."""

    object_id: str
    output: Any
    state: Mapping[str, Any]
    transitions: Sequence[Mapping[str, Any]]


_UNSET = object()


class SpawnHandle:
    """Handle representing a spawned child invocation."""

    __slots__ = ("key", "invocation_id", "_task", "_context", "_result")

    def __init__(
        self,
        key: str,
        invocation_id: str,
        task: "asyncio.Task[Any]",
        context: "Context",
    ) -> None:
        self.key = key
        self.invocation_id = invocation_id
        self._task = task
        self._context = context
        self._result: Any = _UNSET

    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> Any:
        if self._result is not _UNSET:
            return self._result
        try:
            self._result = await self._task
            return self._result
        finally:
            self._context._remove_spawn_handle(self)

    def __await__(self):  # pragma: no cover - delegated to result()
        return self.result().__await__()


@dataclass
class InboundMessage:
    """Message fetched via :meth:`Context.subscribe`."""

    message_id: str
    target: str
    payload: Any
    metadata: Mapping[str, str]
    created_at: int
    status: str
    _context: "Context"

    async def ack(self, *, acked_by: Optional[str] = None, reason: Optional[str] = None) -> None:
        await self._context._ack_message(
            self.message_id,
            target=self.target,
            acked_by=acked_by,
            reason=reason,
        )


class AgentClient:
    """In-process agent runner used by ``ctx.agent``."""

    _state_store: Dict[Tuple[str, str], bytes] = {}

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._logger = logging.getLogger("agnt5.agent-client")

    @classmethod
    def clear_cache(cls) -> None:
        cls._state_store.clear()

    def reset(self, name: Optional[str] = None, *, object_id: Optional[str] = None) -> None:
        """Reset cached local state for agents."""

        if name is None:
            self._state_store.clear()
            return

        if object_id is None:
            keys = [key for key in self._state_store if key[0] == name]
        else:
            keys = [(name, object_id)]

        for key in keys:
            self._state_store.pop(key, None)

    def get_state(self, name: str, object_id: str) -> Optional[Mapping[str, Any]]:
        snapshot = self._state_store.get((name, object_id))
        if snapshot is None:
            return None
        return self._decode_state(snapshot)

    async def call(
        self,
        name: str,
        payload: Any,
        *,
        object_id: Optional[str] = None,
        method: str = "on_message",
        key_args: Optional[Sequence[Any]] = None,
        key_kwargs: Optional[Mapping[str, Any]] = None,
        tools: Optional[Sequence[Any]] = None,
    ) -> AgentCallResult:
        if not name:
            raise ValueError("Agent name must be provided")

        agent_registry = self._get_registry()
        definition = agent_registry.get(name)  # type: ignore[attr-defined]  # Deprecated agent code
        if not definition:
            raise ValueError(f"Stateful agent '{name}' is not registered")

        resolved_object_id = self._resolve_object_id(
            definition,
            payload,
            object_id,
            key_args,
            key_kwargs,
        )

        cache_key = (name, resolved_object_id)
        snapshot = self._state_store.get(cache_key)

        invocation_context = self._build_invocation_context(
            name=name,
            object_id=resolved_object_id,
            method=method,
            tools=tools,
            snapshot=snapshot,
        )

        input_bytes = self._encode_payload(payload)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                self._execute_component,
                name,
                input_bytes,
                invocation_context,
                resolved_object_id,
                method,
                snapshot,
            ),
        )

        return self._process_result(name, resolved_object_id, result, snapshot)

    @staticmethod
    def _get_registry():
        from .entity import get_entity_registry  # Local import to avoid cycle

        return get_entity_registry()

    @staticmethod
    def _execute_component(
        handler_name: str,
        input_data: bytes,
        context: Dict[str, Any],
        object_id: str,
        method: str,
        snapshot: Optional[bytes],
    ) -> "ComponentExecutionResult":
        from .function import execute_component  # Local import to avoid cycle

        return execute_component(
            name=handler_name,
            input_data=input_data,
            context=context,
            component_type="entity",
            method_name=method,
            object_id=object_id,
            state_snapshot=snapshot,
        )

    def _resolve_object_id(
        self,
        definition: Any,  # StatefulAgentDefinition - deprecated
        payload: Any,
        object_id: Optional[str],
        key_args: Optional[Sequence[Any]],
        key_kwargs: Optional[Mapping[str, Any]],
    ) -> str:
        if object_id:
            return str(object_id)

        primary_args = tuple(key_args or ())
        primary_kwargs = dict(key_kwargs or {})

        candidates: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

        if primary_args or primary_kwargs:
            candidates.append((primary_args, primary_kwargs))
        else:
            if isinstance(payload, Mapping):
                mapping_payload = dict(payload)
                candidates.append((tuple(), mapping_payload))
                candidates.append(((mapping_payload,), {}))
            elif isinstance(payload, (list, tuple)):
                candidates.append((tuple(payload), {}))
            else:
                candidates.append(((payload,), {}))

        last_error: Optional[Exception] = None
        resolved: Optional[str] = None

        for args, kwargs in candidates:
            try:
                candidate = definition.key_resolver(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # pragma: no cover - surfaced to user
                raise ValueError(
                    f"Failed to derive object_id for agent '{definition.name}': {exc}"
                ) from exc

            resolved = candidate if isinstance(candidate, str) else str(candidate)
            break

        if resolved is None:
            if last_error is not None:
                raise ValueError(
                    f"Failed to derive object_id for agent '{definition.name}': {last_error}"
                ) from last_error
            raise ValueError(
                f"Failed to derive object_id for agent '{definition.name}': no candidates succeeded"
            )

        if not resolved:
            raise ValueError(f"Agent '{definition.name}' produced an empty object_id")

        return resolved

    def _build_invocation_context(
        self,
        *,
        name: str,
        object_id: str,
        method: str,
        tools: Optional[Sequence[Any]],
        snapshot: Optional[bytes],
    ) -> Dict[str, Any]:
        metadata = dict(self._context._metadata)
        metadata.setdefault("run_id", self._context.run_id)
        if self._context.parent_run_id:
            metadata.setdefault("parent_run_id", self._context.parent_run_id)
        metadata["component_type"] = "agent"
        metadata["object_id"] = object_id
        metadata["method_name"] = method
        metadata.setdefault("headers", dict(self._context.headers()))
        if tools:
            metadata["tools"] = list(tools)

        base: Dict[str, Any] = {
            "invocation_id": f"local-{uuid.uuid4()}",
            "service_name": getattr(self._context, "_service_name", "local"),
            "handler_name": name,
            "metadata": metadata,
            "step_checkpoints": [],
        }

        if snapshot is not None:
            base["state_snapshot"] = snapshot
        if self._context.journal_position is not None:
            base["journal_position"] = self._context.journal_position

        return base

    def _process_result(
        self,
        name: str,
        object_id: str,
        result: "ComponentExecutionResult",
        snapshot: Optional[bytes],
    ) -> AgentCallResult:
        output_bytes = result.output_data or b""
        state_snapshot = snapshot
        transitions: Sequence[Mapping[str, Any]] = []

        if result.state_update is not None:
            update = result.state_update
            if update.new_state:
                state_snapshot = update.new_state
                self._state_store[(name, object_id)] = update.new_state
            output_bytes = update.output_data or output_bytes
            transitions = self._normalise_transitions(update.transitions)

        decoded_output = self._decode_output(output_bytes)
        decoded_state = self._decode_state(state_snapshot)

        return AgentCallResult(
            object_id=object_id,
            output=decoded_output,
            state=decoded_state,
            transitions=transitions,
        )

    @staticmethod
    def _encode_payload(payload: Any) -> bytes:
        if payload is None:
            return b""
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        try:
            return json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Payload for ctx.agent.call must be JSON-serialisable, bytes, or str"
            ) from exc

    @staticmethod
    def _decode_output(data: Optional[bytes]) -> Any:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return data

    @staticmethod
    def _decode_state(snapshot: Optional[bytes]) -> Mapping[str, Any]:
        if not snapshot:
            return {}
        try:
            return json.loads(snapshot.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):  # pragma: no cover
            return {}

    def _normalise_transitions(
        self, transitions: Sequence["StateTransitionPayload"]
    ) -> Sequence[Mapping[str, Any]]:
        normalised: List[Mapping[str, Any]] = []
        for transition in transitions or []:
            normalised.append(
                {
                    "operation": transition.operation,
                    "key": transition.key,
                    "old_value": self._decode_optional_json(transition.old_value),
                    "new_value": self._decode_optional_json(transition.new_value),
                    "timestamp_ms": transition.timestamp_ms,
                }
            )
        return normalised

    @staticmethod
    def _decode_optional_json(data: Optional[bytes]) -> Any:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return data


class _GatewayTransport:
    """Minimal HTTP transport for gateway interactions."""

    def __init__(self, context: "Context") -> None:
        base = os.getenv("AGNT5_GATEWAY_URL", "http://localhost:8080").strip()
        self._base_url = base.rstrip("/") or "http://localhost:8080"
        self._context = context

    def request(
        self, method: str, path: str, body: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        tenant_id = str(self._context.metadata.get("tenant_id") or "").strip()
        if not tenant_id:
            raise RuntimeError("tenant_id metadata is required for signal/timer operations")

        url = urllib.parse.urljoin(self._base_url + "/", path.lstrip("/"))
        headers = {"X-TENANT-ID": tenant_id}
        data: Optional[bytes] = None

        if body is not None:
            try:
                payload = json.dumps(body, default=_json_default).encode("utf-8")
            except (TypeError, ValueError) as exc:  # pragma: no cover
                raise TypeError(f"Request body not serialisable: {exc}") from exc
            headers["Content-Type"] = "application/json"
            data = payload

        request = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Gateway request failed ({exc.code}): {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network issues
            raise RuntimeError(f"Gateway request error: {exc.reason}") from exc

        if not content:
            return {}
        try:
            return json.loads(content.decode("utf-8"))
        except json.JSONDecodeError:
            return {"raw": content.decode("utf-8", errors="ignore")}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, timedelta):
        return int(value.total_seconds() * 1000)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback
            return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


class SignalClient:
    """Durable signal helpers backed by the coordinator gateway."""

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._transport = _GatewayTransport(context)

    async def emit(
        self,
        name: str,
        payload: Any = None,
        *,
        run_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        dedupe_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to emit a signal")

        body: Dict[str, Any] = {"signal_name": name}
        if payload is not None:
            body["payload"] = payload
        if invocation_id:
            body["invocation_id"] = invocation_id
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals",
            body,
        )
        return response

    async def wait(
        self,
        name: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        auto_ack: bool = True,
        dedupe_id: Optional[str] = None,
    ) -> Any:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to wait for a signal")

        body: Dict[str, Any] = {
            "signal_name": name,
            "auto_ack": auto_ack,
        }
        if waiting_step:
            body["waiting_step"] = waiting_step
        if timeout is not None:
            body["timeout_ms"] = int(timeout.total_seconds() * 1000)
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals/wait",
            body,
        )

        if not response.get("delivered"):
            raise TimeoutError("signal wait completed without delivery")

        return response.get("payload")

    async def acknowledge(
        self,
        name: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: str,
        dedupe_id: Optional[str] = None,
        acked_by: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to acknowledge a signal")

        body: Dict[str, Any] = {
            "waiting_step": waiting_step,
        }
        if dedupe_id:
            body["dedupe_id"] = dedupe_id
        if acked_by:
            body["acked_by"] = acked_by

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals/{urllib.parse.quote(name, safe='')}/ack",
            body,
        )

        return response


class TimerClient:
    """Durable timer helpers for scheduling and cancelling waits."""

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._transport = _GatewayTransport(context)

    async def schedule(
        self,
        key: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
        delay: Optional[timedelta] = None,
        fire_at: Optional[datetime] = None,
        schedule_type: Optional[str] = None,
        dedupe_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not key:
            raise ValueError("timer key must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to schedule a timer")

        schedule: Dict[str, Any] = {}
        stype = (schedule_type or "").lower()

        if fire_at is not None:
            stype = stype or "absolute"
            schedule["fire_at"] = fire_at.astimezone(timezone.utc).isoformat()
        elif delay is not None:
            stype = stype or "delay"
            schedule["delay_ms"] = int(delay.total_seconds() * 1000)
        else:
            raise ValueError("either delay or fire_at must be provided")

        body: Dict[str, Any] = {
            "timer_key": key,
            "schedule_type": stype,
            "schedule_payload": schedule,
        }
        if waiting_step:
            body["waiting_step"] = waiting_step
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/timers",
            body,
        )
        return response

    async def cancel(
        self,
        key: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not key:
            raise ValueError("timer key must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to cancel a timer")

        path = f"/runs/{run}/timers/{urllib.parse.quote(key, safe='')}"
        if waiting_step:
            path = f"{path}?step={urllib.parse.quote(waiting_step, safe='')}"

        response = await asyncio.to_thread(
            self._transport.request,
            "DELETE",
            path,
            None,
        )
        return response


@dataclass
class ApprovalResult:
    approval_id: str
    decision: str
    decided_by: Optional[str]
    reason: Optional[str]
    payload: Any
    decided_at: Optional[datetime]


class HumanClient:
    """Helpers for human approval workflows backed by the gateway."""

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._transport = _GatewayTransport(context)
        try:
            interval = float(os.getenv("AGNT5_APPROVAL_POLL_INTERVAL", "0.5"))
        except ValueError:
            interval = 0.5
        self._poll_interval = max(0.1, interval)
        # Default timeout expressed in seconds; 0 disables waiting
        try:
            self._default_timeout = float(os.getenv("AGNT5_APPROVAL_TIMEOUT_SECONDS", "900"))
        except ValueError:
            self._default_timeout = 900.0

    async def approval(
        self,
        name: str,
        payload: Any | None = None,
        *,
        run_id: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        required_roles: Optional[Sequence[str]] = None,
        notify_channels: Optional[Sequence[str]] = None,
    ) -> ApprovalResult:
        if not name:
            raise ValueError("approval name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to request approval")

        body: Dict[str, Any] = {"name": name}
        if payload is not None:
            body["payload"] = payload
        if required_roles:
            body["required_roles"] = list(required_roles)
        if notify_channels:
            body["notify_channels"] = list(notify_channels)
        if timeout is not None:
            timeout_seconds = max(timeout.total_seconds(), 0.0)
            body["timeout_ms"] = int(timeout_seconds * 1000)
        elif self._default_timeout > 0:
            body["timeout_ms"] = int(self._default_timeout * 1000)

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/approvals",
            body,
        )

        approval_id = response.get("approval_id")
        if not approval_id:
            raise RuntimeError("gateway did not return approval_id")

        poll_timeout = timeout.total_seconds() if timeout is not None else self._default_timeout
        poll_timeout = max(poll_timeout, 0.0)
        wait_for_decision = poll_timeout > 0
        deadline = time.monotonic() + poll_timeout if wait_for_decision else None
        pending_payload = copy.deepcopy(payload) if payload is not None else None

        while True:
            list_resp = await asyncio.to_thread(
                self._transport.request,
                "GET",
                f"/runs/{run}/approvals?status=decided",
                None,
            )
            for item in list_resp.get("approvals", []):
                if item.get("approval_id") != approval_id:
                    continue
                decision = str(item.get("decision", "")).lower() or "pending"
                decided_by = item.get("decided_by")
                reason = item.get("reason")
                decided_at = self._parse_timestamp(item.get("decided_at"))
                payload_value = self._decode_payload(item.get("payload"))
                return ApprovalResult(
                    approval_id=approval_id,
                    decision=decision,
                    decided_by=decided_by,
                    reason=reason,
                    payload=payload_value,
                    decided_at=decided_at,
                )

            if not wait_for_decision:
                break

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"approval decision not received before timeout (approval_id={approval_id})"
                )

            await asyncio.sleep(self._poll_interval)

        return ApprovalResult(
            approval_id=approval_id,
            decision="pending",
            decided_by=None,
            reason=None,
            payload=pending_payload,
            decided_at=None,
        )

    def _decode_payload(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            try:
                return json.loads(value.decode("utf-8"))
            except Exception:  # pragma: no cover - fall back to raw
                return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                return None
        return None


class _ContextLlmFacade:
    """Expose ``ctx.llm`` with ``complete`` and ``chat`` helpers."""

    def __init__(self, context: Context) -> None:
        self._context = context

    async def complete(
        self,
        *,
        prompt: str,
        model: str = "gpt-4o-mini",
        schema: Optional[Dict[str, Any]] = None,
    ) -> LlmResponse:
        return await self._context.llm_complete(prompt=prompt, model=model, schema=schema)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str = "gpt-4o-mini",
        tools: Optional[Sequence[Tool]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[LlmStreamChunk], Union[Awaitable[None], None]]] = None,
    ) -> LlmResponse:
        return await self._context.llm_chat(
            messages,
            model=model,
            tools=tools,
            stream=stream,
            on_chunk=on_chunk,
        )


ToolHandler = Callable[..., Union[Awaitable[Any], Any]]


@dataclass
class _RegisteredTool:
    tool: Tool
    handler: ToolHandler


class ToolClient:
    """Lightweight registry for tools declared within a function run."""

    def __init__(self, context: Context) -> None:
        self._context = context
        self._registry: Dict[str, _RegisteredTool] = {}

    def register(
        self,
        name: str,
        *,
        handler: ToolHandler,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tool:
        if not callable(handler):
            raise TypeError("Tool handler must be callable")
        exists = name in self._registry
        tool = Tool(name=name, description=description, input_schema=schema)
        self._registry[name] = _RegisteredTool(tool=tool, handler=handler)
        if exists:
            self._context.log().warning(
                "Tool '%s' was re-registered; overriding previous handler", name
            )
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._registry[name].tool

    def handler(self, name: str) -> ToolHandler:
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._registry[name].handler

    def list(self) -> List[Tool]:
        return [entry.tool for entry in self._registry.values()]

    def clear(self) -> None:
        self._registry.clear()


class ResourcesClient:
    """Container for runtime-provided resource handles."""

    def __init__(self, providers: Mapping[str, Any]):
        self._providers = dict(providers)

    def __getattr__(self, name: str) -> Any:
        if name in self._providers:
            return self._providers[name]
        raise AttributeError(f"Resource '{name}' is not configured for this run")

    def items(self):  # pragma: no cover - convenience helper
        return self._providers.items()


class EvalClient:
    """Stub evaluation client; hooks will integrate with runtime in later phases."""

    def __init__(self, context: Context) -> None:
        self._context = context

    @contextmanager
    def group(self, name: str):  # pragma: no cover - grouping helper
        self._context.log().debug("Starting eval group '%s'", name)
        try:
            yield self
        finally:
            self._context.log().debug("Finished eval group '%s'", name)

    def judge(self, name: str, **kwargs: Any) -> None:
        self._context.log().debug("Eval judge '%s' invoked with %s", name, kwargs)

    def metric(self, name: str, value: Union[int, float], **labels: Any) -> None:
        self._context.metrics().observe(f"eval.{name}", float(value), **labels)


class SecretsClient:
    """Read-only access to secrets injected for the invocation."""

    def __init__(self, secrets: Mapping[str, Any]):
        self._secrets = dict(secrets)

    def get(self, name: str) -> str:
        if name not in self._secrets:
            raise KeyError(f"Secret '{name}' is not available")
        value = self._secrets[name]
        if not isinstance(value, str):
            raise TypeError(f"Secret '{name}' must be a string, got {type(value).__name__}")
        return value


class ConfigClient:
    """Configuration facade exposing feature flags and variants."""

    def __init__(self, config: Mapping[str, Any]):
        raw = dict(config)
        variants = raw.get("variants")
        if isinstance(variants, Mapping):
            self._variants = dict(variants)
        else:
            self._variants = {}
        self._values = raw

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def variant(self, key: str, default: Any = None) -> Any:
        return self._variants.get(key, default)


__all__ = [
    "AgentCallResult",
    "AgentClient",
    "Context",
    "ConfigClient",
    "EvalClient",
    "MemoryClient",
    "MemoryStateTransition",
    "MemoryStateUpdate",
    "MetricsClient",
    "ResourcesClient",
    "SignalClient",
    "HumanClient",
    "ApprovalResult",
    "SecretsClient",
    "TimerClient",
    "SpanContext",
    "StepCheckpoint",
    "ToolClient",
]
