"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from .exceptions import NotImplementedError as AGNT5NotImplementedError
from .exceptions import StateError

T = TypeVar("T")


class Context:
    """
    Execution context for AGNT5 components (functions, entities, workflows).

    Provides APIs for:
    - Orchestration: task(), parallel(), gather(), spawn()
    - State Management: get(), set(), delete()
    - Coordination: signal(), timer(), sleep()
    - AI Integration: llm.generate(), llm.stream()
    - Observability: log(), metrics(), trace_span()
    """

    def __init__(
        self,
        run_id: str,
        component_type: str = "function",
        step_id: Optional[str] = None,
        attempt: int = 0,
        object_id: Optional[str] = None,
        method_name: Optional[str] = None,
    ) -> None:
        """Initialize context with execution metadata."""
        self._run_id = run_id
        self._component_type = component_type
        self._step_id = step_id
        self._attempt = attempt
        self._object_id = object_id
        self._method_name = method_name
        self._state: Dict[str, Any] = {}
        self._checkpoints: Dict[str, Any] = {}
        self._logger = logging.getLogger(f"agnt5.{run_id}")

    # === Execution Metadata ===

    @property
    def run_id(self) -> str:
        """Workflow/run identifier."""
        return self._run_id

    @property
    def step_id(self) -> Optional[str]:
        """Current step identifier."""
        return self._step_id

    @property
    def attempt(self) -> int:
        """Retry attempt number."""
        return self._attempt

    @property
    def component_type(self) -> str:
        """Component type: 'function', 'entity', 'workflow'."""
        return self._component_type

    @property
    def object_id(self) -> Optional[str]:
        """Entity key (for entities only)."""
        return self._object_id

    @property
    def method_name(self) -> Optional[str]:
        """Entity method name (for entities only)."""
        return self._method_name

    # === State Management ===

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state[key] = value

    def delete(self, key: str) -> None:
        """Delete key from state."""
        try:
            del self._state[key]
        except KeyError:
            raise StateError(f"Key '{key}' not found in state")

    # === Checkpointing ===

    async def step(self, name: str, func: Callable[[], Awaitable[T]]) -> T:
        """
        Checkpoint expensive operations.

        If function crashes, won't re-execute this step on retry.
        """
        # Check if step already executed
        if name in self._checkpoints:
            return self._checkpoints[name]

        # Execute and checkpoint
        result = await func()
        self._checkpoints[name] = result
        return result

    # === Orchestration (Workflows Only) ===

    async def task(
        self,
        service_name: str,
        handler_name: str,
        input: Any = None,
    ) -> Any:
        """
        Execute a function and wait for result (workflows only).

        Phase 1: Calls local @function handlers directly.
        Phase 2: Will support distributed function execution across services.

        Args:
            service_name: Service name (ignored in Phase 1)
            handler_name: Function handler name
            input: Input data for the function

        Returns:
            Function result

        Example:
            ```python
            result = await ctx.task(
                service_name="data-processor",
                handler_name="process_data",
                input={"data": [1, 2, 3]}
            )
            ```
        """
        from .function import FunctionRegistry

        # Look up function in registry
        func_config = FunctionRegistry.get(handler_name)
        if func_config is None:
            raise ValueError(f"Function '{handler_name}' not found in registry")

        # Create child context for function execution
        child_ctx = Context(
            run_id=f"{self.run_id}:task:{handler_name}",
            component_type="function",
        )

        # Execute function with input
        if input is not None:
            result = await func_config.handler(child_ctx, input)
        else:
            result = await func_config.handler(child_ctx)

        return result

    async def parallel(self, *tasks: Awaitable[T]) -> List[T]:
        """
        Run multiple tasks in parallel (workflows only).

        Phase 1: Uses asyncio.gather() for in-process parallelism.
        Phase 2: Will add distributed execution across services.

        Args:
            *tasks: Async tasks to run in parallel

        Returns:
            List of results in the same order as tasks

        Example:
            ```python
            result1, result2 = await ctx.parallel(
                fetch_data(source1),
                fetch_data(source2)
            )
            ```
        """
        import asyncio
        return list(await asyncio.gather(*tasks))

    async def gather(self, **tasks: Awaitable[T]) -> Dict[str, T]:
        """
        Run tasks in parallel with named results (workflows only).

        Phase 1: Uses asyncio.gather() for in-process parallelism.
        Phase 2: Will add distributed execution across services.

        Args:
            **tasks: Named async tasks to run in parallel

        Returns:
            Dictionary mapping names to results

        Example:
            ```python
            results = await ctx.gather(
                db=query_database(),
                api=fetch_api(),
                cache=check_cache()
            )
            # Access: results["db"], results["api"], results["cache"]
            ```
        """
        import asyncio
        keys = list(tasks.keys())
        values = list(tasks.values())
        results = await asyncio.gather(*values)
        return dict(zip(keys, results))

    def spawn(self, handler: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> Any:
        """
        Spawn a child function without waiting.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.spawn() will be implemented in Phase 2 with platform integration"
        )

    # === Coordination ===

    async def signal(
        self,
        name: str,
        timeout_ms: Optional[int] = None,
        default: Any = None,
    ) -> Any:
        """
        Wait for external signal.

        Phase 1: Uses asyncio.Event for in-process signaling.
        Phase 2: Will add durable cross-process signal support.

        Args:
            name: Signal name to wait for
            timeout_ms: Optional timeout in milliseconds
            default: Default value if timeout occurs

        Returns:
            Signal payload or default value

        Example:
            ```python
            # In one task:
            await ctx.signal_send("approval", {"approved": True})

            # In another task:
            result = await ctx.signal("approval", timeout_ms=5000)
            ```
        """
        import asyncio

        # Get or create event for this signal
        if not hasattr(self, "_signals"):
            self._signals: Dict[str, tuple[asyncio.Event, Any]] = {}

        if name not in self._signals:
            event = asyncio.Event()
            self._signals[name] = (event, None)

        event, payload = self._signals[name]

        # Wait for signal with optional timeout
        try:
            if timeout_ms:
                await asyncio.wait_for(event.wait(), timeout=timeout_ms / 1000)
            else:
                await event.wait()
            return self._signals[name][1]  # Return payload
        except asyncio.TimeoutError:
            return default

    def signal_send(self, name: str, payload: Any = None) -> None:
        """
        Send a signal to waiting tasks.

        Phase 1: Uses asyncio.Event for in-process signaling.
        Phase 2: Will add durable cross-process signal support.

        Args:
            name: Signal name
            payload: Data to send with signal

        Example:
            ```python
            # Send approval signal
            ctx.signal_send("approval", {"approved": True, "by": "admin"})
            ```
        """
        import asyncio

        # Get or create event for this signal
        if not hasattr(self, "_signals"):
            self._signals: Dict[str, tuple[asyncio.Event, Any]] = {}

        if name not in self._signals:
            event = asyncio.Event()
            self._signals[name] = (event, payload)
        else:
            # Update payload and set event
            event = self._signals[name][0]
            self._signals[name] = (event, payload)

        event.set()

    async def timer(
        self,
        delay_ms: Optional[int] = None,
        cron: Optional[str] = None,
    ) -> None:
        """
        Wait for delay or scheduled time.

        Phase 1: Supports delay_ms using asyncio.sleep (cron not supported).
        Phase 2: Will add durable timers and cron scheduling.

        Args:
            delay_ms: Delay in milliseconds
            cron: Cron expression (Phase 2 only)

        Example:
            ```python
            await ctx.timer(delay_ms=5000)  # Wait 5 seconds
            ```
        """
        if cron is not None:
            raise AGNT5NotImplementedError(
                "ctx.timer(cron=...) will be implemented in Phase 2 with cron support"
            )

        if delay_ms is not None:
            import asyncio
            await asyncio.sleep(delay_ms / 1000)

    async def sleep(self, seconds: int) -> None:
        """
        Durable sleep (alternative to timer).

        Phase 1: Uses asyncio.sleep for in-process delays.
        Phase 2: Will add durable sleep that survives restarts.

        Args:
            seconds: Number of seconds to sleep

        Example:
            ```python
            await ctx.sleep(5)  # Sleep 5 seconds
            ```
        """
        import asyncio
        await asyncio.sleep(seconds)

    # === Observability ===

    def log(self) -> logging.Logger:
        """Get structured logger for this context."""
        return self._logger

    @property
    def logger(self) -> logging.Logger:
        """Alias for log() - get logger."""
        return self._logger

    def metrics(self) -> "MetricsClient":
        """
        Get metrics client.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.metrics() will be implemented in Phase 2 with observability support"
        )

    def trace_span(self) -> "TraceSpan":
        """
        Create trace span.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.trace_span() will be implemented in Phase 2 with observability support"
        )

    # === AI Integration ===

    @property
    def llm(self) -> "LLMClient":
        """
        Get LLM client for generate/stream.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.llm will be implemented in Phase 2 with LLM support"
        )

    @property
    def tools(self) -> "ToolsClient":
        """
        Get tools client for registration.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.tools will be implemented in Phase 2 with tools support"
        )

    # === Memory (Alternative State API) ===

    @property
    def memory(self) -> "MemoryClient":
        """
        Get memory client for durable state operations.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.memory will be implemented in Phase 2 with memory support"
        )

    # === Configuration & Secrets ===

    def secrets(self) -> "SecretsClient":
        """
        Access secrets securely.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.secrets() will be implemented in Phase 2 with secrets support"
        )

    def config(self) -> "ConfigClient":
        """
        Access configuration and feature flags.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.config() will be implemented in Phase 2 with config support"
        )

    def headers(self) -> Dict[str, str]:
        """
        Access request headers.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.headers() will be implemented in Phase 2 with request context"
        )

    # === Messaging ===

    async def send_to(
        self,
        target: str,
        message: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send message to another participant.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.send_to() will be implemented in Phase 2 with messaging support"
        )

    async def subscribe(self, participant_id: str) -> Any:
        """
        Subscribe to messages.

        Raises:
            AGNT5NotImplementedError: Not yet implemented in Phase 1
        """
        raise AGNT5NotImplementedError(
            "ctx.subscribe() will be implemented in Phase 2 with messaging support"
        )

    # === Entity Methods ===

    def entity(self, entity_type: str, key: str) -> "EntityProxy":
        """
        Get entity proxy for method calls.

        Raises:
            AGNT5NotImplementedError: Entities are Phase 2 (Q1 2025)
        """
        raise AGNT5NotImplementedError(
            "ctx.entity() will be implemented in Phase 2 - Entities (Q1 2025)"
        )


# Placeholder classes for type hints
class MetricsClient:
    """Placeholder for metrics client."""

    pass


class TraceSpan:
    """Placeholder for trace span."""

    pass


class LLMClient:
    """Placeholder for LLM client."""

    pass


class ToolsClient:
    """Placeholder for tools client."""

    pass


class MemoryClient:
    """Placeholder for memory client."""

    pass


class SecretsClient:
    """Placeholder for secrets client."""

    pass


class ConfigClient:
    """Placeholder for config client."""

    pass


class EntityProxy:
    """Placeholder for entity proxy."""

    pass
