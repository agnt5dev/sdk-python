"""Events module for AGNT5 SDK.

Provides typed event classes with compile-time correlation enforcement.

Usage:
    from agnt5.events import Started, Completed, Delta, ComponentType, OperationType

    # Lifecycle events
    Started(
        name="my-workflow",
        correlation_id="wf-123",
        parent_correlation_id="root",
        component_type=ComponentType.WORKFLOW,
        input_data={"query": "hello"},
    )

    # Streaming delta events
    Delta(
        name="claude-3-sonnet",
        correlation_id="lm-123",
        parent_correlation_id="agent-456",
        component_type=ComponentType.LM,
        operation=OperationType.THINKING,
        content="Let me think...",
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Union

from edwh_uuid7 import uuid7

from agnt5._serialization import serialize_to_str

logger = logging.getLogger(__name__)


# =============================================================================
# Event Classification
# =============================================================================


def is_sse_only_event(event_type: str) -> bool:
    """Check if an event type is SSE-only (not persisted to journal).

    SSE-only events are streaming/progress events that don't affect replay:
    - output.* (output.start, output.delta, output.stop)
    - lm.stream.* (deprecated streaming)
    - lm.message.* (message deltas)
    - lm.thinking.* (thinking deltas)
    - progress.* (progress updates)
    - log (log events)

    Args:
        event_type: The event type string (e.g., "output.delta", "workflow.started")

    Returns:
        True if the event is SSE-only, False if it's a checkpoint event
    """
    return (
        event_type.startswith("output.")
        or event_type.startswith("lm.stream.")
        or event_type.startswith("lm.message.")
        or event_type.startswith("lm.thinking.")
        or event_type.startswith("progress.")
        or event_type.startswith("log")  # log, log.info, log.warn, log.error, etc.
    )


def is_checkpoint_event(event_type: str) -> bool:
    """Check if an event type is a checkpoint event requiring sync acknowledgement.

    Checkpoint events block until the platform acknowledges persistence.
    This ensures correct event ordering for lifecycle events that affect
    workflow state.

    Checkpoint events include:
    - *.started, *.completed, *.failed, *.paused, *.resumed
    - approval.requested, approval.resolved
    - workflow.state.changed

    This is the inverse of is_sse_only_event().

    Args:
        event_type: The event type string (e.g., "workflow.started", "output.delta")

    Returns:
        True if the event is a checkpoint event, False if it's SSE-only
    """
    return not is_sse_only_event(event_type)


# =============================================================================
# Type Aliases
# =============================================================================


EventData = Union[
    None,
    str,  # Text content
    bytes,  # Binary content
    dict[str, Any],  # Structured JSON object
    list[dict[str, Any]],  # Array of objects (e.g., messages)
]


# =============================================================================
# Enums
# =============================================================================


class ComponentType(str, Enum):
    """Component types for lifecycle events."""

    RUN = "run"
    WORKFLOW = "workflow"
    AGENT = "agent"
    FUNCTION = "function"
    STEP = "step"
    ENTITY = "entity"
    TOOL = "tool"
    LM = "lm"
    SCORER = "scorer"


class OperationType(str, Enum):
    """Operation types for sub-component events."""

    # Execution operations
    ITERATION = "iteration"
    GENERATE = "generate"
    STREAM = "stream"
    STEP = "step"
    # Streaming content blocks
    THINKING = "thinking"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    OUTPUT = "output"


# =============================================================================
# Base Event
# =============================================================================


@dataclass(kw_only=True)
class Event:
    """Base class for all typed events.

    All fields are required and enforced at type-check time.
    Missing any required field is a compile-time error.
    """

    # Source component identifier (e.g., model name, function name)
    name: str

    # Unique ID for this execution span
    correlation_id: str

    # Links to parent span for trace hierarchy
    parent_correlation_id: str

    # Unique event identifier for deduplication
    event_id: str = field(default_factory=lambda: str(uuid7()))

    # Precise timing for ordering/latency
    timestamp_ns: int = field(default_factory=time.time_ns)

    # Wire format identifier (set by subclass)
    event_type: str = field(init=False)

    # Additional key-value context
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dict for transport."""
        result: dict[str, Any] = {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "name": self.name,
            "correlation_id": self.correlation_id,
            "parent_correlation_id": self.parent_correlation_id,
            "timestamp_ns": self.timestamp_ns,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        # Add all other fields from subclass
        for key, value in self.__dict__.items():
            if key.startswith("_") or key in result:
                continue
            if value is not None:
                result[key] = value
        return result


# =============================================================================
# Lifecycle Event Base
# =============================================================================


@dataclass(kw_only=True)
class LifecycleEvent(Event):
    """Base class for lifecycle events (Started, Completed, Failed, etc.).

    Subclasses must set _lifecycle_stage class variable.
    """

    # Lifecycle stage - set by subclass (e.g., "started", "completed")
    _lifecycle_stage: ClassVar[str]

    # Component type - enforced via Enum
    component_type: ComponentType

    # Operation type - None for component-level events
    operation: Optional[OperationType] = None

    def __post_init__(self) -> None:
        if self.operation:
            et = f"{self.component_type.value}.{self.operation.value}.{self._lifecycle_stage}"
        else:
            et = f"{self.component_type.value}.{self._lifecycle_stage}"
        object.__setattr__(self, "event_type", et)


# =============================================================================
# Lifecycle Events
# =============================================================================


@dataclass(kw_only=True)
class Started(LifecycleEvent):
    """Component or operation started execution."""

    _lifecycle_stage: ClassVar[str] = "started"

    # Input provided to the component
    input_data: EventData = None

    # Input format hint (json, text, binary)
    input_type: str = "json"

    # Content block index (for streaming)
    index: int = 0

    # attempt number
    attempt: int = 1


@dataclass(kw_only=True)
class Completed(LifecycleEvent):
    """Component or operation completed successfully."""

    _lifecycle_stage: ClassVar[str] = "completed"

    # Output produced by the component
    output_data: EventData = None

    # Output format hint
    output_type: str = "json"

    # Total execution time in milliseconds
    duration_ms: int = 0

    # Content block index (for streaming)
    index: int = 0


@dataclass(kw_only=True)
class Failed(LifecycleEvent):
    """Component or operation failed with error."""

    _lifecycle_stage: ClassVar[str] = "failed"

    # Error classification code
    error_code: str

    # Human-readable error description
    error_message: str

    # Optional stack trace for debugging
    error_traceback: Optional[str] = None

    # Time spent before failure
    duration_ms: int = 0


@dataclass(kw_only=True)
class Cancelled(LifecycleEvent):
    """Component or operation explicitly stopped."""

    _lifecycle_stage: ClassVar[str] = "cancelled"

    # Reason for cancellation
    reason: str = ""

    # Time spent before cancellation
    duration_ms: int = 0


@dataclass(kw_only=True)
class Timeout(LifecycleEvent):
    """Component or operation exceeded time limit."""

    _lifecycle_stage: ClassVar[str] = "timeout"

    # Configured timeout value that was exceeded
    timeout_ms: int

    # Actual time spent before timeout
    duration_ms: int = 0


@dataclass(kw_only=True)
class Paused(LifecycleEvent):
    """Component or operation awaiting input/approval."""

    _lifecycle_stage: ClassVar[str] = "paused"

    # Why paused (approval_needed, input_required, rate_limited)
    reason: str

    # Context data for resumption
    pause_data: EventData = None

    # Time spent before pausing
    duration_ms: int = 0


@dataclass(kw_only=True)
class Resumed(LifecycleEvent):
    """Component or operation continuing after pause."""

    _lifecycle_stage: ClassVar[str] = "resumed"

    # Input/approval that triggered resume
    resume_data: EventData = None

    # How long the component was paused
    paused_duration_ms: int = 0


# =============================================================================
# HITL Events
# =============================================================================


@dataclass(kw_only=True)
class ApprovalRequested(Event):
    """Request for human approval/decision.

    Emitted when a workflow needs human input before continuing.
    Event type is fixed as: approval.requested
    """

    # Question to ask the user
    question: str

    # Type of input: "text", "approval", "select", "multiselect"
    input_type: str = "approval"

    # Options for approval/select/multiselect (list of dicts with 'id' and 'label')
    options: list[dict[str, str]] = field(default_factory=list)

    # Step key for tracking this approval request
    step_key: str | None = None

    # Previous output to show for context
    previous_output: Any = None

    # Whether to allow a free-text "Something else" option
    allow_custom: bool = False

    # Whether the user can skip this input
    skippable: bool = False

    def __post_init__(self) -> None:
        self.event_type = "approval.requested"


@dataclass(kw_only=True)
class ApprovalResolved(Event):
    """Human responded to an approval request.

    Emitted when the user provides a response (approve, reject, or selection).
    Event type is fixed as: approval.resolved
    """

    # The user's decision: "approved", "rejected", or a custom option ID
    decision: str

    # Optional reason/feedback from user
    reason: str | None = None

    # Who made the decision
    decided_by: str | None = None

    def __post_init__(self) -> None:
        self.event_type = "approval.resolved"


@dataclass(kw_only=True)
class StateChanged(Event):
    """State mutation event for workflows.

    Used to track state changes within workflow execution.
    Event type is fixed as: workflow.state.changed
    """

    # State key that was modified
    key: Optional[str] = None

    # New value (None for delete/clear operations)
    value: Any = None

    # Type of operation: "set", "delete", "clear"
    operation: str = "set"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "workflow.state.changed")


# =============================================================================
# Streaming Events
# =============================================================================


@dataclass(kw_only=True)
class Delta(Event):
    """Streaming delta event for incremental content.

    Used for streaming content blocks (thinking, message, tool_call, output).
    Event type is computed as: {component_type}.{operation}.delta
    """

    # Component type - enforced via Enum
    component_type: ComponentType

    # Operation type - which content block
    operation: OperationType

    # The delta content
    content: Any

    # Content block index
    index: int = 0

    def __post_init__(self) -> None:
        et = f"{self.component_type.value}.{self.operation.value}.delta"
        object.__setattr__(self, "event_type", et)


# =============================================================================
# Output Streaming Events
# =============================================================================


@dataclass(kw_only=True)
class OutputStart(Event):
    """Marks the beginning of user code output streaming."""

    # Content block index
    index: int = 0

    event_type: str = field(default="output.start", init=False)


@dataclass(kw_only=True)
class OutputDelta(Event):
    """Incremental output content from user code."""

    # The delta content
    content: Any

    # Content block index
    index: int = 0

    event_type: str = field(default="output.delta", init=False)


@dataclass(kw_only=True)
class OutputStop(Event):
    """Marks the end of user code output streaming."""

    # Content block index
    index: int = 0

    event_type: str = field(default="output.stop", init=False)


# =============================================================================
# Progress Events
# =============================================================================


@dataclass(kw_only=True)
class ProgressUpdate(Event):
    """Progress indicator event.

    Standalone event for reporting progress - not a lifecycle event.
    """

    # Progress message
    message: Optional[str] = None

    # Completion percentage (0-100)
    percent: Optional[float] = None

    # Current item number
    current: Optional[int] = None

    # Total items
    total: Optional[int] = None

    event_type: str = field(default="progress.update", init=False)


# =============================================================================
# Transport
# =============================================================================


@dataclass
class EventEnvelope:
    """Transport envelope for events."""

    event_type: str
    data: dict[str, Any]
    source_timestamp_ns: int = field(default_factory=time.time_ns)
    content_index: int = 0
    metadata: Optional[dict[str, str]] = None


class EventEmitter:
    """Queues events to the platform."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        base_metadata: Optional[dict[str, str]] = None,
    ) -> None:
        self._run_id = run_id or ""
        self._base_metadata = base_metadata or {}
        self._sequence = 0
        self._worker: Any = None

    def set_worker(self, worker: Any) -> None:
        """Set the worker for queueing events."""
        self._worker = worker

    def emit(self, event: Event) -> EventEnvelope:
        """Emit a typed event to the platform.

        All event metadata (correlation_id, parent_correlation_id, timestamp, etc.)
        is extracted from the Event object.
        """
        event_data = event.to_dict()

        # Extract content_index from event if available (e.g., Delta.index)
        content_index = getattr(event, "index", 0)

        envelope = EventEnvelope(
            event_type=event.event_type,
            data=event_data,
            source_timestamp_ns=event.timestamp_ns,
            content_index=content_index,
            metadata=dict(event.metadata) if event.metadata else None,
        )

        self._queue_event(envelope, event.correlation_id, event.parent_correlation_id)
        return envelope

    async def emit_async(self, event: Event) -> EventEnvelope:
        """Emit a typed event asynchronously.

        Checkpoint gRPC runs on tokio runtime natively via future_into_py,
        avoiding thread pool overhead entirely.
        """
        event_data = event.to_dict()
        content_index = getattr(event, "index", 0)

        envelope = EventEnvelope(
            event_type=event.event_type,
            data=event_data,
            source_timestamp_ns=event.timestamp_ns,
            content_index=content_index,
            metadata=dict(event.metadata) if event.metadata else None,
        )

        await self._queue_event_async(envelope, event.correlation_id, event.parent_correlation_id)
        return envelope

    async def emit_batch_async(self, events: list[Event]) -> None:
        """Emit multiple events in a single AppendBatch RPC.

        Reduces gRPC overhead by batching non-terminal events together.
        Terminal events (.completed, .failed) should still use emit_async() for sync ack.
        """
        if not events or self._worker is None:
            return

        batch_tuples = []
        for event in events:
            event_data = event.to_dict()
            merged_metadata = dict(self._base_metadata)
            if event.metadata:
                merged_metadata.update(event.metadata)
            if event.correlation_id:
                merged_metadata["correlation_id"] = event.correlation_id
            if event.parent_correlation_id:
                merged_metadata["parent_correlation_id"] = event.parent_correlation_id

            self._sequence += 1
            batch_tuples.append((
                self._run_id,
                event.event_type,
                serialize_to_str(event_data),
                self._sequence,
                merged_metadata,
                event.timestamp_ns,
            ))

        await self._worker.emit_event_batch_async(batch_tuples)
        logger.debug(
            f"[EventEmitter.emit_batch_async] Batch emitted: {len(events)} events, "
            f"types={[e.event_type for e in events]}"
        )

    def __call__(self, event: Event) -> EventEnvelope:
        """Callable interface - delegates to emit()."""
        return self.emit(event)

    async def _queue_event_async(
        self,
        envelope: EventEnvelope,
        correlation_id: str,
        parent_correlation_id: str,
    ) -> None:
        """Queue event to the platform via Rust worker (async version).

        For checkpoint events, this method blocks until the platform acknowledges
        that the event has been persisted. This ensures correct event ordering.
        """
        logger.debug(
            f"[EventEmitter._queue_event_async] ENTRY: type={envelope.event_type}, "
            f"run_id={self._run_id}, has_worker={self._worker is not None}"
        )
        if self._worker is None:
            logger.warning(
                f"[EventEmitter._queue_event_async] No worker set, dropping event: "
                f"type={envelope.event_type}, run_id={self._run_id}"
            )
            return

        try:
            merged_metadata = dict(self._base_metadata)
            if envelope.metadata:
                merged_metadata.update(envelope.metadata)

            self._sequence += 1

            # Check if this is a checkpoint event that needs sync acknowledgement
            if is_checkpoint_event(envelope.event_type):
                # Add correlation IDs to metadata for journal persistence
                # Platform extracts these as "correlation_id" and "parent_correlation_id"
                if correlation_id:
                    merged_metadata["correlation_id"] = correlation_id
                if parent_correlation_id:
                    merged_metadata["parent_correlation_id"] = parent_correlation_id

                logger.debug(
                    f"[EventEmitter._queue_event_async] Emitting checkpoint event (sync): "
                    f"type={envelope.event_type}, run_id={self._run_id}, "
                    f"sequence={self._sequence}"
                )

                # Use async emit — runs gRPC on tokio, returns Python awaitable
                await self._worker.emit_event_async(
                    run_id=self._run_id,
                    event_type=envelope.event_type,
                    event_data=serialize_to_str(envelope.data),
                    sequence_number=self._sequence,
                    metadata=merged_metadata,
                    source_timestamp_ns=envelope.source_timestamp_ns,
                    timeout_ms=5000,  # 5 second timeout
                )
                logger.debug(
                    f"[EventEmitter._queue_event_async] Checkpoint event acknowledged: "
                    f"type={envelope.event_type}"
                )
            else:
                logger.debug(
                    f"[EventEmitter._queue_event_async] Queueing observability event (async): "
                    f"type={envelope.event_type}, run_id={self._run_id}, "
                    f"sequence={self._sequence}"
                )

                # Use async queue for observability/streaming events
                self._worker.queue_event(
                    invocation_id=self._run_id,
                    event_type=envelope.event_type,
                    event_data=serialize_to_str(envelope.data),
                    content_index=envelope.content_index,
                    sequence=self._sequence,
                    metadata=merged_metadata,
                    source_timestamp_ns=envelope.source_timestamp_ns,
                    is_streaming=True,
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_correlation_id,
                )
                logger.debug(
                    f"[EventEmitter._queue_event_async] Event queued: type={envelope.event_type}"
                )
        except Exception as e:
            logger.error(f"[EventEmitter._queue_event_async] Failed to queue event: {e}")

    def _queue_event(
        self,
        envelope: EventEnvelope,
        correlation_id: str,
        parent_correlation_id: str,
    ) -> None:
        """Queue event to the platform via Rust worker (sync version).

        Note: For checkpoint events, emit_event_sync blocks until the platform
        acknowledges persistence. For observability events, the async queue is used.
        """
        logger.debug(
            f"[EventEmitter._queue_event] ENTRY: type={envelope.event_type}, "
            f"run_id={self._run_id}, has_worker={self._worker is not None}"
        )
        if self._worker is None:
            logger.warning(
                f"[EventEmitter._queue_event] No worker set, dropping event: "
                f"type={envelope.event_type}, run_id={self._run_id}"
            )
            return

        try:
            merged_metadata = dict(self._base_metadata)
            if envelope.metadata:
                merged_metadata.update(envelope.metadata)

            self._sequence += 1

            # Check if this is a checkpoint event that needs sync acknowledgement
            if is_checkpoint_event(envelope.event_type):
                # Add correlation IDs to metadata for journal persistence
                # Platform extracts these as "correlation_id" and "parent_correlation_id"
                if correlation_id:
                    merged_metadata["correlation_id"] = correlation_id
                if parent_correlation_id:
                    merged_metadata["parent_correlation_id"] = parent_correlation_id

                logger.debug(
                    f"[EventEmitter._queue_event] Emitting checkpoint event (sync): "
                    f"type={envelope.event_type}, run_id={self._run_id}, "
                    f"sequence={self._sequence}"
                )

                # For checkpoint events, use emit_event_sync which blocks until
                # the platform acknowledges the event has been persisted.
                # This is a TRULY SYNCHRONOUS call - no async/await needed.
                try:
                    self._worker.emit_event_sync(
                        run_id=self._run_id,
                        event_type=envelope.event_type,
                        event_data=serialize_to_str(envelope.data),
                        sequence_number=self._sequence,
                        metadata=merged_metadata,
                        source_timestamp_ns=envelope.source_timestamp_ns,
                        timeout_ms=5000,
                    )
                    logger.debug(
                        f"[EventEmitter._queue_event] Checkpoint event acknowledged: "
                        f"type={envelope.event_type}"
                    )
                except Exception as e:
                    # On failure, fall back to async queue to ensure event isn't lost
                    logger.warning(
                        f"[EventEmitter._queue_event] Checkpoint sync emit failed ({e}), "
                        f"falling back to async: type={envelope.event_type}"
                    )
                    self._worker.queue_event(
                        invocation_id=self._run_id,
                        event_type=envelope.event_type,
                        event_data=serialize_to_str(envelope.data),
                        content_index=envelope.content_index,
                        sequence=self._sequence,
                        metadata=merged_metadata,
                        source_timestamp_ns=envelope.source_timestamp_ns,
                        is_streaming=True,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                    )
            else:
                logger.debug(
                    f"[EventEmitter._queue_event] Queueing observability event (async): "
                    f"type={envelope.event_type}, run_id={self._run_id}, "
                    f"sequence={self._sequence}"
                )

                # Use async queue for observability/streaming events
                self._worker.queue_event(
                    invocation_id=self._run_id,
                    event_type=envelope.event_type,
                    event_data=serialize_to_str(envelope.data),
                    content_index=envelope.content_index,
                    sequence=self._sequence,
                    metadata=merged_metadata,
                    source_timestamp_ns=envelope.source_timestamp_ns,
                    is_streaming=True,
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_correlation_id,
                )
                logger.debug(
                    f"[EventEmitter._queue_event] Event queued: type={envelope.event_type}"
                )
        except Exception as e:
            logger.error(f"[EventEmitter._queue_event] Failed to queue event: {e}")

    @property
    def run_id(self) -> str:
        return self._run_id


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Base
    "Event",
    "LifecycleEvent",
    # Enums
    "ComponentType",
    "OperationType",
    # Types
    "EventData",
    # Event classification
    "is_checkpoint_event",
    "is_sse_only_event",
    # Lifecycle events
    "Started",
    "Completed",
    "Failed",
    "Cancelled",
    "Timeout",
    "Paused",
    "Resumed",
    # State events
    "StateChanged",
    # HITL events
    "ApprovalRequested",
    "ApprovalResolved",
    # Streaming events
    "Delta",
    # Output streaming events
    "OutputStart",
    "OutputDelta",
    "OutputStop",
    # Progress events
    "ProgressUpdate",
    # Transport
    "EventEmitter",
    "EventEnvelope",
]
