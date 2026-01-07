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

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Union

from edwh_uuid7 import uuid7

logger = logging.getLogger(__name__)


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

    def __call__(self, event: Event) -> EventEnvelope:
        """Callable interface - delegates to emit()."""
        return self.emit(event)

    def _queue_event(
        self,
        envelope: EventEnvelope,
        correlation_id: str,
        parent_correlation_id: str,
    ) -> None:
        """Queue event to the platform via Rust worker."""
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

            logger.info(
                f"[EventEmitter._queue_event] Queueing event to Rust worker: "
                f"type={envelope.event_type}, run_id={self._run_id}, "
                f"sequence={self._sequence}, correlation_id={correlation_id}"
            )

            self._worker.queue_event(
                invocation_id=self._run_id,
                event_type=envelope.event_type,
                event_data=json.dumps(envelope.data),
                content_index=envelope.content_index,
                sequence=self._sequence,
                metadata=merged_metadata,
                source_timestamp_ns=envelope.source_timestamp_ns,
                is_streaming=True,
                correlation_id=correlation_id,
                parent_event_id=parent_correlation_id,
            )
            logger.debug(
                f"[EventEmitter._queue_event] Event queued successfully: type={envelope.event_type}"
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
    # Streaming events
    "Delta",
    # Progress events
    "ProgressUpdate",
    # Transport
    "EventEmitter",
    "EventEnvelope",
]
