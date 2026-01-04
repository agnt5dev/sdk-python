"""Unified event emission API for AGNT5 SDK.

This module provides a single, consistent way to emit events throughout the SDK.
All events use nanosecond timestamps captured at creation time for correct
logical ordering in the journal.

The EventEmitter uses a unified JournalEventQueue that handles all event types:
- Boundary events: Persisted to journal_events table (workflow.*, agent.*, lm.call.*, etc.)
- SSE-only events: Forwarded to SSE stream but NOT persisted (output.delta, log, etc.)

Events are delivered through the Rust SDK core's unified queue, which handles
both streaming (immediate) and non-streaming (batched) delivery modes.
"""

from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventType


# Task-local storage for parent correlation ID tracking
# This enables automatic parent-child relationship in event hierarchy
_parent_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_parent_correlation_id", default=None
)


def get_parent_correlation_id() -> Optional[str]:
    """Get the current parent correlation ID from task-local storage."""
    return _parent_correlation_id.get()


def set_parent_correlation_id(correlation_id: Optional[str]) -> contextvars.Token:
    """Set the parent correlation ID in task-local storage.

    Returns a token that can be used to restore the previous value.
    """
    return _parent_correlation_id.set(correlation_id)


@contextmanager
def parent_scope(correlation_id: str) -> Generator[None, None, None]:
    """Context manager to set this correlation_id as the parent for child events.

    Usage:
        with parent_scope(workflow_correlation_id):
            # All events emitted here will have parent_correlation_id set
            ctx.emit("agent.iteration.started", {...})
    """
    token = _parent_correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _parent_correlation_id.reset(token)


# SSE-only event type prefixes (not persisted to journal_events)
SSE_ONLY_EVENT_PREFIXES = (
    "output.",          # output.start, output.delta, output.stop
    "lm.stream.",       # lm.stream.start, lm.stream.delta, lm.stream.stop
    "lm.message.",      # lm.message.start, lm.message.delta, lm.message.stop
    "lm.thinking.",     # lm.thinking.start, lm.thinking.delta, lm.thinking.stop
    "progress.",        # progress.update
)


def is_sse_only_event_type(event_type: str) -> bool:
    """Check if an event type is SSE-only (not persisted).

    SSE-only events are forwarded to the SSE stream for real-time display
    but are NOT persisted to the journal_events table.

    Args:
        event_type: The event type string

    Returns:
        True if this event should only go to SSE, False if it should be persisted
    """
    return event_type.startswith(SSE_ONLY_EVENT_PREFIXES) or event_type == "log"


@dataclass
class JournalEvent:
    """Unified event container with nanosecond timestamp.

    All events in the SDK should be created as JournalEvent instances.
    The timestamp is captured at creation time using time.time_ns() for
    correct logical ordering when events are written to the journal.

    Attributes:
        event_type: Event type string (e.g., "workflow.step.started")
        data: Event payload dictionary
        source_timestamp_ns: Nanosecond timestamp when event was created
        content_index: Index for parallel content blocks (streaming)
        sequence: Monotonic sequence number for ordering
        step_key: Optional key for memoization (workflow steps)
        correlation_id: Optional ID for pairing started/completed events
        parent_event_id: Optional parent event ID for hierarchy
        is_sse_only: If True, event goes to SSE only (not persisted)
        metadata: Per-event metadata (name, etc.) - merged with base metadata
    """

    event_type: str
    data: Dict[str, Any]
    source_timestamp_ns: int = field(default_factory=time.time_ns)
    content_index: int = 0
    sequence: int = 0
    step_key: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    is_sse_only: bool = False
    metadata: Optional[Dict[str, str]] = None


class EventEmitter:
    """Unified event emission API.

    Provides a callable emit interface that routes all events through the
    unified JournalEventQueue in the Rust SDK core.

    Events are classified as:
    - Boundary events: Persisted to journal_events (workflow.*, agent.*, lm.call.*, etc.)
    - SSE-only events: Forwarded to SSE only (output.delta, lm.stream.*, log, etc.)

    The EventEmitter is callable, allowing direct use as ctx.emit():
        ctx.emit("workflow.started", {...}, metadata={"name": "Order Processing"})

    Metadata:
        Per-event metadata is merged with base metadata (tenant_id, deployment_id).
        Standard metadata keys:
        - name: Human-readable display name for UI
        - tenant_id: Multi-tenant routing (set at context level)
        - deployment_id: Deployment tracking (set at context level)

    Usage:
        # The emitter is callable - use ctx.emit() directly
        ctx.emit("workflow.started", {"input": data}, metadata={"name": "Order Processing"})
        ctx.emit("agent.completed", {"output": result}, metadata={"name": "Research Agent"})

        # With correlation for pairing started/completed events
        ctx.emit("lm.call.started", {...}, metadata={"name": model}, correlation_id="lm-123")
        ctx.emit("lm.call.completed", {...}, metadata={"name": model}, correlation_id="lm-123")
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        is_streaming: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize EventEmitter.

        Args:
            run_id: Run ID for this invocation
            is_streaming: Whether this is a streaming request
            metadata: Additional metadata (tenant_id, deployment_id, etc.)
        """
        self._run_id = run_id or ""
        self._is_streaming = is_streaming
        self._metadata = metadata or {}
        self._sequence = 0
        self._worker = None  # Will be set when context is initialized

    def set_worker(self, worker) -> None:
        """Set the worker for queueing events.

        Args:
            worker: PyWorker instance for queueing events
        """
        self._worker = worker

    def _next_sequence(self) -> int:
        """Get next sequence number (monotonically increasing)."""
        self._sequence += 1
        return self._sequence

    def __call__(
        self,
        event_type: Union[str, "EventType"],
        data: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, str]] = None,
        content_index: int = 0,
        step_key: Optional[str] = None,
        source_timestamp_ns: Optional[int] = None,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> JournalEvent:
        """Emit an event (callable interface).

        Allows ctx.emit("event_type", data) syntax instead of ctx.emit.emit().

        Args:
            event_type: Event type (string or EventType enum)
            data: Event payload dictionary
            metadata: Per-event metadata (name, etc.) - merged with base metadata
            content_index: Index for parallel content blocks (default: 0)
            step_key: Optional key for memoization lookup
            source_timestamp_ns: Optional override for timestamp (for replays)
            correlation_id: Optional ID for pairing started/completed events
            parent_event_id: Optional parent event ID for hierarchy

        Returns:
            The created JournalEvent for reference

        Example:
            ctx.emit("workflow.started", {"input": data}, metadata={"name": "Order Processing"})
            ctx.emit("agent.completed", {"output": result}, metadata={"name": "Research Agent"})
        """
        return self.emit(
            event_type,
            data,
            metadata=metadata,
            content_index=content_index,
            step_key=step_key,
            source_timestamp_ns=source_timestamp_ns,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def emit(
        self,
        event_type: Union[str, "EventType"],
        data: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, str]] = None,
        content_index: int = 0,
        step_key: Optional[str] = None,
        source_timestamp_ns: Optional[int] = None,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> JournalEvent:
        """Emit an event with timestamp captured NOW.

        Creates a JournalEvent and queues it for delivery through the
        unified JournalEventQueue.

        Args:
            event_type: Event type (string or EventType enum)
            data: Event payload dictionary
            metadata: Per-event metadata (merged with base metadata)
                      Standard keys: name (display name for UI)
            content_index: Index for parallel content blocks (default: 0)
            step_key: Optional key for memoization lookup
            source_timestamp_ns: Optional override for timestamp (for replays)
            correlation_id: Optional ID for pairing started/completed events
            parent_event_id: Optional parent event ID for hierarchy

        Returns:
            The created JournalEvent for reference

        Example:
            # Basic event with name in metadata
            ctx.emit("workflow.started", {"input": data}, metadata={"name": "Order Processing"})

            # With correlation for pairing
            ctx.emit("lm.call.started", {...}, metadata={"name": model}, correlation_id="lm-123")
            ctx.emit("lm.call.completed", {...}, metadata={"name": model}, correlation_id="lm-123")
        """
        # Convert EventType enum to string if needed
        type_str = event_type if isinstance(event_type, str) else event_type.value

        # Determine if this is an SSE-only event
        is_sse_only = is_sse_only_event_type(type_str)

        ts = source_timestamp_ns or time.time_ns()
        seq = self._next_sequence()

        # Auto-inject parent_correlation_id from task-local storage if not explicitly provided
        # This creates the event hierarchy: workflow → agent.iteration → lm.call
        final_metadata = dict(metadata) if metadata else {}
        parent_corr_id = get_parent_correlation_id()
        if parent_corr_id and "parent_correlation_id" not in final_metadata:
            final_metadata["parent_correlation_id"] = parent_corr_id

        # Create event
        event = JournalEvent(
            event_type=type_str,
            data=data,
            source_timestamp_ns=ts,
            content_index=content_index,
            sequence=seq,
            step_key=step_key,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            is_sse_only=is_sse_only,
            metadata=final_metadata if final_metadata else None,
        )

        # Queue for delivery via worker
        self._queue_event(event)

        return event

    def _queue_event(self, event: JournalEvent) -> None:
        """Queue event through the unified JournalEventQueue.

        Uses the worker's queue_event method which routes to the
        Rust SDK core's unified queue.

        Metadata is merged: base metadata (tenant_id, deployment_id) is combined
        with per-event metadata (name, etc.). Per-event values override base.
        """
        if self._worker is None:
            # No worker available - can happen in tests or before initialization
            return

        try:
            # Merge base metadata with per-event metadata
            # Per-event metadata overrides base metadata
            merged_metadata = dict(self._metadata)
            if event.metadata:
                merged_metadata.update(event.metadata)

            # Use the worker's queue_event method
            # This goes through the unified JournalEventQueue in Rust
            self._worker.queue_event(
                invocation_id=self._run_id,
                event_type=event.event_type,
                event_data=json.dumps(event.data),
                content_index=event.content_index,
                sequence=event.sequence,
                metadata=merged_metadata,
                source_timestamp_ns=event.source_timestamp_ns,
                is_streaming=self._is_streaming,
                correlation_id=event.correlation_id,
                parent_event_id=event.parent_event_id,
            )
        except Exception as e:
            # Log but don't fail on queue errors
            import logging
            logging.getLogger(__name__).warning(f"Failed to queue event: {e}")

    @property
    def is_streaming(self) -> bool:
        """Whether emitter is in streaming mode."""
        return self._is_streaming

    @property
    def sequence(self) -> int:
        """Current sequence number (for debugging)."""
        return self._sequence

    @property
    def run_id(self) -> str:
        """Run ID for this emitter."""
        return self._run_id
