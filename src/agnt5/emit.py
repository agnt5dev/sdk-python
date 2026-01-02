"""Unified event emission API for AGNT5 SDK.

This module provides a single, consistent way to emit events throughout the SDK.
All events use nanosecond timestamps captured at creation time for correct
logical ordering in the journal.

The EventEmitter routes events based on streaming mode:
- Streaming mode: immediate delivery via delta queue
- Non-streaming mode: buffered delivery via checkpoint queue
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventType


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
    """

    event_type: str
    data: Dict[str, Any]
    source_timestamp_ns: int = field(default_factory=time.time_ns)
    content_index: int = 0
    sequence: int = 0
    step_key: Optional[str] = None


# Type alias for checkpoint callback
CheckpointCallback = Callable[[Dict[str, Any]], None]

# Type alias for delta callback
# (event_type: str, output_data: str, content_index: int, sequence: int, source_timestamp_ns: int)
DeltaCallback = Callable[[str, str, int, int, int], None]


class EventEmitter:
    """Unified event emission API.

    Provides a single emit() method that routes events to the appropriate
    delivery mechanism based on streaming mode:

    - Streaming mode: Events are delivered immediately via the delta queue
      for real-time SSE streaming to clients.

    - Non-streaming mode: Events are buffered and delivered via the checkpoint
      queue for batch persistence.

    Both paths ultimately write to the same journal table, but streaming mode
    prioritizes low latency while non-streaming mode batches for efficiency.

    Usage:
        # Create emitter (typically done by Context initialization)
        emitter = EventEmitter(
            checkpoint_callback=ctx._checkpoint_callback,
            delta_callback=ctx._delta_callback,
            is_streaming=ctx._is_streaming,
        )

        # Emit events consistently
        emitter.emit("workflow.step.started", {"step_name": "fetch_data"})
        emitter.emit("agent.completed", {"output": result, "iterations": 3})
    """

    def __init__(
        self,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        delta_callback: Optional[DeltaCallback] = None,
        is_streaming: bool = False,
    ) -> None:
        """Initialize EventEmitter.

        Args:
            checkpoint_callback: Callback for buffered checkpoint delivery
            delta_callback: Callback for immediate streaming delivery
            is_streaming: Whether to use streaming (immediate) delivery
        """
        self._checkpoint_callback = checkpoint_callback
        self._delta_callback = delta_callback
        self._is_streaming = is_streaming
        self._sequence = 0

    def _next_sequence(self) -> int:
        """Get next sequence number (monotonically increasing)."""
        self._sequence += 1
        return self._sequence

    def emit(
        self,
        event_type: Union[str, "EventType"],
        data: Dict[str, Any],
        *,
        content_index: int = 0,
        step_key: Optional[str] = None,
        source_timestamp_ns: Optional[int] = None,
    ) -> JournalEvent:
        """Emit an event with timestamp captured NOW.

        Creates a JournalEvent with the current nanosecond timestamp and
        queues it for delivery through the appropriate channel.

        Args:
            event_type: Event type (string or EventType enum)
            data: Event payload dictionary
            content_index: Index for parallel content blocks (default: 0)
            step_key: Optional key for memoization lookup
            source_timestamp_ns: Optional override for timestamp (for replays)

        Returns:
            The created JournalEvent for reference (e.g., to get event_id)

        Example:
            # Basic event
            ctx.emit("workflow.step.started", {"step_name": "process"})

            # With memoization key
            ctx.emit("lm.call.started", {...}, step_key="lm_call_1")
        """
        # Convert EventType enum to string if needed
        type_str = event_type if isinstance(event_type, str) else event_type.value

        # Create event with nanosecond timestamp
        event = JournalEvent(
            event_type=type_str,
            data=data,
            source_timestamp_ns=source_timestamp_ns or time.time_ns(),
            content_index=content_index,
            sequence=self._next_sequence(),
            step_key=step_key,
        )

        # Queue for delivery
        self._queue_event(event)

        return event

    def _queue_event(self, event: JournalEvent) -> None:
        """Queue event using appropriate callback based on streaming mode.

        Streaming mode uses delta_callback for immediate delivery.
        Non-streaming mode uses checkpoint_callback for buffered delivery.
        """
        if self._is_streaming and self._delta_callback:
            # Immediate delivery for streaming requests
            self._delta_callback(
                event.event_type,
                json.dumps(event.data),
                event.content_index,
                event.sequence,
                event.source_timestamp_ns,
            )
        elif self._checkpoint_callback:
            # Buffered delivery for non-streaming requests
            checkpoint = {
                "checkpoint_type": event.event_type,
                "checkpoint_data": event.data,
                "sequence_number": event.sequence,
                "source_timestamp_ns": event.source_timestamp_ns,
            }
            # Add step_key if present (for memoization)
            if event.step_key:
                checkpoint["step_key"] = event.step_key

            self._checkpoint_callback(checkpoint)
        else:
            # No callback available - this can happen in test scenarios
            # or when context is not fully initialized
            pass

    @property
    def is_streaming(self) -> bool:
        """Whether emitter is in streaming mode."""
        return self._is_streaming

    @property
    def sequence(self) -> int:
        """Current sequence number (for debugging)."""
        return self._sequence
