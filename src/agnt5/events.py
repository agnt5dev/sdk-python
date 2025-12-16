"""Streaming event types for AGNT5 SDK.

These events are sent over SSE connections from gateway to clients
during component execution (agents, workflows, functions).

Event types align with the journal event taxonomy defined in:
platform/pkg/repository/dto/journal_event.go
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import time


class StreamEventType(str, Enum):
    """All streaming event types."""

    # Run lifecycle
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_CANCELLED = "run.cancelled"
    RUN_TIMEOUT = "run.timeout"

    # LM streaming lifecycle
    LM_STREAM_STARTED = "lm.stream.started"
    LM_STREAM_COMPLETED = "lm.stream.completed"
    LM_STREAM_FAILED = "lm.stream.failed"

    # LM thinking content blocks (extended thinking / chain-of-thought)
    LM_THINKING_START = "lm.thinking.start"
    LM_THINKING_DELTA = "lm.thinking.delta"
    LM_THINKING_STOP = "lm.thinking.stop"

    # LM message content blocks (assistant output)
    LM_MESSAGE_START = "lm.message.start"
    LM_MESSAGE_DELTA = "lm.message.delta"
    LM_MESSAGE_STOP = "lm.message.stop"

    # LM tool call content blocks (LLM deciding to call a tool)
    LM_TOOL_CALL_START = "lm.tool_call.start"
    LM_TOOL_CALL_DELTA = "lm.tool_call.delta"
    LM_TOOL_CALL_STOP = "lm.tool_call.stop"

    # User code output streaming
    OUTPUT_START = "output.start"
    OUTPUT_DELTA = "output.delta"
    OUTPUT_STOP = "output.stop"

    # Progress indicators
    PROGRESS_UPDATE = "progress.update"

    # Tool execution events
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_ITERATION_STARTED = "agent.iteration.started"
    AGENT_ITERATION_COMPLETED = "agent.iteration.completed"
    AGENT_MAX_ITERATIONS = "agent.max_iterations.reached"

    # HITL: Approval events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_APPROVED = "approval.approved"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_EXPIRED = "approval.expired"

    # HITL: Input events
    INPUT_REQUESTED = "input.requested"
    INPUT_PROVIDED = "input.provided"
    INPUT_EXPIRED = "input.expired"

    # HITL: Feedback events
    FEEDBACK_REQUESTED = "feedback.requested"
    FEEDBACK_PROVIDED = "feedback.provided"
    FEEDBACK_EXPIRED = "feedback.expired"

    # Workflow step events
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_STEP_FAILED = "workflow.step.failed"


# --- Data payloads for streaming events ---


@dataclass
class ContentBlockStart:
    """Start of a content block (thinking, message, tool_call, output)."""

    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index}


@dataclass
class ContentBlockDelta:
    """Delta (incremental content) for a content block."""

    content: str
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "index": self.index}


@dataclass
class ContentBlockStop:
    """End of a content block."""

    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index}


@dataclass
class ToolCallStart:
    """Start of a tool call block."""

    id: str
    name: str
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "index": self.index}


@dataclass
class ToolCallDelta:
    """Delta for tool call input arguments."""

    input_delta: str
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"input_delta": self.input_delta, "index": self.index}


@dataclass
class ToolCallStop:
    """End of a tool call block with complete input."""

    id: str
    name: str
    input: Dict[str, Any]
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "input": self.input, "index": self.index}


@dataclass
class ProgressUpdate:
    """Progress indicator update."""

    message: Optional[str] = None
    percent: Optional[float] = None
    current: Optional[int] = None
    total: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.message is not None:
            d["message"] = self.message
        if self.percent is not None:
            d["percent"] = self.percent
        if self.current is not None:
            d["current"] = self.current
        if self.total is not None:
            d["total"] = self.total
        return d


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    thinking_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.thinking_tokens is not None:
            d["thinking_tokens"] = self.thinking_tokens
        return d


@dataclass
class ErrorDetail:
    """Error details."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"code": self.code, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


# --- Stream Event class ---


@dataclass
class StreamEvent:
    """A streaming event with typed data payload.

    This is the primary class for emitting events during streaming execution.
    Events are serialized and sent via the gRPC response stream to the gateway,
    which then emits them as SSE events to clients.
    """

    event_type: StreamEventType
    data: Dict[str, Any]
    content_index: int = 0
    sequence: int = 0

    def to_response_fields(self) -> Dict[str, Any]:
        """Convert to fields for ExecuteComponentResponse proto.

        Returns dict with:
        - event_type: str
        - content_index: int
        - sequence: int
        - output_data: bytes (JSON-encoded data)
        """
        return {
            "event_type": self.event_type.value,
            "content_index": self.content_index,
            "sequence": self.sequence,
            "output_data": json.dumps(self.data).encode("utf-8") if self.data else b"",
        }

    @classmethod
    def thinking_start(cls, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a thinking.start event."""
        return cls(
            event_type=StreamEventType.LM_THINKING_START,
            data=ContentBlockStart(index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def thinking_delta(cls, content: str, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a thinking.delta event."""
        return cls(
            event_type=StreamEventType.LM_THINKING_DELTA,
            data=ContentBlockDelta(content=content, index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def thinking_stop(cls, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a thinking.stop event."""
        return cls(
            event_type=StreamEventType.LM_THINKING_STOP,
            data=ContentBlockStop(index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def message_start(cls, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a message.start event."""
        return cls(
            event_type=StreamEventType.LM_MESSAGE_START,
            data=ContentBlockStart(index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def message_delta(cls, content: str, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a message.delta event."""
        return cls(
            event_type=StreamEventType.LM_MESSAGE_DELTA,
            data=ContentBlockDelta(content=content, index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def message_stop(cls, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a message.stop event."""
        return cls(
            event_type=StreamEventType.LM_MESSAGE_STOP,
            data=ContentBlockStop(index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def tool_call_start(
        cls, id: str, name: str, index: int = 0, sequence: int = 0
    ) -> "StreamEvent":
        """Create a tool_call.start event."""
        return cls(
            event_type=StreamEventType.LM_TOOL_CALL_START,
            data=ToolCallStart(id=id, name=name, index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def tool_call_delta(cls, input_delta: str, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create a tool_call.delta event."""
        return cls(
            event_type=StreamEventType.LM_TOOL_CALL_DELTA,
            data=ToolCallDelta(input_delta=input_delta, index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def tool_call_stop(
        cls, id: str, name: str, input: Dict[str, Any], index: int = 0, sequence: int = 0
    ) -> "StreamEvent":
        """Create a tool_call.stop event."""
        return cls(
            event_type=StreamEventType.LM_TOOL_CALL_STOP,
            data=ToolCallStop(id=id, name=name, input=input, index=index).to_dict(),
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def output_start(cls, index: int = 0, sequence: int = 0, content_type: str = "text/plain") -> "StreamEvent":
        """Create an output.start event for user code streaming."""
        return cls(
            event_type=StreamEventType.OUTPUT_START,
            data={"index": index, "content_type": content_type},
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def output_delta(cls, content: Any, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create an output.delta event for user code streaming."""
        return cls(
            event_type=StreamEventType.OUTPUT_DELTA,
            data={"content": content, "index": index},
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def output_stop(cls, index: int = 0, sequence: int = 0) -> "StreamEvent":
        """Create an output.stop event for user code streaming."""
        return cls(
            event_type=StreamEventType.OUTPUT_STOP,
            data={"index": index},
            content_index=index,
            sequence=sequence,
        )

    @classmethod
    def progress(
        cls,
        message: Optional[str] = None,
        percent: Optional[float] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        sequence: int = 0,
    ) -> "StreamEvent":
        """Create a progress.update event."""
        return cls(
            event_type=StreamEventType.PROGRESS_UPDATE,
            data=ProgressUpdate(
                message=message, percent=percent, current=current, total=total
            ).to_dict(),
            content_index=0,
            sequence=sequence,
        )

    @classmethod
    def run_completed(cls, output: Any, usage: Optional[TokenUsage] = None, sequence: int = 0) -> "StreamEvent":
        """Create a run.completed event."""
        data: Dict[str, Any] = {"output": output}
        if usage:
            data["usage"] = usage.to_dict()
        return cls(
            event_type=StreamEventType.RUN_COMPLETED,
            data=data,
            content_index=0,
            sequence=sequence,
        )

    @classmethod
    def run_failed(cls, error: ErrorDetail, sequence: int = 0) -> "StreamEvent":
        """Create a run.failed event."""
        return cls(
            event_type=StreamEventType.RUN_FAILED,
            data={"error": error.to_dict()},
            content_index=0,
            sequence=sequence,
        )
