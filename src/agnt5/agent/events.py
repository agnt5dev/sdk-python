"""Agent-specific event classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from ..events import (
    Completed,
    ComponentType,
    Failed,
    OperationType,
    Started,
)


# =============================================================================
# Agent Lifecycle Events
# =============================================================================


@dataclass(kw_only=True)
class AgentStarted(Started):
    """Agent execution started."""

    _event_type: ClassVar[str] = "agent.started"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    agent_model: str = ""
    tool_names: list[str] = field(default_factory=list)
    max_iterations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class AgentCompleted(Completed):
    """Agent execution completed."""

    _event_type: ClassVar[str] = "agent.completed"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    iterations: int = 0
    tool_calls_count: int = 0
    handoff_to: Optional[str] = None
    output_length: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class AgentFailed(Failed):
    """Agent execution failed."""

    _event_type: ClassVar[str] = "agent.failed"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    iterations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


# =============================================================================
# Agent Iteration Events
# =============================================================================


@dataclass(kw_only=True)
class AgentIterationStarted(Started):
    """Agent iteration started."""

    _event_type: ClassVar[str] = "agent.iteration.started"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    operation: OperationType = field(default=OperationType.ITERATION, init=False)
    iteration: int = 0
    max_iterations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class AgentIterationCompleted(Completed):
    """Agent iteration completed."""

    _event_type: ClassVar[str] = "agent.iteration.completed"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    operation: OperationType = field(default=OperationType.ITERATION, init=False)
    iteration: int = 0
    has_tool_calls: bool = False
    tool_calls_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


# =============================================================================
# Tool Call Events (under AGENT)
# =============================================================================


@dataclass(kw_only=True)
class ToolCallStarted(Started):
    """Tool call started."""

    _event_type: ClassVar[str] = "agent.tool_call.started"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    operation: OperationType = field(default=OperationType.TOOL_CALL, init=False)
    tool_name: str = ""
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class ToolCallCompleted(Completed):
    """Tool call completed."""

    _event_type: ClassVar[str] = "agent.tool_call.completed"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    operation: OperationType = field(default=OperationType.TOOL_CALL, init=False)
    tool_name: str = ""
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class ToolCallFailed(Failed):
    """Tool call failed."""

    _event_type: ClassVar[str] = "agent.tool_call.failed"
    component_type: ComponentType = field(default=ComponentType.AGENT, init=False)
    operation: OperationType = field(default=OperationType.TOOL_CALL, init=False)
    tool_name: str = ""
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


__all__ = [
    "AgentStarted",
    "AgentCompleted",
    "AgentFailed",
    "AgentIterationStarted",
    "AgentIterationCompleted",
    "ToolCallStarted",
    "ToolCallCompleted",
    "ToolCallFailed",
]
