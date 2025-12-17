"""Agent execution result."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import Context


class AgentResult:
    """Result from agent execution.

    Attributes:
        output: The final text output from the agent
        tool_calls: List of tool calls made during execution
        context: The execution context
        handoff_to: Name of agent that was handed off to (if any)
        handoff_metadata: Additional metadata from handoff
    """

    def __init__(
        self,
        output: str,
        tool_calls: List[Dict[str, Any]],
        context: "Context",
        handoff_to: Optional[str] = None,
        handoff_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output = output
        self.tool_calls = tool_calls
        self.context = context
        self.handoff_to = handoff_to
        self.handoff_metadata = handoff_metadata or {}

    def __repr__(self) -> str:
        return (
            f"AgentResult(output={self.output[:50]!r}..., "
            f"tool_calls={len(self.tool_calls)}, "
            f"handoff_to={self.handoff_to})"
        )
