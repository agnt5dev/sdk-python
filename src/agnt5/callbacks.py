"""Execution callback types for AGNT5 agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from .agent.core import Agent
    from .agent.result import AgentResult
    from .context import Context
    from .lm import GenerateRequest, GenerateResponse, Message, ToolDefinition
    from .tool import Tool

T = TypeVar("T")
MaybeAwaitable = Union[T, Awaitable[T]]


@dataclass(frozen=True)
class CallbackOverride(Generic[T]):
    """Explicit callback override value.

    Use this when the replacement value may be ``None``. Plain ``None`` means
    "continue normal execution".
    """

    value: T


def override(value: T) -> CallbackOverride[T]:
    """Return an explicit callback override value."""
    return CallbackOverride(value)


@dataclass(frozen=True)
class AgentCallbackContext:
    """Context passed to agent lifecycle callbacks."""

    agent: "Agent"
    context: "Context"
    user_message: str
    history: Optional[List["Message"]] = None
    prompt_context: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ModelCallbackContext:
    """Context passed to model callbacks."""

    agent: "Agent"
    context: "Context"
    iteration: int
    messages: List["Message"]
    tool_definitions: List["ToolDefinition"]


@dataclass(frozen=True)
class ToolCallbackContext:
    """Context passed to tool callbacks."""

    agent: "Agent"
    context: "Context"
    iteration: int
    tool_name: str
    tool_call_id: str
    tool_call: Dict[str, Any]
    arguments: Dict[str, Any]
    tool: Optional["Tool"] = None


BeforeAgentCallback = Callable[
    [AgentCallbackContext],
    MaybeAwaitable[Union["AgentResult", str, Dict[str, Any], CallbackOverride[Any], None]],
]
AfterAgentCallback = Callable[
    [AgentCallbackContext, "AgentResult"],
    MaybeAwaitable[Union["AgentResult", str, Dict[str, Any], CallbackOverride[Any], None]],
]
BeforeModelCallback = Callable[
    [ModelCallbackContext, "GenerateRequest"],
    MaybeAwaitable[Union["GenerateResponse", CallbackOverride[Any], None]],
]
AfterModelCallback = Callable[
    [ModelCallbackContext, "GenerateRequest", "GenerateResponse"],
    MaybeAwaitable[Union["GenerateResponse", CallbackOverride[Any], None]],
]
BeforeToolCallback = Callable[
    [ToolCallbackContext],
    MaybeAwaitable[Union[Any, CallbackOverride[Any], None]],
]
AfterToolCallback = Callable[
    [ToolCallbackContext, Any],
    MaybeAwaitable[Union[Any, CallbackOverride[Any], None]],
]


@dataclass(frozen=True)
class AgentCallbacks:
    """Callbacks attached to an agent execution loop."""

    before_agent: Optional[BeforeAgentCallback] = None
    after_agent: Optional[AfterAgentCallback] = None
    before_model: Optional[BeforeModelCallback] = None
    after_model: Optional[AfterModelCallback] = None
    before_tool: Optional[BeforeToolCallback] = None
    after_tool: Optional[AfterToolCallback] = None


__all__ = [
    "AgentCallbackContext",
    "AgentCallbacks",
    "AfterAgentCallback",
    "AfterModelCallback",
    "AfterToolCallback",
    "BeforeAgentCallback",
    "BeforeModelCallback",
    "BeforeToolCallback",
    "CallbackOverride",
    "MaybeAwaitable",
    "ModelCallbackContext",
    "ToolCallbackContext",
    "override",
]
