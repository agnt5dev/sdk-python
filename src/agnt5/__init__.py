"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management.
"""

from ._compat import _import_error, _rust_available
from .agent import Agent, AgentRegistry, AgentResult, Handoff, agent, handoff
from .client import Client, RunError
from .context import Context
from .entity import (
    DurableEntity,
    SessionEntity,
    MemoryEntity,
    WorkflowEntity,
    EntityInstance,
    EntityRegistry,
    EntityType,
    entity,
)
from .exceptions import (
    AGNT5Error,
    CheckpointError,
    ConfigurationError,
    ExecutionError,
    RetryError,
    StateError,
)
from .function import FunctionRegistry, function
from .tool import Tool, ToolRegistry, tool
from .types import BackoffPolicy, BackoffType, FunctionConfig, RetryPolicy, WorkflowConfig
from .version import _get_version
from .worker import Worker
from .workflow import WorkflowRegistry, workflow

# Expose simplified language model API (recommended)
from . import lm

__version__ = _get_version()

__all__ = [
    # Version
    "__version__",
    # Core components
    "Context",
    "Client",
    "Worker",
    "function",
    "FunctionRegistry",
    "entity",
    "EntityType",
    "EntityInstance",
    "EntityRegistry",
    "DurableEntity",
    "SessionEntity",
    "MemoryEntity",
    "WorkflowEntity",
    "workflow",
    "WorkflowRegistry",
    "tool",
    "Tool",
    "ToolRegistry",
    "agent",
    "Agent",
    "AgentRegistry",
    "AgentResult",
    "Handoff",
    "handoff",
    # Types
    "RetryPolicy",
    "BackoffPolicy",
    "BackoffType",
    "FunctionConfig",
    "WorkflowConfig",
    # Exceptions
    "AGNT5Error",
    "ConfigurationError",
    "ExecutionError",
    "RetryError",
    "StateError",
    "CheckpointError",
    "RunError",
    # Language Model (Simplified API)
    "lm",
]
