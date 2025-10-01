"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management.
"""

from ._compat import _import_error, _rust_available
from .agent import Agent, AgentResult
from .context import Context
from .entity import EntityInstance, EntityType, entity
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

__version__ = _get_version()

__all__ = [
    # Version
    "__version__",
    # Core components
    "Context",
    "function",
    "FunctionRegistry",
    "entity",
    "EntityType",
    "EntityInstance",
    "workflow",
    "WorkflowRegistry",
    "tool",
    "Tool",
    "ToolRegistry",
    "Agent",
    "AgentResult",
    "Worker",
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
]
