"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

Supports functions, entities, workflows, agents, and LLM integration.
"""

from . import events
from . import lm
from .agent import (
    Agent,
    AgentCompleted,
    AgentContext,
    AgentFailed,
    AgentIterationCompleted,
    AgentIterationStarted,
    AgentRegistry,
    AgentResult,
    AgentStarted,
    Handoff,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    agent,
    handoff,
)
from .client import AsyncClient, Client, ReceivedEvent, RunError
from .context import Context
from .entity import (
    Entity,
    EntityRegistry,
    EntityStateAdapter,
    EntityType,
    StateType,
    create_entity_context,
    query,
    with_entity_context,
)
from .events import (
    Cancelled,
    Completed,
    ComponentType,
    Delta,
    Event,
    EventEmitter,
    EventEnvelope,
    Failed,
    LifecycleEvent,
    OperationType,
    OutputDelta,
    OutputStart,
    OutputStop,
    Paused,
    ProgressUpdate,
    Resumed,
    Started,
    StateChanged,
    Timeout,
)
from .exceptions import (
    AGNT5Error,
    ConfigurationError,
    ExecutionError,
    RetryError,
    WaitingForUserInputException,
)
from .function import FunctionContext, FunctionRegistry, function
from .lm import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
    LMFailed,
    LMStarted,
)
from .types import BackoffPolicy, BackoffType, RetryPolicy, WorkflowConfig
from .version import _get_version
from .worker import Worker
from .workflow import WorkflowContext, WorkflowRegistry, workflow

from .tool import Tool, ToolRegistry, tool

# Not yet enabled:
# from .checkpoint import CheckpointClient
# from .exceptions import CheckpointError, StateError
# from .memory import ConversationMemory, MemoryMessage, MemoryMetadata, MemoryResult, MemoryScope, SemanticMemory
# from .tool import AskUserTool, RequestApprovalTool
# from .types import FunctionConfig
# from . import _sentry as sentry

__version__ = _get_version()

__all__ = [
    # Version
    "__version__",
    # Modules
    "events",
    "lm",
    # Core components
    "Context",
    "FunctionContext",
    "Worker",
    "function",
    "FunctionRegistry",
    # Entity components
    "Entity",
    "EntityRegistry",
    "EntityStateAdapter",
    "EntityType",
    "StateType",
    "create_entity_context",
    "query",
    "with_entity_context",
    # Workflow components
    "WorkflowContext",
    "WorkflowRegistry",
    "workflow",
    "WorkflowConfig",
    # Agent components
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "Handoff",
    "agent",
    "handoff",
    # Tool components
    "Tool",
    "ToolRegistry",
    "tool",
    # Agent events
    "AgentCompleted",
    "AgentFailed",
    "AgentIterationCompleted",
    "AgentIterationStarted",
    "AgentStarted",
    "ToolCallCompleted",
    "ToolCallFailed",
    "ToolCallStarted",
    # LM events
    "LMCompleted",
    "LMContentBlockCompleted",
    "LMContentBlockDelta",
    "LMContentBlockStarted",
    "LMFailed",
    "LMStarted",
    # Base events
    "Cancelled",
    "Completed",
    "ComponentType",
    "Delta",
    "Event",
    "EventEmitter",
    "EventEnvelope",
    "Failed",
    "LifecycleEvent",
    "OperationType",
    "OutputDelta",
    "OutputStart",
    "OutputStop",
    "Paused",
    "ProgressUpdate",
    "Resumed",
    "Started",
    "StateChanged",
    "Timeout",
    # Client
    "AsyncClient",
    "Client",
    "ReceivedEvent",
    "RunError",
    # Types
    "BackoffPolicy",
    "BackoffType",
    "RetryPolicy",
    # Exceptions
    "AGNT5Error",
    "ConfigurationError",
    "ExecutionError",
    "RetryError",
    "WaitingForUserInputException",
]
