"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

Supports functions, workflows, agents, and LLM integration.
"""

import logging as _logging
import os as _os

# Configure the agnt5 parent logger so all internal SDK loggers (agnt5.events,
# agnt5.context, etc.) default to INFO. AGNT5_DEBUG=1 lowers to DEBUG.
# This only affects agnt5.* loggers — user loggers are untouched.
_agnt5_logger = _logging.getLogger("agnt5")
if not _agnt5_logger.handlers:
    _debug = _os.environ.get("AGNT5_DEBUG", "").lower() in ("1", "true", "yes")
    _agnt5_logger.setLevel(_logging.DEBUG if _debug else _logging.INFO)

from . import chat
from . import eval
from . import events
from . import lm
from .batch import (
    BatchConfig,
    BatchError,
    BatchItemError,
    BatchItemInput,
    BatchItemResult,
    BatchResult,
    BatchStats,
    BatchStatusResult,
    CancelBatchResult,
)
from .batch_eval import (
    BatchEvalItem,
    BatchEvalItemResult,
    BatchEvalResult,
    BatchEvalStats,
)
from .callbacks import (
    AgentCallbackContext,
    AgentCallbacks,
    AfterAgentCallback,
    AfterModelCallback,
    AfterToolCallback,
    BeforeAgentCallback,
    BeforeModelCallback,
    BeforeToolCallback,
    CallbackOverride,
    ModelCallbackContext,
    ToolCallbackContext,
    override,
)
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
from .chat import ChatBot, SlackConfig
from .client import AsyncClient, Client, ReceivedEvent, RunError
from .responses import (
    EvalResponse,
    Event,
    EventsResponse,
    RunErrorDetail,
    RunResponse,
    RunStatus,
    ScorerResultSummary,
    StatusResponse,
    SubmitLinks,
    SubmitResponse,
)
from .context import Context, LLMRuntimeOptions, RuntimeContext

# Entity API was removed in v0.4.0
# Use State API (ctx.state) and Memory API (ctx.memory) instead
# See migration guide: https://docs.agnt5.dev/migrations/entity-to-state-memory
from .events import (
    ApprovalRequested,
    ApprovalResolved,
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
    Prompt as LMPrompt,
)
from .types import (
    BackoffPolicy,
    BackoffType,
    RetryPolicy,
    TriggerSpec,
    WorkflowConfig,
    event,
    webhook,
)
from .version import _get_version
from .worker import Worker
from ._telemetry import get_logger, set_log_level
from .workflow import WorkflowContext, WorkflowRegistry, workflow
from .state import StateManager, SessionContext, UserContext

from .tool import Tool, ToolRegistry, tool

# Scorer components
from .scorer import (
    ScorerConfig,
    ScorerContext,
    ScorerRegistry,
    get_scorer_config,
    is_scorer,
    run_scorer,
    scorer,
)
from .eval.types import ScorerRequest, ScorerResult

# Memory components (new architecture)
from .memory import (
    ConversationAccessor,
    ConversationMessage,
    KVMemory,
    MemoryAccessor,
    MemoryMessage,
    MemoryMetadata,
    MemoryResult,
    MemoryScope,
    SemanticMemoryProvider,
    SemanticSearchResult,
    WorkingMemory,
)

# Legacy re-exports: ConversationMemory, SemanticMemory emit DeprecationWarning
# on access. GraphMemory/GraphNode/GraphRelationship/GraphTraversalResult are removed.

# Sandbox components
from .sandbox import (
    Sandbox,
    SandboxPool,
    ExecuteCodeResult,
    RunCommandResult,
    WriteFileResult,
    ReadFileResult,
    FileInfo,
    ListFilesResult,
    GitCloneResult,
    GitStatusFile,
    GitStatusResult,
    GitCommitResult,
    GitPushResult,
    SandboxHealthResult,
    StreamEvent,
)
from .sandbox_events import (
    SandboxExecuteCompleted,
    SandboxExecuteFailed,
    SandboxExecuteStarted,
    SandboxFileRead,
    SandboxFileWritten,
)
from .sandbox_tools import sandbox_tools

# MCP (Model Context Protocol) components
from .mcp import (
    MCPClient,
    MCPError,
    MCPServer,
    MCPServerError,
    CallToolResult,
    McpTool,
    McpToolWithServer,
    Prompt,
    Resource,
    ServerCapabilities,
    ServerConfig,
    ServerInfo,
    SseConfig,
    StdioConfig,
    ToolContent,
    TransportType,
)

# Not yet enabled:
# from .checkpoint import CheckpointClient
# from .exceptions import CheckpointError, StateError
# from .tool import AskUserTool, RequestApprovalTool
# from .types import FunctionConfig
# from . import _sentry as sentry

__version__ = _get_version()

__all__ = [
    # Version
    "__version__",
    # Modules
    "chat",
    "eval",
    "events",
    "lm",
    # Chat SDK
    "ChatBot",
    "SlackConfig",
    # Core components
    "Context",
    "LLMRuntimeOptions",
    "RuntimeContext",
    "FunctionContext",
    "Worker",
    "function",
    "FunctionRegistry",
    "get_logger",
    "set_log_level",
    # Entity API was removed in v0.4.0 - use State/Memory APIs
    # Workflow components
    "WorkflowContext",
    "WorkflowRegistry",
    "workflow",
    "WorkflowConfig",
    "TriggerSpec",
    "event",
    # State components
    "StateManager",
    "SessionContext",
    "UserContext",
    # Memory components
    "ConversationAccessor",
    "ConversationMessage",
    "KVMemory",
    "MemoryAccessor",
    "MemoryMessage",
    "MemoryMetadata",
    "MemoryResult",
    "MemoryScope",
    "SemanticMemoryProvider",
    "SemanticSearchResult",
    "WorkingMemory",
    # Agent components
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "Handoff",
    "agent",
    "handoff",
    "AgentCallbackContext",
    "AgentCallbacks",
    "AfterAgentCallback",
    "AfterModelCallback",
    "AfterToolCallback",
    "BeforeAgentCallback",
    "BeforeModelCallback",
    "BeforeToolCallback",
    "CallbackOverride",
    "ModelCallbackContext",
    "ToolCallbackContext",
    "override",
    # Tool components
    "Tool",
    "ToolRegistry",
    "tool",
    # Scorer components
    "scorer",
    "ScorerConfig",
    "ScorerContext",
    "ScorerRegistry",
    "ScorerRequest",
    "ScorerResult",
    "run_scorer",
    "is_scorer",
    "get_scorer_config",
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
    "LMPrompt",
    # Base events
    "ApprovalRequested",
    "ApprovalResolved",
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
    # Batch components
    "BatchConfig",
    "BatchError",
    "BatchItemError",
    "BatchItemInput",
    "BatchItemResult",
    "BatchResult",
    "BatchStats",
    "BatchStatusResult",
    "CancelBatchResult",
    # Batch eval components
    "BatchEvalItem",
    "BatchEvalItemResult",
    "BatchEvalResult",
    "BatchEvalStats",
    # Response types
    "EvalResponse",
    "Event",
    "EventsResponse",
    "RunErrorDetail",
    "RunResponse",
    "RunStatus",
    "ScorerResultSummary",
    "StatusResponse",
    "SubmitLinks",
    "SubmitResponse",
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
    # Sandbox components
    "Sandbox",
    "SandboxPool",
    "ExecuteCodeResult",
    "RunCommandResult",
    "WriteFileResult",
    "ReadFileResult",
    "FileInfo",
    "ListFilesResult",
    "GitCloneResult",
    "GitStatusFile",
    "GitStatusResult",
    "GitCommitResult",
    "GitPushResult",
    "SandboxHealthResult",
    "StreamEvent",
    # MCP components
    "MCPClient",
    "MCPError",
    "MCPServer",
    "MCPServerError",
    "CallToolResult",
    "McpTool",
    "McpToolWithServer",
    "Prompt",
    "Resource",
    "ServerCapabilities",
    "ServerConfig",
    "ServerInfo",
    "SseConfig",
    "StdioConfig",
    "ToolContent",
    "TransportType",
]
