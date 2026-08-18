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

from . import chat, eval, events, improvement, lm, serverless
from ._telemetry import get_logger, set_log_level
from .activation import (
    ActivationEvidence,
    ActivationExecution,
    ActivationRecoveryPolicy,
    ActivationUsage,
    ChildJoinPolicy,
    current_activation,
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
    Skill,
    SkillLoaded,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    agent,
    discover_agents_md,
    discover_skills,
    handoff,
    load_agents_md,
    resolve_skills,
)
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
    AfterAgentCallback,
    AfterModelCallback,
    AfterToolCallback,
    AgentCallbackContext,
    AgentCallbacks,
    BeforeAgentCallback,
    BeforeModelCallback,
    BeforeToolCallback,
    CallbackOverride,
    ModelCallbackContext,
    ToolCallbackContext,
    override,
)
from .chat import ChatBot, SlackConfig
from .client import AsyncClient, Client, ReceivedEvent, RunError
from .context import Context, LLMRuntimeOptions, RuntimeContext
from .eval.types import ScorerRequest, ScorerResult

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
    ActivationError,
    ActivationErrorCode,
    AGNT5Error,
    ConfigurationError,
    ExecutionError,
    RetryError,
    WaitingForUserInputException,
)
from .function import FunctionContext, FunctionRegistry, function
from .improvement import (
    AGNT5ImprovementBlocks,
    BehaviorTopic,
    FixtureImprovementBlocks,
    ImprovementBlocks,
    ImprovementControlPlaneClient,
    ImprovementControlPlaneError,
    ImprovementLoopPolicy,
    ImprovementLoopRequest,
    ImprovementLoopResult,
    ImprovementProposal,
    LoopStatus,
    PromotionAction,
    PromotionDecision,
    QualityCase,
    RepresentativeRun,
    SelfImprovementLoop,
    default_fixture_topic,
)
from .improvement import (
    EvaluationResult as ImprovementEvaluationResult,
)
from .lm import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
    LMFailed,
    LMStarted,
)
from .lm import (
    Prompt as LMPrompt,
)

# MCP (Model Context Protocol) components
from .mcp import (
    CallToolResult,
    MCPClient,
    MCPError,
    MCPServer,
    MCPServerError,
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

# Legacy re-exports: ConversationMemory, SemanticMemory emit DeprecationWarning
# on access. GraphMemory/GraphNode/GraphRelationship/GraphTraversalResult are removed.
# Sandbox components
from .sandbox import (
    ExecuteCodeResult,
    FileInfo,
    GitCloneResult,
    GitCommitResult,
    GitPushResult,
    GitStatusFile,
    GitStatusResult,
    InMemorySandbox,
    ListFilesResult,
    ReadFileResult,
    RunCommandResult,
    Sandbox,
    SandboxHealthResult,
    SandboxPool,
    StreamEvent,
    WriteFileResult,
)
from .sandbox_events import (
    SandboxExecuteCompleted,
    SandboxExecuteFailed,
    SandboxExecuteStarted,
    SandboxFileRead,
    SandboxFileWritten,
)
from .sandbox_providers import (
    CreateSandboxOptions,
    DaytonaSandbox,
    DaytonaSandboxProvider,
    E2BSandbox,
    E2BSandboxProvider,
    NorthflankSandbox,
    NorthflankSandboxProvider,
    SandboxInfo,
    SandboxProviderError,
    TogetherSandbox,
    TogetherSandboxProvider,
    VercelSandbox,
    VercelSandboxProvider,
    load_providers_from_env,
)
from .sandbox_tools import sandbox_tools

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
from .state import SessionContext, StateManager, UserContext
from .tool import Tool, ToolRegistry, tool
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
from .workflow import WorkflowContext, WorkflowRegistry, workflow

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
    "improvement",
    "lm",
    "serverless",
    # Chat SDK
    "ChatBot",
    "SlackConfig",
    # Core components
    "Context",
    "LLMRuntimeOptions",
    "RuntimeContext",
    # Self-improvement loop components
    "AGNT5ImprovementBlocks",
    "BehaviorTopic",
    "ImprovementEvaluationResult",
    "FixtureImprovementBlocks",
    "ImprovementControlPlaneClient",
    "ImprovementControlPlaneError",
    "ImprovementBlocks",
    "ImprovementLoopPolicy",
    "ImprovementLoopRequest",
    "ImprovementLoopResult",
    "ImprovementProposal",
    "LoopStatus",
    "PromotionAction",
    "PromotionDecision",
    "QualityCase",
    "RepresentativeRun",
    "SelfImprovementLoop",
    "default_fixture_topic",
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
    "Skill",
    "SkillLoaded",
    "agent",
    "discover_agents_md",
    "discover_skills",
    "handoff",
    "load_agents_md",
    "resolve_skills",
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
    "ActivationError",
    "ActivationErrorCode",
    "ActivationEvidence",
    "ActivationExecution",
    "ActivationRecoveryPolicy",
    "ChildJoinPolicy",
    "ActivationUsage",
    "current_activation",
    "ConfigurationError",
    "ExecutionError",
    "RetryError",
    "WaitingForUserInputException",
    # Sandbox components
    "InMemorySandbox",
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
    # Sandbox provider integrations
    "CreateSandboxOptions",
    "SandboxInfo",
    "SandboxProviderError",
    "E2BSandboxProvider",
    "E2BSandbox",
    "DaytonaSandboxProvider",
    "DaytonaSandbox",
    "VercelSandboxProvider",
    "VercelSandbox",
    "NorthflankSandboxProvider",
    "NorthflankSandbox",
    "TogetherSandboxProvider",
    "TogetherSandbox",
    "load_providers_from_env",
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
