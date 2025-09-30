"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management, backed by a high-performance
Rust core.
"""

from .version import _get_version
from ._compat import _rust_available, _import_error

# Low-level primitives
from . import durable, llm
from .context import Context, SignalClient, TimerClient, HumanClient, ApprovalResult, SpawnHandle
from .durable import BackoffPolicy, RetryPolicy
from .function import function
from .worker import Worker
from .logging import install_opentelemetry_logging, remove_opentelemetry_logging

# Entity primitive
from .entity import entity, EntityType, EntityProxy

# Tool primitive
from .tools import tool, Tool

# Workflow primitive
from .workflows import (
    FlowDefinition,
    WorkflowStep,
    register_workflow,
    workflow,
    task_step,
    wait_signal_step,
    wait_timer_step,
)

# Agent Development Kit (ADK) - High-level abstractions
from .agent import Agent, AgentResult, StreamEvent, AgentPlan
from .session import Session

# Language Model API
from .language_model import (
    LanguageModel,
    SyncLanguageModel,
    LanguageModelConfig,
    Response,
    StreamChunk,
    Usage,
)

__version__ = _get_version()

__all__ = [
    # Core modules
    "durable",
    "llm",
    # Low-level Primitives
    "function",
    "entity",
    "EntityType",
    "EntityProxy",
    "workflow",
    "register_workflow",
    "FlowDefinition",
    "WorkflowStep",
    "task_step",
    "wait_signal_step",
    "wait_timer_step",
    "Context",
    "SignalClient",
    "TimerClient",
    "HumanClient",
    "ApprovalResult",
    "SpawnHandle",
    "RetryPolicy",
    "BackoffPolicy",
    # Agent Development Kit (ADK)
    "Agent",
    "AgentResult",
    "StreamEvent",
    "AgentPlan",
    "Session",
    "tool",
    "Tool",
    # Language Model API
    "LanguageModel",
    "SyncLanguageModel",
    "LanguageModelConfig",
    "Response",
    "StreamChunk",
    "Usage",
    # Infrastructure
    "Worker",
    "install_opentelemetry_logging",
    "remove_opentelemetry_logging",
    "__version__",
]
