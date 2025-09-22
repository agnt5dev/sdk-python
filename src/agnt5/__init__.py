"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management, backed by a high-performance
Rust core.
"""

from .version import _get_version
from ._compat import _rust_available, _import_error
from . import durable, llm, agent
from .context import Context, SignalClient, TimerClient, HumanClient, ApprovalResult, SpawnHandle
from .durable import BackoffPolicy, RetryPolicy
from .decorators import function
from .worker import Worker
from .logging import install_opentelemetry_logging, remove_opentelemetry_logging
from .workflows import (
    FlowDefinition,
    WorkflowStep,
    register_workflow,
    workflow,
    task_step,
    wait_signal_step,
    wait_timer_step,
)

__version__ = _get_version()

__all__ = [
    'durable',
    'llm',
    'agent',
    'function',
    'Context',
    'SignalClient',
    'TimerClient',
    'HumanClient',
    'ApprovalResult',
    'SpawnHandle',
    'RetryPolicy',
    'BackoffPolicy',
    'workflow',
    'register_workflow',
    'FlowDefinition',
    'WorkflowStep',
    'task_step',
    'wait_signal_step',
    'wait_timer_step',
    'Worker',
    'install_opentelemetry_logging',
    'remove_opentelemetry_logging',
    '__version__',
]
