"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management, backed by a high-performance
Rust core.
"""

from .version import _get_version
# Import compatibility checks
from ._compat import _rust_available, _import_error

# Import decorators
from .decorators import function

# Import high-level Worker
from .worker_manager import Worker

__version__ = _get_version()


# Import the Rust core if available
if _rust_available:
    from ._core import (
        PyWorker
    )

