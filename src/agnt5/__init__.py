"""
AGNT5 Python SDK - Build durable, resilient agent-first applications.

This SDK provides high-level components for building agents, tools, and workflows
with built-in durability guarantees and state management, backed by a high-performance
Rust core.
"""

from .version import _get_version
from ._compat import _rust_available, _import_error


__version__ = _get_version()

__all__ = [
    '__version__',
]
