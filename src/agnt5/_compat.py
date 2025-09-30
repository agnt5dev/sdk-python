"""
Compatibility utilities for the AGNT5 Python SDK.

This module handles runtime compatibility checks and provides utilities
for cross-referencing throughout the project.
"""

# Check if Rust core is available
try:
    from . import _core

    _rust_available = True
    _import_error = None
except ImportError as e:
    _rust_available = False
    _import_error = e
