"""Correlation ID generation utilities for AGNT5 SDK.

This module provides consistent, short correlation IDs for event tracing.
Uses 8-character hex strings (16^8 ≈ 4.3 billion combinations) generated
from a single os.urandom call for minimal syscall overhead.
"""

from __future__ import annotations

import secrets


def generate_cid() -> str:
    """Generate an 8-character correlation ID.

    Uses hex characters (0-9, a-f) for:
    - Single syscall (secrets.token_hex uses one os.urandom call)
    - URL-safe and readable
    - Compact storage in JSONB metadata

    Returns:
        8-character lowercase hex string
    """
    return secrets.token_hex(4)
