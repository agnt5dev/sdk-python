"""Correlation ID generation utilities for AGNT5 SDK.

This module provides consistent, short correlation IDs for event tracing.
Uses 8-character lowercase alphanumeric strings (36^8 ≈ 2.8 trillion combinations).
"""

from __future__ import annotations

import secrets
import string

# Lowercase alphanumeric alphabet (36 characters)
_ALPHABET = string.ascii_lowercase + string.digits


def generate_cid() -> str:
    """Generate an 8-character correlation ID.

    Uses lowercase alphanumeric characters (a-z, 0-9) for:
    - High entropy (36^8 ≈ 2.8 trillion combinations)
    - URL-safe and readable
    - Compact storage in JSONB metadata

    Returns:
        8-character lowercase alphanumeric string
    """
    return "".join(secrets.choice(_ALPHABET) for _ in range(8))
