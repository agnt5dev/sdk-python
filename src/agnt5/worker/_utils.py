"""Worker-specific utilities."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

from .._serialization import normalize_metadata

if TYPE_CHECKING:
    from .._core import PyExecuteComponentResponse

__all__ = ["create_failed_response", "format_error_message", "build_error_metadata"]


def format_error_message(exception: Exception) -> str:
    """Format exception into a consistent error message string."""
    return f"{type(exception).__name__}: {exception!s}"


def build_error_metadata(exception: Exception) -> dict[str, str]:
    """Build normalized error metadata from an exception."""
    stack_trace = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )

    metadata: dict[str, Any] = {
        "error_type": type(exception).__name__,
        "stack_trace": stack_trace,
        "error": True,
    }

    return normalize_metadata(metadata)


def create_failed_response(
    request: Any,
    exception: Exception,
    response_class: type,
) -> "PyExecuteComponentResponse":
    """Create a standardized failed response from an exception."""
    error_msg = format_error_message(exception)
    metadata = build_error_metadata(exception)

    return response_class(
        invocation_id=request.invocation_id,
        success=False,
        output_data=b"",
        state_update=None,
        error_message=error_msg,
        metadata=metadata,
        event_type="run.failed",
        content_index=0,
        sequence=0,
        attempt=getattr(request, "attempt", 0),
    )
