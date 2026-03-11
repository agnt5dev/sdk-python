"""Worker module for AGNT5 SDK."""

import contextvars

from ._core import Worker

# Trace metadata contextvar — set by Rust worker handler (rust-src/worker.rs)
# with the current span's traceparent/tracestate, read by emit_event_sync
# to propagate trace context through WriteCheckpoint calls back to EE.
_trace_metadata: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "_trace_metadata", default={}
)

__all__ = ["Worker"]
