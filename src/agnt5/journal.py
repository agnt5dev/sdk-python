"""Journal client for LLM observability events.

This module provides functions to write observability events to the AGNT5 journal.
These events are used for:
- LLM call tracking (lm.call.started, lm.call.completed, lm.call.failed)
- Cost and usage metrics
- Debugging and tracing
- Real-time SSE streaming

Note: This is for OBSERVABILITY, not memoization. Use CheckpointClient for memoization.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Try to import the Rust core function
try:
    from ._core import write_journal_event as _write_journal_event
    _JOURNAL_AVAILABLE = True
except ImportError:
    _JOURNAL_AVAILABLE = False
    logger.debug("Journal client not available (Rust core not loaded)")


@dataclass
class LMCallStartedEvent:
    """Event data for lm.call.started."""
    model: str
    provider: str
    prompt_hash: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools_count: int = 0
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_hash": self.prompt_hash,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools_count": self.tools_count,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class LMCallCompletedEvent:
    """Event data for lm.call.completed."""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    cost_estimate_usd: Optional[float] = None
    finish_reason: Optional[str] = None
    tool_calls_count: int = 0
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_estimate_usd": self.cost_estimate_usd,
            "finish_reason": self.finish_reason,
            "tool_calls_count": self.tool_calls_count,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class LMCallFailedEvent:
    """Event data for lm.call.failed."""
    model: str
    provider: str
    error_code: str
    error_message: str
    latency_ms: int
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "timestamp_ns": self.timestamp_ns,
        }


async def write_lm_call_started(
    run_id: str,
    trace_id: str,
    span_id: str,
    event: LMCallStartedEvent,
    tenant_id: Optional[str] = None,
) -> None:
    """Write an lm.call.started event to the journal.

    Args:
        run_id: The run ID to associate the event with
        trace_id: Trace ID for correlation
        span_id: Span ID for correlation
        event: The event data
        tenant_id: Optional tenant ID
    """
    if not _JOURNAL_AVAILABLE:
        logger.debug("Journal not available, skipping lm.call.started event")
        return

    try:
        data = json.dumps(event.to_dict()).encode("utf-8")
        await _write_journal_event(
            run_id=run_id,
            event_type="lm.call.started",
            data=data,
            trace_id=trace_id,
            span_id=span_id,
            tenant_id=tenant_id,
            source_timestamp_ns=event.timestamp_ns,
        )
        logger.debug(f"Wrote lm.call.started event for run_id={run_id}")
    except Exception as e:
        logger.warning(f"Failed to write lm.call.started event: {e}")


async def write_lm_call_completed(
    run_id: str,
    trace_id: str,
    span_id: str,
    event: LMCallCompletedEvent,
    tenant_id: Optional[str] = None,
) -> None:
    """Write an lm.call.completed event to the journal.

    Args:
        run_id: The run ID to associate the event with
        trace_id: Trace ID for correlation
        span_id: Span ID for correlation
        event: The event data
        tenant_id: Optional tenant ID
    """
    if not _JOURNAL_AVAILABLE:
        logger.debug("Journal not available, skipping lm.call.completed event")
        return

    try:
        data = json.dumps(event.to_dict()).encode("utf-8")
        await _write_journal_event(
            run_id=run_id,
            event_type="lm.call.completed",
            data=data,
            trace_id=trace_id,
            span_id=span_id,
            tenant_id=tenant_id,
            source_timestamp_ns=event.timestamp_ns,
        )
        logger.debug(f"Wrote lm.call.completed event for run_id={run_id}, tokens={event.total_tokens}")
    except Exception as e:
        logger.warning(f"Failed to write lm.call.completed event: {e}")


async def write_lm_call_failed(
    run_id: str,
    trace_id: str,
    span_id: str,
    event: LMCallFailedEvent,
    tenant_id: Optional[str] = None,
) -> None:
    """Write an lm.call.failed event to the journal.

    Args:
        run_id: The run ID to associate the event with
        trace_id: Trace ID for correlation
        span_id: Span ID for correlation
        event: The event data
        tenant_id: Optional tenant ID
    """
    if not _JOURNAL_AVAILABLE:
        logger.debug("Journal not available, skipping lm.call.failed event")
        return

    try:
        data = json.dumps(event.to_dict()).encode("utf-8")
        await _write_journal_event(
            run_id=run_id,
            event_type="lm.call.failed",
            data=data,
            trace_id=trace_id,
            span_id=span_id,
            tenant_id=tenant_id,
            source_timestamp_ns=event.timestamp_ns,
        )
        logger.debug(f"Wrote lm.call.failed event for run_id={run_id}, error={event.error_code}")
    except Exception as e:
        logger.warning(f"Failed to write lm.call.failed event: {e}")
