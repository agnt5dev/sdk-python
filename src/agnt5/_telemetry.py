"""OpenTelemetry integration for Python logging."""

import json
import logging
from typing import Any, Dict, MutableMapping, Optional

# Standard logging kwargs that should NOT be treated as custom attributes
_STANDARD_LOGGING_KWARGS = frozenset({
    'exc_info', 'stack_info', 'stacklevel', 'extra'
})


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that allows keyword arguments as log attributes.

    Usage: ctx.logger.info("message", attr1="value1", attr2="value2")
    """

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Extract custom attributes from kwargs into extra['agnt5_attrs']."""
        custom_attrs = {}
        standard_kwargs = {}

        for key, value in kwargs.items():
            if key in _STANDARD_LOGGING_KWARGS:
                standard_kwargs[key] = value
            else:
                # Convert non-string values to their string representation
                if isinstance(value, (dict, list)):
                    custom_attrs[key] = json.dumps(value)
                elif not isinstance(value, str):
                    custom_attrs[key] = str(value)
                else:
                    custom_attrs[key] = value

        # Merge with any existing extra dict
        extra = standard_kwargs.get('extra', {})
        if isinstance(extra, dict):
            extra = dict(extra)  # Make a copy
        else:
            extra = {}

        # Add custom attributes and any default extra from adapter
        if custom_attrs:
            extra['agnt5_attrs'] = custom_attrs
        if self.extra:
            extra.update(self.extra)

        standard_kwargs['extra'] = extra
        return msg, standard_kwargs


class OpenTelemetryHandler(logging.Handler):
    """Forwards Python logs to Rust OpenTelemetry system and emits log events for SSE streaming."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        try:
            from ._core import log_from_python
            self._log_from_python = log_from_python
        except ImportError as e:
            import warnings
            warnings.warn(f"Rust telemetry bridge unavailable: {e}", RuntimeWarning)
            self._log_from_python = None

    def emit(self, record: logging.LogRecord) -> None:
        """Forward log record to Rust and emit log event for SSE streaming."""
        if self._log_from_python is None:
            return

        # Filter gRPC internal logs
        if record.name.startswith(('grpc.', 'h2.', '_grpc_', 'h2-')):
            return

        try:
            message = self.format(record)

            # Include exception traceback if present
            if record.exc_info:
                if self.formatter:
                    exc_text = self.formatter.formatException(record.exc_info)
                else:
                    import traceback
                    exc_text = ''.join(traceback.format_exception(*record.exc_info))
                message = f"{message}\n{exc_text}"

            # Extract correlation IDs (added by _CorrelationFilter)
            trace_id = getattr(record, 'trace_id', None)
            span_id = getattr(record, 'span_id', None)
            run_id = getattr(record, 'run_id', None)
            attributes = getattr(record, 'agnt5_attrs', None)

            # Forward to OTLP for observability storage
            self._log_from_python(
                level=record.levelname,
                message=message,
                target=record.name,
                module_path=record.module,
                filename=record.pathname,
                line=record.lineno,
                trace_id=trace_id,
                span_id=span_id,
                run_id=run_id,
                attributes=attributes,
            )

            # Also emit as event for SSE streaming (if we have an active context)
            try:
                from .context import get_current_context
                from .events import Event
                import time
                import sys

                ctx = get_current_context()
                print(f"[TELEMETRY DEBUG] get_current_context() = {ctx}", file=sys.stderr, flush=True)

                if ctx is not None and hasattr(ctx, 'emit'):
                    print(f"[TELEMETRY DEBUG] Context has emit, creating log event for: {record.levelname} - {message[:50]}", file=sys.stderr, flush=True)

                    # Create log event with proper correlation IDs from context
                    log_event_data = {
                        "event_type": f"log.{record.levelname.lower()}",
                        "name": record.name,
                        "correlation_id": ctx._correlation_id if hasattr(ctx, '_correlation_id') else "",
                        "parent_correlation_id": ctx._parent_correlation_id if hasattr(ctx, '_parent_correlation_id') else "",
                        "timestamp_ns": time.time_ns(),
                        "level": record.levelname,
                        "message": message,
                        "target": record.name,
                        "module_path": record.module,
                        "filename": record.pathname,
                        "line": record.lineno,
                        "metadata": {},
                    }

                    # Add custom attributes if present
                    if attributes:
                        log_event_data["attributes"] = attributes

                    # Queue the log event for SSE streaming
                    # Use raw event emission since we're constructing the dict directly
                    if hasattr(ctx, '_event_emitter') and ctx._event_emitter:
                        print(f"[TELEMETRY DEBUG] Queueing log event: event_type={log_event_data['event_type']}, correlation_id={log_event_data['correlation_id']}", file=sys.stderr, flush=True)
                        from .events import EventEnvelope
                        envelope = EventEnvelope(
                            event_type=log_event_data["event_type"],
                            data=log_event_data,
                            source_timestamp_ns=log_event_data["timestamp_ns"],
                            content_index=0,
                            metadata=log_event_data.get("metadata"),
                        )
                        ctx._event_emitter._queue_event(
                            envelope,
                            log_event_data["correlation_id"],
                            log_event_data["parent_correlation_id"],
                        )
                        print(f"[TELEMETRY DEBUG] Log event queued successfully", file=sys.stderr, flush=True)
                    else:
                        print(f"[TELEMETRY DEBUG] No event emitter found on context", file=sys.stderr, flush=True)
                else:
                    print(f"[TELEMETRY DEBUG] No active context or context has no emit method", file=sys.stderr, flush=True)
            except Exception as e:
                # Silently fail log event emission - don't break logging
                print(f"[TELEMETRY DEBUG] Exception during log event emission: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)

        except Exception:
            self.handleError(record)


def setup_context_logger(logger: logging.Logger, log_level: Optional[int] = None) -> None:
    """Configure a Context logger with OpenTelemetry and console handlers."""
    logger.handlers.clear()

    # OpenTelemetry handler (forwards to Rust)
    otel_handler = OpenTelemetryHandler()
    otel_handler.setLevel(logging.DEBUG)
    otel_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(otel_handler)

    # Console handler (fallback for local testing)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    logger.setLevel(log_level or logging.DEBUG)
    logger.propagate = False


def setup_module_logger(module_name: str, log_level: Optional[int] = None) -> logging.Logger:
    """Create and configure a logger for a module."""
    logger = logging.getLogger(module_name)
    setup_context_logger(logger, log_level or logging.INFO)
    return logger
