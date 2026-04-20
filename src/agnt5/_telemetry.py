"""OpenTelemetry integration for Python logging."""

import json
import logging
import warnings
from typing import Any, Dict, MutableMapping, Optional, Union

# Module-level default log level (set via set_log_level())
_default_log_level: Optional[int] = None
_telemetry_initialized = False

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

    def __init__(self, level: int = logging.NOTSET, context: Any = None):
        super().__init__(level)
        # Direct context reference — bypasses contextvars which don't propagate
        # across run_in_executor thread boundaries reliably.
        self._ctx_ref = context
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
                from .events import EventEnvelope
                import time

                # Use direct reference first (works across threads), fall back to contextvar
                ctx = self._ctx_ref
                if ctx is None:
                    from .context import get_current_context
                    ctx = get_current_context()

                if ctx is not None and hasattr(ctx, 'emit'):
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

                    if attributes:
                        log_event_data["attributes"] = attributes

                    emitter = ctx._get_emitter() if hasattr(ctx, '_get_emitter') else None
                    if emitter:
                        envelope = EventEnvelope(
                            event_type=log_event_data["event_type"],
                            data=log_event_data,
                            source_timestamp_ns=log_event_data["timestamp_ns"],
                            content_index=0,
                            metadata=log_event_data.get("metadata"),
                        )
                        emitter._queue_event(
                            envelope,
                            log_event_data["correlation_id"],
                            log_event_data["parent_correlation_id"],
                        )
            except Exception:
                pass

        except Exception:
            self.handleError(record)


def init_sdk_telemetry(service_name: str, service_version: str) -> bool:
    """Initialize Rust OTEL telemetry early for Python worker startup logs."""
    global _telemetry_initialized

    if _telemetry_initialized:
        return True

    try:
        from ._core import init_telemetry
    except ImportError as e:
        warnings.warn(f"Rust telemetry init unavailable: {e}", RuntimeWarning)
        return False

    try:
        init_telemetry(service_name, service_version)
        _telemetry_initialized = True
        return True
    except Exception as e:
        warnings.warn(f"Failed to initialize SDK telemetry: {e}", RuntimeWarning)
        return False


def ensure_root_otel_handler() -> bool:
    """Attach a non-invasive OTEL handler to the root logger once.

    This preserves user console/file handlers configured via logging.basicConfig()
    while also forwarding records to the Rust OTEL bridge.
    """
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, OpenTelemetryHandler):
            return True

    otel_handler = OpenTelemetryHandler()
    otel_handler.setLevel(logging.DEBUG)
    otel_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(otel_handler)

    # Leave the logger's level untouched so we don't widen the user's local output.
    return True


def _is_debug_enabled() -> bool:
    """Check if AGNT5_DEBUG is enabled."""
    import os
    debug_val = os.environ.get("AGNT5_DEBUG", "").lower()
    return debug_val in ("1", "true", "yes")


def set_log_level(level: Union[int, str]) -> None:
    """Set the default log level for context loggers.

    Call this in your app.py to control what log levels appear in console output.

    Args:
        level: Log level as int (e.g., logging.DEBUG) or string (e.g., "DEBUG", "INFO")

    Example:
        import agnt5
        import logging

        agnt5.set_log_level(logging.DEBUG)  # Show all logs
        agnt5.set_log_level("INFO")         # Show INFO and above
    """
    global _default_log_level

    if isinstance(level, str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level_upper = level.upper()
        if level_upper not in level_map:
            raise ValueError(f"Invalid log level: {level}. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        _default_log_level = level_map[level_upper]
    else:
        _default_log_level = level


def setup_context_logger(logger: logging.Logger, log_level: Optional[int] = None, context: Any = None) -> None:
    """Configure a Context logger with OpenTelemetry and console handlers.

    Args:
        context: Optional Context instance stored directly on the OTel handler,
                 so log events can be emitted for SSE streaming even from
                 run_in_executor threads where contextvars don't propagate.
    """
    logger.handlers.clear()

    # Priority: explicit log_level > user-configured default > AGNT5_DEBUG > INFO
    if log_level is not None:
        effective_level = log_level
    elif _default_log_level is not None:
        effective_level = _default_log_level
    elif _is_debug_enabled():
        effective_level = logging.DEBUG
    else:
        effective_level = logging.INFO

    # OpenTelemetry handler (forwards to Rust + SSE streaming)
    otel_handler = OpenTelemetryHandler(context=context)
    otel_handler.setLevel(logging.DEBUG)  # Always forward to OTel
    otel_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(otel_handler)

    # Console handler - respects AGNT5_DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(effective_level)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)  # Let handlers filter
    logger.propagate = False


def setup_module_logger(module_name: str, log_level: Optional[int] = None) -> logging.Logger:
    """Create and configure a logger for a module."""
    logger = logging.getLogger(module_name)
    # Default to INFO for module loggers, but respect AGNT5_DEBUG
    debug_enabled = _is_debug_enabled()
    default_level = logging.DEBUG if debug_enabled else logging.INFO
    setup_context_logger(logger, log_level or default_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with AGNT5-consistent formatting.

    Use this to get a logger that follows AGNT5 conventions:
    - Respects AGNT5_DEBUG environment variable
    - Uses consistent [LEVEL] format
    - Integrates with OpenTelemetry when running in a worker

    Args:
        name: Logger name (typically __name__ or your service name)

    Returns:
        Configured logger instance

    Example:
        from agnt5 import get_logger

        logger = get_logger(__name__)
        logger.info("Starting worker...")
    """
    return setup_module_logger(name)
