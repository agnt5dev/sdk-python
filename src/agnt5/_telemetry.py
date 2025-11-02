"""
OpenTelemetry integration for Python logging.

This module bridges Python's standard logging to Rust's tracing/OpenTelemetry system,
ensuring all logs from ctx.logger are sent to both the console and OTLP exporters.
"""

import logging
from typing import Optional


class OpenTelemetryHandler(logging.Handler):
    """
    Custom logging handler that forwards Python logs to Rust OpenTelemetry system.

    This handler routes all Python log records through the Rust `log_from_python()`
    function, which integrates with the tracing ecosystem. This ensures:

    1. Logs are sent to OpenTelemetry OTLP exporter
    2. Logs appear in console output (via Rust's fmt layer)
    3. Logs inherit span context (invocation.id, trace_id, etc.)
    4. Structured logging with proper attributes

    The Rust side handles both console output and OTLP export, so we only
    need one handler on the Python side.
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize the OpenTelemetry handler.

        Args:
            level: Minimum log level to process (default: NOTSET processes all)
        """
        super().__init__(level)

        # Import Rust bridge function
        try:
            from ._core import log_from_python
            self._log_from_python = log_from_python
        except ImportError as e:
            # Fallback if Rust core not available (development/testing)
            import warnings
            warnings.warn(
                f"Failed to import Rust telemetry bridge: {e}. "
                "Logs will not be sent to OpenTelemetry.",
                RuntimeWarning
            )
            self._log_from_python = None

    def emit(self, record: logging.LogRecord):
        """
        Process a log record and forward to Rust telemetry.

        Args:
            record: Python logging record to process
        """
        if self._log_from_python is None:
            # No Rust bridge available, silently skip
            return

        # Filter out gRPC internal logs to avoid noise
        # These are low-level HTTP/2 protocol logs that aren't useful for application debugging
        if record.name.startswith(('grpc.', 'h2.', '_grpc_', 'h2-')):
            return

        try:
            # Format the message (applies any formatters)
            message = self.format(record)

            # Include exception traceback if present (from logger.exception() or exc_info=True)
            if record.exc_info:
                # Use formatter to format the exception, or fall back to basic formatting
                if self.formatter:
                    exc_text = self.formatter.formatException(record.exc_info)
                else:
                    # Fallback: use basic traceback formatting
                    import traceback
                    exc_text = ''.join(traceback.format_exception(*record.exc_info))
                message = f"{message}\n{exc_text}"

            # Extract correlation IDs from LogRecord attributes (added by _CorrelationFilter)
            # These ensure logs can be correlated with distributed traces in observability backends
            trace_id = getattr(record, 'trace_id', None)
            span_id = getattr(record, 'span_id', None)
            run_id = getattr(record, 'run_id', None)

            # Forward to Rust tracing system
            # Rust side will:
            # - Add to current span context (inherits invocation.id)
            # - Attach correlation IDs as span attributes for OTLP export
            # - Send to OTLP exporter with trace context
            # - Print to console via fmt layer
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
            )
        except Exception:
            # Don't let logging errors crash the application
            # Use handleError to report the issue via logging system
            self.handleError(record)


def setup_context_logger(logger: logging.Logger, log_level: Optional[int] = None) -> None:
    """
    Configure a Context logger with OpenTelemetry integration.

    This function:
    1. Removes any existing handlers (avoid duplicates)
    2. Adds OpenTelemetry handler for OTLP + console output (when Worker is running)
    3. Adds console handler for local testing (fallback)
    4. Sets appropriate log level
    5. Disables propagation to avoid duplicate logs

    Args:
        logger: Logger instance to configure
        log_level: Optional log level (default: DEBUG)
    """
    # Remove existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Add OpenTelemetry handler (for Worker/platform execution)
    otel_handler = OpenTelemetryHandler()
    otel_handler.setLevel(logging.DEBUG)

    # Use simple formatter - Rust side handles structured logging
    formatter = logging.Formatter('%(message)s')
    otel_handler.setFormatter(formatter)

    logger.addHandler(otel_handler)

    # Add console handler for local testing (fallback when Rust bridge not available)
    # This ensures logs appear when testing functions locally without Worker
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Console format includes level, message, and exception info if present
    # exc_info=True in the format string means "include traceback if present"
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s',
        # Python automatically appends exception traceback when exc_info is set
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # Set log level (default to DEBUG to let handlers filter)
    if log_level is None:
        log_level = logging.DEBUG
    logger.setLevel(log_level)

    # Don't propagate to root logger (we handle everything ourselves)
    logger.propagate = False


def setup_module_logger(module_name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Create and configure a logger for a module with OpenTelemetry integration.

    Convenience function for setting up loggers in SDK modules.

    Args:
        module_name: Name of the module (e.g., "agnt5.worker")
        log_level: Optional log level (default: INFO for modules)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)

    # For module loggers, default to INFO level
    if log_level is None:
        log_level = logging.INFO

    setup_context_logger(logger, log_level)
    return logger
