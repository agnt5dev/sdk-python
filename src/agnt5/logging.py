"""
OpenTelemetry logging integration for AGNT5 Python SDK.

This module provides a logging handler that forwards Python logs to the Rust core
for integration with OpenTelemetry. All logs are automatically correlated with 
traces and sent to the OTLP collector.
"""

import logging
import os
from typing import Optional

from ._compat import _rust_available

if _rust_available:
    from ._core import log_from_python


class OpenTelemetryHandler(logging.Handler):
    """
    A logging handler that forwards Python logs to Rust for OpenTelemetry integration.
    
    This handler automatically captures all Python logs and forwards them to the
    Rust core where they are integrated with OpenTelemetry tracing and sent to
    the OTLP collector. Logs are automatically correlated with active traces.
    """
    
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        
        if not _rust_available:
            raise RuntimeError("OpenTelemetry logging handler requires Rust core")
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Forward a log record to Rust for OpenTelemetry integration.
        
        Args:
            record: The Python log record to forward
        """
        try:
            # Format the message
            message = self.format(record)
            
            # Extract metadata for Rust
            level = record.levelname
            target = record.name  # Logger name (e.g., 'agnt5.worker')
            module_path = getattr(record, 'module', record.name)
            filename = getattr(record, 'pathname', None)
            line = getattr(record, 'lineno', None)
            
            # Make filename relative if it's absolute
            if filename and os.path.isabs(filename):
                try:
                    # Try to make it relative to current working directory
                    filename = os.path.relpath(filename)
                except ValueError:
                    # If relpath fails (e.g., different drives on Windows), use basename
                    filename = os.path.basename(filename)
            
            # Forward to Rust core - silently ignore if telemetry not ready yet
            try:
                log_from_python(
                    level=level,
                    message=message,
                    target=target,
                    module_path=module_path,
                    filename=filename,
                    line=line
                )
            except Exception:
                # Silently ignore if Rust telemetry system not ready yet
                # This handles the timing issue during startup
                pass
            
        except Exception as e:
            # Don't let logging errors crash the application
            # Use handleError to maintain Python logging standards
            self.handleError(record)


def install_opentelemetry_logging(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> OpenTelemetryHandler:
    """
    Install OpenTelemetry logging handler on a logger.
    
    Args:
        logger: Logger to install handler on. If None, uses root logger.
        level: Minimum log level to forward to OpenTelemetry
        format_string: Optional format string for log messages
        
    Returns:
        The installed OpenTelemetryHandler instance
        
    Example:
        # Install on root logger (captures all logs)
        install_opentelemetry_logging()
        
        # Install on specific logger
        logger = logging.getLogger('my_app')
        install_opentelemetry_logging(logger, level=logging.DEBUG)
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Create handler
    handler = OpenTelemetryHandler(level=level)
    
    # Set formatter if provided
    if format_string:
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
    
    # Install handler
    logger.addHandler(handler)
    
    return handler


def remove_opentelemetry_logging(logger: Optional[logging.Logger] = None) -> None:
    """
    Remove OpenTelemetry logging handlers from a logger.
    
    Args:
        logger: Logger to remove handlers from. If None, uses root logger.
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Remove all OpenTelemetryHandler instances
    handlers_to_remove = [
        handler for handler in logger.handlers 
        if isinstance(handler, OpenTelemetryHandler)
    ]
    
    for handler in handlers_to_remove:
        logger.removeHandler(handler)