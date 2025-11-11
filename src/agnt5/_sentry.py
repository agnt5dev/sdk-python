"""Sentry integration for AGNT5 SDK error tracking and monitoring.

This module provides opt-in Sentry integration for capturing errors and exceptions
in the AGNT5 SDK. Sentry is only initialized if AGNT5_SENTRY_DSN is set.

Environment Variables:
    AGNT5_SENTRY_DSN: Sentry project DSN (required to enable Sentry)
    AGNT5_SENTRY_ENVIRONMENT: Environment tag (default: "development")
    AGNT5_SENTRY_TRACES_SAMPLE_RATE: APM trace sampling rate (default: 0.1)
    AGNT5_SENTRY_ENABLED: Explicitly enable/disable Sentry (default: auto from DSN)

Example:
    export AGNT5_SENTRY_DSN="https://abc123@o123.ingest.sentry.io/456"
    export AGNT5_SENTRY_ENVIRONMENT="production"
    export AGNT5_SENTRY_TRACES_SAMPLE_RATE="0.2"
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_sentry_initialized = False
_sentry_available = False

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    _sentry_available = True
except ImportError:
    logger.debug("sentry-sdk not installed, Sentry integration disabled")
    _sentry_available = False


def is_sentry_enabled() -> bool:
    """Check if Sentry integration is enabled and initialized.

    Returns:
        True if Sentry is available and initialized, False otherwise
    """
    return _sentry_initialized and _sentry_available


def initialize_sentry(
    service_name: str,
    service_version: str,
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: Optional[float] = None,
) -> bool:
    """Initialize Sentry SDK for error tracking and performance monitoring.

    This function is idempotent - calling it multiple times will not reinitialize Sentry.

    Args:
        service_name: Name of the service (used as transaction name prefix)
        service_version: Version of the service (used as release tag)
        dsn: Sentry DSN (if None, reads from AGNT5_SENTRY_DSN env var)
        environment: Environment tag (if None, reads from AGNT5_SENTRY_ENVIRONMENT, defaults to "development")
        traces_sample_rate: APM sampling rate 0.0-1.0 (if None, reads from AGNT5_SENTRY_TRACES_SAMPLE_RATE, defaults to 0.1)

    Returns:
        True if Sentry was initialized, False if disabled or unavailable

    Example:
        >>> initialize_sentry("my-service", "1.0.0")
        True
    """
    global _sentry_initialized

    # Check if already initialized
    if _sentry_initialized:
        logger.debug("Sentry already initialized, skipping")
        return True

    # Check if Sentry SDK is available
    if not _sentry_available:
        logger.debug("Sentry SDK not available, skipping initialization")
        return False

    # Check explicit disable flag
    enabled_flag = os.getenv("AGNT5_SENTRY_ENABLED", "").lower()
    if enabled_flag in ("false", "0", "no"):
        logger.info("Sentry explicitly disabled via AGNT5_SENTRY_ENABLED")
        return False

    # Get DSN from parameter or environment
    sentry_dsn = dsn or os.getenv("AGNT5_SENTRY_DSN")
    if not sentry_dsn:
        logger.debug("AGNT5_SENTRY_DSN not set, Sentry integration disabled")
        return False

    # Get environment and sampling rate
    sentry_env = environment or os.getenv("AGNT5_SENTRY_ENVIRONMENT", "development")
    sample_rate_str = os.getenv("AGNT5_SENTRY_TRACES_SAMPLE_RATE", "0.1")
    if traces_sample_rate is None:
        try:
            traces_sample_rate = float(sample_rate_str)
        except ValueError:
            logger.warning(
                f"Invalid AGNT5_SENTRY_TRACES_SAMPLE_RATE: {sample_rate_str}, using default 0.1"
            )
            traces_sample_rate = 0.1

    # Configure logging integration
    # Capture ERROR and above automatically
    logging_integration = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors and above as events
    )

    try:
        # Initialize Sentry SDK
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=sentry_env,
            release=f"{service_name}@{service_version}",
            traces_sample_rate=traces_sample_rate,
            integrations=[logging_integration],
            # Add default tags
            default_integrations=True,
            # Enable performance monitoring
            enable_tracing=True,
            # Attach stack traces to messages
            attach_stacktrace=True,
            # Max breadcrumbs to keep
            max_breadcrumbs=100,
        )

        # Set global tags
        sentry_sdk.set_tag("service", service_name)
        sentry_sdk.set_tag("version", service_version)

        _sentry_initialized = True
        logger.info(
            f"Sentry initialized for {service_name}@{service_version} "
            f"(env: {sentry_env}, sample_rate: {traces_sample_rate})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)
        return False


def capture_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    level: str = "error",
) -> Optional[str]:
    """Capture an exception and send it to Sentry.

    Args:
        exception: The exception to capture
        context: Additional context data to attach
        tags: Tags to add to this event
        level: Severity level (error, warning, info)

    Returns:
        Event ID if captured, None if Sentry not initialized

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     capture_exception(e, context={"run_id": "123"}, tags={"component": "workflow"})
    """
    if not is_sentry_enabled():
        return None

    with sentry_sdk.push_scope() as scope:
        # Add tags
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        # Add context
        if context:
            scope.set_context("additional_context", context)

        # Set level
        scope.level = level

        # Capture exception
        event_id = sentry_sdk.capture_exception(exception)
        return event_id


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Capture a message and send it to Sentry.

    Args:
        message: The message to capture
        level: Severity level (error, warning, info, debug)
        context: Additional context data to attach
        tags: Tags to add to this event

    Returns:
        Event ID if captured, None if Sentry not initialized

    Example:
        >>> capture_message("Unusual behavior detected", level="warning", tags={"component": "agent"})
    """
    if not is_sentry_enabled():
        return None

    with sentry_sdk.push_scope() as scope:
        # Add tags
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        # Add context
        if context:
            scope.set_context("additional_context", context)

        # Set level
        scope.level = level

        # Capture message
        event_id = sentry_sdk.capture_message(message, level=level)
        return event_id


def add_breadcrumb(
    message: str,
    category: str = "default",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Add a breadcrumb to the current scope.

    Breadcrumbs are a trail of events that led up to an error.

    Args:
        message: Breadcrumb message
        category: Breadcrumb category (e.g., "execution", "state", "api")
        level: Severity level
        data: Additional data

    Example:
        >>> add_breadcrumb("Starting workflow execution", category="workflow", data={"workflow_id": "123"})
    """
    if not is_sentry_enabled():
        return

    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {},
    )


def set_user(user_id: Optional[str] = None, **kwargs: Any) -> None:
    """Set user information for the current scope.

    Args:
        user_id: User ID
        **kwargs: Additional user attributes (email, username, etc.)

    Example:
        >>> set_user(user_id="user123", email="user@example.com")
    """
    if not is_sentry_enabled():
        return

    user_data = {}
    if user_id:
        user_data["id"] = user_id
    user_data.update(kwargs)

    sentry_sdk.set_user(user_data)


def set_context(name: str, context: Dict[str, Any]) -> None:
    """Set context information for the current scope.

    Args:
        name: Context name (e.g., "runtime", "execution")
        context: Context data

    Example:
        >>> set_context("runtime", {"run_id": "123", "tenant_id": "tenant456"})
    """
    if not is_sentry_enabled():
        return

    sentry_sdk.set_context(name, context)


def set_tag(key: str, value: str) -> None:
    """Set a tag for the current scope.

    Tags are searchable key-value pairs.

    Args:
        key: Tag key
        value: Tag value

    Example:
        >>> set_tag("component_type", "workflow")
    """
    if not is_sentry_enabled():
        return

    sentry_sdk.set_tag(key, value)


def flush(timeout: float = 2.0) -> None:
    """Flush pending Sentry events.

    This should be called before shutdown to ensure all events are sent.

    Args:
        timeout: Maximum time to wait in seconds

    Example:
        >>> flush(timeout=5.0)
    """
    if not is_sentry_enabled():
        return

    sentry_sdk.flush(timeout=timeout)
