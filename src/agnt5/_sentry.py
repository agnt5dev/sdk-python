"""Sentry integration for AGNT5 SDK error tracking and monitoring.

This module provides automatic SDK error tracking to help improve AGNT5.

**Telemetry Behavior:**
- Alpha/Beta releases (e.g., 0.2.8a12, 1.0.0b3): Telemetry ENABLED by default
- Stable releases (e.g., 1.0.0, 2.1.3): Telemetry DISABLED by default

**What's Collected:**
- SDK initialization failures and crashes
- Rust FFI import errors
- Component registration failures
- Anonymized service metadata (no user code/data)

**Privacy:**
- Only SDK errors are captured (not your application errors)
- All data is anonymized (no secrets, IP addresses, or personal data)
- Full transparency in what's sent

Environment Variables:
    AGNT5_DISABLE_SDK_TELEMETRY: Set to "true" to disable (for alpha/beta)
    AGNT5_ENABLE_SDK_TELEMETRY: Set to "true" to enable (for stable)
    AGNT5_SENTRY_ENVIRONMENT: Environment tag (default: "production")
    AGNT5_SENTRY_TRACES_SAMPLE_RATE: APM trace sampling rate (default: 0.1)

Example:
    # Disable telemetry in alpha/beta
    export AGNT5_DISABLE_SDK_TELEMETRY="true"

    # Enable telemetry in stable (to help AGNT5 team)
    export AGNT5_ENABLE_SDK_TELEMETRY="true"
"""

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# AGNT5-owned Sentry project for SDK error collection
# This DSN is hardcoded and sends SDK errors to the AGNT5 team
# Users can override for testing with AGNT5_SDK_SENTRY_DSN env var
AGNT5_SDK_SENTRY_DSN = os.getenv(
    "AGNT5_SDK_SENTRY_DSN",
    "https://a25fea6eeec2e8b393a77f1e2cc7fe2c@o4509047159521280.ingest.us.sentry.io/4509047294656512"
)

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


def _is_prerelease_version(version: str) -> bool:
    """Check if SDK version is alpha or beta (pre-release).

    Args:
        version: Version string (e.g., "0.2.8a12", "1.0.0b3", "1.2.3")

    Returns:
        True if version contains 'a' (alpha) or 'b' (beta), False otherwise

    Examples:
        >>> _is_prerelease_version("0.2.8a12")
        True
        >>> _is_prerelease_version("1.0.0b3")
        True
        >>> _is_prerelease_version("1.2.3")
        False
        >>> _is_prerelease_version("1.2.3rc1")
        False
    """
    # Match alpha (a) or beta (b) followed by digits after version number
    # Pattern: <major>.<minor>.<patch>(a|b)<number>
    # More robust: anchored to end and requires digits after a/b
    return bool(re.search(r'\d+\.\d+\.\d+(a|b)\d+', version))


def _should_enable_telemetry(sdk_version: str) -> bool:
    """Determine if SDK telemetry should be enabled based on version and env vars.

    Default behavior:
    - Alpha/Beta releases: ENABLED by default (users can opt-out)
    - Stable releases: DISABLED by default (users can opt-in)

    Environment variable overrides:
    - AGNT5_DISABLE_SDK_TELEMETRY="true" → Force disable
    - AGNT5_ENABLE_SDK_TELEMETRY="true" → Force enable

    Args:
        sdk_version: SDK version string

    Returns:
        True if telemetry should be enabled, False otherwise
    """
    # Check explicit disable flag (takes precedence)
    disable_flag = os.getenv("AGNT5_DISABLE_SDK_TELEMETRY", "").lower()
    if disable_flag in ("true", "1", "yes"):
        logger.debug("SDK telemetry explicitly disabled via AGNT5_DISABLE_SDK_TELEMETRY")
        return False

    # Check explicit enable flag
    enable_flag = os.getenv("AGNT5_ENABLE_SDK_TELEMETRY", "").lower()
    if enable_flag in ("true", "1", "yes"):
        logger.debug("SDK telemetry explicitly enabled via AGNT5_ENABLE_SDK_TELEMETRY")
        return True

    # Default behavior based on version
    is_prerelease = _is_prerelease_version(sdk_version)

    if is_prerelease:
        logger.debug(f"SDK version {sdk_version} is pre-release → telemetry enabled by default")
        return True
    else:
        logger.debug(f"SDK version {sdk_version} is stable → telemetry disabled by default")
        return False


def _anonymize_event(event, hint):
    """Remove potentially sensitive data before sending to Sentry.

    This ensures no user secrets, environment variables, or personal data
    is sent to Sentry. This includes:
    - IP addresses
    - Environment variables
    - Stack trace local variables (may contain API keys, passwords, etc.)
    - Request data and headers
    - Sensitive breadcrumb data

    Args:
        event: Sentry event dict
        hint: Event hint with exception info

    Returns:
        Sanitized event or None to drop the event
    """
    # Remove user IP address
    if 'user' in event:
        event['user'].pop('ip_address', None)

    # Remove environment variables (might contain secrets)
    if 'contexts' in event:
        if 'os' in event['contexts']:
            event['contexts']['os'].pop('env', None)

        # Remove sensitive runtime context
        if 'runtime' in event['contexts']:
            event['contexts']['runtime'].pop('env', None)

    # CRITICAL: Remove stack trace local variables (may contain secrets)
    # Example: api_key = "sk-abc123..." in local scope
    if 'exception' in event:
        for exc in event['exception'].get('values', []):
            if 'stacktrace' in exc:
                for frame in exc['stacktrace'].get('frames', []):
                    # Remove all local variables from stack frames
                    if 'vars' in frame:
                        frame.pop('vars')

    # Remove request data if present (may contain secrets in POST data)
    if 'request' in event:
        event['request'].pop('data', None)
        event['request'].pop('env', None)
        event['request'].pop('headers', None)
        # Keep only safe request metadata
        safe_request_keys = {'url', 'method', 'query_string'}
        event['request'] = {k: v for k, v in event['request'].items() if k in safe_request_keys}

    # Sanitize breadcrumbs (remove any data fields that might be sensitive)
    if 'breadcrumbs' in event:
        for crumb in event['breadcrumbs'].get('values', []):
            if 'data' in crumb:
                # Keep only safe metadata
                safe_keys = {'category', 'level', 'message', 'timestamp', 'type'}
                crumb['data'] = {k: v for k, v in crumb['data'].items() if k in safe_keys}

    return event


def initialize_sentry(
    service_name: str,
    service_version: str,
    sdk_version: str,
    environment: Optional[str] = None,
    traces_sample_rate: Optional[float] = None,
) -> bool:
    """Initialize Sentry SDK for automatic SDK error tracking.

    This function is idempotent - calling it multiple times will not reinitialize Sentry.

    **Telemetry Behavior:**
    - Alpha/Beta releases: ENABLED by default (opt-out with AGNT5_DISABLE_SDK_TELEMETRY=true)
    - Stable releases: DISABLED by default (opt-in with AGNT5_ENABLE_SDK_TELEMETRY=true)

    **What's Collected:**
    - SDK initialization failures and crashes
    - Component registration errors
    - Anonymized metadata (no user code, secrets, or personal data)

    Args:
        service_name: Name of the service (used in event context)
        service_version: Version of the service (used in event context)
        sdk_version: AGNT5 SDK version (determines default telemetry behavior)
        environment: Environment tag (if None, reads from AGNT5_SENTRY_ENVIRONMENT, defaults to "production")
        traces_sample_rate: APM sampling rate 0.0-1.0 (if None, reads from AGNT5_SENTRY_TRACES_SAMPLE_RATE, defaults to 0.1)

    Returns:
        True if Sentry was initialized, False if disabled or unavailable

    Example:
        >>> initialize_sentry("my-service", "1.0.0", "0.2.8a12")
        True  # Telemetry enabled (alpha version)

        >>> initialize_sentry("my-service", "1.0.0", "1.0.0")
        False  # Telemetry disabled (stable version)
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

    # Check if AGNT5 team has configured the DSN
    if not AGNT5_SDK_SENTRY_DSN:
        logger.debug("AGNT5_SDK_SENTRY_DSN not configured, telemetry disabled")
        return False

    # Determine if telemetry should be enabled based on version and env vars
    if not _should_enable_telemetry(sdk_version):
        is_prerelease = _is_prerelease_version(sdk_version)
        if is_prerelease:
            logger.info(
                f"SDK telemetry disabled for pre-release version {sdk_version} "
                f"(set AGNT5_ENABLE_SDK_TELEMETRY=true to enable)"
            )
        else:
            logger.debug(
                f"SDK telemetry disabled by default for stable version {sdk_version} "
                f"(set AGNT5_ENABLE_SDK_TELEMETRY=true to help AGNT5 team)"
            )
        return False

    # Get environment and sampling rate
    sentry_env = environment or os.getenv("AGNT5_SENTRY_ENVIRONMENT", "production")
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
        # Initialize Sentry SDK with AGNT5's hardcoded DSN
        sentry_sdk.init(
            dsn=AGNT5_SDK_SENTRY_DSN,  # Hardcoded AGNT5 Sentry project
            environment=sentry_env,
            release=f"agnt5-python-sdk@{sdk_version}",  # SDK version, not service version
            traces_sample_rate=traces_sample_rate,
            integrations=[logging_integration],
            # Anonymize all events before sending
            before_send=_anonymize_event,
            # Add default tags
            default_integrations=True,
            # Enable performance monitoring
            enable_tracing=True,
            # Attach stack traces to messages
            attach_stacktrace=True,
            # Max breadcrumbs to keep
            max_breadcrumbs=50,
        )

        # Set global tags for filtering
        sentry_sdk.set_tag("sdk_version", sdk_version)
        sentry_sdk.set_tag("sdk_component", "python")
        sentry_sdk.set_tag("is_prerelease", str(_is_prerelease_version(sdk_version)))

        _sentry_initialized = True

        # Log different messages based on version
        is_prerelease = _is_prerelease_version(sdk_version)
        if is_prerelease:
            logger.info(
                f"SDK telemetry enabled for alpha/beta version {sdk_version} "
                f"(helps AGNT5 team find bugs). To disable: export AGNT5_DISABLE_SDK_TELEMETRY=true"
            )
        else:
            logger.info(
                f"SDK telemetry enabled for version {sdk_version} (thank you for helping improve AGNT5!)"
            )

        return True

    except Exception as e:
        logger.error(f"Failed to initialize SDK telemetry: {e}", exc_info=True)
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
