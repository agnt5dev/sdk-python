"""Shared helpers for third-party framework capture integrations.

Capture integrations observe calls made by third-party libraries (openai,
google-adk, ...) inside AGNT5 worker components and translate them into
canonical journal events. They must never break the user's call: every emit
is best-effort, and the absence of an ambient Context means passthrough.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

from .._ids import generate_cid
from ..context import Context, get_current_context
from ..events import Event
from ..lm.events import LMCompleted, LMFailed, LMStarted

logger = logging.getLogger(__name__)

# Interim capture labels until the envelope-level `source`/`capture_mode`
# fields (tpp-sdk-support task 0.3) land. All-string metadata is the only
# payload the runtime's parse_llm_event reads.
SOURCE_METADATA_KEY = "source"
CAPTURE_MODE_METADATA_KEY = "capture_mode"
OBSERVED_CAPTURE_MODE = "observed"

_OFF_VALUES = ("off", "0", "false", "no")
_INVALID_CONTENT_MODES_LOGGED: set[str] = set()


def master_capture_enabled() -> bool:
    """AGNT5_CAPTURE=off is the master kill switch (default on)."""
    return os.getenv("AGNT5_CAPTURE", "").lower() not in _OFF_VALUES


def library_capture_enabled(flag: str) -> bool:
    """Per-library gate, e.g. AGNT5_CAPTURE_OPENAI=0 (default on)."""
    return os.getenv(flag, "").lower() not in _OFF_VALUES


def supported_package_version(
    distribution: str,
    *,
    minimum: tuple[int, int, int],
    max_major_exclusive: int,
) -> bool:
    """Whether an installed instrumentation target is in its tested API band."""
    try:
        installed = version(distribution)
    except PackageNotFoundError:
        return False
    except Exception:
        logger.debug("failed to inspect %s version; capture disabled", distribution, exc_info=True)
        return False
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", installed)
    if match is None:
        logger.debug("cannot parse %s version %r; capture disabled", distribution, installed)
        return False
    parsed = tuple(int(part or 0) for part in match.groups())
    supported = parsed >= minimum and parsed[0] < max_major_exclusive
    if not supported:
        logger.debug(
            "%s %s is outside the supported capture band >=%s.%s.%s,<%s",
            distribution,
            installed,
            *minimum,
            max_major_exclusive,
        )
    return supported


def content_capture_enabled() -> bool:
    """Whether prompts/responses are captured under the resolved policy."""
    return content_capture_mode() != "metadata-only"


def content_capture_mode() -> str:
    """Resolve ``full`` / ``redacted`` / ``metadata-only`` capture mode.

    The explicit mode is preferred.  The legacy boolean remains supported so
    existing deployments using ``AGNT5_LLM_CAPTURE_CONTENT=off`` tighten to
    metadata-only rather than unexpectedly capturing content.
    """
    configured = os.getenv("AGNT5_CAPTURE_CONTENT_MODE", "").strip().lower()
    if configured:
        if configured in {"full", "redacted", "metadata-only"}:
            return configured
        if configured not in _INVALID_CONTENT_MODES_LOGGED:
            _INVALID_CONTENT_MODES_LOGGED.add(configured)
            logger.warning(
                "invalid AGNT5_CAPTURE_CONTENT_MODE=%r; defaulting to metadata-only",
                configured,
            )
        return "metadata-only"
    if os.getenv("AGNT5_LLM_CAPTURE_CONTENT", "").lower() in _OFF_VALUES:
        return "metadata-only"
    return "full"


def capture_text_limit() -> int:
    """Maximum retained characters per captured string (default 32 KiB)."""
    try:
        return max(256, int(os.getenv("AGNT5_CAPTURE_MAX_CONTENT_CHARS", "32768")))
    except ValueError:
        return 32768


def capture_payload(value: Any) -> Any:
    """Return a bounded, JSON-safe payload under the active capture policy."""
    mode = content_capture_mode()
    if mode == "metadata-only":
        return None
    try:
        return _sanitize_payload(value, redacted=mode == "redacted", depth=0)
    except Exception:
        logger.debug("capture payload sanitization failed", exc_info=True)
        return f"<{type(value).__name__}>"


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credentials",
        "password",
        "secret",
        "set_cookie",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)
_BEARER_RE = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_GOOGLE_KEY_RE = re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _bounded_string(value: str, *, redacted: bool) -> str:
    if redacted:
        value = _BEARER_RE.sub("Bearer [REDACTED]", value)
        value = _OPENAI_KEY_RE.sub("[REDACTED]", value)
        value = _JWT_RE.sub("[REDACTED]", value)
        value = _GOOGLE_KEY_RE.sub("[REDACTED]", value)
        value = _GITHUB_TOKEN_RE.sub("[REDACTED]", value)
    limit = capture_text_limit()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated]"


def _sanitize_payload(value: Any, *, redacted: bool, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_string(value, redacted=redacted)
    if isinstance(value, bytes):
        return _bounded_string(value.decode("utf-8", errors="replace"), redacted=redacted)
    if depth >= 8:
        return "[max-depth]"
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(exclude_none=True)
        except Exception:
            return f"<{type(value).__name__}>"
    elif is_dataclass(value) and not isinstance(value, type):
        try:
            value = asdict(value)
        except Exception:
            return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        mapping_result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                mapping_result["__truncated__"] = "additional entries omitted"
                break
            key_text = str(key)
            if redacted and _is_sensitive_key(key_text):
                mapping_result[key_text] = "[REDACTED]"
            else:
                mapping_result[key_text] = _sanitize_payload(
                    item, redacted=redacted, depth=depth + 1
                )
        return mapping_result
    if isinstance(value, (list, tuple, set, frozenset)):
        if isinstance(value, (list, tuple)):
            items = list(value[:129])
        else:
            items = list(itertools.islice(iter(value), 129))
        list_result = [
            _sanitize_payload(item, redacted=redacted, depth=depth + 1) for item in items[:128]
        ]
        if len(items) == 129:
            list_result.append("[additional items omitted]")
        return list_result
    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return _bounded_string(text, redacted=redacted)


def observed_metadata(source: str, name: str, **values: Any) -> dict[str, str]:
    """Metadata shared by all captured events until envelope fields land."""
    result = {
        "name": str(name),
        SOURCE_METADATA_KEY: source,
        CAPTURE_MODE_METADATA_KEY: OBSERVED_CAPTURE_MODE,
    }
    result.update({key: str(value) for key, value in values.items()})
    return result


def capture_error_message(error: Any) -> str:
    """Apply the content policy and bounds to an error description."""
    mode = content_capture_mode()
    if mode == "metadata-only":
        return "[error details omitted by capture policy]"
    try:
        message = str(error)
    except Exception:
        message = f"<{type(error).__name__}>"
    return _bounded_string(message, redacted=mode == "redacted")


def ambient_context() -> Optional[Context]:
    """The Context of the enclosing component execution, or None."""
    try:
        return get_current_context()
    except Exception:
        return None


def safe_emit(ctx: Context, event: Event) -> None:
    """Emit without ever propagating a failure into the user's call."""
    try:
        ctx.emit_observed(event)
    except Exception:
        logger.debug("capture emit failed for %s", event.event_type, exc_info=True)


async def safe_emit_async(ctx: Context, event: Event) -> None:
    """Async twin of safe_emit."""
    try:
        await ctx.emit_observed_async(event)
    except Exception:
        logger.debug("capture emit failed for %s", event.event_type, exc_info=True)


def new_capture_span(ctx: Context) -> tuple[str, str]:
    """Allocate a correlation id and snapshot the parent.

    The parent must be snapshotted before the wrapper's first await:
    ctx correlation fields are mutable shared state, and concurrent calls
    under asyncio.gather would otherwise interleave.
    """
    return generate_cid(), ctx.correlation_id


def build_lm_started(
    *,
    source: str,
    name: str,
    model: str,
    provider: str,
    correlation_id: str,
    parent_correlation_id: str,
    input_data: Optional[dict[str, Any]],
) -> LMStarted:
    return LMStarted(
        name=name,
        correlation_id=correlation_id,
        parent_correlation_id=parent_correlation_id,
        model=model,
        provider=provider,
        input_data=input_data,
        metadata=observed_metadata(source, name),
    )


def build_lm_completed(
    *,
    source: str,
    name: str,
    model: str,
    provider: str,
    correlation_id: str,
    parent_correlation_id: str,
    duration_ms: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    finish_reason: Optional[str] = None,
    output_data: Optional[dict[str, Any]] = None,
) -> LMCompleted:
    # Keep string metadata for older runtimes while also populating typed wire
    # fields for current consumers.
    return LMCompleted(
        name=name,
        correlation_id=correlation_id,
        parent_correlation_id=parent_correlation_id,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        finish_reason=finish_reason,
        output_data=output_data,
        duration_ms=duration_ms,
        metadata=observed_metadata(
            source,
            name,
            model=model,
            provider=provider,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        ),
    )


def build_lm_failed(
    *,
    source: str,
    name: str,
    model: str,
    provider: str,
    correlation_id: str,
    parent_correlation_id: str,
    duration_ms: int,
    error: BaseException,
) -> LMFailed:
    mode = content_capture_mode()
    error_message = capture_error_message(error)
    error_traceback = None
    if mode != "metadata-only":
        try:
            error_traceback = _bounded_string(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                redacted=mode == "redacted",
            )
        except Exception:
            logger.debug("capture error traceback serialization failed", exc_info=True)
    return LMFailed(
        name=name,
        correlation_id=correlation_id,
        parent_correlation_id=parent_correlation_id,
        model=model,
        provider=provider,
        error_code=type(error).__name__,
        error_message=error_message,
        error_traceback=error_traceback,
        duration_ms=duration_ms,
        metadata=observed_metadata(
            source,
            name,
            model=model,
            provider=provider,
            duration_ms=duration_ms,
        ),
    )
