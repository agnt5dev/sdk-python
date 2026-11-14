"""Content policy for SDK-authored telemetry, separate from durable execution data."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

PayloadRedactor = Callable[[Any], Any]
_OFF_VALUES = {"off", "0", "false", "no"}
_ON_VALUES = {"on", "1", "true", "yes"}
_SENSITIVE_KEYS = {
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
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def content_capture_mode() -> str:
    """Explicit modes win; absent or invalid opt-in selects metadata-only."""
    configured = os.getenv("AGNT5_CAPTURE_CONTENT_MODE", "").strip().lower()
    if configured:
        return configured if configured in {"full", "redacted"} else "metadata-only"
    legacy = os.getenv("AGNT5_LLM_CAPTURE_CONTENT", "").strip().lower()
    return "full" if legacy in _ON_VALUES else "metadata-only"


def is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def redact_text(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _copy_payload(value: Any, *, depth: int = 0) -> Any:
    if depth >= 16:
        return "[max-depth]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if value in {"[REDACTED]", "[max-depth]", "[additional items omitted]"}:
            return value
        # Decode embedded JSON before key redaction, never after truncation.
        if value.lstrip().startswith(("{", "[", '"')):
            try:
                value = json.loads(value)
            except ValueError:
                return "[unparseable payload omitted]"
            return _copy_payload(value, depth=depth + 1)
        return redact_text(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                result["__truncated__"] = "additional entries omitted"
                break
            if not isinstance(key, str):
                raise TypeError("capture requires string keys")
            result[key] = (
                "[REDACTED]" if is_sensitive_key(key) else _copy_payload(item, depth=depth + 1)
            )
        return result
    if isinstance(value, (list, tuple)):
        result = [_copy_payload(item, depth=depth + 1) for item in value[:128]]
        if len(value) > 128:
            result.append("[additional items omitted]")
        return result
    raise TypeError("unsupported capture value")


def _span_byte_limit() -> int:
    try:
        return max(0, int(os.getenv("AGNT5_SPAN_ATTRIBUTE_VALUE_MAX_BYTES", "8192")))
    except ValueError:
        return 0


class SpanPayloadPolicy:
    """Opt-in span capture with isolated application redaction and hard byte bounds."""

    def __init__(self, redactor: PayloadRedactor | None = None) -> None:
        if redactor is not None and not callable(redactor):
            raise TypeError("span_payload_redactor must be callable")
        self._redactor = redactor

    def capture(self, value: Any) -> tuple[str, bool] | None:
        limit = _span_byte_limit()
        if (
            not limit
            or os.getenv("AGNT5_CAPTURE", "").strip().lower() in _OFF_VALUES
            or content_capture_mode() == "metadata-only"
        ):
            return None
        try:
            copied = _copy_payload(value)
            if self._redactor is not None:
                copied = self._redactor(copied)
                if copied is None:
                    return None
                copied = _copy_payload(copied)
            data = json.dumps(copied, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(data) <= limit:
                return data.decode("utf-8"), False
            marker = b"...[truncated]"[:limit]
            preview = data[: limit - len(marker)].decode("utf-8", errors="ignore")
            return preview + marker.decode("ascii"), True
        except Exception:
            # Capture errors must not leak exception text or affect execution.
            return None

    def attributes(self, direction: str, value: Any) -> dict[str, str]:
        captured = self.capture(value)
        return {f"{direction}.data": captured[0]} if captured is not None else {}
