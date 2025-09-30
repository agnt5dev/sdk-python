"""Public entrypoint for the next-generation Context implementation."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Mapping, Optional

from .._compat import _rust_available
from .config import ContextConfig
from .core import (
    ContextNotReadyError,
    CoreContext,
    LlmNamespace,
    SignalNamespace,
    TaskNamespace,
    TimerNamespace,
)

__all__ = [
    "Context",
    "ContextConfig",
    "ContextNotReadyError",
    "ContextVersion",
    "CoreContext",
    "LlmNamespace",
    "SignalNamespace",
    "TaskNamespace",
    "TimerNamespace",
    "get_context_version",
]


class ContextVersion(Enum):
    LEGACY = "legacy"
    CORE = "core"


def get_context_version(env: Optional[Mapping[str, str]] = None) -> ContextVersion:
    """Resolve whether the legacy or core implementation should be used."""

    env = env or os.environ
    value = env.get("AGNT5_SDK_USE_CORE_CONTEXT", "").strip().lower()
    if value in {"1", "true", "core", "next"}:
        return ContextVersion.CORE
    return ContextVersion.LEGACY


class Context:
    """Factory that returns either the legacy or core-backed Context."""

    def __new__(
        cls,
        *args: Any,
        config: Optional[ContextConfig] = None,
        runtime_handle: Any = None,
        **kwargs: Any,
    ) -> Any:
        version = get_context_version()
        if version is ContextVersion.LEGACY:
            from agnt5.archive.context import Context as LegacyContext

            return LegacyContext(*args, **kwargs)

        if args:
            if config is not None:
                raise TypeError(
                    "Context configuration supplied as both positional and keyword arguments"
                )
            if not isinstance(args[0], ContextConfig):
                raise TypeError(
                    "Core Context expects a ContextConfig as the first argument"
                )
            config = args[0]
            args = args[1:]

        if args:
            raise TypeError(
                "Core Context does not accept additional positional arguments"
            )

        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for core Context: {unexpected}")

        if config is None:
            raise TypeError("Core Context requires a ContextConfig instance")

        if not _rust_available:
            raise RuntimeError(
                "Rust core extension not available; cannot instantiate core Context"
            )

        return CoreContext(config=config, runtime_handle=runtime_handle)
