"""LM-specific event classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from ..events import (
    Completed,
    ComponentType,
    Delta,
    Failed,
    OperationType,
    Started,
)


# =============================================================================
# LM Content Block Events (for streaming)
# =============================================================================


@dataclass(kw_only=True)
class LMContentBlockStarted(Started):
    """LM content block started (message or thinking)."""

    _event_type: ClassVar[str] = "lm.content_block.started"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    block_type: str = "text"  # "text" or "thinking"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class LMContentBlockDelta(Delta):
    """LM content block delta (streaming text)."""

    _event_type: ClassVar[str] = "lm.content_block.delta"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    operation: OperationType = field(default=OperationType.MESSAGE, init=False)
    block_type: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class LMContentBlockCompleted(Completed):
    """LM content block completed."""

    _event_type: ClassVar[str] = "lm.content_block.completed"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    block_type: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


# =============================================================================
# LM Call Lifecycle Events
# =============================================================================


@dataclass(kw_only=True)
class LMStarted(Started):
    """LM generation call started."""

    _event_type: ClassVar[str] = "lm.started"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    model: str = ""
    provider: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class LMCompleted(Completed):
    """LM generation call completed."""

    _event_type: ClassVar[str] = "lm.completed"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class LMFailed(Failed):
    """LM generation call failed."""

    _event_type: ClassVar[str] = "lm.failed"
    component_type: ComponentType = field(default=ComponentType.LM, init=False)
    model: str = ""
    provider: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


__all__ = [
    "LMContentBlockStarted",
    "LMContentBlockDelta",
    "LMContentBlockCompleted",
    "LMStarted",
    "LMCompleted",
    "LMFailed",
]
