"""Sandbox-specific event classes for journal and observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .events import Completed, ComponentType, Failed, Started


@dataclass(kw_only=True)
class SandboxExecuteStarted(Started):
    """Sandbox code execution started."""

    _event_type: ClassVar[str] = "sandbox.execute.started"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    language: str = ""
    code_length: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class SandboxExecuteCompleted(Completed):
    """Sandbox code execution completed."""

    _event_type: ClassVar[str] = "sandbox.execute.completed"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    language: str = ""
    exit_code: int = 0
    execution_time_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class SandboxExecuteFailed(Failed):
    """Sandbox code execution failed."""

    _event_type: ClassVar[str] = "sandbox.execute.failed"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    language: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class SandboxFileWritten(Completed):
    """File written to sandbox workspace."""

    _event_type: ClassVar[str] = "sandbox.file.written"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    file_path: str = ""
    file_size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)


@dataclass(kw_only=True)
class SandboxFileRead(Completed):
    """File read from sandbox workspace."""

    _event_type: ClassVar[str] = "sandbox.file.read"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    file_path: str = ""
    file_size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)
