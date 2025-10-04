"""Type definitions and protocols for AGNT5 SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, TypeVar, Union

# Type aliases
JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
HandlerFunc = Callable[..., Awaitable[Any]]

T = TypeVar("T")


class BackoffType(str, Enum):
    """Backoff strategy for retry policies."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryPolicy:
    """Configuration for function retry behavior."""

    max_attempts: int = 3
    initial_interval_ms: int = 1000
    max_interval_ms: int = 60000

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_interval_ms < 0:
            raise ValueError("initial_interval_ms must be non-negative")
        if self.max_interval_ms < self.initial_interval_ms:
            raise ValueError("max_interval_ms must be >= initial_interval_ms")


@dataclass
class BackoffPolicy:
    """Configuration for retry backoff strategy."""

    type: BackoffType = BackoffType.EXPONENTIAL
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")


@dataclass
class FunctionConfig:
    """Configuration for a function handler."""

    name: str
    handler: HandlerFunc
    retries: Optional[RetryPolicy] = None
    backoff: Optional[BackoffPolicy] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class WorkflowConfig:
    """Configuration for a workflow handler."""

    name: str
    handler: HandlerFunc
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None


class ContextProtocol(Protocol):
    """Protocol defining the Context interface."""

    @property
    def run_id(self) -> str:
        """Workflow/run identifier."""
        ...

    @property
    def step_id(self) -> Optional[str]:
        """Current step identifier."""
        ...

    @property
    def attempt(self) -> int:
        """Retry attempt number."""
        ...

    @property
    def component_type(self) -> str:
        """Component type: 'function', 'entity', 'workflow'."""
        ...

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        ...

    def delete(self, key: str) -> None:
        """Delete key from state."""
        ...
