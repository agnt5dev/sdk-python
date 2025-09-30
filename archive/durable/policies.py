"""Retry and backoff policy configuration for durable functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BackoffPolicy:
    """Configuration describing how retries are spaced out."""

    strategy: str = "exponential"
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        strategy_normalised = self.strategy.lower()
        if strategy_normalised not in {"exponential", "fixed"}:
            raise ValueError(
                f"BackoffPolicy.strategy must be 'exponential' or 'fixed', got '{self.strategy}'."
            )
        object.__setattr__(self, "strategy", strategy_normalised)

        if self.initial_delay_seconds <= 0:
            raise ValueError("BackoffPolicy.initial_delay_seconds must be positive")
        if self.max_delay_seconds <= 0:
            raise ValueError("BackoffPolicy.max_delay_seconds must be positive")
        if self.multiplier <= 0:
            raise ValueError("BackoffPolicy.multiplier must be positive")
        if self.jitter < 0:
            raise ValueError("BackoffPolicy.jitter must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "initial_delay_seconds": self.initial_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "multiplier": self.multiplier,
            "jitter": self.jitter,
        }


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration for a durable function or step."""

    max_attempts: int = 3
    max_duration_seconds: Optional[float] = None
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("RetryPolicy.max_duration_seconds must be positive when set")

    def with_backoff(self, backoff: Optional[BackoffPolicy]) -> "RetryPolicy":
        if backoff is None or backoff == self.backoff:
            return self
        return RetryPolicy(
            max_attempts=self.max_attempts,
            max_duration_seconds=self.max_duration_seconds,
            backoff=backoff,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "max_attempts": self.max_attempts,
            "backoff": self.backoff.to_dict(),
        }
        if self.max_duration_seconds is not None:
            payload["max_duration_seconds"] = self.max_duration_seconds
        return payload
