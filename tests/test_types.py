"""Tests for type definitions and validation."""

import pytest

from agnt5.types import BackoffPolicy, BackoffType, RetryPolicy


class TestRetryPolicy:
    """Test RetryPolicy validation."""

    def test_default_values(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.initial_interval_ms == 1000
        assert policy.max_interval_ms == 60000

    def test_custom_values(self) -> None:
        policy = RetryPolicy(max_attempts=5, initial_interval_ms=500, max_interval_ms=30000)
        assert policy.max_attempts == 5
        assert policy.initial_interval_ms == 500
        assert policy.max_interval_ms == 30000

    def test_max_attempts_validation(self) -> None:
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryPolicy(max_attempts=0)

    def test_initial_interval_validation(self) -> None:
        with pytest.raises(ValueError, match="initial_interval_ms must be non-negative"):
            RetryPolicy(initial_interval_ms=-100)

    def test_max_interval_validation(self) -> None:
        with pytest.raises(ValueError, match="max_interval_ms must be >= initial_interval_ms"):
            RetryPolicy(initial_interval_ms=5000, max_interval_ms=1000)


class TestBackoffPolicy:
    """Test BackoffPolicy validation."""

    def test_default_values(self) -> None:
        policy = BackoffPolicy()
        assert policy.type == BackoffType.EXPONENTIAL
        assert policy.multiplier == 2.0

    def test_custom_values(self) -> None:
        policy = BackoffPolicy(type=BackoffType.LINEAR, multiplier=1.5)
        assert policy.type == BackoffType.LINEAR
        assert policy.multiplier == 1.5

    def test_multiplier_validation(self) -> None:
        with pytest.raises(ValueError, match="multiplier must be positive"):
            BackoffPolicy(multiplier=0)

        with pytest.raises(ValueError, match="multiplier must be positive"):
            BackoffPolicy(multiplier=-1.5)


class TestBackoffType:
    """Test BackoffType enum."""

    def test_enum_values(self) -> None:
        assert BackoffType.CONSTANT == "constant"
        assert BackoffType.LINEAR == "linear"
        assert BackoffType.EXPONENTIAL == "exponential"

    def test_enum_membership(self) -> None:
        assert BackoffType.CONSTANT in BackoffType
        assert BackoffType.LINEAR in BackoffType
        assert BackoffType.EXPONENTIAL in BackoffType
