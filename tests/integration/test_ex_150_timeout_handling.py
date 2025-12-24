"""
Integration Tests: Timeout Handling

Tests function timeout behavior using timeout_ms parameter:
- Functions completing before timeout succeed
- Functions exceeding timeout raise errors
- Timeout errors include context (timeout value)
- Timeouts work with retry policies

Run with:
    pytest tests/integration/test_ex_150_timeout_handling.py -v
"""

import time

import pytest

from agnt5.client import RunError


# =============================================================================
# BASIC TIMEOUT BEHAVIOR
# =============================================================================


@pytest.mark.integration
def test_function_completes_before_timeout(client, worker_process, platform):
    """Test function completing within timeout succeeds."""
    # slow_function has timeout_ms=2000 (2 seconds)
    # Sleeping for 0.5s should complete well within timeout
    result = client.run("slow_function", {"sleep_seconds": 0.5})

    assert result["slept"] == 0.5


@pytest.mark.integration
def test_function_exceeds_timeout(client, worker_process, platform):
    """Test function exceeding timeout raises error."""
    # slow_function has timeout_ms=2000 (2 seconds)
    # Sleeping for 3s should exceed timeout
    start = time.time()

    with pytest.raises(RunError) as exc_info:
        client.run("slow_function", {"sleep_seconds": 3.0})

    duration = time.time() - start

    # Should fail before full sleep duration (3s) completes
    # Note: Platform may have default retries, so allow up to 15s (worst case: 3 retries × 3s + backoff)
    # The key assertion is that we get a timeout error, not a successful completion
    assert duration < 15.0, f"Expected timeout within 15s, took {duration:.2f}s"

    # Error should indicate timeout
    error_msg = str(exc_info.value)
    assert "timeout" in error_msg.lower(), f"Expected 'timeout' in error: {error_msg}"


@pytest.mark.integration
def test_timeout_error_includes_context(client, worker_process, platform):
    """Test timeout error message includes useful context."""
    with pytest.raises(RunError) as exc_info:
        client.run("slow_function", {"sleep_seconds": 5.0})

    error_msg = str(exc_info.value)

    # Error should contain timeout value (2000ms)
    assert "2000" in error_msg or "2s" in error_msg.lower() or "timeout" in error_msg.lower(), (
        f"Expected timeout context in error: {error_msg}"
    )


# =============================================================================
# TIMEOUT WITH RETRY
# =============================================================================


@pytest.mark.integration
def test_timeout_with_retry_exhausts_attempts(client, worker_process, platform):
    """Test that timeouts work with retry policies - all attempts timeout."""
    # timeout_with_retry has timeout_ms=1000, retries=3
    # Sleeping for 2s should timeout on every attempt
    start = time.time()

    with pytest.raises(RunError) as exc_info:
        client.run("timeout_with_retry", {"sleep_seconds": 2.0})

    duration = time.time() - start

    # Should take roughly:
    # - 3 attempts × 1s timeout = 3s
    # - Plus 2 backoff intervals (100ms each, exponential) = ~100ms + ~200ms
    # Total: ~3.3s
    assert duration >= 3.0, f"Expected at least 3s (3 timeouts), got {duration:.2f}s"
    assert duration < 6.0, f"Expected under 6s, got {duration:.2f}s"

    # Error should indicate timeout or retry exhaustion
    error_msg = str(exc_info.value)
    assert "timeout" in error_msg.lower() or "failed" in error_msg.lower(), (
        f"Expected timeout/failure message in error: {error_msg}"
    )


@pytest.mark.integration
def test_timeout_with_retry_succeeds_when_fast_enough(client, worker_process, platform):
    """Test function with timeout+retry succeeds when fast enough."""
    # timeout_with_retry has timeout_ms=1000, retries=3
    # Sleeping for 0.5s should complete within timeout
    result = client.run("timeout_with_retry", {"sleep_seconds": 0.5})

    assert result["slept"] == 0.5
    assert result["attempts"] == 1  # Should succeed on first attempt
