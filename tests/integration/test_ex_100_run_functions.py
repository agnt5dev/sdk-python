"""
Integration Tests: run() - Synchronous Function Invocation

Tests synchronous function execution using client.run():
- Blocks until completion and returns result directly
- Basic functions (greet, add, process_data)
- Error handling (failing_function)
- Retry policies (backoff timing, concurrent retries, error preservation)

Run with:
    pytest tests/integration/test_ex_100_run_functions.py -v

TODO: Additional tests to add:
- Timeout handling: Test behavior when function exceeds timeout
- Large payload handling: Test with large inputs/outputs (e.g., 1MB+ data)
- Custom function names: Test name= parameter in @function decorator (currently not registering)
"""

import concurrent.futures
import time

import pytest

from agnt5.client import RunError


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================


@pytest.mark.integration
def test_greet(client, worker_process, platform):
    """Test basic greeting function."""
    result = client.run("greet", {"name": "Alice"})
    assert result == "Hello, Alice!"


@pytest.mark.integration
def test_greet_different_names(client, worker_process):
    """Test greeting with different inputs."""
    test_cases = [
        {"name": "Bob", "expected": "Hello, Bob!"},
        {"name": "World", "expected": "Hello, World!"},
        {"name": "", "expected": "Hello, !"},
    ]

    for case in test_cases:
        result = client.run("greet", {"name": case["name"]})
        assert result == case["expected"], f"Failed for name={case['name']}"


@pytest.mark.integration
def test_add(client, worker_process):
    """Test add function with different inputs."""
    test_cases = [
        {"a": 5, "b": 3, "expected": 8},
        {"a": 0, "b": 0, "expected": 0},
        {"a": -5, "b": 5, "expected": 0},
        {"a": 100, "b": 200, "expected": 300},
    ]

    for case in test_cases:
        result = client.run("add", {"a": case["a"], "b": case["b"]})
        assert result == case["expected"], f"Failed for a={case['a']}, b={case['b']}"


@pytest.mark.integration
def test_process_data(client, worker_process):
    """Test process_data function."""
    result = client.run("process_data", {"data": "hello world"})

    assert result["processed"] == "HELLO WORLD"
    assert result["length"] == 11


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_failing_function_success(client, worker_process):
    """Test failing_function when should_fail=False."""
    result = client.run("failing_function", {"should_fail": False})

    assert result["success"] is True
    assert "No error" in result["message"]


@pytest.mark.integration
def test_failing_function_value_error(client, worker_process):
    """Test failing_function raises ValueError."""
    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"should_fail": True, "error_type": "ValueError"})

    assert "ValueError" in str(exc_info.value)


@pytest.mark.integration
def test_failing_function_runtime_error(client, worker_process):
    """Test failing_function raises RuntimeError."""
    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"should_fail": True, "error_type": "RuntimeError"})

    assert "RuntimeError" in str(exc_info.value)


@pytest.mark.integration
def test_worker_continues_after_error(client, worker_process):
    """Test that worker continues functioning after an error."""
    # Cause an error
    with pytest.raises(RunError):
        client.run("failing_function", {"should_fail": True})

    # Worker should still work
    result = client.run("greet", {"name": "StillWorking"})
    assert result == "Hello, StillWorking!"


@pytest.mark.integration
def test_non_retry_function_fails_immediately(client, worker_process):
    """Function without retry config should fail immediately without retry."""
    start = time.time()

    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"should_fail": True, "error_type": "ValueError"})

    duration = time.time() - start

    # Should fail without retry backoff (under 5s accounting for network overhead)
    assert duration < 5.0, f"Expected failure under 5s, got {duration:.2f}s"
    assert "ValueError" in str(exc_info.value)


# =============================================================================
# RETRY POLICIES - Platform-level retry orchestration
# =============================================================================


@pytest.mark.integration
def test_retry_succeeds_on_second_attempt(client, worker_process):
    """Test function that fails once then succeeds (retries work)."""
    # retry_eventually_succeeds is configured with max_attempts=3, fails once, then succeeds
    result = client.run("retry_eventually_succeeds", {})
    assert result["success"] is True
    assert result["attempts"] == 2  # Failed once, succeeded on 2nd attempt


@pytest.mark.integration
def test_retry_exhausts_all_attempts(client, worker_process):
    """Test function that fails all retries."""
    # retry_always_fails is configured with max_attempts=2, always fails
    with pytest.raises(RunError) as exc_info:
        client.run("retry_always_fails", {})

    assert "Always fails" in str(exc_info.value)


@pytest.mark.integration
def test_retry_respects_backoff_timing(client, worker_process):
    """Verify retry backoff timing is approximately correct."""
    start = time.time()
    result = client.run("retry_eventually_succeeds", {})
    duration = time.time() - start

    # retry_eventually_succeeds has initial_interval_ms=100
    # Should take at least ~100ms backoff between attempt 1 and 2
    assert result["success"] is True
    assert result["attempts"] == 2
    # At least 80ms (accounting for network and jitter)
    assert duration >= 0.08, f"Expected at least 80ms, got {duration*1000:.0f}ms"
    # But not too long (should be under 2s)
    assert duration < 2.0, f"Expected under 2s, got {duration:.2f}s"


@pytest.mark.integration
def test_concurrent_retry_functions(client, worker_process):
    """Multiple retry functions executing concurrently."""

    def invoke_retry():
        return client.run("retry_eventually_succeeds", {})

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(invoke_retry) for _ in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed after retries
    assert all(r["success"] for r in results), f"Some invocations failed: {results}"
    assert all(r["attempts"] == 2 for r in results), f"Expected 2 attempts each: {results}"


@pytest.mark.integration
def test_retry_preserves_error_message(client, worker_process):
    """Final error should contain original exception message."""
    with pytest.raises(RunError) as exc_info:
        client.run("retry_always_fails", {})

    error_msg = str(exc_info.value)
    # Should contain the error message from the function
    assert "Always fails" in error_msg, f"Expected 'Always fails' in error: {error_msg}"
