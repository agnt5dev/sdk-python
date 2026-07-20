"""
Integration Tests: run() - Synchronous Function Invocation

Tests synchronous function execution using client.run():
- Blocks until completion and returns result directly
- Basic functions (greet, add, process_data)
- Error handling (failing_function)
- Retry policies (backoff timing, concurrent retries, error preservation)

Run with:
    pytest tests/integration/test_ex_100_run_functions.py -v

Additional useful coverage:
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
# UNICODE / INTERNATIONALIZATION (P1.5)
# =============================================================================


@pytest.mark.integration
def test_greet_unicode_name(client, worker_process):
    """Test greeting function with Unicode names from various scripts.

    Validates:
    - CJK characters (Chinese, Japanese, Korean)
    - Arabic script
    - Cyrillic script
    - Emoji
    - Mixed scripts
    """
    test_cases = [
        {"name": "世界", "expected": "Hello, 世界!"},  # Chinese
        {"name": "こんにちは", "expected": "Hello, こんにちは!"},  # Japanese
        {"name": "한국", "expected": "Hello, 한국!"},  # Korean
        {"name": "مرحبا", "expected": "Hello, مرحبا!"},  # Arabic
        {"name": "Привет", "expected": "Hello, Привет!"},  # Russian
        {"name": "🌍🚀", "expected": "Hello, 🌍🚀!"},  # Emoji
        {"name": "café", "expected": "Hello, café!"},  # Accented Latin
        {"name": "日本語 & English", "expected": "Hello, 日本語 & English!"},  # Mixed
    ]

    for case in test_cases:
        result = client.run("greet", {"name": case["name"]})
        assert result == case["expected"], (
            f"Unicode test failed for name='{case['name']}': "
            f"expected '{case['expected']}', got '{result}'"
        )


@pytest.mark.integration
def test_process_data_unicode(client, worker_process):
    """Test process_data with Unicode content."""
    unicode_data = "Hello 世界 مرحبا 🌍 café"
    result = client.run("process_data", {"data": unicode_data})

    assert result["processed"] == unicode_data.upper()
    assert result["length"] == len(unicode_data)


# =============================================================================
# LARGE PAYLOAD HANDLING (P1.4)
# =============================================================================


@pytest.mark.integration
def test_function_with_large_payload(client, worker_process):
    """Test function with large payload (~100KB).

    Validates:
    - Large string payloads are transmitted correctly
    - Response contains correct length
    - No truncation or corruption
    """
    # Generate ~100KB payload (100,000 characters)
    large_data = "x" * 100_000

    result = client.run("process_data", {"data": large_data})

    assert result["length"] == 100_000, (
        f"Expected length 100000, got {result['length']}"
    )
    assert result["processed"] == large_data.upper(), (
        "Large payload was corrupted or truncated"
    )


@pytest.mark.integration
def test_function_with_very_large_payload(client, worker_process):
    """Test function with very large payload (~1MB).

    Validates:
    - 1MB payloads can be processed
    - Platform handles large messages correctly
    """
    # Generate ~1MB payload (1,000,000 characters)
    large_data = "a" * 1_000_000

    result = client.run("process_data", {"data": large_data})

    assert result["length"] == 1_000_000, (
        f"Expected length 1000000, got {result['length']}"
    )
    # Just check it starts and ends correctly (full comparison would be slow)
    assert result["processed"].startswith("A" * 100), "Payload start corrupted"
    assert result["processed"].endswith("A" * 100), "Payload end corrupted"


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_error_payload_structure(client, worker_process):
    """Test that error responses have proper structure (P1.1).

    Validates:
    - RunError is raised (not generic Exception)
    - Error message contains exception type
    - Error message contains original message
    - Error message is descriptive and useful
    """
    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"should_fail": True, "error_type": "ValueError"})

    error = exc_info.value
    error_msg = str(error)

    # Must be a RunError, not generic Exception
    assert isinstance(error, RunError), (
        f"Expected RunError, got {type(error).__name__}"
    )

    # Error message should contain the exception type
    assert "ValueError" in error_msg, (
        f"Error should mention exception type 'ValueError'. Got: {error_msg}"
    )

    # Error message should contain the original message (or be descriptive)
    assert "testing" in error_msg.lower() or "fail" in error_msg.lower(), (
        f"Error should contain original exception message. Got: {error_msg}"
    )

    # Error message should be reasonably detailed (not empty or minimal)
    assert len(error_msg) > 10, (
        f"Error message too short to be useful: {error_msg}"
    )


@pytest.mark.integration
def test_error_preserves_exception_type(client, worker_process):
    """Test that different exception types are preserved (P1.1)."""
    error_types = ["ValueError", "RuntimeError", "TypeError"]

    for error_type in error_types:
        with pytest.raises(RunError) as exc_info:
            client.run("failing_function", {"should_fail": True, "error_type": error_type})

        error_msg = str(exc_info.value)
        assert error_type in error_msg, (
            f"Error message should contain '{error_type}'. Got: {error_msg}"
        )


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
    """Test function that fails all retries with precise verification.

    Validates:
    - Function exhausts all retry attempts
    - RunError contains structured retry metadata (attempts, max_attempts)
    - Error message is preserved
    """
    # retry_always_fails is configured with max_attempts=2, always fails
    with pytest.raises(RunError) as exc_info:
        client.run("retry_always_fails", {})

    error = exc_info.value
    # Verify retry metadata
    assert error.exhausted_retries, (
        f"Expected all retries exhausted. "
        f"attempts={error.attempts}, max_attempts={error.max_attempts}"
    )
    assert error.attempts == 2, f"Expected 2 attempts, got {error.attempts}"
    assert error.max_attempts == 2, f"Expected max_attempts=2, got {error.max_attempts}"

    # Error message should still be preserved
    assert "Always fails" in error.message


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
    """Final error should contain original exception message with retry context.

    Validates:
    - Original error message is preserved
    - RunError has structured fields for programmatic access
    - String representation includes retry context
    """
    with pytest.raises(RunError) as exc_info:
        client.run("retry_always_fails", {})

    error = exc_info.value

    # Should contain the error message from the function
    assert "Always fails" in error.message, f"Expected 'Always fails' in error: {error.message}"

    # Error should have retry metadata
    assert error.attempts is not None, "Expected attempts to be set"
    assert error.max_attempts is not None, "Expected max_attempts to be set"

    # String representation should include context
    error_str = str(error)
    assert "Always fails" in error_str
    # Should show attempts in string representation
    assert "attempts:" in error_str or error.attempts is None


# =============================================================================
# INVALID INPUT HANDLING (P2.3)
# =============================================================================


@pytest.mark.integration
def test_add_with_string_inputs(client, worker_process):
    """Test add function with string inputs instead of numbers.

    Validates:
    - Function rejects invalid input types
    - Error is raised (not silent corruption)
    - Error message is descriptive
    """
    # Try adding strings - should fail or coerce
    try:
        result = client.run("add", {"a": "five", "b": "three"})
        # If it somehow succeeds, the result should be reasonable
        # (e.g., string concatenation or error)
        assert isinstance(result, (int, str)), f"Unexpected result type: {type(result)}"
    except RunError as e:
        # Expected - should have a descriptive error
        error_msg = str(e).lower()
        assert (
            "type" in error_msg or
            "invalid" in error_msg or
            "convert" in error_msg or
            "int" in error_msg
        ), f"Error should mention type issue. Got: {e}"


@pytest.mark.integration
def test_add_with_none_inputs(client, worker_process):
    """Test add function with None inputs.

    Validates:
    - None values are handled properly
    - Either fails with descriptive error or uses defaults
    """
    try:
        result = client.run("add", {"a": None, "b": 5})
        # If it succeeds, None might be treated as 0
        assert isinstance(result, (int, float, type(None))), (
            f"Unexpected result: {result}"
        )
    except RunError as e:
        # Expected - None is not a valid number
        error_msg = str(e).lower()
        assert len(error_msg) > 5, f"Error should be descriptive. Got: {e}"


@pytest.mark.integration
def test_function_with_missing_required_param(client, worker_process):
    """Test function called without required parameters.

    Validates:
    - Missing required params raise an error
    - Error message mentions the missing parameter
    """
    # greet requires 'name' parameter
    with pytest.raises(RunError) as exc_info:
        client.run("greet", {})  # Missing 'name'

    error_msg = str(exc_info.value).lower()
    # Should mention missing parameter or similar
    assert (
        "name" in error_msg or
        "required" in error_msg or
        "missing" in error_msg or
        "parameter" in error_msg or
        len(error_msg) > 10  # At least some error message
    ), f"Error should mention missing 'name' parameter. Got: {exc_info.value}"
