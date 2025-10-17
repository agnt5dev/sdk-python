"""
Integration Tests: Function Execution

Tests function invocation, retry logic, and error handling using the Client API.

These tests validate:
- Basic synchronous function calls work
- Async function submission and polling works
- Retry logic handles transient failures
- Errors propagate correctly to client

Expected Results (Week 1 - RED Phase):
- test_sync_function_call: ✅ PASS (basic execution works)
- test_async_function_submission: ✅ PASS (async API works)
- test_function_retry_on_failure: ⚠️  May FAIL (retry logic issues)
- test_function_error_propagation: ✅ PASS (errors propagate)
"""

import pytest

@pytest.mark.integration
def test_sync_function_call(client, worker_process):
    """
    Test synchronous function invocation via client.

    This should PASS - validates basic client.run() functionality.
    """
    result = client.run("greet", {"name": "Alice"})

    assert result is not None
    assert "message" in result
    assert result["message"] == "Hello, Alice!"


@pytest.mark.integration
def test_sync_function_with_different_inputs(client, worker_process):
    """
    Test function with various input values.

    This should PASS - validates input serialization/deserialization.
    """
    # Test different names
    names = ["Bob", "Charlie", "Diana", "Eve"]

    for name in names:
        result = client.run("greet", {"name": name})
        assert result["message"] == f"Hello, {name}!"


@pytest.mark.integration
def test_async_function_submission(client, worker_process):
    """
    Test async function submission and polling.

    This should PASS - validates client.submit() and client.wait_for_result().
    """
    # Submit long-running task
    run_id = client.submit("long_task", {"duration": 1})

    assert run_id is not None
    assert isinstance(run_id, str)

    # Wait for completion
    result = client.wait_for_result(run_id, timeout=10)

    assert result["status"] == "completed"
    assert result["duration"] == 1


@pytest.mark.skip(reason="Retry error filtering not yet implemented")
@pytest.mark.integration
def test_function_retry_on_failure(client, worker_process):
    """
    Test function retry logic works end-to-end.

    ⚠️  May FAIL - depends on retry implementation completeness.

    This validates:
    - Function fails N times
    - Retry logic kicks in
    - Function eventually succeeds
    - Client receives final success result
    """
    # Function fails twice, succeeds on 3rd attempt
    result = client.run("flaky_function", {"fail_count": 2})

    assert result["succeeded"] is True
    assert result["attempts"] == 3, (
        f"Expected 3 attempts (2 failures + 1 success), but got {result['attempts']}. "
        "Retry logic may not be working correctly."
    )


@pytest.mark.integration
def test_function_retry_exhaustion(client, worker_process):
    """
    Test function fails after exhausting retries.

    This validates retry limits are enforced.
    """
    from agnt5 import RunError

    # Function configured with max 5 retries, fail 10 times
    # Should fail after exhausting retries
    with pytest.raises(RunError) as exc_info:
        client.run("flaky_function", {"fail_count": 10})

    assert "Simulated failure" in str(exc_info.value)


@pytest.mark.integration
def test_function_error_propagation(client, worker_process):
    """
    Test errors propagate correctly to client.

    This should PASS - validates error handling path.
    """
    from agnt5 import RunError

    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"error": "Test error message"})

    # Error message should be propagated
    assert "Test error message" in str(exc_info.value)


@pytest.mark.integration
def test_function_with_invalid_input(client, worker_process):
    """
    Test function with invalid input type.

    This should properly handle validation errors.
    """
    from agnt5 import RunError

    # Try to call greet without required 'name' parameter
    with pytest.raises((RunError, TypeError, KeyError)):
        client.run("greet", {})


@pytest.mark.integration
def test_function_parallel_execution(client, worker_process):
    """
    Test multiple functions can execute in parallel.

    This should PASS - validates concurrent function execution.
    """
    # Submit multiple function calls
    run_ids = []
    for i in range(10):
        run_id = client.submit("greet", {"name": f"User{i}"})
        run_ids.append(run_id)

    # Wait for all to complete
    results = []
    for run_id in run_ids:
        result = client.wait_for_result(run_id, timeout=10)
        results.append(result)

    # Verify all completed
    assert len(results) == 10
    for i, result in enumerate(results):
        assert result["message"] == f"Hello, User{i}!"


@pytest.mark.integration
def test_function_call_nonexistent(client, worker_process):
    """
    Test calling non-existent function returns error.

    This should properly handle function not found errors.
    """
    from agnt5 import RunError

    with pytest.raises(RunError) as exc_info:
        client.run("nonexistent_function", {"data": "test"})

    # Should indicate function not found
    error_msg = str(exc_info.value).lower()
    assert "not found" in error_msg or "unknown" in error_msg or "does not exist" in error_msg


@pytest.mark.integration
def test_function_with_timeout(client, worker_process):
    """
    Test function execution with timeout.

    This validates timeout handling.
    """
    # Submit long task
    run_id = client.submit("long_task", {"duration": 10})

    # Try to wait with short timeout
    # Should raise timeout error or return status indicating not completed
    try:
        result = client.wait_for_result(run_id, timeout=1)
        # If it doesn't timeout, result should indicate still running
        assert result.get("status") in ["pending", "running"]
    except Exception as e:
        # Timeout exception is acceptable
        assert "timeout" in str(e).lower()


@pytest.mark.integration
@pytest.mark.skip(reason="Retry error filtering not yet implemented")
def test_function_retry_only_on_retryable_errors(client, worker_process):
    """
    Test retry logic only retries retryable errors.

    ⏭️  SKIPPED - Requires retry error filtering implementation.

    This validates:
    - Transient errors (network, timeout) are retried
    - Permanent errors (validation, auth) are NOT retried
    - Retry budget is not wasted on non-retryable errors
    """
    # TODO: Implement retryable vs non-retryable error types
    # TODO: Test that permanent errors fail immediately
    # TODO: Test that transient errors are retried
    pass
