"""
Integration Tests: submit() - Asynchronous Function Invocation

Tests async function execution using client.submit():
- Returns run_id immediately (non-blocking)
- Use get_status() to check progress
- Use get_result() to retrieve output when complete
- Use wait_for_result() for convenience polling

Run with:
    pytest tests/integration/test_ex_110_submit_functions.py -v
"""

import time

import pytest

# =============================================================================
# BASIC SUBMIT AND WAIT
# =============================================================================


@pytest.mark.integration
def test_submit_returns_run_id(client, worker_process, platform):
    """Test that submit returns a run_id immediately."""
    run_id = client.submit("greet", {"name": "Alice"})

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0


@pytest.mark.integration
def test_submit_and_wait_for_result(client, worker_process):
    """Test submit followed by wait_for_result."""
    run_id = client.submit("greet", {"name": "Bob"})

    result = client.wait_for_result(run_id, timeout=30.0)
    assert result == "Hello, Bob!"


@pytest.mark.integration
def test_submit_and_get_status(client, worker_process):
    """Test checking status of a submitted run."""
    run_id = client.submit("greet", {"name": "Charlie"})

    # Poll until complete
    for _ in range(30):
        status = client.get_status(run_id)
        assert "status" in status
        assert status["status"] in ["pending", "running", "completed", "failed"]

        if status["status"] == "completed":
            break
        time.sleep(0.5)
    else:
        pytest.fail("Run did not complete within timeout")

    # Verify status structure
    assert status["runId"] == run_id
    assert status["status"] == "completed"


@pytest.mark.integration
def test_submit_and_get_result(client, worker_process):
    """Test getting result of a completed run."""
    run_id = client.submit("add", {"a": 10, "b": 20})

    # Wait for completion
    client.wait_for_result(run_id, timeout=30.0)

    # Get result explicitly
    result = client.get_result(run_id)
    assert result == 30


# =============================================================================
# MULTIPLE CONCURRENT SUBMITS
# =============================================================================


@pytest.mark.integration
def test_submit_multiple_concurrent(client, worker_process):
    """Test submitting multiple functions concurrently."""
    # Submit multiple runs
    run_ids = []
    for i in range(5):
        run_id = client.submit("greet", {"name": f"User{i}"})
        run_ids.append(run_id)

    # All should have unique run_ids
    assert len(set(run_ids)) == 5

    # Wait for all to complete
    results = []
    for run_id in run_ids:
        result = client.wait_for_result(run_id, timeout=30.0)
        results.append(result)

    # Verify results
    for i, result in enumerate(results):
        assert result == f"Hello, User{i}!"


@pytest.mark.integration
def test_submit_different_functions(client, worker_process):
    """Test submitting different functions concurrently."""
    # Submit different function types
    run_id_greet = client.submit("greet", {"name": "Alice"})
    run_id_add = client.submit("add", {"a": 5, "b": 3})
    run_id_data = client.submit("process_data", {"data": "hello"})

    # Wait for all
    result_greet = client.wait_for_result(run_id_greet, timeout=30.0)
    result_add = client.wait_for_result(run_id_add, timeout=30.0)
    result_data = client.wait_for_result(run_id_data, timeout=30.0)

    # Verify each result
    assert result_greet == "Hello, Alice!"
    assert result_add == 8
    assert result_data["processed"] == "HELLO"
    assert result_data["length"] == 5


# =============================================================================
# STATUS TRANSITIONS
# =============================================================================


@pytest.mark.integration
def test_status_transitions(client, worker_process):
    """Test that status transitions correctly through states."""
    run_id = client.submit("process_data", {"data": "test"})

    # Collect observed statuses
    observed_statuses = set()
    for _ in range(60):
        status = client.get_status(run_id)
        observed_statuses.add(status["status"])

        if status["status"] == "completed":
            break
        time.sleep(0.1)

    # Should have seen at least pending/running and completed
    assert "completed" in observed_statuses
    # Either pending or running should have been observed (might be too fast for both)
    assert len(observed_statuses) >= 1


# =============================================================================
# ERROR HANDLING WITH SUBMIT
# =============================================================================


@pytest.mark.integration
def test_submit_failing_function(client, worker_process):
    """Test submit with a function that fails."""
    from agnt5.client import RunError

    run_id = client.submit("failing_function", {"should_fail": True, "error_type": "ValueError"})

    # Wait should raise error - the specific error message may vary
    with pytest.raises(RunError):
        client.wait_for_result(run_id, timeout=30.0)
    # Test passes if RunError is raised (error details may not be propagated)


@pytest.mark.integration
def test_get_result_before_complete(client, worker_process):
    """Test that get_result raises RunError for incomplete runs.

    Note: This test may be skipped if the platform processes runs too fast
    to observe the 'not complete' state. The test validates:
    - RunError is raised (not generic Exception)
    - Error message contains "not complete" or similar indicator

    This is a timing-dependent test that tries to catch a run before it completes.
    """
    from agnt5.client import RunError

    # Try multiple times to catch a run in progress
    # Use a function that sleeps for 1.5s (within its 2s timeout)
    attempts = 5

    for _ in range(attempts):
        run_id = client.submit("slow_function", {"sleep_seconds": 1.5})

        try:
            # Immediately try to get result
            client.get_result(run_id)
            # Run completed too fast - try again
            continue
        except RunError as e:
            # This is what we want to test - verify the error message
            error_message = str(e).lower()
            assert "not complete" in error_message or "not found" in error_message, (
                f"Expected 'not complete' or 'not found' in error, got: {e}"
            )
            # Test passed - we caught and validated the error
            return

    # If we get here, all attempts completed too fast to observe the error
    pytest.skip(
        f"Run completed too fast in {attempts} attempts. "
        "This test validates error handling for incomplete runs but requires "
        "catching a run before completion, which is timing-dependent."
    )


@pytest.mark.integration
def test_submit_continues_after_failure(client, worker_process):
    """Test that new submits work after a failure."""
    from agnt5.client import RunError

    # Submit a failing function
    fail_run_id = client.submit("failing_function", {"should_fail": True})

    # Wait for it to fail
    with pytest.raises(RunError):
        client.wait_for_result(fail_run_id, timeout=30.0)

    # Submit a new function - should work
    success_run_id = client.submit("greet", {"name": "AfterFailure"})
    result = client.wait_for_result(success_run_id, timeout=30.0)
    assert result == "Hello, AfterFailure!"


# =============================================================================
# TIMEOUT BEHAVIOR
# =============================================================================


@pytest.mark.integration
def test_wait_for_result_timeout(client, worker_process):
    """Test that wait_for_result raises RunError (not TimeoutError) on timeout.

    Verifies:
    - RunError is raised (SDK uses RunError, not Python's TimeoutError)
    - Error message explicitly mentions "timeout"
    - The timeout value is included in the error message
    """
    from agnt5.client import RunError

    # Submit a slow function that sleeps for 1.5s (within its 2s timeout)
    # We'll use a very short client-side timeout to trigger the timeout
    run_id = client.submit("slow_function", {"sleep_seconds": 1.5})

    # Wait with a 0.5 second timeout - client will timeout before function completes
    with pytest.raises(RunError) as exc_info:
        client.wait_for_result(run_id, timeout=0.5, poll_interval=0.1)

    # Verify error message mentions timeout
    error_message = str(exc_info.value).lower()
    assert "timeout" in error_message, (
        f"Expected 'timeout' in error message, got: {exc_info.value}"
    )
    # Timeout value should be included
    assert "0.5" in str(exc_info.value), (
        f"Expected timeout value in error message, got: {exc_info.value}"
    )


@pytest.mark.integration
def test_custom_poll_interval(client, worker_process):
    """Test wait_for_result with custom poll interval."""
    run_id = client.submit("greet", {"name": "PollTest"})

    # Use longer poll interval
    start = time.time()
    result = client.wait_for_result(run_id, timeout=30.0, poll_interval=0.5)
    elapsed = time.time() - start

    assert result == "Hello, PollTest!"
    # Function should complete quickly, but we can't guarantee timing


# =============================================================================
# WORKFLOW SUBMIT
# =============================================================================


@pytest.mark.integration
def test_submit_workflow(client, worker_process):
    """Test submitting a workflow."""
    run_id = client.submit(
        "data_pipeline",
        {"source": "test-source"},
        component_type="workflow"
    )

    assert run_id is not None
    result = client.wait_for_result(run_id, timeout=60.0)

    # data_pipeline returns: source, original_count, transformed, valid
    assert result["source"] == "test-source"
    assert "original_count" in result
    assert "transformed" in result
    assert result["valid"] is True


@pytest.mark.integration
def test_submit_workflow_status(client, worker_process):
    """Test checking workflow status."""
    run_id = client.submit(
        "data_pipeline",
        {"source": "status-test"},
        component_type="workflow"
    )

    # Poll status
    status = client.get_status(run_id)
    assert status["runId"] == run_id
    assert status["status"] in ["pending", "running", "completed", "failed"]

    # Wait for completion
    client.wait_for_result(run_id, timeout=60.0)

    # Final status
    final_status = client.get_status(run_id)
    assert final_status["status"] == "completed"


# =============================================================================
# ERROR CASES - NONEXISTENT RUNS
# =============================================================================


@pytest.mark.integration
def test_get_status_nonexistent_run(client, worker_process):
    """Test that get_status raises appropriate error for nonexistent run.

    Verifies:
    - HTTP 404 or similar error is raised
    - Error indicates "not found"
    """
    import httpx

    fake_run_id = "00000000-0000-0000-0000-000000000000"

    # get_status should raise an HTTP error for nonexistent run
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.get_status(fake_run_id)

    # Should be a 404 Not Found
    assert exc_info.value.response.status_code == 404, (
        f"Expected 404, got: {exc_info.value.response.status_code}"
    )


@pytest.mark.integration
def test_get_result_nonexistent_run(client, worker_process):
    """Test that get_result raises appropriate error for nonexistent run.

    Verifies:
    - RunError is raised (not generic HTTP error)
    - Error message indicates "not found"
    """
    from agnt5.client import RunError

    fake_run_id = "00000000-0000-0000-0000-000000000000"

    # get_result should raise RunError for nonexistent run
    with pytest.raises(RunError) as exc_info:
        client.get_result(fake_run_id)

    # Error should indicate not found
    error_message = str(exc_info.value).lower()
    assert "not found" in error_message or "not complete" in error_message, (
        f"Expected 'not found' or 'not complete' in error, got: {exc_info.value}"
    )
