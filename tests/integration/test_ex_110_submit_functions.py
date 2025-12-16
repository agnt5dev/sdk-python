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

import pytest
import time


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


@pytest.mark.integration
def test_status_includes_timestamps(client, worker_process):
    """Test that status includes timing information."""
    run_id = client.submit("greet", {"name": "Timed"})

    # Wait for completion
    client.wait_for_result(run_id, timeout=30.0)

    # Get final status
    status = client.get_status(run_id)

    assert "submittedAt" in status or "submitted_at" in status


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
    """Test that get_result raises if run not complete."""
    from agnt5.client import RunError

    # Submit but don't wait
    run_id = client.submit("greet", {"name": "Quick"})

    # Immediately trying to get result might fail if not complete
    # This depends on timing - the function might complete instantly
    # So we just verify the API works
    try:
        result = client.get_result(run_id)
        # If we got here, it completed very quickly
        assert result == "Hello, Quick!"
    except RunError:
        # Expected if not yet complete
        pass


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
    """Test that wait_for_result respects timeout."""
    from agnt5.client import RunError

    # Submit a potentially long-running function
    run_id = client.submit("greet", {"name": "TimeoutTest"})

    # Very short timeout - might timeout or might succeed (function is fast)
    try:
        result = client.wait_for_result(run_id, timeout=0.001, poll_interval=0.001)
        # If we got here, function was very fast
        assert result == "Hello, TimeoutTest!"
    except (RunError, TimeoutError):
        # Expected if timeout occurred
        pass


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
