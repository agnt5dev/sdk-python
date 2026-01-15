"""
Integration tests for functions.

Tests basic function execution using fixtures and verifies journal events.
"""

import pytest
from test_helpers import print_journal_events, verify_journal_events


@pytest.mark.integration
def test_add(client, worker_process, platform):
    """Test basic addition function."""
    result = client.run("add", {"a": 5, "b": 3})
    assert result == 8

    # negative numbers
    result = client.run("add", {"a": -5, "b": 3})
    assert result == -2

    # zero
    result = client.run("add", {"a": 0, "b": 0})
    assert result == 0


@pytest.mark.integration
def test_add_journal_events(client, worker_process, platform):
    """Test that journal events are recorded for function execution."""
    # Execute the function
    result = client.run("add", {"a": 10, "b": 20})
    assert result == 30

    # Verify journal events in expected order
    run_id, event_types = verify_journal_events(
        platform,
        expected_events=[
            "run.enqueued",
            "run.assigned",
            "run.started",
            "function.started",
            "function.completed",
            "run.completed",
        ],
    )

    # Print for debugging
    print_journal_events(run_id, event_types)


# ============================================================================
# Error Scenario Tests
# ============================================================================


@pytest.mark.integration
def test_missing_parameters(client, worker_process, platform):
    """Test function called without required parameters."""
    from agnt5.client import RunError

    # Call add without parameters - should fail
    with pytest.raises(RunError) as exc_info:
        client.run("add", {})

    # Validate error
    error = exc_info.value
    error_msg = str(error).lower()
    # Error should mention missing parameters or type error
    assert "error" in error_msg or "missing" in error_msg or "required" in error_msg

    # Verify journal events (includes enqueue and assignment)
    verify_journal_events(
        platform,
        expected_events=[
            "run.enqueued",
            "run.assigned",
            "run.started",
            "function.started",
            "function.failed",
            "run.failed",
        ],
    )


@pytest.mark.integration
def test_invalid_parameter_type(client, worker_process, platform):
    """Test function with wrong parameter type."""
    from agnt5.client import RunError

    # Call type_strict_function with string instead of int
    with pytest.raises(RunError) as exc_info:
        client.run("type_strict_function", {"value": "not_an_int"})

    # Validate error contains TypeError information
    error = exc_info.value
    error_msg = str(error).lower()
    assert "type" in error_msg or "error" in error_msg

    # Verify journal events (includes enqueue and assignment)
    verify_journal_events(
        platform,
        expected_events=[
            "run.enqueued",
            "run.assigned",
            "run.started",
            "function.started",
            "function.failed",
            "run.failed",
        ],
    )


# @pytest.mark.integration
# def test_runtime_error_divide_by_zero(client, worker_process, platform):
#     """Test function that raises ZeroDivisionError."""
#     from agnt5.client import RunError

#     # Call divide with zero divisor
#     with pytest.raises(RunError) as exc_info:
#         client.run("divide", {"a": 10, "b": 0})

#     # Validate error mentions division by zero
#     error = exc_info.value
#     error_msg = str(error).lower()
#     assert "division" in error_msg or "zero" in error_msg or "error" in error_msg

#     # Verify journal events (includes enqueue and assignment)
#     verify_journal_events(
#         platform,
#         expected_events=[
#             "run.enqueued",
#             "run.assigned",
#             "run.started",
#             "function.started",
#             "function.failed",
#             "run.failed",
#         ],
#     )


# @pytest.mark.integration
# def test_function_timeout(client, worker_process, platform):
#     """Test function that exceeds timeout."""
#     from agnt5.client import RunError

#     # Client timeout is 60s, so use 70s delay
#     # Note: This test will take time proportional to the timeout
#     with pytest.raises(RunError) as exc_info:
#         client.run("slow_function", {"delay_seconds": 70})

#     # Validate error mentions timeout
#     error = exc_info.value
#     error_msg = str(error).lower()
#     assert "timeout" in error_msg or "time" in error_msg

#     # Verify journal events - timeout generates specific events
#     verify_journal_events(
#         platform,
#         expected_events=[
#             "run.started",
#             "function.started",
#             # Timeout events may vary - could be function.failed or function.timeout
#             "run.failed",  # At minimum, run should fail
#         ],
#     )


# @pytest.mark.integration
# def test_function_failure_no_retry(client, worker_process, platform):
#     """Test function that fails immediately."""
#     from agnt5.client import RunError

#     # Call function that always fails
#     with pytest.raises(RunError) as exc_info:
#         client.run("failing_function", {"message": "test error"})

#     # Validate error message
#     error = exc_info.value
#     error_msg = str(error).lower()
#     assert "test error" in error_msg or "intentional" in error_msg or "failure" in error_msg

#     # Verify journal events (includes enqueue and assignment)
#     verify_journal_events(
#         platform,
#         expected_events=[
#             "run.enqueued",
#             "run.assigned",
#             "run.started",
#             "function.started",
#             "function.failed",
#             "run.failed",
#         ],
#     )
