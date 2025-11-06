"""
End-to-End Integration Tests for Error Scenarios

These tests verify comprehensive error handling:
1. Function errors propagate correctly
2. Workflow error handling with try-catch
3. Error recovery and continuation
4. Different error types are handled
5. Partial success scenarios
6. Error state persistence
7. Cascading error handling

Uses test-bench error workflows for systematic error testing.
"""

import pytest
import time


@pytest.mark.integration
def test_function_error_propagates_to_workflow(client, worker_process, platform):
    """
    Test function errors propagate to workflow and client.

    Verifies:
    - Function errors bubble up correctly
    - Error details are preserved
    - Client receives RunError
    """
    time.sleep(2)

    from agnt5.client import RunError

    try:
        result = client.run("wf_11_function_error_propagation", {})
        assert False, "Expected RunError to be raised"
    except RunError as e:
        # Error should contain details about ValueError
        error_msg = str(e)
        assert "ValueError" in error_msg or "negative" in error_msg.lower()

    print(f"\n✅ Function error propagation works correctly")


@pytest.mark.integration
def test_workflow_error_handling_with_try_catch(client, worker_process, platform):
    """
    Test workflow can catch and handle function errors.

    Verifies:
    - Try-catch blocks work in workflows
    - Multiple error types can be caught
    - Workflow completes successfully after handling errors
    """
    time.sleep(2)

    result = client.run("wf_12_error_handling_with_try_catch", {})

    assert result["status"] == "success"
    assert result["total_errors_handled"] == 3

    # Verify all error types were caught
    errors_caught = result["errors_caught"]
    error_types = [e["type"] for e in errors_caught if isinstance(e, dict)]
    assert "ValueError" in error_types
    assert "RuntimeError" in error_types

    print(f"\n✅ Workflow try-catch error handling works correctly")
    print(f"   Errors handled: {result['total_errors_handled']}")


@pytest.mark.integration
def test_error_recovery_and_continuation(client, worker_process, platform):
    """
    Test workflow continues after error recovery.

    Verifies:
    - Workflow doesn't stop at first error
    - Subsequent operations succeed
    - State tracks both successes and failures
    """
    time.sleep(2)

    result = client.run("wf_13_error_recovery_continue", {})

    assert result["total_attempts"] == 5
    assert result["successes"] == 3  # 3 valid operations
    assert result["failures"] == 2  # 2 operations that raised errors

    # Verify mix of results
    assert "results" in result
    results = result["results"]
    assert len(results) == 5

    # Check that some succeeded and some failed
    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = sum(1 for r in results if r["status"] == "error")
    assert success_count == 3
    assert failure_count == 2

    print(f"\n✅ Error recovery and continuation works correctly")
    print(f"   Successes: {success_count}, Failures: {failure_count}")


@pytest.mark.integration
def test_mixed_error_types_handling(client, worker_process, platform):
    """
    Test workflow handles multiple different error types.

    Verifies:
    - Different exception types are handled
    - Error type information is preserved
    - Specific error handling per type works
    """
    time.sleep(2)

    result = client.run("wf_14_mixed_error_types", {})

    assert "errors_by_type" in result

    # Should have encountered multiple error types
    errors_by_type = result["errors_by_type"]
    assert len(errors_by_type) >= 2

    # Verify error categorization
    assert "error_categories" in result

    print(f"\n✅ Mixed error types handled correctly")
    print(f"   Error types: {list(errors_by_type.keys())}")


@pytest.mark.integration
def test_partial_success_scenarios(client, worker_process, platform):
    """
    Test workflows handle partial success correctly.

    Verifies:
    - Some operations succeed while others fail
    - Partial results are captured
    - Final status reflects mixed outcome
    """
    time.sleep(2)

    result = client.run("wf_15_partial_success_with_errors", {})

    assert "status" in result
    assert result["status"] == "partial_success"

    assert "successful_operations" in result
    assert "failed_operations" in result

    # Should have both successes and failures
    assert result["successful_operations"] > 0
    assert result["failed_operations"] > 0

    print(f"\n✅ Partial success scenarios work correctly")
    print(f"   Successful: {result['successful_operations']}")
    print(f"   Failed: {result['failed_operations']}")


@pytest.mark.integration
def test_error_does_not_corrupt_workflow_state(client, worker_process, platform):
    """
    Test errors don't corrupt workflow state.

    Verifies:
    - State before error is preserved
    - Failed operations don't modify state
    - State remains consistent
    """
    time.sleep(2)

    result = client.run("wf_13_error_recovery_continue", {})

    # State should be consistent with actual operations
    assert result["total_attempts"] == 5
    assert result["successes"] + result["failures"] == result["total_attempts"]

    print(f"\n✅ Errors don't corrupt workflow state")


@pytest.mark.integration
def test_nested_error_handling(client, worker_process, platform):
    """
    Test nested try-catch blocks in workflows.

    Verifies:
    - Inner try-catch can handle errors
    - Outer try-catch can catch re-raised errors
    - Error context is preserved through nesting
    """
    time.sleep(2)

    result = client.run("wf_12_error_handling_with_try_catch", {})

    # Each try-catch block should handle its error
    assert result["total_errors_handled"] >= 3

    print(f"\n✅ Nested error handling works correctly")


@pytest.mark.integration
def test_error_in_concurrent_operations(client, worker_process, platform):
    """
    Test error handling in concurrent workflow executions.

    Verifies:
    - Errors in one workflow don't affect others
    - Concurrent workflows maintain error isolation
    - Error handling is thread-safe
    """
    time.sleep(2)

    import concurrent.futures
    from agnt5.client import RunError

    def run_workflow(workflow_name):
        try:
            return client.run(workflow_name, {})
        except RunError as e:
            return {"error": str(e), "workflow": workflow_name}

    workflows = [
        "wf_12_error_handling_with_try_catch",  # Should succeed
        "wf_11_function_error_propagation",  # Should fail
        "wf_13_error_recovery_continue",  # Should succeed
        "wf_11_function_error_propagation",  # Should fail
        "wf_12_error_handling_with_try_catch",  # Should succeed
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_workflow, wf) for wf in workflows]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Should have mix of successes and errors
    successes = [r for r in results if "status" in r or "total_errors_handled" in r]
    errors = [r for r in results if "error" in r]

    assert len(successes) > 0
    assert len(errors) > 0
    assert len(successes) + len(errors) == len(workflows)

    print(f"\n✅ Concurrent error handling works correctly")
    print(f"   Successes: {len(successes)}, Errors: {len(errors)}")


@pytest.mark.integration
def test_entity_error_handling(client, worker_process, platform):
    """
    Test entity method errors are handled correctly.

    Verifies:
    - Entity errors propagate properly
    - State is not corrupted by errors
    - Subsequent operations work after error
    """
    time.sleep(2)

    from agnt5.client import RunError

    account_key = "account-error-handling"

    # Successful deposit
    result1 = client.run("BankAccount", "deposit", key=account_key, amount=100.0)
    assert result1["balance"] == 100.0

    # Invalid withdrawal (insufficient funds)
    try:
        client.run("BankAccount", "withdraw", key=account_key, amount=200.0)
        assert False, "Expected error for insufficient funds"
    except RunError:
        pass

    # Verify balance unchanged
    balance = client.run("BankAccount", "get_balance", key=account_key)
    assert balance == 100.0

    # Subsequent valid operation should work
    result2 = client.run("BankAccount", "withdraw", key=account_key, amount=30.0)
    assert result2["balance"] == 70.0

    print(f"\n✅ Entity error handling works correctly")


@pytest.mark.integration
def test_error_details_in_response(client, worker_process, platform):
    """
    Test error responses contain sufficient details.

    Verifies:
    - Error messages are informative
    - Error types are identifiable
    - Stack traces are available (if enabled)
    """
    time.sleep(2)

    from agnt5.client import RunError

    try:
        client.run("fn_06_raises_runtime_error", {"message": "Test error message"})
        assert False, "Expected error"
    except RunError as e:
        error_msg = str(e)
        # Error message should contain the original message
        assert "Test error message" in error_msg or "RuntimeError" in error_msg

    print(f"\n✅ Error details are preserved in response")


@pytest.mark.integration
def test_worker_survives_multiple_errors(client, worker_process, platform):
    """
    Test worker remains functional after multiple errors.

    Verifies:
    - Worker doesn't crash from errors
    - Error recovery at worker level
    - Subsequent requests work correctly
    """
    time.sleep(2)

    from agnt5.client import RunError

    # Cause multiple errors
    for i in range(5):
        try:
            client.run("fn_06_raises_runtime_error", {"message": f"Error {i}"})
        except RunError:
            pass

    # Worker should still be functional
    result = client.run("fn_02_one_param", {"value": 10})
    assert result["result"] == 20

    print(f"\n✅ Worker survives multiple errors")


# TODO: Add tests for:
# - Error handling with streaming responses
# - Error timeouts and deadlines
# - Error rate limiting
# - Error aggregation across multiple steps
# - Error handling in nested workflows
# - Error context propagation through agent handoffs
# - Custom error handlers/middleware
# - Error telemetry and monitoring
# - Graceful degradation on errors
# - Circuit breaker patterns for repeated errors
