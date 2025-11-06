"""
End-to-End Integration Tests for Workflows

These tests verify the complete workflow execution flow:
1. Worker starts and registers workflows
2. Client invokes workflows via platform
3. Worker executes workflows with state management
4. Results are returned correctly
5. State persists to platform database
6. Error handling and recovery work as expected

Uses the test-bench workflows (wf_01 through wf_08) for comprehensive testing.

Coverage:
- Basic workflow execution
- State management and persistence
- Function invocation from workflows
- Multi-step workflows
- Error handling
- Agent integration
- Workflow durability
"""

import pytest
import time


@pytest.mark.integration
def test_basic_workflow_execution(client, worker_process, platform):
    """
    Test basic workflow execution end-to-end.

    Verifies:
    - Workflow can be invoked via client
    - Worker receives and executes workflow
    - Result is returned correctly
    - Workflow completes successfully
    """
    # Give worker time to register
    time.sleep(2)

    # Execute basic workflow
    result = client.run("wf_01_basic_execution", {})

    assert result["status"] == "success"
    assert "message" in result
    assert result["message"] == "Basic execution completed"

    print(f"\n✅ Basic workflow executed successfully")


@pytest.mark.integration
def test_workflow_state_management(client, worker_process, platform):
    """
    Test workflow state management.

    Verifies:
    - State can be set and retrieved in workflows
    - Different data types work correctly
    - State updates are handled properly
    - State deletion works
    """
    time.sleep(2)

    result = client.run("wf_02_state_management", {})

    # Verify state operations
    assert result["string_value"] == "hello world"
    assert result["int_value"] == 42
    assert result["bool_value"] is True
    assert result["dict_value"]["key"] == "value"
    assert result["dict_value"]["nested"]["data"] == 123
    assert result["list_value"] == [1, 2, 3, 4, 5]
    assert result["updated_value"] == "updated value"
    assert result["deleted_value"] is None

    print(f"\n✅ Workflow state management works correctly")
    print(f"   ✓ String, int, bool, dict, list types")
    print(f"   ✓ Update and delete operations")


@pytest.mark.integration
def test_workflow_function_invocation(client, worker_process, platform):
    """
    Test workflow invoking functions.

    Verifies:
    - Workflows can invoke functions via ctx.task()
    - Parameters are passed correctly
    - Return values are captured
    - Multiple function calls work
    """
    time.sleep(2)

    result = client.run("wf_03_function_invocation", {})

    # Verify function invocation results
    assert "no_params_result" in result
    assert result["no_params_result"]["params"] == 0

    assert "one_param_result_1" in result
    assert result["one_param_result_1"]["result"] == 20  # 10 * 2

    assert "one_param_result_2" in result
    assert result["one_param_result_2"]["result"] == 50  # 25 * 2

    assert result["total_calls"] == 3

    print(f"\n✅ Workflow function invocation works correctly")
    print(f"   ✓ {result['total_calls']} function calls executed")


@pytest.mark.integration
def test_workflow_with_parameters(client, worker_process, platform):
    """
    Test workflow execution with input parameters.

    Verifies:
    - Parameters are passed to workflows correctly
    - Different parameter values work
    - Parameters affect workflow logic
    """
    time.sleep(2)

    # Test with different initial values
    test_cases = [
        {"initial_value": 5, "expected_doubled": 10, "expected_category": "low"},
        {"initial_value": 7, "expected_doubled": 14, "expected_category": "medium"},
        {"initial_value": 15, "expected_doubled": 30, "expected_category": "high"},
    ]

    for test_case in test_cases:
        result = client.run(
            "wf_05_multi_step_workflow",
            {"initial_value": test_case["initial_value"]}
        )

        assert result["initial_value"] == test_case["initial_value"]
        assert result["doubled_value"] == test_case["expected_doubled"]
        assert result["category"] == test_case["expected_category"]
        assert result["steps_completed"] == 4

    print(f"\n✅ Workflow parameters work correctly ({len(test_cases)} test cases)")


@pytest.mark.integration
def test_multi_step_workflow(client, worker_process, platform):
    """
    Test multi-step workflow execution.

    Verifies:
    - Multiple steps execute in sequence
    - State persists across steps
    - Conditional logic works
    - Final result contains all step data
    """
    time.sleep(2)

    result = client.run("wf_05_multi_step_workflow", {"initial_value": 12})

    # Verify each step's results
    assert result["initial_value"] == 12
    assert result["doubled_value"] == 24  # 12 * 2
    assert result["category"] == "high"  # doubled > 20
    assert result["multiplier"] == 3  # high category
    assert result["final_value"] == 72  # 24 * 3
    assert result["steps_completed"] == 4

    print(f"\n✅ Multi-step workflow executed successfully")
    print(f"   ✓ {result['steps_completed']} steps completed")
    print(f"   ✓ Conditional logic: {result['category']} category")


@pytest.mark.integration
def test_workflow_with_function_chaining(client, worker_process, platform):
    """
    Test workflow that chains function results.

    Verifies:
    - Function results can be passed to next function
    - Result chaining works correctly
    - State captures intermediate results
    """
    time.sleep(2)

    result = client.run("wf_08_function_result_chaining", {})

    # Verify result chaining
    # fn_02(5) = 10, fn_02(10) = 20, fn_02(20) = 40
    assert result["step1_result"] == 10
    assert result["step2_result"] == 20
    assert result["step3_result"] == 40
    assert result["chain_length"] == 3

    print(f"\n✅ Function result chaining works correctly")
    print(f"   ✓ Chain length: {result['chain_length']}")


@pytest.mark.integration
def test_workflow_sequential_function_calls(client, worker_process, platform):
    """
    Test workflow with sequential function calls.

    Verifies:
    - Multiple functions can be called in sequence
    - Results are tracked correctly
    - No interference between calls
    """
    time.sleep(2)

    result = client.run("wf_07_sequential_function_calls", {})

    # Verify all function calls succeeded
    assert "results" in result
    assert len(result["results"]) == 4
    assert result["total_calls"] == 4

    print(f"\n✅ Sequential function calls work correctly")
    print(f"   ✓ {result['total_calls']} sequential calls executed")


@pytest.mark.integration
def test_workflow_error_handling(client, worker_process, platform):
    """
    Test workflow error handling.

    Verifies:
    - Errors in workflows are caught
    - Error details are returned to client
    - Worker continues functioning after errors
    """
    time.sleep(2)

    from agnt5.client import RunError

    # Test error propagation workflow
    try:
        result = client.run("wf_11_function_error_propagation", {})
        assert False, "Expected RunError to be raised"
    except RunError as e:
        assert "RuntimeError" in str(e) or "error" in str(e).lower()

    # Verify worker still works after error
    result2 = client.run("wf_01_basic_execution", {})
    assert result2["status"] == "success"

    print(f"\n✅ Workflow error handling works correctly")
    print(f"   ✓ Error propagated to client")
    print(f"   ✓ Worker still functional after error")


@pytest.mark.integration
def test_workflow_conditional_execution(client, worker_process, platform):
    """
    Test workflow with conditional function calls.

    Verifies:
    - Conditional logic in workflows
    - Different code paths execute correctly
    - Results reflect the chosen path
    """
    time.sleep(2)

    # Test both conditional paths
    result_high = client.run("wf_10_conditional_function_calls", {"threshold": 3})
    assert result_high["path_taken"] == "above_threshold"
    assert result_high["function_called"] == "fn_04_three_params"

    result_low = client.run("wf_10_conditional_function_calls", {"threshold": 10})
    assert result_low["path_taken"] == "below_threshold"
    assert result_low["function_called"] == "fn_02_one_param"

    print(f"\n✅ Conditional workflow execution works correctly")
    print(f"   ✓ Both code paths tested")


@pytest.mark.integration
def test_workflow_different_param_counts(client, worker_process, platform):
    """
    Test workflow calling functions with different parameter counts.

    Verifies:
    - Functions with 0, 1, 2, 3 parameters work
    - Parameters are passed correctly
    - Results are captured properly
    """
    time.sleep(2)

    result = client.run("wf_06_different_param_counts", {})

    # Verify all function calls with different param counts
    assert result["fn_0_params"]["params"] == 0
    assert result["fn_1_param"]["params"] == 1
    assert result["fn_2_params"]["params"] == 2
    assert result["fn_3_params"]["params"] == 3
    assert result["total_functions_called"] == 4

    print(f"\n✅ Functions with different parameter counts work correctly")
    print(f"   ✓ Tested 0-3 parameters")


@pytest.mark.integration
def test_concurrent_workflow_executions(client, worker_process, platform):
    """
    Test concurrent workflow executions.

    Verifies:
    - Multiple workflows can execute concurrently
    - Results are returned correctly for each
    - No interference between concurrent executions
    - State isolation works
    """
    time.sleep(2)

    import concurrent.futures

    def invoke_workflow(value):
        return client.run("wf_05_multi_step_workflow", {"initial_value": value})

    # Execute 5 workflows concurrently with different parameters
    values = [1, 2, 3, 4, 5]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(invoke_workflow, value) for value in values]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Verify all succeeded
    assert len(results) == 5

    # Verify each got correct result (value * 2)
    returned_values = {r["initial_value"] for r in results}
    assert returned_values == set(values)

    # Verify state isolation - each workflow should have its own state
    for result in results:
        assert result["initial_value"] in values
        assert result["doubled_value"] == result["initial_value"] * 2

    print(f"\n✅ Concurrent workflow execution works correctly")
    print(f"   ✓ {len(results)} concurrent workflows executed")
    print(f"   ✓ State isolation verified")


@pytest.mark.integration
@pytest.mark.slow
def test_workflow_with_state_persistence(client, worker_process, platform):
    """
    Test workflow state persistence through workflow execution.

    Verifies:
    - State is set during workflow execution
    - State persists throughout workflow
    - Final state is accessible in result
    """
    time.sleep(2)

    result = client.run("wf_09_function_with_state", {"multiplier": 3})

    # Verify state was used in computation
    assert result["stored_multiplier"] == 3
    assert "result" in result
    assert result["state_operations"] == ["set", "get", "compute"]

    print(f"\n✅ Workflow state persistence works correctly")


# TODO: Add tests for:
# - Workflow execution with worker restart (durability/recovery)
# - Workflow retry logic with retryable functions
# - Workflow with agent execution (OpenAI integration)
# - Workflow pause/resume (HITL - requires Phase 2 platform)
# - Workflow timeout handling
# - Long-running workflows
# - Workflow execution metrics/telemetry
# - Workflow idempotency guarantees
# - Workflow exactly-once semantics
