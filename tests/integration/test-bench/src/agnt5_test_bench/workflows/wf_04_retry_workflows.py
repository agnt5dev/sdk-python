"""
Retry Workflows - Test retry behavior and exhaustion

Test Coverage:
✅ Retry exhaustion scenarios
✅ Eventual success after retries
✅ Retry count tracking
✅ Multiple retry configurations
✅ State persistence across retries

MCP Verification Points:
- Logs: Retry attempts logged with attempt numbers
- Traces: Multiple function attempt spans
- Events: function.retry, function.retry_exhausted
"""

from agnt5 import workflow, WorkflowContext
from ..functions import (
    fn_02_one_param,
    fn_08_retry_succeeds_on_attempt_2,
    fn_09_retry_exhausted,
)


@workflow
async def wf_16_retry_eventual_success(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Function succeeds after retries.

    Validates:
    - Function retries on failure
    - Eventually succeeds after configured retries
    - Retry count tracked correctly

    Returns:
        Success with retry information

    MCP Verification Points:
        - Logs: Multiple attempt logs followed by success
        - Traces: Multiple function attempt spans
        - Events: function.retry events then function.completed
    """
    ctx.logger.info("=== WF_16: Retry Eventual Success ===")
    ctx.logger.info("Calling function that fails once then succeeds...")

    # This function fails on first attempt, succeeds on second
    result = await ctx.task(fn_08_retry_succeeds_on_attempt_2)

    ctx.logger.info(f"✓ Function succeeded: {result}")

    return {
        "status": "success",
        "result": result,
        "expected_attempts": 2,
    }


@workflow
async def wf_17_retry_exhaustion(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Function exhausts all retries and fails.

    Validates:
    - All retry attempts are made
    - Error propagates after exhaustion
    - Retry count matches configuration

    Returns:
        This workflow should raise RuntimeError after retries exhausted

    MCP Verification Points:
        - Logs: All retry attempts logged
        - Traces: All attempt spans visible
        - Events: Multiple function.retry then function.failed
    """
    ctx.logger.info("=== WF_17: Retry Exhaustion ===")
    ctx.logger.info("Calling function that always fails (will exhaust retries)...")

    # This function always fails - should retry max_attempts times
    result = await ctx.task(fn_09_retry_exhausted)

    # Should not reach here
    ctx.logger.error("Should not reach this line!")
    return {"unexpected": "Should have thrown error"}


@workflow
async def wf_18_retry_with_error_handling(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Handle retry exhaustion with try-catch.

    Validates:
    - Retry exhaustion can be caught
    - Error contains retry information
    - Workflow continues after handling

    Returns:
        Success with error information

    MCP Verification Points:
        - Logs: Retry attempts then error caught
        - Events: Workflow completes despite function failure
    """
    ctx.logger.info("=== WF_18: Retry with Error Handling ===")

    retry_info = {}

    # Test 1: Eventual success
    ctx.logger.info("Test 1: Function with eventual success...")
    try:
        result = await ctx.task(fn_08_retry_succeeds_on_attempt_2)
        ctx.logger.info(f"✓ Success: {result}")
        retry_info["test1"] = {"status": "success", "result": result}
    except Exception as e:
        ctx.logger.error(f"✗ Unexpected error: {e}")
        retry_info["test1"] = {"status": "error", "error": str(e)}

    # Test 2: Retry exhaustion
    ctx.logger.info("Test 2: Function that exhausts retries...")
    try:
        result = await ctx.task(fn_09_retry_exhausted)
        ctx.logger.error("Should not succeed")
        retry_info["test2"] = {"status": "unexpected_success", "result": result}
    except RuntimeError as e:
        ctx.logger.info(f"✓ Caught retry exhaustion error: {e}")
        retry_info["test2"] = {"status": "retry_exhausted", "error": str(e)}

    ctx.logger.info("Both retry scenarios handled")

    return {
        "status": "completed",
        "retry_scenarios_tested": 2,
        "results": retry_info,
    }


@workflow
async def wf_19_multiple_retry_operations(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Multiple operations with different retry behaviors.

    Validates:
    - Multiple retrying functions in sequence
    - State tracking across retries
    - Mixed success and failure scenarios

    Returns:
        Results from multiple retry operations

    MCP Verification Points:
        - Logs: Multiple retry sequences
        - Traces: Multiple function retry spans
        - Entity: State updated between retry operations
    """
    ctx.logger.info("=== WF_19: Multiple Retry Operations ===")

    ctx.state.set("operations", [])
    ctx.state.set("total_attempts", 0)

    # Operation 1: Function with retries (succeeds)
    ctx.logger.info("Operation 1: Retry with eventual success")
    try:
        result1 = await ctx.task(fn_08_retry_succeeds_on_attempt_2)
        ctx.logger.info(f"✓ Op1 success: {result1}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 1, "status": "success", "result": result1})
        ctx.state.set("operations", ops)
    except Exception as e:
        ctx.logger.error(f"✗ Op1 error: {e}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 1, "status": "error", "error": str(e)})
        ctx.state.set("operations", ops)

    # Operation 2: Regular function (no retries needed)
    ctx.logger.info("Operation 2: Regular function (no retry needed)")
    try:
        result2 = await ctx.task(fn_02_one_param, value=15)
        ctx.logger.info(f"✓ Op2 success: {result2}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 2, "status": "success", "result": result2})
        ctx.state.set("operations", ops)
    except Exception as e:
        ctx.logger.error(f"✗ Op2 error: {e}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 2, "status": "error", "error": str(e)})
        ctx.state.set("operations", ops)

    # Operation 3: Function that exhausts retries
    ctx.logger.info("Operation 3: Retry exhaustion")
    try:
        result3 = await ctx.task(fn_09_retry_exhausted)
        ctx.logger.error("Should not succeed")

        ops = ctx.state.get("operations")
        ops.append({"operation": 3, "status": "unexpected_success", "result": result3})
        ctx.state.set("operations", ops)
    except RuntimeError as e:
        ctx.logger.info(f"✓ Op3 exhausted retries: {e}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 3, "status": "retry_exhausted", "error": str(e)})
        ctx.state.set("operations", ops)

    # Operation 4: Another retry with eventual success
    ctx.logger.info("Operation 4: Another retry success")
    try:
        result4 = await ctx.task(fn_08_retry_succeeds_on_attempt_2)
        ctx.logger.info(f"✓ Op4 success: {result4}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 4, "status": "success", "result": result4})
        ctx.state.set("operations", ops)
    except Exception as e:
        ctx.logger.error(f"✗ Op4 error: {e}")

        ops = ctx.state.get("operations")
        ops.append({"operation": 4, "status": "error", "error": str(e)})
        ctx.state.set("operations", ops)

    operations = ctx.state.get("operations")
    success_count = sum(1 for op in operations if op["status"] == "success")
    error_count = sum(1 for op in operations if op["status"] in ["error", "retry_exhausted"])

    ctx.logger.info(f"Completed: {success_count} successes, {error_count} failures")

    return {
        "total_operations": len(operations),
        "successes": success_count,
        "failures": error_count,
        "operations": operations,
    }


@workflow
async def wf_20_retry_state_persistence(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Verify state persists across retry attempts.

    Validates:
    - State remains consistent during retries
    - State can be updated between attempts
    - Workflow state independent of function retries

    Returns:
        State tracking through retry operations

    MCP Verification Points:
        - Logs: State values logged before/after retries
        - Entity: State persistence visible across attempts
    """
    ctx.logger.info("=== WF_20: Retry State Persistence ===")

    # Set initial state
    ctx.state.set("before_retry", "initial_value")
    ctx.state.set("retry_operations", [])

    ctx.logger.info(f"Initial state: {ctx.state.get('before_retry')}")

    # Operation with retries
    ctx.logger.info("Executing operation with retries...")
    try:
        result = await ctx.task(fn_08_retry_succeeds_on_attempt_2)
        ctx.logger.info(f"Operation completed: {result}")

        # Update state after successful retry
        ctx.state.set("after_retry", "retry_succeeded")

        retry_ops = ctx.state.get("retry_operations")
        retry_ops.append({
            "function": "fn_08_retry_succeeds_on_attempt_2",
            "status": "success",
            "attempts": result.get("attempts"),
        })
        ctx.state.set("retry_operations", retry_ops)

    except Exception as e:
        ctx.logger.error(f"Operation failed: {e}")
        ctx.state.set("after_retry", "retry_failed")

    # Verify state persisted
    before = ctx.state.get("before_retry")
    after = ctx.state.get("after_retry")
    retry_ops = ctx.state.get("retry_operations")

    ctx.logger.info(f"State before: {before}")
    ctx.logger.info(f"State after: {after}")
    ctx.logger.info(f"Retry operations: {retry_ops}")

    return {
        "state_before_retry": before,
        "state_after_retry": after,
        "retry_operations": retry_ops,
        "state_persisted": before is not None and after is not None,
    }


__all__ = [
    "wf_16_retry_eventual_success",
    "wf_17_retry_exhaustion",
    "wf_18_retry_with_error_handling",
    "wf_19_multiple_retry_operations",
    "wf_20_retry_state_persistence",
]
