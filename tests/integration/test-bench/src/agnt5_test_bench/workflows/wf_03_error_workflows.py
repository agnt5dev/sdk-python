"""
Error Handling Workflows - Test error propagation and recovery

Test Coverage:
✅ Function error propagation through workflows
✅ Try-catch error handling
✅ Different error types (ValueError, RuntimeError, Custom)
✅ Error recovery and continuation
✅ Partial success scenarios

MCP Verification Points:
- Logs: Error messages, stack traces, recovery attempts
- Traces: Error spans with exception attributes
- Events: error.occurred, error.handled, workflow.failed
"""

from agnt5 import workflow, WorkflowContext
from ..functions import (
    fn_02_one_param,
    fn_05_raises_value_error,
    fn_06_raises_runtime_error,
    fn_07_raises_custom_error,
)


@workflow
async def wf_11_function_error_propagation(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Function errors propagate to workflow.

    Validates:
    - Function errors bubble up to workflow
    - Error details are preserved
    - Workflow fails with correct error

    Returns:
        This workflow should raise ValueError

    MCP Verification Points:
        - Logs: Error logged before propagation
        - Traces: Error span with exception details
        - Events: function.failed, workflow.failed
    """
    ctx.logger.info("=== WF_11: Function Error Propagation ===")
    ctx.logger.info("Calling function that will raise ValueError...")

    # This should raise ValueError and propagate to workflow
    result = await ctx.task(fn_05_raises_value_error, value=-5)

    # Should not reach here
    ctx.logger.error("Should not reach this line!")
    return {"unexpected": "This should not be returned"}


@workflow
async def wf_12_error_handling_with_try_catch(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Handle function errors with try-catch.

    Validates:
    - Errors can be caught and handled
    - Workflow continues after error handling
    - Error details accessible in catch block

    Returns:
        Success with error information

    MCP Verification Points:
        - Logs: Error caught and handled
        - Traces: Workflow span shows recovery
        - Events: error.occurred but workflow.completed
    """
    ctx.logger.info("=== WF_12: Error Handling with Try-Catch ===")

    errors_caught = []

    # Test 1: Handle ValueError
    ctx.logger.info("Test 1: Catching ValueError...")
    try:
        await ctx.task(fn_05_raises_value_error, value=-10)
        errors_caught.append("none")
    except ValueError as e:
        ctx.logger.info(f"✓ Caught ValueError: {e}")
        errors_caught.append({"type": "ValueError", "message": str(e)})

    # Test 2: Handle RuntimeError
    ctx.logger.info("Test 2: Catching RuntimeError...")
    try:
        await ctx.task(fn_06_raises_runtime_error, message="Test error message")
        errors_caught.append("none")
    except RuntimeError as e:
        ctx.logger.info(f"✓ Caught RuntimeError: {e}")
        errors_caught.append({"type": "RuntimeError", "message": str(e)})

    # Test 3: Handle custom error
    ctx.logger.info("Test 3: Catching custom error...")
    try:
        await ctx.task(fn_07_raises_custom_error)
        errors_caught.append("none")
    except Exception as e:
        ctx.logger.info(f"✓ Caught {type(e).__name__}: {e}")
        errors_caught.append({"type": type(e).__name__, "message": str(e)})

    ctx.logger.info(f"Successfully handled {len(errors_caught)} errors")

    return {
        "status": "success",
        "errors_caught": errors_caught,
        "total_errors_handled": len(errors_caught),
    }


@workflow
async def wf_13_error_recovery_continue(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Continue execution after handling errors.

    Validates:
    - Workflow continues after error recovery
    - Subsequent operations succeed
    - State updated correctly through error scenarios

    Returns:
        Results including both errors and successes

    MCP Verification Points:
        - Logs: Mix of error handling and successful operations
        - Entity: State reflects partial completion
    """
    ctx.logger.info("=== WF_13: Error Recovery and Continue ===")

    results = []
    ctx.state.set("total_attempts", 0)
    ctx.state.set("successes", 0)
    ctx.state.set("failures", 0)

    # Mix of valid and invalid operations
    operations = [
        {"value": 5, "should_succeed": True},
        {"value": -3, "should_succeed": False},  # Negative value -> error
        {"value": 10, "should_succeed": True},
        {"value": -1, "should_succeed": False},  # Negative value -> error
        {"value": 7, "should_succeed": True},
    ]

    for idx, op in enumerate(operations):
        ctx.logger.info(f"Operation {idx + 1}/{len(operations)}: value={op['value']}")
        ctx.state.set("total_attempts", idx + 1)

        try:
            result = await ctx.task(fn_05_raises_value_error, value=op["value"])
            ctx.logger.info(f"✓ Success: {result}")
            results.append({"operation": idx + 1, "status": "success", "result": result})
            ctx.state.set("successes", ctx.state.get("successes") + 1)
        except ValueError as e:
            ctx.logger.info(f"✗ Error handled: {e}")
            results.append({"operation": idx + 1, "status": "error", "error": str(e)})
            ctx.state.set("failures", ctx.state.get("failures") + 1)

    successes = ctx.state.get("successes")
    failures = ctx.state.get("failures")

    ctx.logger.info(f"Completed: {successes} successes, {failures} failures")

    return {
        "status": "completed_with_errors",
        "total_attempts": len(operations),
        "successes": successes,
        "failures": failures,
        "results": results,
    }


@workflow
async def wf_14_mixed_error_types(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Handle different error types in single workflow.

    Validates:
    - Multiple error types caught correctly
    - Type-specific error handling
    - Workflow completes despite multiple errors

    Returns:
        Summary of error handling by type

    MCP Verification Points:
        - Logs: Different error types logged
        - Traces: Multiple error spans with different types
    """
    ctx.logger.info("=== WF_14: Mixed Error Types ===")

    error_summary = {
        "ValueError": 0,
        "RuntimeError": 0,
        "CustomError": 0,
        "Unknown": 0,
    }

    # Test ValueError
    ctx.logger.info("Test 1: ValueError")
    try:
        await ctx.task(fn_05_raises_value_error, value=-1)
    except ValueError as e:
        ctx.logger.info(f"Caught ValueError: {e}")
        error_summary["ValueError"] += 1
    except Exception as e:
        error_summary["Unknown"] += 1

    # Test RuntimeError
    ctx.logger.info("Test 2: RuntimeError")
    try:
        await ctx.task(fn_06_raises_runtime_error, message="Intentional error")
    except RuntimeError as e:
        ctx.logger.info(f"Caught RuntimeError: {e}")
        error_summary["RuntimeError"] += 1
    except Exception as e:
        error_summary["Unknown"] += 1

    # Test CustomError
    ctx.logger.info("Test 3: CustomError")
    try:
        await ctx.task(fn_07_raises_custom_error)
    except Exception as e:
        ctx.logger.info(f"Caught {type(e).__name__}: {e}")
        if "Custom" in type(e).__name__:
            error_summary["CustomError"] += 1
        else:
            error_summary["Unknown"] += 1

    # Test another ValueError
    ctx.logger.info("Test 4: Another ValueError")
    try:
        await ctx.task(fn_05_raises_value_error, value=-99)
    except ValueError as e:
        ctx.logger.info(f"Caught ValueError: {e}")
        error_summary["ValueError"] += 1
    except Exception as e:
        error_summary["Unknown"] += 1

    ctx.logger.info(f"Error summary: {error_summary}")

    return {
        "status": "completed",
        "error_types_handled": error_summary,
        "total_errors": sum(error_summary.values()),
    }


@workflow
async def wf_15_partial_success_with_errors(ctx: WorkflowContext, threshold: int = 50) -> dict:
    """
    Test Scenario: Workflow with some operations succeeding and some failing.

    Validates:
    - Partial success scenarios
    - State tracking through mixed results
    - Summary of successes vs failures

    Args:
        threshold: Value threshold for validation

    Returns:
        Partial results with success/failure counts

    MCP Verification Points:
        - Logs: Mixed success and error messages
        - Entity: State shows partial progress
        - Events: Both success and error events
    """
    ctx.logger.info("=== WF_15: Partial Success with Errors ===")
    ctx.logger.info(f"Threshold: {threshold}")

    ctx.state.set("processed", [])
    ctx.state.set("success_count", 0)
    ctx.state.set("error_count", 0)

    # Process various values
    values = [100, -5, 75, -10, 30, 120, -3]

    for idx, value in enumerate(values):
        ctx.logger.info(f"Processing {idx + 1}/{len(values)}: {value}")

        try:
            # Try to process value
            result = await ctx.task(fn_05_raises_value_error, value=value)

            # Check if result meets threshold
            squared = result["squared"]
            if squared >= threshold:
                ctx.logger.info(f"✓ Value {value} squared to {squared} (meets threshold)")
                status = "success"
                ctx.state.set("success_count", ctx.state.get("success_count") + 1)
            else:
                ctx.logger.info(f"⚠ Value {value} squared to {squared} (below threshold)")
                status = "below_threshold"

            processed = ctx.state.get("processed")
            processed.append({"value": value, "squared": squared, "status": status})
            ctx.state.set("processed", processed)

        except ValueError as e:
            ctx.logger.info(f"✗ Error processing {value}: {e}")
            ctx.state.set("error_count", ctx.state.get("error_count") + 1)

            processed = ctx.state.get("processed")
            processed.append({"value": value, "status": "error", "error": str(e)})
            ctx.state.set("processed", processed)

    success_count = ctx.state.get("success_count")
    error_count = ctx.state.get("error_count")
    processed = ctx.state.get("processed")

    ctx.logger.info(f"Final: {success_count} successes, {error_count} errors")

    return {
        "threshold": threshold,
        "total_processed": len(values),
        "success_count": success_count,
        "error_count": error_count,
        "results": processed,
    }


__all__ = [
    "wf_11_function_error_propagation",
    "wf_12_error_handling_with_try_catch",
    "wf_13_error_recovery_continue",
    "wf_14_mixed_error_types",
    "wf_15_partial_success_with_errors",
]
