"""
Function Invocation Workflows - Test function calling patterns

Test Coverage:
✅ Function calls with different parameter counts
✅ Sequential function calls
✅ Parallel function patterns (data flow)
✅ Function result chaining
✅ Function state interaction

MCP Verification Points:
- Logs: Function invocation order and results
- Traces: Function spans within workflow spans
- Events: function.started, function.completed
"""

from agnt5 import workflow, WorkflowContext
from ..functions import (
    fn_01_no_params,
    fn_02_one_param,
    fn_03_two_params,
    fn_04_three_params,
)


@workflow
async def wf_06_different_param_counts(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Call functions with different parameter counts.

    Validates:
    - Functions with 0-3 parameters work correctly
    - Parameters are passed correctly
    - Return values captured correctly

    Returns:
        Results from all function calls

    MCP Verification Points:
        - Logs: Each function call with its parameters
        - Traces: Four function spans within workflow
    """
    ctx.logger.info("=== WF_06: Different Parameter Counts ===")

    # No parameters
    ctx.logger.info("Calling fn_01_no_params...")
    result0 = await ctx.task(fn_01_no_params)
    ctx.logger.info(f"Result 0 params: {result0}")

    # One parameter
    ctx.logger.info("Calling fn_02_one_param(value=7)...")
    result1 = await ctx.task(fn_02_one_param, value=7)
    ctx.logger.info(f"Result 1 param: {result1}")

    # Two parameters
    ctx.logger.info("Calling fn_03_two_params(a=10, b=5)...")
    result2 = await ctx.task(fn_03_two_params, a=10, b=5)
    ctx.logger.info(f"Result 2 params: {result2}")

    # Three parameters
    ctx.logger.info("Calling fn_04_three_params(x=2, y=3, z=4)...")
    result3 = await ctx.task(fn_04_three_params, x=2, y=3, z=4)
    ctx.logger.info(f"Result 3 params: {result3}")

    return {
        "no_params": result0,
        "one_param": result1,
        "two_params": result2,
        "three_params": result3,
        "total_calls": 4,
    }


@workflow
async def wf_07_sequential_function_calls(ctx: WorkflowContext, start_value: int = 5) -> dict:
    """
    Test Scenario: Sequential function calls with dependent values.

    Validates:
    - Functions execute in order
    - Output of one function feeds into next
    - State tracks progress through sequence

    Args:
        start_value: Starting value for computation chain

    Returns:
        Results of each step in the sequence

    MCP Verification Points:
        - Logs: Sequence of operations clearly logged
        - Traces: Sequential function spans
        - Entity: State updated after each step
    """
    ctx.logger.info("=== WF_07: Sequential Function Calls ===")
    ctx.logger.info(f"Starting with value: {start_value}")

    # Step 1: Double the value
    ctx.logger.info("Step 1: Double the value")
    result1 = await ctx.task(fn_02_one_param, value=start_value)
    doubled = result1["result"]
    ctx.state.set("step1_result", doubled)
    ctx.logger.info(f"After doubling: {doubled}")

    # Step 2: Add 10 to the doubled value
    ctx.logger.info("Step 2: Add 10 to doubled value")
    result2 = await ctx.task(fn_03_two_params, a=doubled, b=10)
    added = result2["result"]
    ctx.state.set("step2_result", added)
    ctx.logger.info(f"After adding 10: {added}")

    # Step 3: Use in three-parameter function
    ctx.logger.info("Step 3: Use in three-parameter calculation (x * y + z)")
    result3 = await ctx.task(fn_04_three_params, x=added, y=2, z=5)
    final = result3["result"]
    ctx.state.set("step3_result", final)
    ctx.logger.info(f"Final result: {final}")

    return {
        "start_value": start_value,
        "step1_doubled": doubled,
        "step2_added": added,
        "step3_final": final,
        "computation": f"((({start_value} * 2) + 10) * 2) + 5 = {final}",
    }


@workflow
async def wf_08_function_result_chaining(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Chain function results directly.

    Validates:
    - Function results can be used immediately
    - No explicit state needed for simple chains
    - Complex computations from simple functions

    Returns:
        Chained computation results

    MCP Verification Points:
        - Logs: Each computation step
        - Traces: Nested function invocations
    """
    ctx.logger.info("=== WF_08: Function Result Chaining ===")

    # Get initial values
    ctx.logger.info("Getting initial values...")
    val1 = await ctx.task(fn_02_one_param, value=5)  # 5 * 2 = 10
    val2 = await ctx.task(fn_02_one_param, value=3)  # 3 * 2 = 6

    ctx.logger.info(f"val1: {val1['result']}, val2: {val2['result']}")

    # Chain results
    ctx.logger.info("Chaining results...")
    sum_result = await ctx.task(fn_03_two_params, a=val1["result"], b=val2["result"])  # 10 + 6 = 16
    ctx.logger.info(f"Sum: {sum_result['result']}")

    # Use sum in final calculation
    ctx.logger.info("Final calculation...")
    final = await ctx.task(fn_04_three_params, x=sum_result["result"], y=2, z=8)  # 16 * 2 + 8 = 40
    ctx.logger.info(f"Final: {final['result']}")

    return {
        "val1": val1["result"],
        "val2": val2["result"],
        "sum": sum_result["result"],
        "final": final["result"],
        "formula": "(5*2 + 3*2) * 2 + 8 = 40",
    }


@workflow
async def wf_09_function_with_state(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Combine function calls with state management.

    Validates:
    - Functions and state work together
    - State can track function results
    - Computed state from multiple function calls

    Returns:
        Accumulated results with statistics

    MCP Verification Points:
        - Logs: State and function operations interleaved
        - Entity: State accumulation visible
    """
    ctx.logger.info("=== WF_09: Function with State ===")

    # Initialize state
    ctx.state.set("results", [])
    ctx.state.set("total", 0)
    ctx.state.set("count", 0)

    # Call functions and accumulate results
    values = [2, 5, 8, 3, 7]

    for idx, val in enumerate(values):
        ctx.logger.info(f"Processing value {idx + 1}/{len(values)}: {val}")

        # Call function
        result = await ctx.task(fn_02_one_param, value=val)
        doubled = result["result"]

        # Update state
        results = ctx.state.get("results")
        results.append(doubled)
        ctx.state.set("results", results)

        total = ctx.state.get("total")
        ctx.state.set("total", total + doubled)

        ctx.state.set("count", idx + 1)

        ctx.logger.info(f"Doubled: {doubled}, Running total: {ctx.state.get('total')}")

    # Compute statistics
    final_results = ctx.state.get("results")
    final_total = ctx.state.get("total")
    final_count = ctx.state.get("count")
    average = final_total / final_count if final_count > 0 else 0

    ctx.logger.info(f"Final statistics: total={final_total}, count={final_count}, avg={average}")

    return {
        "input_values": values,
        "doubled_values": final_results,
        "total": final_total,
        "count": final_count,
        "average": average,
    }


@workflow
async def wf_10_conditional_function_calls(ctx: WorkflowContext, operation: str, a: int, b: int) -> dict:
    """
    Test Scenario: Conditional function invocation based on parameters.

    Validates:
    - Different functions called based on conditions
    - Branch logic in workflows
    - Parameter-driven function selection

    Args:
        operation: Operation to perform (double, add, multiply)
        a: First operand
        b: Second operand

    Returns:
        Result of the selected operation

    MCP Verification Points:
        - Logs: Branch taken and function called
        - Traces: Only selected function appears in trace
    """
    ctx.logger.info("=== WF_10: Conditional Function Calls ===")
    ctx.logger.info(f"Operation: {operation}, a={a}, b={b}")

    result = None

    if operation == "double":
        ctx.logger.info("Branch: double (using fn_02_one_param)")
        result = await ctx.task(fn_02_one_param, value=a)
        operation_used = "fn_02_one_param"

    elif operation == "add":
        ctx.logger.info("Branch: add (using fn_03_two_params)")
        result = await ctx.task(fn_03_two_params, a=a, b=b)
        operation_used = "fn_03_two_params"

    elif operation == "multiply":
        ctx.logger.info("Branch: multiply (using fn_04_three_params with x * y + 0)")
        result = await ctx.task(fn_04_three_params, x=a, y=b, z=0)
        operation_used = "fn_04_three_params"

    else:
        ctx.logger.warning(f"Unknown operation: {operation}")
        return {
            "error": f"Unknown operation: {operation}",
            "valid_operations": ["double", "add", "multiply"],
        }

    ctx.logger.info(f"Result: {result}")

    return {
        "operation": operation,
        "operands": {"a": a, "b": b},
        "function_used": operation_used,
        "result": result.get("result"),
        "full_result": result,
    }


__all__ = [
    "wf_06_different_param_counts",
    "wf_07_sequential_function_calls",
    "wf_08_function_result_chaining",
    "wf_09_function_with_state",
    "wf_10_conditional_function_calls",
]
