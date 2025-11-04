"""
Simple Workflows - Basic workflow execution patterns

Test Coverage:
✅ Basic workflow execution
✅ State management (set/get/delete)
✅ Simple agent integration
✅ Function invocation from workflows
✅ Return value handling

MCP Verification Points:
- Logs: Workflow execution steps, state operations
- Traces: Workflow spans, function call spans
- Events: workflow.started, workflow.completed, state.updated
- Entity: Workflow state persistence
"""

from agnt5 import workflow, WorkflowContext, Agent
from ..functions import fn_01_no_params, fn_02_one_param


@workflow
async def wf_01_basic_execution(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Basic workflow execution with no dependencies.

    Validates:
    - Workflow starts and completes successfully
    - Logging works correctly
    - Simple return values

    Returns:
        Success result dictionary

    MCP Verification Points:
        - Logs: "wf_01_basic_execution started", "wf_01_basic_execution completed"
        - Traces: Single workflow span
        - Events: workflow.started, workflow.completed
    """
    ctx.logger.info("=== WF_01: Basic Execution ===")
    ctx.logger.info("Executing basic workflow with no dependencies")

    result = {"status": "success", "message": "Basic execution completed"}

    ctx.logger.info(f"Result: {result}")
    return result


@workflow
async def wf_02_state_management(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Workflow state get/set/delete operations.

    Validates:
    - State can be set and retrieved
    - State persists within workflow
    - State can be deleted
    - Different data types work correctly

    Returns:
        State operation results

    MCP Verification Points:
        - Logs: State operations and values
        - Entity: State keys in workflow entity
        - Events: state.set, state.get, state.delete events
    """
    ctx.logger.info("=== WF_02: State Management ===")

    # Set various types of state
    ctx.logger.info("Setting state values...")
    ctx.state.set("string_value", "hello world")
    ctx.state.set("int_value", 42)
    ctx.state.set("float_value", 3.14)
    ctx.state.set("bool_value", True)
    ctx.state.set("dict_value", {"key": "value", "nested": {"data": 123}})
    ctx.state.set("list_value", [1, 2, 3, 4, 5])

    # Get and verify state
    ctx.logger.info("Retrieving state values...")
    string_val = ctx.state.get("string_value")
    int_val = ctx.state.get("int_value")
    float_val = ctx.state.get("float_value")
    bool_val = ctx.state.get("bool_value")
    dict_val = ctx.state.get("dict_value")
    list_val = ctx.state.get("list_value")

    ctx.logger.info(f"String: {string_val}")
    ctx.logger.info(f"Int: {int_val}")
    ctx.logger.info(f"Float: {float_val}")
    ctx.logger.info(f"Bool: {bool_val}")
    ctx.logger.info(f"Dict: {dict_val}")
    ctx.logger.info(f"List: {list_val}")

    # Test state update
    ctx.logger.info("Updating state value...")
    ctx.state.set("string_value", "updated value")
    updated_val = ctx.state.get("string_value")
    ctx.logger.info(f"Updated string: {updated_val}")

    # Test state deletion
    ctx.logger.info("Deleting state value...")
    ctx.state.delete("float_value")
    deleted_val = ctx.state.get("float_value")  # Should be None
    ctx.logger.info(f"Deleted float (should be None): {deleted_val}")

    return {
        "string_value": string_val,
        "int_value": int_val,
        "bool_value": bool_val,
        "dict_value": dict_val,
        "list_value": list_val,
        "updated_value": updated_val,
        "deleted_value": deleted_val,
    }


@workflow
async def wf_03_function_invocation(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Invoke functions from workflow using ctx.task().

    Validates:
    - Functions can be called from workflows
    - Parameters are passed correctly
    - Return values are captured
    - Multiple function calls in sequence

    Returns:
        Function invocation results

    MCP Verification Points:
        - Logs: Function invocation logs
        - Traces: Workflow span contains function spans
        - Events: function.invoked events
    """
    ctx.logger.info("=== WF_03: Function Invocation ===")

    # Call function with no parameters
    ctx.logger.info("Calling fn_01_no_params...")
    result1 = await ctx.task(fn_01_no_params)
    ctx.logger.info(f"fn_01 result: {result1}")

    # Call function with parameter
    ctx.logger.info("Calling fn_02_one_param with value=10...")
    result2 = await ctx.task(fn_02_one_param, value=10)
    ctx.logger.info(f"fn_02 result: {result2}")

    # Call same function again with different parameter
    ctx.logger.info("Calling fn_02_one_param with value=25...")
    result3 = await ctx.task(fn_02_one_param, value=25)
    ctx.logger.info(f"fn_02 result (2nd call): {result3}")

    return {
        "no_params_result": result1,
        "one_param_result_1": result2,
        "one_param_result_2": result3,
        "total_calls": 3,
    }


@workflow
async def wf_04_agent_integration(ctx: WorkflowContext, task: str = "What is 2+2?") -> dict:
    """
    Test Scenario: Execute agent within workflow context.

    Validates:
    - Agent can be created and executed in workflow
    - Agent has access to workflow context
    - Conversation state is managed
    - Agent results are captured correctly

    Args:
        task: Task for the agent to execute

    Returns:
        Agent execution results

    MCP Verification Points:
        - Logs: Agent execution logs
        - Traces: Agent spans within workflow span
        - Entity: Agent conversation state in workflow
    """
    ctx.logger.info("=== WF_04: Agent Integration ===")
    ctx.logger.info(f"Task: {task}")

    # Create simple agent
    simple_agent = Agent(
        name="SimpleWorkflowAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Keep responses brief and concise.",
        temperature=0.7,
    )

    # Execute agent
    ctx.logger.info("Running agent...")
    result = await simple_agent.run(task, context=ctx)
    ctx.logger.info(f"Agent output: {result.output}")

    # Store result in state
    ctx.state.set("agent_result", result.output)

    return {
        "task": task,
        "agent_output": result.output,
        "agent_name": "SimpleWorkflowAgent",
    }


@workflow
async def wf_05_multi_step_workflow(ctx: WorkflowContext, initial_value: int = 5) -> dict:
    """
    Test Scenario: Multi-step workflow combining state, functions, and logic.

    Validates:
    - Multiple operations in sequence
    - State management across steps
    - Function invocations with computed values
    - Conditional logic in workflows

    Args:
        initial_value: Starting value for computation

    Returns:
        Complete workflow results

    MCP Verification Points:
        - Logs: Each step logged clearly
        - Traces: Multiple spans showing workflow progression
        - Entity: State updates at each step
    """
    ctx.logger.info("=== WF_05: Multi-Step Workflow ===")
    ctx.logger.info(f"Initial value: {initial_value}")

    # Step 1: Store initial value
    ctx.logger.info("Step 1: Store initial value")
    ctx.state.set("current_value", initial_value)
    ctx.state.set("step", 1)

    # Step 2: Double the value using function
    ctx.logger.info("Step 2: Double the value")
    result = await ctx.task(fn_02_one_param, value=initial_value)
    doubled = result["result"]
    ctx.state.set("doubled_value", doubled)
    ctx.state.set("step", 2)
    ctx.logger.info(f"Doubled: {doubled}")

    # Step 3: Process with conditional logic
    ctx.logger.info("Step 3: Conditional processing")
    if doubled > 20:
        ctx.state.set("category", "high")
        multiplier = 3
    elif doubled > 10:
        ctx.state.set("category", "medium")
        multiplier = 2
    else:
        ctx.state.set("category", "low")
        multiplier = 1

    ctx.logger.info(f"Category: {ctx.state.get('category')}")

    # Step 4: Final computation
    ctx.logger.info("Step 4: Final computation")
    final_value = doubled * multiplier
    ctx.state.set("final_value", final_value)
    ctx.state.set("step", 4)
    ctx.state.set("completed", True)

    ctx.logger.info(f"Final value: {final_value}")

    return {
        "initial_value": initial_value,
        "doubled_value": doubled,
        "category": ctx.state.get("category"),
        "multiplier": multiplier,
        "final_value": final_value,
        "steps_completed": 4,
    }


__all__ = [
    "wf_01_basic_execution",
    "wf_02_state_management",
    "wf_03_function_invocation",
    "wf_04_agent_integration",
    "wf_05_multi_step_workflow",
]
