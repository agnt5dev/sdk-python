"""
Agent Tool Workflows - Agents using tools

Test Coverage:
✅ Agent with single tool
✅ Agent with multiple tools
✅ Tool selection and execution
✅ Tool results in agent output
✅ Multi-tool coordination

MCP Verification Points:
- Logs: Tool invocations, arguments, results
- Traces: Tool call spans within agent spans
- Events: tool.started, tool.completed
"""

from agnt5 import workflow, WorkflowContext, Agent
from ..tools import (
    calculate_total,
    search_database,
    format_report,
    validate_data,
)


# Module-level agent with tools
calculator_agent = Agent(
    name="CalculatorAgent",
    model="openai/gpt-4o-mini",
    instructions="You are a calculator assistant. Use tools to perform calculations.",
    tools=[calculate_total],
    temperature=0.7,
    max_iterations=5,
)


@workflow
async def wf_26_agent_with_single_tool(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent with single tool.

    Validates:
    - Agent can invoke tool
    - Tool arguments passed correctly
    - Tool results used in response

    Returns:
        Agent output with tool usage

    MCP Verification Points:
        - Logs: Tool invocation and result
        - Traces: Agent span contains tool span
        - Events: agent.tool_call, tool.completed
    """
    ctx.logger.info("=== WF_26: Agent with Single Tool ===")

    # Ask agent to use the tool
    ctx.logger.info("Asking agent to calculate sum...")
    result = await calculator_agent.run(
        "Calculate the sum of these numbers: 10, 25, 33, 17, 8",
        context=ctx,
    )

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Tool calls made: {len(result.tool_calls)}")

    return {
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "agent_name": "CalculatorAgent",
    }


@workflow
async def wf_27_agent_with_multiple_tools(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent with multiple tools available.

    Validates:
    - Agent can choose appropriate tool
    - Multiple tools available for selection
    - Correct tool selected for task

    Returns:
        Agent output with tool selection info

    MCP Verification Points:
        - Logs: Tool selection decision
        - Traces: Selected tool span
        - Events: Multiple tools available, one used
    """
    ctx.logger.info("=== WF_27: Agent with Multiple Tools ===")

    # Agent with multiple tools
    multi_tool_agent = Agent(
        name="MultiToolAgent",
        model="openai/gpt-4o-mini",
        instructions="You have access to multiple tools. Choose the right tool for the task.",
        tools=[calculate_total, search_database, format_report],
        temperature=0.7,
        max_iterations=5,
    )

    # Ask agent to search (should use search_database)
    ctx.logger.info("Task 1: Search database")
    result1 = await multi_tool_agent.run(
        "Search for information about analytics",
        context=ctx,
    )
    ctx.logger.info(f"Task 1 output: {result1.output[:100]}...")
    ctx.logger.info(f"Task 1 tool calls: {len(result1.tool_calls)}")

    # Ask agent to calculate (should use calculate_total)
    ctx.logger.info("Task 2: Calculate")
    result2 = await multi_tool_agent.run(
        "What is the sum of 5, 10, and 15?",
        context=ctx,
    )
    ctx.logger.info(f"Task 2 output: {result2.output}")
    ctx.logger.info(f"Task 2 tool calls: {len(result2.tool_calls)}")

    return {
        "task1_output": result1.output,
        "task1_tool_calls": len(result1.tool_calls),
        "task2_output": result2.output,
        "task2_tool_calls": len(result2.tool_calls),
        "total_tool_calls": len(result1.tool_calls) + len(result2.tool_calls),
    }


@workflow
async def wf_28_agent_multi_tool_coordination(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent using multiple tools in sequence.

    Validates:
    - Agent can chain multiple tool calls
    - Results from one tool feed into next
    - Complex multi-step reasoning with tools

    Returns:
        Agent output from multi-tool workflow

    MCP Verification Points:
        - Logs: Multiple tool invocations in sequence
        - Traces: Multiple tool spans in order
        - Events: Sequential tool.started/completed events
    """
    ctx.logger.info("=== WF_28: Agent Multi-Tool Coordination ===")

    # Agent with tools for data processing
    data_agent = Agent(
        name="DataAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a data processing assistant. "
            "Use tools to search, validate, and format data. "
            "IMPORTANT: Execute one tool at a time and wait for results before calling the next tool. "
            "Always pass the results from one tool as arguments to the next tool when needed. "
            "For example, if you search for data, pass that data to validate_data before formatting."
        ),
        tools=[search_database, validate_data, format_report],
        temperature=0.7,
        max_iterations=10,
    )

    # Complex task requiring multiple tools
    ctx.logger.info("Executing complex multi-tool task...")
    result = await data_agent.run(
        (
            "Step 1: Use search_database to search for 'analytics' records (limit 3). "
            "Step 2: Take the results from search_database and pass them to validate_data "
            "to check that each record has 'title' and 'category' fields. "
            "Step 3: Format the validation results as a detailed report."
        ),
        context=ctx,
    )

    ctx.logger.info(f"Agent output: {result.output[:200]}...")
    ctx.logger.info(f"Total tool calls: {len(result.tool_calls)}")

    # Log each tool call
    for idx, tool_call in enumerate(result.tool_calls):
        ctx.logger.info(f"Tool call {idx + 1}: {tool_call.get('name', 'unknown')}")

    return {
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "tool_calls": [tc.get("name") for tc in result.tool_calls],
        "agent_name": "DataAgent",
    }


@workflow
async def wf_29_agent_tool_with_parameters(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent tools with various parameter types.

    Validates:
    - Tools receive correct parameter types
    - Lists, dicts, and primitives work
    - Parameter validation

    Returns:
        Agent tool execution results

    MCP Verification Points:
        - Logs: Tool parameters logged
        - Traces: Tool call arguments visible
    """
    ctx.logger.info("=== WF_29: Agent Tool with Parameters ===")

    param_agent = Agent(
        name="ParamAgent",
        model="openai/gpt-4o-mini",
        instructions="You use tools with various parameter types. Pass parameters correctly.",
        tools=[calculate_total, validate_data],
        temperature=0.7,
        max_iterations=5,
    )

    # Test 1: List parameter
    ctx.logger.info("Test 1: List parameter")
    result1 = await param_agent.run(
        "Calculate the average of these numbers: 12, 18, 24, 30, 36",
        context=ctx,
    )
    ctx.logger.info(f"Result 1: {result1.output}")

    # Test 2: Dict and list parameters
    ctx.logger.info("Test 2: Dict and list parameters")
    result2 = await param_agent.run(
        (
            "Validate this data has required fields 'name' and 'email': "
            '{"name": "Alice", "email": "alice@example.com", "age": 30}'
        ),
        context=ctx,
    )
    ctx.logger.info(f"Result 2: {result2.output}")

    return {
        "test1_output": result1.output,
        "test1_tool_calls": len(result1.tool_calls),
        "test2_output": result2.output,
        "test2_tool_calls": len(result2.tool_calls),
        "total_tests": 2,
    }


@workflow
async def wf_30_agent_tool_iteration(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent iterating with tools.

    Validates:
    - Agent can make multiple tool calls
    - Iteration count respected
    - Results build on each other

    Returns:
        Agent iteration results

    MCP Verification Points:
        - Logs: Multiple iterations logged
        - Traces: Multiple tool call spans
        - Events: Iteration count visible
    """
    ctx.logger.info("=== WF_30: Agent Tool Iteration ===")

    iteration_agent = Agent(
        name="IterationAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You solve problems step by step. "
            "Use tools multiple times if needed to build up the answer."
        ),
        tools=[calculate_total],
        temperature=0.7,
        max_iterations=10,  # Allow multiple iterations
    )

    # Task that may require multiple tool calls
    ctx.logger.info("Executing task with potential iterations...")
    result = await iteration_agent.run(
        (
            "Calculate the sum of 5, 10, 15, then calculate the average of that sum "
            "with 20 and 25. Show your work."
        ),
        context=ctx,
    )

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Total tool calls: {len(result.tool_calls)}")
    ctx.logger.info(f"Iterations used: {result.iteration_count if hasattr(result, 'iteration_count') else 'unknown'}")

    return {
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "agent_name": "IterationAgent",
    }


__all__ = [
    "calculator_agent",  # Export module-level agent
    "wf_26_agent_with_single_tool",
    "wf_27_agent_with_multiple_tools",
    "wf_28_agent_multi_tool_coordination",
    "wf_29_agent_tool_with_parameters",
    "wf_30_agent_tool_iteration",
]
