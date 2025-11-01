"""
Agent + Tools Test Workflows

This module contains test cases for agents using tools (but not other agents as tools).
Tests single and multiple tool usage patterns.

Test Coverage:
- Agent with single tool
- Agent with multiple tools
- Tool selection and execution
- Multi-tool coordination

Shared Tools:
- simple_calculator: Mathematical expression evaluator
- search_web: Simulated web search
- get_weather: Simulated weather API

See also:
- agent_basic_workflows.py: Agents without tools
- agent_collaboration_workflows.py: Agents using other agents as tools
"""

from typing import Dict, List
from agnt5 import workflow, WorkflowContext, Agent, tool, Context


# ============================================================================
# SHARED TOOLS (Reusable across workflows)
# ============================================================================


@tool(auto_schema=True)
async def simple_calculator(ctx: Context, expression: str) -> float:
    """
    Evaluate a simple mathematical expression.

    Args:
        expression: Mathematical expression (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of the calculation
    """
    ctx.logger.info(f"[TOOL] simple_calculator: evaluating '{expression}'")
    try:
        # Simple eval (in production, use safe expression evaluator)
        result = float(eval(expression))
        ctx.logger.info(f"[TOOL] simple_calculator: result = {result}")
        return result
    except Exception as e:
        ctx.logger.error(f"[TOOL] simple_calculator: error = {e}")
        return 0.0


@tool(auto_schema=True)
async def search_web(ctx: Context, query: str) -> List[Dict]:
    """
    Search the web for information (simulated).

    Args:
        query: Search query string

    Returns:
        List of search results with title and snippet
    """
    ctx.logger.info(f"[TOOL] search_web: searching for '{query}'")

    # Simulate search results
    results = [
        {
            "title": f"Article: {query}",
            "url": f"https://example.com/article-{query.replace(' ', '-')}",
            "snippet": f"Detailed information about {query}...",
        },
        {
            "title": f"Tutorial: {query}",
            "url": f"https://tutorial.com/{query.replace(' ', '-')}",
            "snippet": f"Learn all about {query} with examples...",
        },
    ]

    ctx.logger.info(f"[TOOL] search_web: found {len(results)} results")
    return results


@tool(auto_schema=True)
async def get_weather(ctx: Context, location: str) -> Dict:
    """
    Get current weather for a location (simulated).

    Args:
        location: City name or location

    Returns:
        Weather information including temperature and conditions
    """
    ctx.logger.info(f"[TOOL] get_weather: checking weather for '{location}'")

    # Simulate weather data
    weather = {
        "location": location,
        "temperature": 22,
        "units": "celsius",
        "conditions": "Partly cloudy",
        "humidity": 65,
    }

    ctx.logger.info(f"[TOOL] get_weather: {location} is {weather['temperature']}°C, {weather['conditions']}")
    return weather


# ============================================================================
# TEST WORKFLOW 2: Agent with Simple Tool
# ============================================================================


@workflow
async def test_agent_with_simple_tool(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 2: Workflow + Agent + Simple Tool

    Workflow with an agent that has access to one tool (calculator).
    Agent should autonomously decide to use the tool when appropriate.

    Args:
        task: Task description (e.g., "Calculate 25 * 4 + 10")

    Returns:
        Result with agent output and tool usage metadata

    MCP Verification Points:
        - Logs: Tool invocation logs
        - Traces: 3+ spans (workflow, agent.run, tool.invoke)
        - Events: Tool invocation events
        - Entity: Agent data with tool call history
    """
    ctx.logger.info("=== TEST 2: Agent with Simple Tool ===")
    ctx.logger.info(f"Task: {task}")

    # Create agent with calculator tool
    agent = Agent(
        name="CalculatorAgent",
        model="openai/gpt-4o-mini",
        instructions="""You are a helpful assistant with access to a calculator.
        Use the simple_calculator tool when you need to perform mathematical calculations.
        Be precise and show your work.""",
        tools=[simple_calculator],
        temperature=0.7,
        max_iterations=5,
    )

    # Run agent
    ctx.logger.info("Executing agent with calculator tool...")
    result = await agent.run(task, context=ctx)

    # Extract tool usage info
    tool_calls = result.tool_calls
    tools_used = [tc["name"] for tc in tool_calls]

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Tools used: {tools_used}")

    return {
        "scenario": "test_agent_with_simple_tool",
        "output": result.output,
        "tool_calls_count": len(tool_calls),
        "tools_used": tools_used,
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }


# ============================================================================
# TEST WORKFLOW 3: Agent with Multiple Tools
# ============================================================================


@workflow
async def test_agent_with_multiple_tools(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 3: Workflow + Agent + Multiple Tools

    Agent with 2-3 tools available. Agent autonomously selects which tools
    to use based on the task requirements.

    Args:
        task: Complex task requiring multiple tools

    Returns:
        Result with agent output and multi-tool usage metadata

    MCP Verification Points:
        - Logs: Multiple tool invocations
        - Traces: 4+ spans (workflow, agent.run, multiple tool.invoke)
        - Events: Multiple tool invocation events
        - Entity: Complete tool call history
    """
    ctx.logger.info("=== TEST 3: Agent with Multiple Tools ===")
    ctx.logger.info(f"Task: {task}")

    # Create agent with multiple tools
    agent = Agent(
        name="MultiToolAgent",
        model="openai/gpt-4o-mini",
        instructions="""You are a research assistant with access to multiple tools:
        - search_web: Search for information online
        - simple_calculator: Perform mathematical calculations
        - get_weather: Get current weather information

        Use the appropriate tools to complete tasks effectively.
        Explain your reasoning and the results from each tool.""",
        tools=[search_web, simple_calculator, get_weather],
        temperature=0.7,
        max_iterations=10,
    )

    # Run agent
    ctx.logger.info("Executing agent with multiple tools...")
    result = await agent.run(task, context=ctx)

    # Extract tool usage info
    tool_calls = result.tool_calls
    tools_used = list(set(tc["name"] for tc in tool_calls))
    tool_call_sequence = [{"name": tc["name"], "iteration": tc["iteration"]} for tc in tool_calls]

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Unique tools used: {tools_used}")
    ctx.logger.info(f"Total tool calls: {len(tool_calls)}")

    return {
        "scenario": "test_agent_with_multiple_tools",
        "output": result.output,
        "tool_calls_count": len(tool_calls),
        "unique_tools_used": tools_used,
        "tool_call_sequence": tool_call_sequence,
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
