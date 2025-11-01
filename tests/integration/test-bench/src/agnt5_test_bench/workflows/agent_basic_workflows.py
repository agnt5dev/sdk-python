"""
Basic Agent Test Workflows

This module contains the simplest agent test cases - agents without tools,
demonstrating pure agent reasoning and conversation capabilities.

Test Coverage:
- Agent initialization and execution
- Basic agent reasoning without tools
- Workflow + Agent integration
- Agent conversation state management

See also:
- agent_tools_workflows.py: Agents with tools
- agent_collaboration_workflows.py: Agents using other agents
"""

from agnt5 import workflow, WorkflowContext, Agent


@workflow
async def test_agent_basic(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 1: Workflow + Agent (Basic)

    A simple workflow that executes a single agent with a basic task.
    No tools, just agent reasoning.

    Args:
        task: Task description for the agent

    Returns:
        Result with agent output and metadata

    MCP Verification Points:
        - Logs: Workflow start/end, agent execution
        - Traces: 2 spans (workflow, agent.run)
        - Events: run.started, run.completed
        - Entity: Workflow entity with agent conversation data
    """
    ctx.logger.info("=== TEST 1: Basic Agent ===")
    ctx.logger.info(f"Task: {task}")

    # Create simple agent
    agent = Agent(
        name="BasicAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise and clear.",
        temperature=0.7,
    )

    # Run agent
    ctx.logger.info("Executing agent...")
    result = await agent.run(task, context=ctx)

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Tool calls: {len(result.tool_calls)}")

    return {
        "scenario": "test_agent_basic",
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
