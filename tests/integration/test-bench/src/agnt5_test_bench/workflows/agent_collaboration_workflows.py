"""
Agent Collaboration Test Workflows (Agent-as-Tool Pattern)

This module contains test cases for agents using other agents as tools.
Tests the agent-as-tool delegation pattern where a coordinator agent
invokes specialist agents to complete subtasks.

Test Coverage:
- Coordinator + Specialist pattern
- Nested agent execution
- Tool sharing between agents
- Multi-level delegation

Shared Components:
- domain_specific_analysis: Tool for specialist agents
- create_domain_specialist(): Helper to create specialist agent

See also:
- agent_tools_workflows.py: Agents with regular tools
- agent_handoff_workflows.py: Explicit handoff pattern (different from agent-as-tool)
"""

from typing import Dict
from agnt5 import workflow, WorkflowContext, Agent, tool, Context
from .agent_tools_workflows import search_web


# ============================================================================
# SHARED TOOLS FOR COLLABORATION
# ============================================================================


@tool(auto_schema=True)
async def domain_specific_analysis(ctx: Context, data: str) -> Dict:
    """
    Perform domain-specific analysis on data (for specialist agent).

    Args:
        data: Data to analyze

    Returns:
        Analysis results with insights and recommendations
    """
    ctx.logger.info(f"[TOOL] domain_specific_analysis: analyzing '{data[:50]}...'")

    analysis = {
        "summary": f"Analysis of: {data[:50]}...",
        "key_insights": ["Insight 1: Important finding", "Insight 2: Notable pattern"],
        "recommendations": ["Recommendation 1: Action item", "Recommendation 2: Follow-up"],
        "confidence": 0.85,
    }

    ctx.logger.info(f"[TOOL] domain_specific_analysis: confidence = {analysis['confidence']}")
    return analysis


# ============================================================================
# AGENT FACTORY HELPERS
# ============================================================================


def create_domain_specialist() -> Agent:
    """
    Create a domain specialist agent with analysis capabilities.

    Returns:
        Agent configured as a domain specialist
    """
    return Agent(
        name="DomainSpecialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a domain specialist for data analysis.
        Use your analysis tool to provide expert insights.
        Be thorough and technical in your responses.""",
        tools=[domain_specific_analysis],
        temperature=0.7,
        max_iterations=5,
    )


# ============================================================================
# TEST WORKFLOW 4: Agent with Agent as Tool
# ============================================================================


@workflow
async def test_agent_with_agent_as_tool(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 4: Workflow + Agent + Simple Tool + Agent as Tool

    Coordinator agent uses a specialist agent as a tool. Each agent has
    its own tools. Tests the agent-as-tool delegation pattern.

    Args:
        task: Task requiring specialist expertise

    Returns:
        Result with coordinator output and delegation metadata

    MCP Verification Points:
        - Logs: Nested agent invocations
        - Traces: 5+ spans (workflow, coordinator.run, specialist invocation, tools)
        - Events: Multiple agent execution events
        - Entity: Multiple entities (coordinator + specialist)
    """
    ctx.logger.info("=== TEST 4: Agent with Agent as Tool ===")
    ctx.logger.info(f"Task: {task}")

    # Create specialist agent with domain-specific tool
    specialist_agent = create_domain_specialist()

    # Create coordinator agent that uses specialist as a tool
    coordinator_agent = Agent(
        name="Coordinator",
        model="openai/gpt-4o-mini",
        instructions="""You are a coordinator agent. You can:
        - Search for information (search_web)
        - Delegate complex analysis to the DomainSpecialist agent

        When you need expert analysis, use the DomainSpecialist.
        Synthesize results from all sources into a coherent response.""",
        tools=[search_web, specialist_agent],  # Agent as tool!
        temperature=0.7,
        max_iterations=10,
    )

    # Run coordinator
    ctx.logger.info("Executing coordinator agent...")
    result = await coordinator_agent.run(task, context=ctx)

    # Extract delegation info
    tool_calls = result.tool_calls
    agent_delegations = [tc for tc in tool_calls if tc["name"] == "DomainSpecialist"]
    regular_tool_calls = [tc for tc in tool_calls if tc["name"] != "DomainSpecialist"]

    ctx.logger.info(f"Coordinator output: {result.output}")
    ctx.logger.info(f"Delegated to specialist: {len(agent_delegations)} times")
    ctx.logger.info(f"Regular tool calls: {len(regular_tool_calls)}")

    # Check how many agents are in the workflow entity
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    return {
        "scenario": "test_agent_with_agent_as_tool",
        "output": result.output,
        "tool_calls_count": len(tool_calls),
        "agent_delegations": len(agent_delegations),
        "regular_tool_calls": len(regular_tool_calls),
        "agents_in_workflow": all_agents,
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
