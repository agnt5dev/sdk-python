"""
Comprehensive Integration Test - The Ultimate Test

This module contains the comprehensive integration test that combines
ALL features of the AGNT5 platform in a single workflow.

Features Tested:
✅ Workflows + Agents
✅ Multiple tools (calculator, weather, search)
✅ Agent-as-tool pattern (specialist delegation)
✅ HITL (human-in-the-loop) interactions
✅ Complete observability (logs, traces, events, entities)

This is the ultimate test for validating:
- End-to-end platform functionality
- Full observability stack
- Complex multi-agent coordination
- All collaboration patterns
- State management
- Trace propagation

See also:
- agent_basic_workflows.py: Simple agent tests
- agent_tools_workflows.py: Tool usage
- agent_collaboration_workflows.py: Agent-as-tool
- agent_handoff_workflows.py: Handoff patterns
- agent_hitl_workflows.py: HITL interactions
"""

from agnt5 import workflow, WorkflowContext, Agent
from agnt5.tool import RequestApprovalTool
from .agent_tools_workflows import simple_calculator, get_weather
from .agent_collaboration_workflows import domain_specific_analysis
from .agent_handoff_workflows import search_web


# ============================================================================
# MODULE-LEVEL AGENTS (for auto-registration)
# ============================================================================

# Research specialist with search and analysis tools
research_specialist = Agent(
    name="ResearchSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a research specialist. You can:
    - Search the web for information (search_web)
    - Perform data analysis (domain_specific_analysis)

    Be thorough and provide detailed insights with evidence.""",
    tools=[search_web, domain_specific_analysis],
    temperature=0.7,
    max_iterations=5,
)


# ============================================================================
# TEST WORKFLOW 10: Comprehensive Scenario (All Features)
# ============================================================================


@workflow
async def test_comprehensive_scenario(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 10: Complete Integration Test (ALL FEATURES)

    Combines ALL features in a single workflow:
    - Coordinator agent with multiple tools
    - Specialist agent as a tool (with its own tools)
    - HITL for approval
    - Multi-step complex workflow
    - Full observability verification

    Args:
        task: Complex task requiring all capabilities

    Returns:
        Comprehensive result with full metadata

    MCP Verification Points:
        - Logs: Complete execution trace with all agent interactions
        - Traces: 7+ spans (workflow, coordinator, specialist, tools, HITL)
        - Events: Complete event sequence (start, tool calls, delegation, HITL, completion)
        - Entity: Multiple entities with full state (coordinator + specialist)

    This is the ultimate test for observability and functionality.

    Example task:
        "Research AI adoption trends, calculate average adoption rate from [25, 45, 62, 78, 85],
         check weather in San Francisco, and request approval for the final report."

    Flow:
        1. Coordinator receives complex task
        2. Uses simple_calculator for calculations
        3. Uses get_weather for weather info
        4. Delegates research to ResearchSpecialist (agent-as-tool)
        5. Specialist uses search_web and domain_specific_analysis
        6. Coordinator requests approval via HITL
        7. Returns comprehensive synthesized result

    Verifies:
        - All tool types work together
        - Agent delegation patterns
        - HITL integration
        - Complete trace propagation
        - Entity state consistency
    """
    ctx.logger.info("=== TEST 10: Comprehensive Scenario (All Features) ===")
    ctx.logger.info(f"Task: {task}")

    # Create coordinator with all capabilities
    # NOTE: This agent is created inline (not module-level) because it uses
    # RequestApprovalTool which requires runtime WorkflowContext.
    # It will not be auto-registered, but that's expected for this comprehensive test.
    coordinator_agent = Agent(
        name="ComprehensiveCoordinator",
        model="openai/gpt-4o-mini",
        instructions="""You are a comprehensive coordinator agent with full capabilities:
        - Calculator for math (simple_calculator)
        - Weather information (get_weather)
        - Research specialist agent (ResearchSpecialist) for complex research
        - Human approval (request_approval) for important decisions

        Follow this approach:
        1. Gather information using available tools
        2. Delegate complex analysis to the specialist
        3. Request approval for any significant actions or reports
        4. Synthesize all results into a comprehensive response

        Be methodical, explain your reasoning, and use all relevant tools.""",
        tools=[
            simple_calculator,
            get_weather,
            research_specialist,  # Agent as tool (module-level)
            RequestApprovalTool(ctx),  # HITL - needs WorkflowContext
        ],
        temperature=0.7,
        max_iterations=15,
    )

    # Run comprehensive workflow
    ctx.logger.info("Executing comprehensive coordinator...")
    result = await coordinator_agent.run(task, context=ctx)

    # Extract comprehensive metadata
    tool_calls = result.tool_calls
    agent_delegations = [tc for tc in tool_calls if tc["name"] == "ResearchSpecialist"]
    hitl_interactions = [tc for tc in tool_calls if tc["name"] == "request_approval"]
    regular_tools = [
        tc for tc in tool_calls
        if tc["name"] not in ["ResearchSpecialist", "request_approval"]
    ]

    # Get all agents in workflow
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    ctx.logger.info(f"Coordinator output: {result.output[:100]}...")
    ctx.logger.info(f"Total tool calls: {len(tool_calls)}")
    ctx.logger.info(f"Agent delegations: {len(agent_delegations)}")
    ctx.logger.info(f"HITL interactions: {len(hitl_interactions)}")
    ctx.logger.info(f"Regular tools: {len(regular_tools)}")
    ctx.logger.info(f"Agents in workflow: {all_agents}")

    return {
        "scenario": "test_comprehensive_scenario",
        "output": result.output,
        "total_tool_calls": len(tool_calls),
        "agent_delegations": len(agent_delegations),
        "hitl_interactions": len(hitl_interactions),
        "regular_tool_calls": len(regular_tools),
        "agents_in_workflow": all_agents,
        "tool_call_breakdown": {
            "agent_as_tool": [tc["name"] for tc in agent_delegations],
            "hitl": [tc["name"] for tc in hitl_interactions],
            "regular": [tc["name"] for tc in regular_tools],
        },
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
