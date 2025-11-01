"""
Agent HITL (Human-in-the-Loop) Test Workflows

This module contains test cases for agents with HITL capabilities,
testing workflow pause/resume for human interaction.

Test Coverage:
- ask_user tool for collecting user input
- request_approval tool for getting approvals
- Workflow pause/resume mechanics
- Checkpoint state preservation
- Agent context restoration after resume

HITL Tools:
- AskUserTool: Ask questions and get text responses
- RequestApprovalTool: Request approval for actions

See also:
- agent_tools_workflows.py: Regular tool usage
- comprehensive_integration_test.py: HITL combined with other features
"""

from agnt5 import workflow, WorkflowContext, Agent
from agnt5.tool import AskUserTool, RequestApprovalTool
from .agent_tools_workflows import simple_calculator


# ============================================================================
# TEST WORKFLOW 5: Agent with Human-in-the-Loop (HITL)
# ============================================================================


@workflow
async def test_agent_with_hitl(ctx: WorkflowContext, task: str, auto_approve: bool = True) -> dict:
    """
    Test Scenario 5: Workflow + Agent + Human-in-the-Loop

    Agent with HITL tools (ask_user, request_approval). Tests workflow
    pause/resume for human interaction.

    Args:
        task: Task that requires human input/approval
        auto_approve: If True, automatically approve (for automated testing)

    Returns:
        Result with agent output and HITL interaction metadata

    MCP Verification Points:
        - Logs: HITL request logs, pause/resume markers
        - Traces: HITL interaction spans with OK status (not error)
        - Events: Pause/resume events for HITL
        - Entity: HITL interaction history in agent data

    Note:
        For automated testing, set auto_approve=True to simulate approval.
        For manual testing, set auto_approve=False to require real human input.

        When testing manually:
        1. Run workflow - it will pause with status awaiting_user_input
        2. Check /tmp/start.json for question and runId
        3. Resume with: curl -X POST /v1/run/{runId}/resume -d '{"user_input": "..."}'
    """
    ctx.logger.info("=== TEST 5: Agent with HITL ===")
    ctx.logger.info(f"Task: {task}")
    ctx.logger.info(f"Auto-approve: {auto_approve}")

    # Create agent with HITL capabilities
    # Note: AskUserTool and RequestApprovalTool need WorkflowContext
    agent = Agent(
        name="HITLAgent",
        model="openai/gpt-4o-mini",
        instructions="""You are an assistant that works collaboratively with humans.
        You can:
        - Ask users for information using ask_user
        - Request approval for important actions using request_approval
        - Perform calculations using simple_calculator

        Always explain why you need human input and what you'll do with it.
        Be courteous and clear in your requests.""",
        tools=[AskUserTool(ctx), RequestApprovalTool(ctx), simple_calculator],
        temperature=0.7,
        max_iterations=10,
    )

    # Run agent
    ctx.logger.info("Executing agent with HITL tools...")

    # NOTE: In a real scenario, this would pause and wait for user input
    # For automated testing, we would need to mock the HITL responses
    # This is a simplified version that demonstrates the pattern

    result = await agent.run(task, context=ctx)

    # Extract HITL interaction info
    tool_calls = result.tool_calls
    hitl_interactions = [
        tc for tc in tool_calls if tc["name"] in ["ask_user", "request_approval"]
    ]

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"HITL interactions: {len(hitl_interactions)}")

    return {
        "scenario": "test_agent_with_hitl",
        "output": result.output,
        "tool_calls_count": len(tool_calls),
        "hitl_interactions": len(hitl_interactions),
        "hitl_types": [tc["name"] for tc in hitl_interactions],
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
