"""Human-in-the-Loop (HITL) Workflows for Testing.

These workflows test workflow pause/resume functionality.
"""

from agnt5 import workflow, WorkflowContext


@workflow(name="approval_workflow_hitl")
async def approval_workflow_hitl(ctx: WorkflowContext, message: str) -> dict:
    """Test Scenario: Workflow that pauses for approval.

    MCP Verification Points:
    - Journal: Check for workflow.paused event
    - Entity State: Verify workflow state includes pause metadata
    """
    ctx.logger.info("=== HITL: Approval Workflow ===")
    ctx.logger.info(f"Processing message: {message}")

    # In a real scenario, this would call ctx.wait_for_user()
    # For now, just return success to indicate workflow completed
    # The actual pause functionality requires platform support

    return {
        "status": "completed",
        "message": message,
        "processed": True,
    }


@workflow(name="multi_choice_workflow")
async def multi_choice_workflow(ctx: WorkflowContext, task: str) -> dict:
    """Test Scenario: Workflow with multi-choice user input.

    MCP Verification Points:
    - Journal: Check for workflow.choice_presented event
    - Entity State: Verify choice options in workflow state
    """
    ctx.logger.info("=== HITL: Multi-Choice Workflow ===")
    ctx.logger.info(f"Task: {task}")

    # In a real scenario, this would call ctx.wait_for_user() with choices
    # For now, return a simulated choice

    return {
        "status": "completed",
        "task": task,
        "choice": "option_a",
    }


@workflow(name="text_input_workflow")
async def text_input_workflow(ctx: WorkflowContext, prompt: str = "Enter text") -> dict:
    """Test Scenario: Workflow that pauses for text input.

    MCP Verification Points:
    - Journal: Check for workflow.input_requested event
    - Entity State: Verify prompt is stored in state
    """
    ctx.logger.info("=== HITL: Text Input Workflow ===")
    ctx.logger.info(f"Prompt: {prompt}")

    # In a real scenario, this would call ctx.wait_for_user() for text input
    # Simulate user providing input
    user_input = "Sample user response"

    return {
        "status": "completed",
        "prompt": prompt,
        "user_input": user_input,
    }


__all__ = [
    "approval_workflow_hitl",
    "multi_choice_workflow",
    "text_input_workflow",
]
