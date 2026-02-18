"""
Example: AGNT5 Human-in-the-Loop (HITL)

This example demonstrates HITL patterns:
- Workflow-level pauses with ctx.wait_for_user()
- Different input types (text, approval, choice)
- Agent-level HITL with AskUserTool

HITL enables workflows and agents to:
- Pause execution and wait for user input
- Request approval before critical actions
- Collect information interactively
- Resume processing after user response

Each workflow can be:
- Run standalone: python ex_08_hitl.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("approval_workflow", {...}, component_type="workflow")

Prerequisites:
- Set OPENAI_API_KEY for agent-based HITL examples
"""

import asyncio
import os

from agnt5 import workflow, WorkflowContext, Agent
from agnt5.tool import AskUserTool


# =============================================================================
# WORKFLOW-LEVEL HITL
# =============================================================================


@workflow
async def hitl_text_input(ctx: WorkflowContext, prompt: str) -> dict:
    """
    Workflow that pauses to collect text input from user.

    Demonstrates the text input type for free-form responses.

    Args:
        prompt: Initial prompt to show user

    Returns:
        Dict with user input and processed result

    Example:
        result = client.run("text_input_workflow", {
            "prompt": "Please enter your name"
        }, component_type="workflow")
    """
    ctx.logger.info(f"[text_input_workflow] Prompt: {prompt}")

    # Pause and wait for user text input
    user_input = await ctx.wait_for_user(
        question=prompt,
        input_type="text",
    )

    ctx.logger.info(f"[text_input_workflow] Received: {user_input}")

    # Process the input
    return {
        "prompt": prompt,
        "user_input": user_input,
        "processed": user_input.upper() if user_input else "",
        "char_count": len(user_input) if user_input else 0,
    }


@workflow
async def hitl_approval(ctx: WorkflowContext, action: str, resource: str) -> dict:
    """
    Workflow that requires user approval before proceeding.

    Demonstrates the approval input type with approve/reject options.

    Args:
        action: The action to be approved (e.g., "delete", "publish")
        resource: The resource the action applies to

    Returns:
        Dict with approval status and result

    Example:
        result = client.run("approval_workflow", {
            "action": "delete",
            "resource": "user-data-2024"
        }, component_type="workflow")
    """
    ctx.logger.info(f"[approval_workflow] Action: {action} on {resource}")

    # Store action details in state
    ctx.state.set("pending_action", action)
    ctx.state.set("pending_resource", resource)

    # Request approval
    decision = await ctx.wait_for_user(
        question=f"Do you approve the following action?\n\nAction: {action}\nResource: {resource}",
        input_type="approval",
        options=[
            {"id": "approve", "label": "Approve"},
            {"id": "reject", "label": "Reject"},
        ],
    )

    ctx.logger.info(f"[approval_workflow] Decision: {decision}")

    # Execute based on decision
    if decision == "approve":
        # Simulate executing the action
        result = {
            "status": "completed",
            "action": action,
            "resource": resource,
            "message": f"Successfully executed {action} on {resource}",
        }
    else:
        result = {
            "status": "rejected",
            "action": action,
            "resource": resource,
            "message": f"Action {action} on {resource} was rejected by user",
        }

    return result


@workflow
async def hitl_select(ctx: WorkflowContext, question: str, options: list) -> dict:
    """
    Workflow that presents single-select options to user.

    Demonstrates the select input type for choosing one option.

    Args:
        question: Question to ask the user
        options: List of option strings to present

    Returns:
        Dict with selected choice and processing result

    Example:
        result = client.run("select_workflow", {
            "question": "Select your preferred language",
            "options": ["Python", "JavaScript", "Go", "Rust"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"[select_workflow] Question: {question}")

    # Build select options
    select_options = [
        {"id": opt.lower().replace(" ", "_"), "label": opt}
        for opt in options
    ]

    # Present single-select to user
    selection = await ctx.wait_for_user(
        question=question,
        input_type="select",
        options=select_options,
    )

    ctx.logger.info(f"[select_workflow] Selection: {selection}")

    # Find the selected option label
    selected_label = next(
        (opt["label"] for opt in select_options if opt["id"] == selection),
        selection
    )

    return {
        "question": question,
        "available_options": options,
        "selected_id": selection,
        "selected_label": selected_label,
    }


@workflow
async def hitl_multiselect(ctx: WorkflowContext, question: str, options: list) -> dict:
    """
    Workflow that presents multi-select options to user.

    Demonstrates the multiselect input type for choosing multiple options.

    Args:
        question: Question to ask the user
        options: List of option strings to present

    Returns:
        Dict with selected choices and processing result

    Example:
        result = client.run("multiselect_workflow", {
            "question": "Which topics interest you?",
            "options": ["AI", "Web", "Data", "Security"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"[multiselect_workflow] Question: {question}")

    # Build multiselect options
    choice_options = [
        {"id": opt.lower().replace(" ", "_"), "label": opt}
        for opt in options
    ]

    # Present multi-select to user
    selection = await ctx.wait_for_user(
        question=question,
        input_type="multiselect",
        options=choice_options,
    )

    ctx.logger.info(f"[multiselect_workflow] Selection: {selection}")

    return {
        "question": question,
        "available_options": options,
        "selected": selection,
    }


@workflow
async def hitl_custom_and_skip(ctx: WorkflowContext) -> dict:
    """
    Workflow demonstrating allow_custom and skippable modifiers.

    Shows how to let users provide free-text answers or skip questions.

    Example:
        result = client.run("custom_and_skip_workflow", {},
                            component_type="workflow")
    """
    ctx.logger.info("[custom_and_skip] Starting")

    # Select with custom option — user can pick from list or type their own
    color = await ctx.wait_for_user(
        question="What's your favourite colour?",
        input_type="select",
        options=[
            {"id": "red", "label": "Red"},
            {"id": "blue", "label": "Blue"},
            {"id": "green", "label": "Green"},
        ],
        allow_custom=True,
    )
    ctx.logger.info(f"[custom_and_skip] Color: {color}")

    # Skippable question — user can skip without answering
    nickname = await ctx.wait_for_user(
        question="Do you have a nickname? (optional)",
        input_type="text",
        skippable=True,
    )
    ctx.logger.info(f"[custom_and_skip] Nickname: {nickname}")

    return {
        "color": color,
        "nickname": nickname,
        "skipped_nickname": nickname is None,
    }


# =============================================================================
# MULTI-STEP HITL WORKFLOW
# =============================================================================


@workflow
async def hitl_onboarding(ctx: WorkflowContext, welcome_message: str) -> dict:
    """
    Multi-step onboarding workflow with multiple HITL interactions.

    Demonstrates chaining multiple wait_for_user calls.

    Args:
        welcome_message: Welcome message to display

    Returns:
        Dict with collected user information

    Example:
        result = client.run("onboarding_workflow", {
            "welcome_message": "Welcome to our service!"
        }, component_type="workflow")
    """
    ctx.logger.info("[onboarding_workflow] Starting onboarding")

    # Step 1: Get name
    name = await ctx.wait_for_user(
        question=f"{welcome_message}\n\nPlease enter your name:",
        input_type="text",
    )
    ctx.state.set("user_name", name)
    ctx.logger.info(f"[onboarding_workflow] Name: {name}")

    # Step 2: Get preferences
    preference = await ctx.wait_for_user(
        question=f"Hello {name}! What type of content interests you most?",
        input_type="select",
        options=[
            {"id": "tech", "label": "Technology"},
            {"id": "business", "label": "Business"},
            {"id": "science", "label": "Science"},
            {"id": "arts", "label": "Arts & Culture"},
        ],
    )
    ctx.state.set("preference", preference)
    ctx.logger.info(f"[onboarding_workflow] Preference: {preference}")

    # Step 3: Confirm settings
    confirm = await ctx.wait_for_user(
        question=f"Confirm your settings:\n- Name: {name}\n- Interest: {preference}\n\nIs this correct?",
        input_type="approval",
        options=[
            {"id": "confirm", "label": "Confirm"},
            {"id": "restart", "label": "Start Over"},
        ],
    )

    if confirm == "restart":
        return {
            "status": "cancelled",
            "message": "User requested to restart onboarding",
        }

    return {
        "status": "completed",
        "user_name": name,
        "preference": preference,
        "message": f"Welcome aboard, {name}! Your preference for {preference} has been saved.",
    }


# =============================================================================
# AGENT-LEVEL HITL
# =============================================================================


@workflow
async def hitl_agent_questions(ctx: WorkflowContext, task: str) -> dict:
    """
    Workflow with an agent that can ask questions during execution.

    Demonstrates AskUserTool for agent-initiated HITL.

    Args:
        task: Task for the agent to work on

    Returns:
        Dict with agent response and interaction details

    Example:
        result = client.run("agent_with_questions", {
            "task": "Help me plan a trip to Japan"
        }, component_type="workflow")
    """
    ctx.logger.info(f"[agent_with_questions] Task: {task[:50]}...")

    # Create agent with AskUserTool
    planner = Agent(
        name="planner",
        model="openai/gpt-4o-mini",
        instructions="""You are a helpful planning assistant.
When working on tasks, you may need to clarify details with the user.
Use the ask_user tool to ask questions when you need more information.
Ask one question at a time and wait for the response before proceeding.
After gathering enough information, provide a complete plan.""",
        tools=[AskUserTool(ctx)],  # Agent can ask questions!
        temperature=0.7,
        max_iterations=15,
    )

    result = await planner.run_sync(task, context=ctx)

    # Count questions asked
    questions_asked = [
        tc for tc in result.tool_calls
        if tc.get("name") == "ask_user"
    ]

    return {
        "task": task,
        "response": result.output,
        "questions_count": len(questions_asked),
        "total_tool_calls": len(result.tool_calls),
    }


# =============================================================================
# CONDITIONAL APPROVAL WORKFLOW
# =============================================================================


@workflow
async def hitl_conditional_approval(ctx: WorkflowContext, amount: float, description: str) -> dict:
    """
    Workflow that only requires approval for large amounts.

    Demonstrates conditional HITL based on business logic.

    Args:
        amount: Transaction amount
        description: Transaction description

    Returns:
        Dict with transaction result

    Example:
        # Small amount - auto-approved
        result = client.run("conditional_approval", {
            "amount": 50.0,
            "description": "Office supplies"
        }, component_type="workflow")

        # Large amount - requires approval
        result = client.run("conditional_approval", {
            "amount": 5000.0,
            "description": "Server upgrade"
        }, component_type="workflow")
    """
    ctx.logger.info(f"[conditional_approval] Amount: ${amount}, Desc: {description}")

    APPROVAL_THRESHOLD = 1000.0

    if amount >= APPROVAL_THRESHOLD:
        # Large amount requires approval
        ctx.logger.info("[conditional_approval] Amount exceeds threshold, requesting approval")

        decision = await ctx.wait_for_user(
            question=f"Approve transaction?\n\nAmount: ${amount:,.2f}\nDescription: {description}\n\n(Approval required for amounts >= ${APPROVAL_THRESHOLD:,.2f})",
            input_type="approval",
            options=[
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
            ],
        )

        if decision != "approve":
            return {
                "status": "rejected",
                "amount": amount,
                "description": description,
                "message": "Transaction rejected by approver",
                "required_approval": True,
            }

        approval_status = "approved_by_user"
    else:
        # Small amount auto-approved
        approval_status = "auto_approved"

    return {
        "status": "completed",
        "amount": amount,
        "description": description,
        "approval_status": approval_status,
        "required_approval": amount >= APPROVAL_THRESHOLD,
        "message": f"Transaction of ${amount:,.2f} processed successfully",
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (simulated without platform)."""
    import logging
    from agnt5 import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Human-in-the-Loop (HITL) Example")
    print("=" * 60)

    print("""
Note: HITL workflows are designed to run on the platform where
they can pause and resume based on user input.

In standalone mode, we demonstrate the structure but cannot
actually pause for user input.

To test HITL workflows:
1. Start the platform: just platform standalone python
2. Run: client.run("approval_workflow", {...}, component_type="workflow")
3. The workflow will pause and wait for your input
4. Respond via the platform UI or API
""")

    # Show workflow structure without running
    print("\n--- Available HITL Workflows ---")
    print("1. text_input_workflow - Collects free-form text input")
    print("2. approval_workflow - Binary approve/reject decision")
    print("3. select_workflow - Single-select from options")
    print("4. multiselect_workflow - Multi-select from options")
    print("5. onboarding_workflow - Multi-step user onboarding")
    print("6. agent_with_questions - Agent that can ask questions")
    print("7. conditional_approval - Approval only for large amounts")
    print("8. custom_and_skip_workflow - allow_custom + skippable modifiers")

    # Demonstrate conditional approval auto-approve case
    print("\n--- Conditional Approval (Auto-Approve Case) ---")
    ctx = Context(
        run_id="hitl-demo",
        correlation_id="hitl-demo",
        parent_correlation_id="",
    )

    # Note: This would normally pause, but we'll show the auto-approve path
    print("For amount < $1000, transaction is auto-approved")
    print("For amount >= $1000, workflow pauses for user approval")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
