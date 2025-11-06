"""Human-in-the-Loop (HITL) Workflows for Testing.

These workflows test workflow pause/resume functionality using ctx.wait_for_user()
and agent-based HITL patterns with AskUserTool and RequestApprovalTool.

Test Coverage:
✅ Approval-based user input (direct)
✅ Multi-choice user input (direct)
✅ Free-text user input (direct)
✅ Agent with AskUserTool (interactive Q&A)
✅ Agent with RequestApprovalTool (approval flow)
✅ Workflow pause/resume behavior
✅ State persistence across pause

MCP Verification Points:
- Logs: User input requests, pause events, resume events, agent reasoning
- Traces: Workflow spans showing pause points, agent spans with tool calls
- Events: workflow.paused, workflow.resumed, agent.started, tool.invoked
- Entity: Workflow state preservation during pause, agent conversation history
"""

from agnt5 import workflow, WorkflowContext, Agent
from agnt5.tool import AskUserTool, RequestApprovalTool


@workflow
async def wf_36_approval_workflow_hitl(ctx: WorkflowContext, message: str) -> dict:
    """Test Scenario: Workflow that pauses for approval.

    Demonstrates human-in-the-loop approval pattern where a workflow
    processes input, pauses for user decision, then continues based
    on the approval/rejection.

    Args:
        message: Message to process that requires approval

    Returns:
        Result dictionary with approval status and processing result

    MCP Verification Points:
        - Logs: Initial processing, pause request, approval decision
        - Events: workflow.paused (with approval options)
        - Entity State: Message and approval status in workflow state
    """
    ctx.logger.info("=== HITL: Approval Workflow ===")
    ctx.logger.info(f"Processing message: {message}")

    # Step 1: Store initial data in state
    ctx.state.set("original_message", message)
    ctx.state.set("status", "pending_approval")
    ctx.logger.info("State: Message stored, awaiting approval")

    # Step 2: Analyze the message (pre-approval processing)
    word_count = len(message.split())
    char_count = len(message)
    ctx.state.set("word_count", word_count)
    ctx.state.set("char_count", char_count)
    ctx.logger.info(f"Analysis: {word_count} words, {char_count} characters")

    # Step 3: Pause for user approval
    ctx.logger.info("⏸️  Requesting user approval...")
    decision = await ctx.wait_for_user(
        question=f"Approve processing of message: '{message}'? (Words: {word_count}, Chars: {char_count})",
        input_type="approval",
        options=[
            {"id": "approve", "label": "Approve"},
            {"id": "reject", "label": "Reject"},
        ],
    )

    # Step 4: Process based on decision
    ctx.logger.info(f"▶️  User decision received: {decision}")
    ctx.state.set("approval_decision", decision)

    if decision == "approve":
        ctx.state.set("status", "approved")
        ctx.logger.info("✅ Message approved - processing")

        # Perform approved processing
        processed_message = message.upper()
        ctx.state.set("processed_message", processed_message)

        return {
            "status": "approved",
            "message": message,
            "processed": processed_message,
            "word_count": word_count,
            "char_count": char_count,
        }
    else:
        ctx.state.set("status", "rejected")
        ctx.logger.info("❌ Message rejected - no processing")

        return {
            "status": "rejected",
            "message": message,
            "processed": False,
            "reason": "User rejected approval",
        }


@workflow
async def wf_37_multi_choice_workflow(ctx: WorkflowContext, task: str) -> dict:
    """Test Scenario: Workflow with multi-choice user input.

    Demonstrates human-in-the-loop multi-choice pattern where a workflow
    presents multiple options to the user and processes based on selection.

    Args:
        task: Task description requiring choice selection

    Returns:
        Result dictionary with selected choice and processing result

    MCP Verification Points:
        - Logs: Task presentation, choice options, selection processing
        - Events: workflow.paused (with choice options)
        - Entity State: Task and choice selection in workflow state
    """
    ctx.logger.info("=== HITL: Multi-Choice Workflow ===")
    ctx.logger.info(f"Task: {task}")

    # Step 1: Store task in state
    ctx.state.set("task", task)
    ctx.state.set("status", "awaiting_choice")

    # Step 2: Present processing options to user
    ctx.logger.info("⏸️  Presenting processing options to user...")
    choice = await ctx.wait_for_user(
        question=f"How should we process task '{task}'?",
        input_type="choice",
        options=[
            {"id": "fast", "label": "Fast Processing (lower quality)"},
            {"id": "balanced", "label": "Balanced Processing (medium quality)"},
            {"id": "thorough", "label": "Thorough Processing (highest quality)"},
        ],
    )

    # Step 3: Process based on choice
    ctx.logger.info(f"▶️  User selected: {choice}")
    ctx.state.set("processing_mode", choice)
    ctx.state.set("status", "processing")

    # Simulate different processing based on choice
    processing_results = {
        "fast": {
            "duration": "1 second",
            "quality": "70%",
            "details": "Quick analysis completed",
        },
        "balanced": {
            "duration": "5 seconds",
            "quality": "85%",
            "details": "Standard analysis completed",
        },
        "thorough": {
            "duration": "15 seconds",
            "quality": "95%",
            "details": "Deep analysis completed with comprehensive review",
        },
    }

    result = processing_results.get(
        choice,
        {"duration": "unknown", "quality": "unknown", "details": "Invalid choice"},
    )

    ctx.logger.info(f"✅ Processing complete: {result['details']}")
    ctx.state.set("status", "completed")
    ctx.state.set("result", result)

    return {
        "task": task,
        "choice": choice,
        "duration": result["duration"],
        "quality": result["quality"],
        "details": result["details"],
    }


@workflow
async def wf_38_text_input_workflow(ctx: WorkflowContext, prompt: str = "Enter text") -> dict:
    """Test Scenario: Workflow that pauses for text input.

    Demonstrates human-in-the-loop free-text input pattern where a workflow
    collects arbitrary text from the user and incorporates it into processing.

    Args:
        prompt: Prompt message to display to user

    Returns:
        Result dictionary with user input and processing result

    MCP Verification Points:
        - Logs: Prompt display, text input reception, processing
        - Events: workflow.paused (with text input prompt)
        - Entity State: Prompt and user input in workflow state
    """
    ctx.logger.info("=== HITL: Text Input Workflow ===")
    ctx.logger.info(f"Prompt: {prompt}")

    # Step 1: Store prompt in state
    ctx.state.set("prompt", prompt)
    ctx.state.set("status", "awaiting_input")

    # Step 2: Request text input from user
    ctx.logger.info(f"⏸️  Requesting user input: {prompt}")
    user_input = await ctx.wait_for_user(
        question=prompt,
        input_type="text",
    )

    # Step 3: Process the user's text input
    ctx.logger.info(f"▶️  User input received: {user_input}")
    ctx.state.set("user_input", user_input)
    ctx.state.set("status", "processing")

    # Analyze the input
    input_length = len(user_input)
    word_count = len(user_input.split())
    has_numbers = any(char.isdigit() for char in user_input)
    has_special = any(not char.isalnum() and not char.isspace() for char in user_input)

    analysis = {
        "length": input_length,
        "word_count": word_count,
        "has_numbers": has_numbers,
        "has_special_chars": has_special,
    }

    ctx.logger.info(f"✅ Input analysis: {analysis}")
    ctx.state.set("analysis", analysis)
    ctx.state.set("status", "completed")

    # Create a processed response
    response = f"Received '{user_input}' ({word_count} words, {input_length} chars)"

    return {
        "prompt": prompt,
        "user_input": user_input,
        "response": response,
        "analysis": analysis,
    }


@workflow
async def wf_39_agent_with_ask_user_tool(
    ctx: WorkflowContext, task: str = "Help me plan a birthday party"
) -> dict:
    """Test Scenario: Agent that uses AskUserTool to gather information.

    Demonstrates agent-based HITL where the agent autonomously decides when
    to ask the user for clarifying information. The agent can ask multiple
    questions in sequence to gather all necessary details.

    Args:
        task: Initial task description from user

    Returns:
        Result dictionary with final output and interaction details

    MCP Verification Points:
        - Logs: Agent reasoning, tool invocations, user questions/responses
        - Events: Multiple workflow.paused/resumed events for each question
        - Entity State: Agent conversation history preserved across pauses
        - Traces: Agent span containing multiple ask_user tool call spans
    """
    ctx.logger.info("=== HITL: Agent with AskUserTool ===")
    ctx.logger.info(f"Initial task: {task}")

    # Store initial task in state
    ctx.state.set("task", task)
    ctx.state.set("status", "agent_starting")

    # Initialize agent with AskUserTool
    # The agent will autonomously decide when to ask questions
    planner_agent = Agent(
        name="PlannerAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a helpful planning assistant. When you need more information "
            "from the user to complete their task, use the ask_user tool to ask "
            "clarifying questions. Ask one question at a time. Gather all necessary "
            "details before providing your final plan. Keep responses concise."
        ),
        tools=[AskUserTool(ctx)],  # Agent can ask questions
        temperature=0.7,
        max_iterations=15,  # Allow multiple rounds of Q&A
    )

    # Run agent - it will autonomously decide when to ask questions
    ctx.logger.info("🤖 Starting agent execution...")
    ctx.state.set("status", "agent_running")

    result = await planner_agent.run(task, context=ctx)

    ctx.logger.info(f"✅ Agent completed with output: {result.output[:100]}...")

    # Extract tool calls to analyze interaction
    ask_user_calls = [tc for tc in result.tool_calls if tc.get("name") == "ask_user"]

    ctx.logger.info(f"📊 Agent asked {len(ask_user_calls)} questions")
    ctx.state.set("questions_asked", len(ask_user_calls))
    ctx.state.set("status", "completed")

    return {
        "task": task,
        "output": result.output,
        "questions_asked": len(ask_user_calls),
        "tool_calls": result.tool_calls,
        "agent_iterations": len(result.tool_calls) + 1,
    }


@workflow
async def wf_40_agent_with_approval_tool(
    ctx: WorkflowContext, action_request: str = "Deploy the new feature to production"
) -> dict:
    """Test Scenario: Agent that requests approval before executing actions.

    Demonstrates agent-based approval workflow where the agent analyzes
    a requested action, explains the implications, and requests user
    approval before proceeding.

    Args:
        action_request: The action being requested

    Returns:
        Result dictionary with approval status and agent output

    MCP Verification Points:
        - Logs: Agent reasoning, approval requests, user decisions
        - Events: workflow.paused/resumed for approval request
        - Entity State: Action details and approval status in state
        - Traces: Agent span with request_approval tool call span
    """
    ctx.logger.info("=== HITL: Agent with RequestApprovalTool ===")
    ctx.logger.info(f"Action request: {action_request}")

    # Store request in state
    ctx.state.set("action_request", action_request)
    ctx.state.set("status", "analyzing_request")

    # Initialize agent with RequestApprovalTool
    deployment_agent = Agent(
        name="DeploymentAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a deployment safety assistant. When a user requests an action, "
            "analyze it carefully and identify any risks or requirements. Then use the "
            "request_approval tool to get user confirmation before proceeding. "
            "Explain what will happen if approved. Keep responses concise."
        ),
        tools=[RequestApprovalTool(ctx)],  # Agent can request approval
        temperature=0.7,
        max_iterations=10,
    )

    # Run agent - it will analyze and request approval
    ctx.logger.info("🤖 Starting agent analysis...")
    ctx.state.set("status", "agent_running")

    result = await deployment_agent.run(action_request, context=ctx)

    ctx.logger.info(f"✅ Agent completed with output: {result.output[:100]}...")

    # Extract approval tool calls
    approval_calls = [tc for tc in result.tool_calls if tc.get("name") == "request_approval"]

    approval_status = "approved" if approval_calls else "not_required"
    if approval_calls:
        # Check the result of the approval call
        last_approval = approval_calls[-1]
        if "result" in last_approval:
            approval_status = last_approval["result"]

    ctx.logger.info(f"📊 Approval status: {approval_status}")
    ctx.state.set("approval_status", approval_status)
    ctx.state.set("status", "completed")

    return {
        "action_request": action_request,
        "output": result.output,
        "approval_status": approval_status,
        "approval_calls": len(approval_calls),
        "tool_calls": result.tool_calls,
    }


__all__ = [
    "wf_36_approval_workflow_hitl",
    "wf_37_multi_choice_workflow",
    "wf_38_text_input_workflow",
    "wf_39_agent_with_ask_user_tool",
    "wf_40_agent_with_approval_tool",
]
