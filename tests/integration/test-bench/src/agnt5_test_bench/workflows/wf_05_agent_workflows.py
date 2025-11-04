"""
Agent Workflows - Basic agent execution patterns

Test Coverage:
✅ Agent initialization and execution
✅ Agent without tools (pure reasoning)
✅ Agent conversation state management
✅ Multiple agent turns
✅ Agent + workflow state interaction

MCP Verification Points:
- Logs: Agent execution, prompts, responses
- Traces: Agent spans with LLM calls
- Events: agent.started, agent.completed, llm.call
- Entity: Agent conversation state in workflow
"""

from agnt5 import workflow, WorkflowContext, Agent


# Module-level agent for testing auto-registration
test_agent = Agent(
    name="TestAgent",
    model="openai/gpt-4o-mini",
    instructions="You are a helpful assistant. Be concise and clear.",
    temperature=0.7,
)


@workflow
async def wf_21_basic_agent_execution(ctx: WorkflowContext, task: str = "What is 5+5?") -> dict:
    """
    Test Scenario: Basic agent execution without tools.

    Validates:
    - Agent can be executed in workflow
    - Agent reasoning works correctly
    - Results are captured
    - No tool calls needed

    Args:
        task: Task for the agent to execute

    Returns:
        Agent execution results

    MCP Verification Points:
        - Logs: Task and agent output
        - Traces: Workflow span contains agent span
        - Events: agent.started, agent.completed
        - Entity: Agent conversation stored in workflow state
    """
    ctx.logger.info("=== WF_21: Basic Agent Execution ===")
    ctx.logger.info(f"Task: {task}")

    # Execute agent
    ctx.logger.info("Executing agent...")
    result = await test_agent.run(task, context=ctx)

    ctx.logger.info(f"Agent output: {result.output}")
    ctx.logger.info(f"Tool calls: {len(result.tool_calls)}")

    return {
        "task": task,
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "agent_name": "TestAgent",
    }


@workflow
async def wf_22_agent_conversation(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Multi-turn agent conversation.

    Validates:
    - Conversation state maintained across turns
    - Agent remembers context
    - Each turn builds on previous

    Returns:
        Results from conversation turns

    MCP Verification Points:
        - Logs: Each conversation turn
        - Entity: Conversation history accumulated
        - Events: Multiple agent.started/completed pairs
    """
    ctx.logger.info("=== WF_22: Agent Conversation ===")

    conversation_agent = Agent(
        name="ConversationAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Remember what the user tells you.",
        temperature=0.7,
    )

    # Turn 1: Initial message
    ctx.logger.info("Turn 1: Setting context")
    turn1 = await conversation_agent.run("My favorite color is blue.", context=ctx)
    ctx.logger.info(f"Turn 1 response: {turn1.output}")

    # Turn 2: Test memory
    ctx.logger.info("Turn 2: Testing memory")
    turn2 = await conversation_agent.run("What is my favorite color?", context=ctx)
    ctx.logger.info(f"Turn 2 response: {turn2.output}")

    # Turn 3: Follow-up
    ctx.logger.info("Turn 3: Follow-up question")
    turn3 = await conversation_agent.run("Why did I mention that color?", context=ctx)
    ctx.logger.info(f"Turn 3 response: {turn3.output}")

    return {
        "turns": 3,
        "turn1_output": turn1.output,
        "turn2_output": turn2.output,
        "turn3_output": turn3.output,
        "agent_name": "ConversationAgent",
    }


@workflow
async def wf_23_agent_with_workflow_state(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent + workflow state interaction.

    Validates:
    - Agent can be combined with workflow state
    - State accessible before and after agent
    - Independent state management

    Returns:
        Combined agent and state results

    MCP Verification Points:
        - Logs: State operations and agent execution
        - Entity: Both workflow state and agent state present
    """
    ctx.logger.info("=== WF_23: Agent with Workflow State ===")

    # Set workflow state before agent
    ctx.logger.info("Setting workflow state...")
    ctx.state.set("user_name", "Alice")
    ctx.state.set("preference", "detailed explanations")
    ctx.state.set("counter", 0)

    # Get state values
    user_name = ctx.state.get("user_name")
    preference = ctx.state.get("preference")

    # Execute agent with context
    ctx.logger.info(f"Executing agent for user: {user_name}")
    agent_result = await test_agent.run(
        f"Explain the concept of variables in programming. "
        f"The user prefers {preference}.",
        context=ctx,
    )
    ctx.logger.info(f"Agent output: {agent_result.output[:100]}...")

    # Update state after agent
    ctx.state.set("counter", 1)
    ctx.state.set("agent_executed", True)
    ctx.state.set("agent_output_length", len(agent_result.output))

    return {
        "user_name": user_name,
        "preference": preference,
        "agent_output": agent_result.output,
        "agent_output_length": ctx.state.get("agent_output_length"),
        "counter": ctx.state.get("counter"),
    }


@workflow
async def wf_24_multiple_agents_in_workflow(ctx: WorkflowContext, topic: str = "Python") -> dict:
    """
    Test Scenario: Multiple independent agents in single workflow.

    Validates:
    - Multiple agents can be created and used
    - Each agent has independent conversation state
    - Different agents for different tasks

    Args:
        topic: Topic for the agents to discuss

    Returns:
        Results from multiple agents

    MCP Verification Points:
        - Logs: Multiple agent executions
        - Entity: Multiple agent conversation states
        - Events: Multiple agent.started/completed events
    """
    ctx.logger.info("=== WF_24: Multiple Agents in Workflow ===")
    ctx.logger.info(f"Topic: {topic}")

    # Agent 1: Expert explainer
    ctx.logger.info("Agent 1: Expert explainer")
    expert = Agent(
        name="Expert",
        model="openai/gpt-4o-mini",
        instructions="You are an expert who provides detailed technical explanations.",
        temperature=0.7,
    )

    expert_result = await expert.run(f"Explain {topic} in technical terms.", context=ctx)
    ctx.logger.info(f"Expert response: {expert_result.output[:100]}...")

    # Agent 2: Beginner teacher
    ctx.logger.info("Agent 2: Beginner teacher")
    teacher = Agent(
        name="Teacher",
        model="openai/gpt-4o-mini",
        instructions="You are a teacher who explains things simply for beginners.",
        temperature=0.7,
    )

    teacher_result = await teacher.run(f"Explain {topic} to a complete beginner.", context=ctx)
    ctx.logger.info(f"Teacher response: {teacher_result.output[:100]}...")

    # Agent 3: Summarizer
    ctx.logger.info("Agent 3: Summarizer")
    summarizer = Agent(
        name="Summarizer",
        model="openai/gpt-4o-mini",
        instructions="You create brief summaries.",
        temperature=0.7,
    )

    summarizer_result = await summarizer.run(f"Summarize the key points about {topic}.", context=ctx)
    ctx.logger.info(f"Summarizer response: {summarizer_result.output[:100]}...")

    return {
        "topic": topic,
        "expert_output": expert_result.output,
        "teacher_output": teacher_result.output,
        "summarizer_output": summarizer_result.output,
        "total_agents": 3,
    }


@workflow
async def wf_25_agent_with_structured_output(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Agent producing structured information.

    Validates:
    - Agent can produce structured responses
    - Output can be parsed and used
    - Results stored in workflow state

    Returns:
        Structured agent output

    MCP Verification Points:
        - Logs: Agent prompt and response
        - Entity: Structured data in workflow state
    """
    ctx.logger.info("=== WF_25: Agent with Structured Output ===")

    # Create agent for structured output
    structured_agent = Agent(
        name="StructuredAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You provide structured information. "
            "Always format your response as clear bullet points."
        ),
        temperature=0.7,
    )

    # Request structured information
    ctx.logger.info("Requesting structured output...")
    result = await structured_agent.run(
        "List 5 programming best practices with brief explanations.",
        context=ctx,
    )

    ctx.logger.info(f"Agent output: {result.output}")

    # Store in state
    ctx.state.set("structured_output", result.output)
    ctx.state.set("output_length", len(result.output))

    # Count lines (rough estimate of structure)
    lines = result.output.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    ctx.logger.info(f"Output has {len(non_empty_lines)} non-empty lines")

    return {
        "output": result.output,
        "output_length": len(result.output),
        "line_count": len(non_empty_lines),
        "agent_name": "StructuredAgent",
    }


__all__ = [
    "test_agent",  # Export module-level agent
    "wf_21_basic_agent_execution",
    "wf_22_agent_conversation",
    "wf_23_agent_with_workflow_state",
    "wf_24_multiple_agents_in_workflow",
    "wf_25_agent_with_structured_output",
]
