"""
Memory Architecture Test Workflows

Test workflows for verifying session-scoped and user-scoped memory functionality.
"""

from agnt5 import workflow, WorkflowContext, Agent


@workflow
async def test_session_memory_workflow(ctx: WorkflowContext, message: str) -> dict:
    """
    Test session-scoped memory across multiple invocations.

    This workflow should be called multiple times with the SAME session_id
    to verify that the agent remembers previous conversations.

    Args:
        message: User message to send to agent

    Returns:
        Dict with agent response and session info

    Example Usage:
        # First call
        client.run("test_session_memory_workflow",
                   {"message": "My name is Alice"},
                   component_type="workflow",
                   session_id="sess-123")

        # Second call (should remember name)
        client.run("test_session_memory_workflow",
                   {"message": "What's my name?"},
                   component_type="workflow",
                   session_id="sess-123")  # Same session!
    """
    ctx.logger.info("=== Session Memory Test ===")
    ctx.logger.info(f"Session ID: {ctx.session_id}")
    ctx.logger.info(f"User ID: {ctx.user_id}")
    ctx.logger.info(f"Run ID: {ctx.run_id}")
    ctx.logger.info(f"Message: {message}")

    # Create agent - it will auto-detect session scope from context
    agent = Agent(
        name="SessionTestAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a helpful assistant. Remember information from previous messages "
            "in this conversation. Keep responses brief (1-2 sentences)."
        ),
    )

    # Run agent - it will automatically use session-scoped memory
    result = await agent.run(message, context=ctx)

    ctx.logger.info(f"Agent response: {result.output}")

    # Get conversation history to verify memory
    workflow_entity = ctx._workflow_entity
    agent_data = workflow_entity.get_agent_data("SessionTestAgent")
    message_count = len(agent_data.get("messages", [])) if agent_data else 0

    ctx.logger.info(f"Total messages in session: {message_count}")

    return {
        "response": result.output,
        "session_id": ctx.session_id,
        "message_count": message_count,
        "memory_scope": "session",
    }


@workflow
async def test_user_memory_workflow(ctx: WorkflowContext, message: str) -> dict:
    """
    Test user-scoped memory across multiple sessions.

    This workflow should be called with the SAME user_id but DIFFERENT session_ids
    to verify that user-level memory persists across sessions.

    Args:
        message: User message to send to agent

    Returns:
        Dict with agent response and user info

    Example Usage:
        # First session
        client.run("test_user_memory_workflow",
                   {"message": "I prefer Python programming"},
                   component_type="workflow",
                   session_id="sess-001",
                   user_id="user-alice")

        # Different session, same user (should remember preference)
        client.run("test_user_memory_workflow",
                   {"message": "What programming language do I prefer?"},
                   component_type="workflow",
                   session_id="sess-002",  # Different session!
                   user_id="user-alice")    # Same user!
    """
    ctx.logger.info("=== User Memory Test ===")
    ctx.logger.info(f"User ID: {ctx.user_id}")
    ctx.logger.info(f"Session ID: {ctx.session_id}")
    ctx.logger.info(f"Run ID: {ctx.run_id}")
    ctx.logger.info(f"Message: {message}")

    # Create agent - it will auto-detect user scope if user_id is present
    agent = Agent(
        name="UserTestAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a personal assistant. Remember user preferences and information "
            "across all conversations. Keep responses brief (1-2 sentences)."
        ),
    )

    # Run agent - it will automatically use user-scoped memory if user_id is set
    result = await agent.run(message, context=ctx)

    ctx.logger.info(f"Agent response: {result.output}")

    # Get conversation history
    workflow_entity = ctx._workflow_entity
    agent_data = workflow_entity.get_agent_data("UserTestAgent")
    message_count = len(agent_data.get("messages", [])) if agent_data else 0

    # Determine actual memory scope used
    memory_scope = "user" if ctx.user_id else ("session" if ctx.session_id != ctx.run_id else "run")

    ctx.logger.info(f"Total messages for user: {message_count}")
    ctx.logger.info(f"Memory scope: {memory_scope}")

    return {
        "response": result.output,
        "user_id": ctx.user_id,
        "session_id": ctx.session_id,
        "message_count": message_count,
        "memory_scope": memory_scope,
    }


@workflow
async def test_multi_agent_session_workflow(ctx: WorkflowContext, task: str) -> dict:
    """
    Test multiple agents in the same session sharing conversation context.

    This demonstrates the multi-agent collaboration pattern where all agents
    in the same session can see each other's conversation history.

    Args:
        task: Task description for agents

    Returns:
        Dict with results from both agents

    Example Usage:
        client.run("test_multi_agent_session_workflow",
                   {"task": "Research AI trends and then analyze the data"},
                   component_type="workflow",
                   session_id="collab-123")
    """
    ctx.logger.info("=== Multi-Agent Session Test ===")
    ctx.logger.info(f"Session ID: {ctx.session_id}")
    ctx.logger.info(f"Task: {task}")

    # Create two different agents
    researcher = Agent(
        name="ResearchAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a research assistant. Gather information and present findings briefly.",
    )

    analyst = Agent(
        name="AnalystAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a data analyst. Analyze information and provide insights briefly.",
    )

    # Agent 1: Research
    ctx.logger.info("Running ResearchAgent...")
    research_result = await researcher.run(
        f"Research this topic briefly: {task}",
        context=ctx
    )
    ctx.logger.info(f"Research findings: {research_result.output[:100]}...")

    # Agent 2: Analysis (in the same session)
    ctx.logger.info("Running AnalystAgent...")
    analysis_result = await analyst.run(
        "Based on the research findings above, provide a brief analysis.",
        context=ctx
    )
    ctx.logger.info(f"Analysis: {analysis_result.output[:100]}...")

    # Verify both agents are stored in workflow
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    ctx.logger.info(f"Agents in session: {all_agents}")

    return {
        "research": research_result.output,
        "analysis": analysis_result.output,
        "session_id": ctx.session_id,
        "agents_in_session": all_agents,
        "memory_scope": "session",
    }
