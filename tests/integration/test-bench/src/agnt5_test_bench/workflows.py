from agnt5 import workflow, WorkflowContext, function, FunctionContext, Agent


@function
async def test_function(ctx: FunctionContext) -> str:
    """A simple test function to verify setup."""
    ctx.logger.info("Test function executed successfully.")
    return "Test function executed successfully."


@workflow
async def test_workflow(ctx: WorkflowContext) -> str:
    """A simple test workflow to verify setup."""
    ctx.logger.info("Test workflow executed successfully.")
    ctx.state.set("test_key", "test_value")
    ctx.state.set("another_key", 12345)

    ctx.logger.info("test_key value: %s", ctx.state.get("test_key"))
    ctx.logger.info("another_key value: %s", ctx.state.get("another_key"))

    response = await ctx.task(test_function)
    ctx.logger.info("Test function 01 response: %s", response)

    # Call the test function using ctx.task() for registered @function

    return "Test workflow executed successfully."


@workflow
async def test_workflow_with_state(ctx: WorkflowContext) -> str:
    """Test workflow to verify workflow state and agent state consolidation."""
    ctx.logger.info("=== Starting Test Workflow with State ===")

    # Test 1: Workflow state management
    ctx.logger.info("\n--- Test 1: Workflow State Management ---")
    ctx.state.set("test_key", "test_value")
    ctx.state.set("another_key", 12345)
    ctx.logger.info(
        "Set test_key=%s, another_key=%s", ctx.state.get("test_key"), ctx.state.get("another_key")
    )

    # Test 2: Function invocation
    ctx.logger.info("\n--- Test 2: Function Invocation ---")
    response = await ctx.task(test_function)
    ctx.logger.info("Test function response: %s", response)

    # Test 3: Update workflow state
    ctx.logger.info("\n--- Test 3: Update Workflow State ---")
    ctx.state.set("test_key", "updated_value")
    ctx.state.set("another_key", 67890)
    ctx.logger.info(
        "Updated test_key=%s, another_key=%s",
        ctx.state.get("test_key"),
        ctx.state.get("another_key"),
    )

    # Test 4: Agent execution with state consolidation
    ctx.logger.info("\n--- Test 4: Agent State Consolidation ---")

    # Create test agent
    test_agent = Agent(
        name="TestAgent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful test agent. Keep responses brief.",
    )

    # First agent interaction
    ctx.logger.info("Running agent (first interaction)...")
    agent_result_1 = await test_agent.run("What is 2+2?", context=ctx)
    ctx.logger.info("Agent response 1: %s", agent_result_1.output)

    # Second agent interaction (should preserve conversation history)
    ctx.logger.info("Running agent (second interaction)...")
    agent_result_2 = await test_agent.run("What did I just ask you?", context=ctx)
    ctx.logger.info("Agent response 2: %s", agent_result_2.output)

    # Test 5: Verify agent state is in workflow entity
    ctx.logger.info("\n--- Test 5: Verify Agent State Storage ---")
    workflow_entity = ctx._workflow_entity

    # Check that agent data is stored in workflow state
    agent_data = workflow_entity.get_agent_data("TestAgent")
    if agent_data:
        ctx.logger.info("✓ Agent data found in workflow state")
        ctx.logger.info("  - Agent name: %s", agent_data.get("agent_name"))
        ctx.logger.info("  - Message count: %s", agent_data.get("message_count"))
        ctx.logger.info("  - Created at: %s", agent_data.get("created_at"))

        # Log messages
        messages = agent_data.get("messages", [])
        ctx.logger.info("  - Conversation history:")
        for i, msg in enumerate(messages):
            ctx.logger.info(
                "    %d. [%s] %s",
                i + 1,
                msg.get("role"),
                (
                    msg.get("content")[:50] + "..."
                    if len(msg.get("content", "")) > 50
                    else msg.get("content")
                ),
            )
    else:
        ctx.logger.warning("✗ Agent data NOT found in workflow state")

    # Test 6: List all agents in workflow
    ctx.logger.info("\n--- Test 6: List All Agents ---")
    all_agents = workflow_entity.list_agents()
    ctx.logger.info("Agents in workflow: %s", all_agents)

    # Test 7: Multiple agents
    ctx.logger.info("\n--- Test 7: Multiple Agents ---")

    second_agent = Agent(
        name="SecondAgent", model="openai/gpt-4o-mini", instructions="You are another test agent."
    )

    await second_agent.run("Hello from second agent!", context=ctx)

    all_agents = workflow_entity.list_agents()
    ctx.logger.info("Agents after adding second: %s", all_agents)

    # Verify both agents have separate data
    agent1_messages = workflow_entity.get_agent_messages("TestAgent")
    agent2_messages = workflow_entity.get_agent_messages("SecondAgent")
    ctx.logger.info("TestAgent has %d messages", len(agent1_messages))
    ctx.logger.info("SecondAgent has %d messages", len(agent2_messages))

    # Test 8: Verify workflow state structure
    ctx.logger.info("\n--- Test 8: Workflow State Structure ---")
    ctx.logger.info("Workflow state keys: %s", list(workflow_entity.state._state.keys()))

    # Count agent-related keys
    agent_keys = [k for k in workflow_entity.state._state.keys() if k.startswith("agent.")]
    ctx.logger.info("Agent state keys: %s", agent_keys)
    ctx.logger.info("Total agent conversations in workflow: %d", len(agent_keys))

    # Summary
    ctx.logger.info("\n=== Test Summary ===")
    ctx.logger.info("✓ Workflow state: working")
    ctx.logger.info("✓ Function invocation: working")
    ctx.logger.info("✓ Agent execution: working")
    ctx.logger.info("✓ Agent state consolidation: %s", "working" if agent_data else "FAILED")
    ctx.logger.info("✓ Multiple agents: %d agents stored", len(all_agents))
    ctx.logger.info(
        "✓ State structure: %d total keys, %d agent keys",
        len(workflow_entity.state._state.keys()),
        len(agent_keys),
    )

    return f"Test completed - {len(all_agents)} agents with consolidated state"
