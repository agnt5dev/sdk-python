"""
Example: AGNT5 Agent Streaming

This example demonstrates streaming events from AI agents:
- Agent lifecycle events (started, completed, failed)
- LLM message events (start, delta, stop)
- Tool call events (when tools are used)

Streaming enables real-time visibility into agent execution:
- See what the agent is doing in real-time
- Get LLM tokens as they're generated
- Monitor tool invocations
- Build responsive UIs

Each function can be:
- Run standalone: python ex_11_agent_streaming.py
- Registered with platform: imported by app.py
- Invoked via client: client.stream("stream_agent_chat", {"message": "..."})
"""

import asyncio
import json
import os

from agnt5 import function, FunctionContext, Agent, Context, tool


# =============================================================================
# TOOLS FOR STREAMING AGENT
# =============================================================================


@tool(auto_schema=True)
async def calculate(ctx: Context, expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "15 * 23")

    Returns:
        Result of the calculation as a string
    """
    ctx.logger.info(f"[calculate] Evaluating: {expression}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


# =============================================================================
# STREAMING AGENT FUNCTIONS
# =============================================================================


@function
async def stream_agent_chat(ctx: FunctionContext, message: str):
    """
    Stream events from a simple chat agent.

    Yields agent lifecycle events and can be consumed via client.stream().

    Args:
        message: The user message to process

    Yields:
        Dict with event_type and event data

    Example:
        for event in client.stream("stream_agent_chat", {"message": "Hello!"}):
            print(f"{event['event_type']}: {event.get('data', {})}")
    """
    if not os.getenv("OPENAI_API_KEY"):
        yield {"event_type": "error", "message": "OPENAI_API_KEY not set"}
        return

    agent = Agent(
        name="streaming_assistant",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise and friendly.",
        temperature=0.7,
        max_iterations=5,
    )

    ctx.logger.info(f"[stream_agent_chat] Starting agent with message: {message[:50]}...")

    # Stream events from the agent - yield Event objects directly
    # Worker handles Event serialization correctly
    async for event in agent.run(message, context=ctx):
        yield event


@function
async def stream_agent_with_tools(ctx: FunctionContext, message: str):
    """
    Stream events from an agent that can use tools.

    Shows tool invocation events in the stream.

    Args:
        message: The user message to process

    Yields:
        Dict with event_type and event data including tool calls

    Example:
        for event in client.stream("stream_agent_with_tools", {"message": "What is 15 * 23?"}):
            if event["event_type"] == "agent.tool_call.started":
                print(f"Tool: {event['data']['tool_name']}")
    """
    if not os.getenv("OPENAI_API_KEY"):
        yield {"event_type": "error", "message": "OPENAI_API_KEY not set"}
        return

    agent = Agent(
        name="calculator_streaming",
        model="openai/gpt-4o-mini",
        instructions="""You are a helpful math assistant.
Use the calculate tool to perform mathematical operations.
Show your work and explain the result.""",
        tools=[calculate],
        temperature=0.3,
        max_iterations=10,
    )

    ctx.logger.info(f"[stream_agent_with_tools] Starting agent with message: {message[:50]}...")

    # Stream events from the agent - yield Event objects directly
    async for event in agent.run(message, context=ctx):
        yield event


@function
async def stream_agent_simple(ctx: FunctionContext, message: str):
    """
    Minimal streaming agent for testing.

    Just streams start and completion events without tools.

    Args:
        message: The user message to process

    Yields:
        Dict with event_type and minimal data

    Example:
        events = list(client.stream("stream_agent_simple", {"message": "Hi"}))
        assert events[0]["event_type"] == "agent.started"
        assert events[-1]["event_type"] == "agent.completed"
    """
    if not os.getenv("OPENAI_API_KEY"):
        yield {"event_type": "error", "message": "OPENAI_API_KEY not set"}
        return

    agent = Agent(
        name="simple_streaming",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Give very brief responses.",
        temperature=0.5,
        max_iterations=3,
    )

    ctx.logger.info(f"[stream_agent_simple] Processing: {message[:50]}...")

    # Stream events from the agent - yield Event objects directly
    async for event in agent.run(message, context=ctx):
        yield event


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main():
    """Run examples standalone (without platform)."""
    print("=" * 60)
    print("AGNT5 Agent Streaming Examples")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Agent streaming requires an API key.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return

    # Example 1: Simple chat streaming
    print("\n--- Example 1: Simple Chat Streaming ---\n")
    ctx = Context(
        run_id="streaming-example-1",
        correlation_id="streaming-example-1",
        parent_correlation_id="",
    )

    event_count = 0
    async for event in stream_agent_chat(ctx, "What is Python?"):
        event_count += 1
        print(f"Event {event_count}: {event['event_type']}")
        if event.get("data"):
            # Print just key fields, not entire data
            if "output" in event["data"]:
                output = event["data"]["output"]
                if len(output) > 100:
                    output = output[:100] + "..."
                print(f"  Output: {output}")

    # Example 2: Agent with tools
    print("\n--- Example 2: Agent with Tools Streaming ---\n")
    ctx = Context(
        run_id="streaming-example-2",
        correlation_id="streaming-example-2",
        parent_correlation_id="",
    )

    event_count = 0
    async for event in stream_agent_with_tools(ctx, "Calculate 15 * 23 + 100"):
        event_count += 1
        print(f"Event {event_count}: {event['event_type']}")
        if event["event_type"] == "agent.tool_call.started":
            print(f"  Tool: {event['data'].get('tool_name', 'N/A')}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
