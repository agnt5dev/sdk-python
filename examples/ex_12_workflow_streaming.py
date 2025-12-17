"""
Example: AGNT5 Workflow Streaming

This example demonstrates streaming from agents within workflows:
- Workflow steps automatically detect async generators
- Streaming events from nested agents are forwarded to client
- Final agent output is extracted and returned to the workflow

When a workflow step calls an agent that returns an async generator,
the workflow step executor:
1. Detects the async generator
2. Forwards all events via delta queue (agent.started, lm.message.delta, etc.)
3. Extracts the final output from agent.completed
4. Returns the output to the workflow for the next step

Each workflow can be:
- Run standalone: python ex_12_workflow_streaming.py
- Registered with platform: imported by app.py
- Streamed via client: client.stream("research_workflow", {...})
"""

import asyncio
import os

from agnt5 import workflow, WorkflowContext, Agent


# =============================================================================
# STREAMING AGENT WORKFLOW
# =============================================================================


@workflow
async def research_workflow(ctx: WorkflowContext, topic: str) -> dict:
    """
    Research workflow with streaming agent.

    The agent.run() returns an async generator. The workflow step executor
    automatically detects this and:
    - Forwards streaming events to the client
    - Extracts the final output for the next step

    Args:
        topic: Topic to research

    Returns:
        Dict with research and summary

    Example:
        # Streaming - receive events in real-time
        for event in client.stream("research_workflow", {"topic": "Python"}):
            print(f"Event: {event['event_type']}")
    """
    ctx.logger.info(f"Starting research workflow for: {topic}")

    # Create research agent
    researcher = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="You are a research assistant. Provide 3 key facts about the topic.",
        temperature=0.7,
    )

    # Step 1: Research - the async generator is auto-detected
    # Events like agent.started, lm.message.delta, agent.completed are forwarded
    research = await ctx.step(
        "research",
        researcher.run(f"Research: {topic}", context=ctx),
    )
    ctx.logger.info(f"Research result: {research[:100] if research else 'None'}...")

    # Create summarizer agent
    summarizer = Agent(
        name="summarizer",
        model="openai/gpt-4o-mini",
        instructions="You are a summarizer. Create a one-sentence summary.",
        temperature=0.5,
    )

    # Step 2: Summarize - also streams events
    summary = await ctx.step(
        "summarize",
        summarizer.run(f"Summarize: {research}", context=ctx),
    )
    ctx.logger.info(f"Summary: {summary}")

    return {
        "topic": topic,
        "research": research,
        "summary": summary,
    }


@workflow
async def mixed_workflow(ctx: WorkflowContext, x: int = 10, y: int = 5) -> dict:
    """
    Workflow mixing streaming agents and regular steps.

    Demonstrates that both streaming and non-streaming steps work together.

    Args:
        x: First number
        y: Second number

    Returns:
        Dict with calculation and description

    Example:
        for event in client.stream("mixed_workflow", {"x": 10, "y": 5}):
            print(f"Event: {event['event_type']}")
    """
    ctx.logger.info(f"Starting mixed workflow: {x} + {y}")

    # Step 1: Regular non-streaming step
    async def add(a: int, b: int) -> int:
        return a + b

    result = await ctx.step("calculate", add(x, y))
    ctx.logger.info(f"Calculation: {x} + {y} = {result}")

    # Step 2: Streaming agent step
    agent = Agent(
        name="describer",
        model="openai/gpt-4o-mini",
        instructions="Describe numbers in a fun way. Be brief.",
        temperature=0.8,
    )

    description = await ctx.step(
        "describe",
        agent.run(f"Describe the number {result}", context=ctx),
    )
    ctx.logger.info(f"Description: {description}")

    return {
        "x": x,
        "y": y,
        "sum": result,
        "description": description,
    }


@workflow
async def simple_agent_workflow(ctx: WorkflowContext, message: str = "Hello") -> dict:
    """
    Simple workflow with a single streaming agent step.

    Minimal example for testing workflow streaming.

    Args:
        message: Message to send to agent

    Returns:
        Dict with agent response

    Example:
        for event in client.stream("simple_agent_workflow", {"message": "Hi"}):
            print(f"Event: {event['event_type']}")
    """
    ctx.logger.info(f"Simple agent workflow: {message}")

    agent = Agent(
        name="simple_agent",
        model="openai/gpt-4o-mini",
        instructions="You are helpful. Be very brief.",
        temperature=0.5,
    )

    # This step returns an async generator - auto-detected by step executor
    response = await ctx.step(
        "respond",
        agent.run(message, context=ctx),
    )

    return {
        "message": message,
        "response": response,
    }


@workflow(chat=True)
async def chat_assistant(ctx: WorkflowContext, message: str) -> dict:
    """
    Chat-enabled workflow for multi-turn conversations.

    This workflow supports the /chat endpoint for SSE streaming.
    Use session_id to maintain conversation continuity.

    Args:
        message: User message to process

    Returns:
        Dict with assistant response

    Example:
        # Via chat endpoint (SSE streaming)
        curl -X POST http://localhost:34181/v1/chat/workflow/chat_assistant \\
          -H "Content-Type: application/json" \\
          -H "Accept: text/event-stream" \\
          -d '{"message": "Hello!", "session_id": "sess-123"}'
    """
    ctx.logger.info(f"Chat assistant: {message}")

    agent = Agent(
        name="chat_assistant",
        model="openai/gpt-4o-mini",
        instructions="You are a friendly chat assistant. Be helpful and conversational.",
        temperature=0.7,
    )

    # Stream response through the agent
    response = await ctx.step(
        "respond",
        agent.run(message, context=ctx),
    )

    return {
        "message": message,
        "response": response,
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main():
    """Run examples standalone (without platform)."""
    from agnt5 import Context

    print("=" * 60)
    print("AGNT5 Workflow Streaming Examples")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Workflow streaming requires an API key.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return

    ctx = Context(run_id="workflow-streaming-demo")

    # Example 1: Simple agent workflow
    print("\n--- Example 1: Simple Agent Workflow ---\n")
    result = await simple_agent_workflow(ctx, message="What is 2+2?")
    print(f"Result: {result}")

    # Example 2: Mixed workflow
    print("\n--- Example 2: Mixed Workflow ---\n")
    result = await mixed_workflow(ctx, x=7, y=3)
    print(f"Result: {result}")

    # Example 3: Research workflow
    print("\n--- Example 3: Research Workflow ---\n")
    result = await research_workflow(ctx, topic="Python decorators")
    print(f"Result: {result}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
