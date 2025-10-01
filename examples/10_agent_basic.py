"""
Example: Basic Agent Usage

This example demonstrates:
- Creating an agent with LLM integration
- Agent with tools for autonomous task execution
- Multi-turn conversations
- Tool orchestration

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=your-key-here
"""

import asyncio
import os
from typing import Dict, List

from agnt5 import Agent, Context, tool
from agnt5.lm import OpenAILanguageModel


# Define tools for the agent
@tool(auto_schema=True)
async def search_web(ctx: Context, query: str) -> List[Dict]:
    """Search the web for information.

    Args:
        query: Search query string
    """
    ctx.logger.info(f"Searching web for: {query}")

    # Simulate web search results
    return [
        {
            "title": f"Article about {query}",
            "url": f"https://example.com/article-{query.replace(' ', '-')}",
            "snippet": f"This article discusses {query} in detail..."
        },
        {
            "title": f"Tutorial: {query}",
            "url": f"https://tutorial.com/{query.replace(' ', '-')}",
            "snippet": f"Learn everything about {query} with examples..."
        }
    ]


@tool(auto_schema=True)
async def calculate(ctx: Context, expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "5 * 3")
    """
    ctx.logger.info(f"Calculating: {expression}")

    # Simple evaluation (in production, use a safe evaluator)
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        ctx.logger.error(f"Calculation error: {e}")
        return 0.0


@tool(auto_schema=True)
async def get_current_time(ctx: Context) -> str:
    """Get the current time.

    Returns:
        Current time as a formatted string
    """
    from datetime import datetime
    ctx.logger.info("Getting current time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def example_simple_agent():
    """Example: Simple agent without tools."""
    print("=== Example 1: Simple Agent (No Tools) ===\n")

    # Create language model
    lm = OpenAILanguageModel()

    # Create simple agent
    agent = Agent(
        name="assistant",
        model=lm,
        instructions="You are a helpful assistant. Be concise and friendly.",
        model_name="gpt-4o-mini",
        temperature=0.7,
    )

    # Run agent
    result = await agent.run("Explain what an API is in simple terms")

    print(f"Agent: {result.output}\n")
    print(f"Tool calls: {len(result.tool_calls)}\n")


async def example_agent_with_tools():
    """Example: Agent with tool orchestration."""
    print("=== Example 2: Agent with Tools ===\n")

    # Create language model
    lm = OpenAILanguageModel()

    # Create agent with tools
    agent = Agent(
        name="research_assistant",
        model=lm,
        instructions="""You are a research assistant. You can:
        - Search the web for information
        - Perform calculations
        - Get the current time

        Use tools when needed to provide accurate information.""",
        tools=[search_web, calculate, get_current_time],
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_iterations=5,
    )

    # Task 1: Research question
    print("Task 1: Research question\n")
    result = await agent.run("What are the latest trends in AI? Search for information.")
    print(f"Agent: {result.output}\n")
    print(f"Tool calls made: {len(result.tool_calls)}")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"  {i}. {tc['name']} (iteration {tc['iteration']})")
    print()

    # Task 2: Calculation
    print("Task 2: Calculation task\n")
    result = await agent.run("Calculate 15 * 27 + 100")
    print(f"Agent: {result.output}\n")
    print(f"Tool calls made: {len(result.tool_calls)}")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"  {i}. {tc['name']}: {tc['arguments']}")
    print()

    # Task 3: Time query
    print("Task 3: Time query\n")
    result = await agent.run("What time is it now?")
    print(f"Agent: {result.output}\n")
    print(f"Tool calls made: {len(result.tool_calls)}")
    print()


async def example_multi_turn_chat():
    """Example: Multi-turn conversation."""
    print("=== Example 3: Multi-Turn Conversation ===\n")

    # Create language model
    lm = OpenAILanguageModel()

    # Create conversational agent
    agent = Agent(
        name="tutor",
        model=lm,
        instructions="""You are a patient math tutor.
        Explain concepts clearly and ask follow-up questions to check understanding.""",
        model_name="gpt-4o-mini",
        temperature=0.7,
    )

    # Conversation
    messages = []

    # Turn 1
    response, messages = await agent.chat("What is a derivative?", messages)
    print(f"User: What is a derivative?")
    print(f"Agent: {response}\n")

    # Turn 2
    response, messages = await agent.chat("Can you give me an example?", messages)
    print(f"User: Can you give me an example?")
    print(f"Agent: {response}\n")

    # Turn 3
    response, messages = await agent.chat("How do I calculate it?", messages)
    print(f"User: How do I calculate it?")
    print(f"Agent: {response}\n")

    print(f"Total messages in conversation: {len(messages)}\n")


async def example_agent_with_state():
    """Example: Agent using context state."""
    print("=== Example 4: Agent with State Management ===\n")

    @tool(auto_schema=True)
    async def save_note(ctx: Context, note: str) -> str:
        """Save a note to memory.

        Args:
            note: Note text to save
        """
        notes = ctx.get("notes", [])
        notes.append(note)
        ctx.set("notes", notes)
        return f"Saved note #{len(notes)}"

    @tool(auto_schema=True)
    async def list_notes(ctx: Context) -> List[str]:
        """List all saved notes."""
        return ctx.get("notes", [])

    # Create agent with state tools
    lm = OpenAILanguageModel()
    agent = Agent(
        name="note_keeper",
        model=lm,
        instructions="""You are a note-taking assistant.
        Help users save and retrieve notes.""",
        tools=[save_note, list_notes],
        model_name="gpt-4o-mini",
        temperature=0.7,
    )

    # Create context to persist state
    ctx = Context(run_id="note-session")

    # Save some notes
    result1 = await agent.run("Save a note: Buy groceries", context=ctx)
    print(f"Agent: {result1.output}\n")

    result2 = await agent.run("Save another note: Call dentist", context=ctx)
    print(f"Agent: {result2.output}\n")

    # List notes
    result3 = await agent.run("Show me all my notes", context=ctx)
    print(f"Agent: {result3.output}\n")

    # Check state
    print(f"Notes in context state: {ctx.get('notes', [])}\n")


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-key-here\n")
        return

    print("=== AGNT5 Agent Examples ===\n")
    print("Phase 1: Simple agent with external LLM integration")
    print("Phase 2: Platform-backed agents with durability and multi-agent coordination\n")

    try:
        # Run examples
        await example_simple_agent()
        await example_agent_with_tools()
        await example_multi_turn_chat()
        await example_agent_with_state()

        print("=== All Examples Complete ===")

    except ImportError:
        print("ERROR: OpenAI package not installed")
        print("Install with: pip install openai")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
