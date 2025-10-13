"""
Example: Basic Agent Usage

This example demonstrates:
- Creating an agent with LLM integration (multiple providers supported)
- Agent with tools for autonomous task execution
- Multi-turn conversations
- Tool orchestration

Prerequisites:
    Set environment variables for providers you want to use:
    - OPENAI_API_KEY=your-key
    - ANTHROPIC_API_KEY=your-key
    - GROQ_API_KEY=your-key
"""

import asyncio
import os
from typing import Dict, List

from agnt5 import Agent, Context, lm, tool


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

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create simple agent with model string
    agent = Agent(
        name="assistant",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise and friendly.",
        temperature=0.7,
    )

    # Run agent
    result = await agent.run("Explain what an API is in simple terms")

    print(f"Agent: {result.output}\n")
    print(f"Tool calls: {len(result.tool_calls)}\n")


async def example_agent_with_tools():
    """Example: Agent with tool orchestration."""
    print("=== Example 2: Agent with Tools ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create agent with tools using model string
    agent = Agent(
        name="research_assistant",
        model="openai/gpt-4o-mini",
        instructions="""You are a research assistant. You can:
        - Search the web for information
        - Perform calculations
        - Get the current time

        Use tools when needed to provide accurate information.""",
        tools=[search_web, calculate, get_current_time],
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

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create conversational agent with model string
    agent = Agent(
        name="tutor",
        model="openai/gpt-4o-mini",
        instructions="""You are a patient math tutor.
        Explain concepts clearly and ask follow-up questions to check understanding.""",
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
    """Example: Agent with stateful tools using closure."""
    print("=== Example 4: Agent with State Management ===\n")

    # State storage for notes (shared across tool calls via closure)
    notes_storage: List[str] = []

    @tool(auto_schema=True)
    async def save_note(ctx: Context, note: str) -> str:
        """Save a note to memory.

        Args:
            note: Note text to save
        """
        notes_storage.append(note)
        return f"Saved note #{len(notes_storage)}"

    @tool(auto_schema=True)
    async def list_notes(ctx: Context) -> List[str]:
        """List all saved notes."""
        return notes_storage

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create agent with state tools using model string
    agent = Agent(
        name="note_keeper",
        model="openai/gpt-4o-mini",
        instructions="""You are a note-taking assistant.
        Help users save and retrieve notes.""",
        tools=[save_note, list_notes],
        temperature=0.7,
    )

    # Save some notes
    result1 = await agent.run("Save a note: Buy groceries")
    print(f"Agent: {result1.output}\n")

    result2 = await agent.run("Save another note: Call dentist")
    print(f"Agent: {result2.output}\n")

    # List notes
    result3 = await agent.run("Show me all my notes")
    print(f"Agent: {result3.output}\n")

    # Check state
    print(f"Notes in storage: {notes_storage}\n")


async def example_multi_provider_agents():
    """Example: Use agents with different LLM providers."""
    print("=== Example 5: Multi-Provider Agents ===\n")

    providers = []

    # Test with OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI GPT-4o-mini", "openai/gpt-4o-mini"))

    # Test with Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic Claude 3.5 Haiku", "anthropic/claude-3-5-haiku-20241022"))

    # Test with Groq
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq Llama 3.3 70B", "groq/llama-3.3-70b-versatile"))

    if not providers:
        print("⚠️  Skipped: No API keys set\n")
        return

    question = "What are the three most important skills for a software engineer?"

    for provider_name, model in providers:
        print(f"\n[{provider_name}]")

        agent = Agent(
            name=f"advisor_{provider_name.lower().replace(' ', '_')}",
            model=model,
            instructions="You are a career advisor. Give concise, practical advice.",
            temperature=0.7,
        )

        result = await agent.run(question)
        print(f"  {result.output}\n")

    print()


async def example_advanced_model_config():
    """Example: Using ModelConfig for advanced configuration."""
    print("=== Example 6: Advanced Model Configuration ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    from agnt5.lm import ModelConfig

    # Create custom model configuration
    # Useful for custom endpoints, timeouts, or special headers
    config = ModelConfig(
        timeout=60,  # Custom timeout in seconds
        # base_url="https://custom-api.example.com",  # Custom endpoint
        # headers={"X-Custom-Header": "value"},  # Custom headers
    )

    # Create agent with advanced configuration
    agent = Agent(
        name="custom_agent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant with custom configuration.",
        temperature=0.7,
        max_tokens=150,  # Limit response length
        model_config=config,
    )

    result = await agent.run("Explain what custom API configuration means in one sentence.")
    print(f"Agent: {result.output}\n")
    print("✅ Successfully used ModelConfig for advanced settings\n")


async def main():
    print("=== AGNT5 Agent Examples ===\n")
    print("Language models with multi-provider support")
    print("Supported providers: OpenAI, Anthropic, Groq, Azure, Bedrock, OpenRouter\n")

    # Check if at least one provider is configured
    has_provider = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
    ])

    if not has_provider:
        print("⚠️  WARNING: No API keys found. Please set at least one:")
        print("  - OPENAI_API_KEY=your-key")
        print("  - ANTHROPIC_API_KEY=your-key")
        print("  - GROQ_API_KEY=your-key")
        print("  - OPENROUTER_API_KEY=your-key\n")

    try:
        # Run examples
        await example_simple_agent()
        await example_agent_with_tools()
        await example_multi_turn_chat()
        await example_agent_with_state()
        await example_multi_provider_agents()
        await example_advanced_model_config()

        print("=== All Examples Complete ===")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
