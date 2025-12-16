"""
Example: AGNT5 Agents

This example demonstrates basic AI agent features:
- Creating agents with Agent class
- Using different LLM providers (OpenAI, Anthropic)
- Multi-turn conversations with session persistence
- Agent configuration (temperature, max_iterations, etc.)

Agents are AI-powered components that can:
- Process natural language instructions
- Maintain conversation context
- Be composed with tools and other agents

Each agent can be:
- Run standalone: python ex_06_agents.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("simple_agent", {"message": "Hello"})

Prerequisites:
- Set OPENAI_API_KEY for OpenAI models
- Set ANTHROPIC_API_KEY for Anthropic models
"""

import asyncio
import os

from agnt5 import function, FunctionContext, Agent


# =============================================================================
# BASIC AGENTS
# =============================================================================


@function
async def simple_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Simple agent that responds to a message.

    Uses OpenAI GPT-4o-mini by default.

    Args:
        message: User message to process

    Returns:
        Dict with agent response and metadata

    Example:
        result = client.run("simple_agent", {"message": "What is Python?"})
    """
    ctx.logger.info(f"[simple_agent] Processing: {message[:50]}...")

    agent = Agent(
        name="assistant",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise and friendly.",
        temperature=0.7,
        max_iterations=5,
    )

    result = await agent.run(message)

    return {
        "message": message,
        "response": result.output,
        "model": "openai/gpt-4o-mini",
    }


@function
async def claude_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Agent using Anthropic Claude.

    Args:
        message: User message to process

    Returns:
        Dict with agent response and metadata

    Example:
        result = client.run("claude_agent", {"message": "Explain recursion"})
    """
    ctx.logger.info(f"[claude_agent] Processing: {message[:50]}...")

    agent = Agent(
        name="claude_assistant",
        model="anthropic/claude-3-haiku-20240307",
        instructions="You are Claude, a helpful AI assistant. Be clear and thorough.",
        temperature=0.7,
        max_iterations=5,
    )

    result = await agent.run(message)

    return {
        "message": message,
        "response": result.output,
        "model": "anthropic/claude-3-haiku-20240307",
        "provider": "anthropic",
    }


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================


@function
async def code_assistant(ctx: FunctionContext, question: str, language: str = "python") -> dict:
    """
    Coding assistant agent specialized for programming questions.

    Args:
        question: Programming question or task
        language: Programming language context (default: python)

    Returns:
        Dict with code explanation/solution

    Example:
        result = client.run("code_assistant", {
            "question": "How do I read a file?",
            "language": "python"
        })
    """
    ctx.logger.info(f"[code_assistant] Question about {language}: {question[:50]}...")

    agent = Agent(
        name="code_assistant",
        model="openai/gpt-4o-mini",
        instructions=f"""You are an expert {language} programmer and teacher.
When answering questions:
1. Provide clear, working code examples
2. Explain the code step by step
3. Mention best practices and common pitfalls
4. Keep explanations concise but complete""",
        temperature=0.3,  # Lower temperature for more deterministic code
        max_iterations=5,
    )

    result = await agent.run(question)

    return {
        "question": question,
        "language": language,
        "response": result.output,
        "model": "openai/gpt-4o-mini",
    }


@function
async def creative_writer(ctx: FunctionContext, prompt: str, style: str = "neutral") -> dict:
    """
    Creative writing agent for generating text content.

    Args:
        prompt: Writing prompt or topic
        style: Writing style (neutral, formal, casual, humorous)

    Returns:
        Dict with generated content

    Example:
        result = client.run("creative_writer", {
            "prompt": "Write a short story about a robot",
            "style": "humorous"
        })
    """
    ctx.logger.info(f"[creative_writer] Style={style}, Prompt: {prompt[:50]}...")

    style_instructions = {
        "neutral": "Write in a clear, balanced tone.",
        "formal": "Write in a professional, formal tone with proper structure.",
        "casual": "Write in a friendly, conversational tone.",
        "humorous": "Write with wit and humor, include clever observations.",
    }

    agent = Agent(
        name="creative_writer",
        model="openai/gpt-4o-mini",
        instructions=f"""You are a creative writer.
{style_instructions.get(style, style_instructions['neutral'])}
Focus on engaging content that captivates the reader.""",
        temperature=0.9,  # Higher temperature for creativity
        max_iterations=5,
    )

    result = await agent.run(prompt)

    return {
        "prompt": prompt,
        "style": style,
        "response": result.output,
        "model": "openai/gpt-4o-mini",
    }


# =============================================================================
# MULTI-TURN CONVERSATION
# =============================================================================


@function
async def conversation_agent(
    ctx: FunctionContext,
    message: str,
    history: list = None,
) -> dict:
    """
    Agent that maintains conversation history.

    Pass previous messages in history to continue a conversation.

    Args:
        message: Current user message
        history: Previous conversation messages (list of {role, content} dicts)

    Returns:
        Dict with response and updated history

    Example:
        # First message
        result = client.run("conversation_agent", {"message": "My name is Alice"})

        # Follow-up (pass history from previous response)
        result = client.run("conversation_agent", {
            "message": "What is my name?",
            "history": result["history"]
        })
    """
    ctx.logger.info(f"[conversation_agent] Message: {message[:50]}...")

    agent = Agent(
        name="conversationalist",
        model="openai/gpt-4o-mini",
        instructions="""You are a friendly conversational assistant.
Remember context from previous messages and maintain a coherent conversation.
Be helpful, engaging, and personable.""",
        temperature=0.7,
        max_iterations=5,
    )

    # Build history if provided
    conversation_history = history if history else []

    # Run agent with history
    result = await agent.run(message, history=conversation_history)

    # Update history with new exchange
    updated_history = conversation_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.output},
    ]

    return {
        "message": message,
        "response": result.output,
        "history": updated_history,
        "turn_count": len(updated_history) // 2,
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging
    from agnt5 import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. OpenAI agent calls will fail.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return

    print("=" * 60)
    print("AGNT5 Agents Example")
    print("=" * 60)

    ctx = Context(run_id="agents-example")

    # Simple agent
    print("\n--- Simple Agent ---")
    result = await simple_agent(ctx, "What is the capital of France?")
    print(f"Response: {result['response'][:200]}...")

    # Code assistant
    print("\n--- Code Assistant ---")
    result = await code_assistant(ctx, "How do I reverse a string?", "python")
    print(f"Response: {result['response'][:300]}...")

    # Creative writer
    print("\n--- Creative Writer ---")
    result = await creative_writer(ctx, "Write a haiku about programming", "humorous")
    print(f"Response: {result['response']}")

    # Multi-turn conversation
    print("\n--- Multi-turn Conversation ---")
    result1 = await conversation_agent(ctx, "Hi! My name is Bob.")
    print(f"Turn 1: {result1['response'][:100]}...")

    result2 = await conversation_agent(ctx, "What is my name?", history=result1["history"])
    print(f"Turn 2: {result2['response'][:100]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
