"""
Synchronous LLM Example

This example demonstrates synchronous (non-streaming) LLM usage with multiple providers.
Shows the new simplified API that auto-detects providers from model prefixes.

Prerequisites:
    Set environment variables for the providers you want to use:
    - OPENAI_API_KEY=your-key          # For OpenAI
    - ANTHROPIC_API_KEY=your-key       # For Anthropic
    - GROQ_API_KEY=your-key            # For Groq
    - OPENROUTER_API_KEY=your-key      # For OpenRouter

Usage:
    python 12_llm_sync.py
"""

import asyncio
import os

from agnt5 import lm


async def example_openai():
    """Example: OpenAI GPT-4o with simple prompt."""
    print("=== Example 1: OpenAI GPT-4o (Simple) ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Simple one-liner generation
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Explain what AGNT5 is in one sentence.",
        temperature=0.7,
        max_tokens=100
    )

    print(f"Response: {response.text}\n")
    if response.usage:
        print(f"Token usage: {response.usage.total_tokens} tokens")
        print(f"  Prompt: {response.usage.prompt_tokens}")
        print(f"  Completion: {response.usage.completion_tokens}\n")


async def example_anthropic():
    """Example: Anthropic Claude."""
    print("=== Example 2: Anthropic Claude ===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Skipped: ANTHROPIC_API_KEY not set\n")
        return

    response = await lm.generate(
        model="anthropic/claude-3-5-haiku-20241022",
        prompt="What are the three most important principles of good software design?",
        temperature=0.7,
        max_tokens=200
    )

    print(f"Response: {response.text}\n")
    if response.usage:
        print(f"Token usage: {response.usage.total_tokens} tokens\n")


async def example_groq():
    """Example: Groq (fast inference)."""
    print("=== Example 3: Groq (Fast Inference) ===\n")

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Skipped: GROQ_API_KEY not set\n")
        return

    response = await lm.generate(
        model="groq/llama-3.3-70b-versatile",
        prompt="Write a haiku about programming.",
        temperature=0.9,
        max_tokens=100
    )

    print(f"Response: {response.text}\n")
    if response.usage:
        print(f"Token usage: {response.usage.total_tokens} tokens\n")


async def example_openrouter():
    """Example: OpenRouter (access to multiple models)."""
    print("=== Example 4: OpenRouter ===\n")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Skipped: OPENROUTER_API_KEY not set\n")
        return

    response = await lm.generate(
        model="openrouter/anthropic/claude-3.5-haiku",
        prompt="What is the capital of France?",
        temperature=0.3,
        max_tokens=50
    )

    print(f"Response: {response.text}\n")
    if response.usage:
        print(f"Token usage: {response.usage.total_tokens} tokens\n")


async def example_multi_turn_conversation():
    """Example: Multi-turn conversation using messages."""
    print("=== Example 5: Multi-Turn Conversation ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # First turn
    response1 = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful math tutor.",
        messages=[
            {"role": "user", "content": "What is calculus?"}
        ],
        temperature=0.7,
        max_tokens=150
    )

    print(f"User: What is calculus?")
    print(f"Assistant: {response1.text}\n")

    # Second turn (continue conversation)
    response2 = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful math tutor.",
        messages=[
            {"role": "user", "content": "What is calculus?"},
            {"role": "assistant", "content": response1.text},
            {"role": "user", "content": "Can you give me a simple example?"}
        ],
        temperature=0.7,
        max_tokens=200
    )

    print(f"User: Can you give me a simple example?")
    print(f"Assistant: {response2.text}\n")


async def example_temperature_comparison():
    """Example: Compare different temperature settings."""
    print("=== Example 6: Temperature Comparison ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    prompt = "Tell me a creative story idea in one sentence."
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt=prompt,
            temperature=temp,
            max_tokens=100
        )

        print(f"Temperature {temp}:")
        print(f"  {response.text}\n")


async def main():
    print("=== AGNT5 Synchronous LLM Examples ===")
    print("Simplified API with auto-detection from model prefix\n")

    try:
        # Run all examples
        await example_openai()
        await example_anthropic()
        await example_groq()
        await example_openrouter()
        await example_multi_turn_conversation()
        await example_temperature_comparison()

        print("=== All Examples Complete ===")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
