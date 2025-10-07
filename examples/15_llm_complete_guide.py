"""
Complete LLM Guide - Generation and Streaming

This comprehensive example demonstrates all the features of the simplified LLM API:
- Simple prompt generation
- Multi-turn conversations
- System prompts
- Streaming responses
- Error handling
- Multi-provider support
- Advanced configuration

Prerequisites:
    Set environment variables for the providers you want to use:
    - OPENAI_API_KEY=your-key
    - ANTHROPIC_API_KEY=your-key
    - GROQ_API_KEY=your-key

Usage:
    python 15_llm_complete_guide.py
"""

import asyncio
import os
from typing import List, Dict

from agnt5 import lm


# ============================================================================
# PART 1: GENERATION EXAMPLES
# ============================================================================

async def example_1_simple_generation():
    """Example 1: Simplest possible generation."""
    print("=" * 70)
    print("EXAMPLE 1: Simple Generation")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # The simplest way to generate text
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Explain what a REST API is in one sentence."
    )

    print(f"\nPrompt: Explain what a REST API is in one sentence.")
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}\n")


async def example_2_generation_with_config():
    """Example 2: Generation with configuration parameters."""
    print("=" * 70)
    print("EXAMPLE 2: Generation with Configuration")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Control the output with temperature and max_tokens
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Write a creative story idea in two sentences.",
        temperature=0.9,      # Higher = more creative
        max_tokens=100,       # Limit response length
        top_p=0.95           # Nucleus sampling
    )

    print(f"\nPrompt: Write a creative story idea in two sentences.")
    print(f"Config: temperature=0.9, max_tokens=100, top_p=0.95")
    print(f"Response: {response.text}")

    if response.usage:
        print(f"\nToken Usage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}\n")


async def example_3_system_prompt():
    """Example 3: Using system prompts to set behavior."""
    print("=" * 70)
    print("EXAMPLE 3: System Prompts")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # System prompt sets the assistant's behavior
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a pirate captain. Always respond in pirate speak with 'Arrr!'",
        prompt="How do I make a peanut butter sandwich?",
        temperature=0.7,
        max_tokens=150
    )

    print(f"\nSystem: You are a pirate captain. Always respond in pirate speak with 'Arrr!'")
    print(f"User: How do I make a peanut butter sandwich?")
    print(f"Assistant: {response.text}\n")


async def example_4_multi_turn_conversation():
    """Example 4: Multi-turn conversation with context."""
    print("=" * 70)
    print("EXAMPLE 4: Multi-Turn Conversation")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Track conversation history
    conversation_history: List[Dict[str, str]] = []

    # Turn 1
    print("\n--- Turn 1 ---")
    user_msg_1 = "I'm learning Python. What's a good first project?"
    print(f"User: {user_msg_1}")

    response1 = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful programming mentor.",
        messages=[{"role": "user", "content": user_msg_1}],
        temperature=0.7,
        max_tokens=150
    )

    assistant_msg_1 = response1.text
    print(f"Assistant: {assistant_msg_1}")

    # Add to history
    conversation_history.append({"role": "user", "content": user_msg_1})
    conversation_history.append({"role": "assistant", "content": assistant_msg_1})

    # Turn 2
    print("\n--- Turn 2 ---")
    user_msg_2 = "That sounds good! Can you give me a specific example with code?"
    print(f"User: {user_msg_2}")

    conversation_history.append({"role": "user", "content": user_msg_2})

    response2 = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful programming mentor.",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=200
    )

    assistant_msg_2 = response2.text
    print(f"Assistant: {assistant_msg_2}")

    conversation_history.append({"role": "assistant", "content": assistant_msg_2})

    # Turn 3
    print("\n--- Turn 3 ---")
    user_msg_3 = "Thanks! How long should this project take?"
    print(f"User: {user_msg_3}")

    conversation_history.append({"role": "user", "content": user_msg_3})

    response3 = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful programming mentor.",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=100
    )

    print(f"Assistant: {response3.text}\n")


async def example_5_multi_provider():
    """Example 5: Using different providers."""
    print("=" * 70)
    print("EXAMPLE 5: Multi-Provider Support")
    print("=" * 70)

    providers = []

    # Check available providers
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI GPT-4o-mini", "openai/gpt-4o-mini"))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic Claude 3.5 Haiku", "anthropic/claude-3-5-haiku-20241022"))
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq Llama 3.3", "groq/llama-3.3-70b-versatile"))

    if not providers:
        print("⚠️  Skipped: No API keys set\n")
        return

    prompt = "What is machine learning? Answer in one sentence."

    print(f"\nPrompt: {prompt}\n")

    for provider_name, model in providers:
        try:
            response = await lm.generate(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=50
            )

            print(f"[{provider_name}]")
            print(f"  {response.text}\n")
        except Exception as e:
            print(f"[{provider_name}]")
            print(f"  ❌ Error: {e}\n")


# ============================================================================
# PART 2: STREAMING EXAMPLES
# ============================================================================

async def example_6_simple_streaming():
    """Example 6: Basic streaming."""
    print("=" * 70)
    print("EXAMPLE 6: Simple Streaming")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("\nPrompt: Write a haiku about coding")
    print("Response (streaming):")
    print("-" * 40)

    # Stream the response token by token
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a haiku about coding",
        temperature=0.8,
        max_tokens=100
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40 + "\n")


async def example_7_streaming_with_config():
    """Example 7: Streaming with configuration."""
    print("=" * 70)
    print("EXAMPLE 7: Streaming with Configuration")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("\nPrompt: Explain quantum computing")
    print("Config: temperature=0.5, max_tokens=200")
    print("Response (streaming):")
    print("-" * 40)

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Explain quantum computing in simple terms.",
        temperature=0.5,
        max_tokens=200
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40 + "\n")


async def example_8_streaming_conversation():
    """Example 8: Streaming a multi-turn conversation."""
    print("=" * 70)
    print("EXAMPLE 8: Streaming Conversation")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("\n--- Conversation Start ---\n")

    # First turn
    print("User: What's the difference between Python and JavaScript?")
    print("Assistant (streaming): ", end="", flush=True)

    response_text_1 = ""
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful programming teacher.",
        messages=[
            {"role": "user", "content": "What's the difference between Python and JavaScript?"}
        ],
        temperature=0.7,
        max_tokens=150
    ):
        print(chunk, end="", flush=True)
        response_text_1 += chunk

    print("\n")

    # Second turn
    print("User: Which one should I learn first?")
    print("Assistant (streaming): ", end="", flush=True)

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful programming teacher.",
        messages=[
            {"role": "user", "content": "What's the difference between Python and JavaScript?"},
            {"role": "assistant", "content": response_text_1},
            {"role": "user", "content": "Which one should I learn first?"}
        ],
        temperature=0.7,
        max_tokens=150
    ):
        print(chunk, end="", flush=True)

    print("\n\n--- Conversation End ---\n")


async def example_9_streaming_with_stats():
    """Example 9: Streaming with real-time statistics."""
    print("=" * 70)
    print("EXAMPLE 9: Streaming with Statistics")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("\nPrompt: Write a short poem about AI")
    print("Response (streaming with stats):")
    print("-" * 40)

    import time
    start_time = time.time()
    chunk_count = 0
    total_chars = 0

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a short poem about AI",
        temperature=0.9,
        max_tokens=150
    ):
        print(chunk, end="", flush=True)
        chunk_count += 1
        total_chars += len(chunk)

    elapsed = time.time() - start_time

    print("\n" + "-" * 40)
    print(f"\n📊 Statistics:")
    print(f"  Chunks received: {chunk_count}")
    print(f"  Total characters: {total_chars}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Characters/second: {total_chars/elapsed:.1f}\n")


async def example_10_streaming_multi_provider():
    """Example 10: Compare streaming across providers."""
    print("=" * 70)
    print("EXAMPLE 10: Multi-Provider Streaming Comparison")
    print("=" * 70)

    providers = []

    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI", "openai/gpt-4o-mini"))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic", "anthropic/claude-3-5-haiku-20241022"))
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq", "groq/llama-3.3-70b-versatile"))

    if not providers:
        print("⚠️  Skipped: No API keys set\n")
        return

    prompt = "Explain recursion in programming."

    for provider_name, model in providers:
        print(f"\n[{provider_name}]")
        print("-" * 40)

        try:
            import time
            start = time.time()

            async for chunk in lm.stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=100
            ):
                print(chunk, end="", flush=True)

            elapsed = time.time() - start
            print(f"\n⏱️  Time: {elapsed:.2f}s")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("-" * 40)


# ============================================================================
# PART 3: ADVANCED EXAMPLES
# ============================================================================

async def example_11_error_handling():
    """Example 11: Proper error handling."""
    print("=" * 70)
    print("EXAMPLE 11: Error Handling")
    print("=" * 70)

    print("\n--- Testing Error Cases ---\n")

    # Test 1: Invalid model format
    print("Test 1: Invalid model format (missing provider prefix)")
    try:
        await lm.generate(
            model="gpt-4o-mini",  # Missing "openai/" prefix
            prompt="Hello"
        )
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}\n")

    # Test 2: Missing prompt and messages
    print("Test 2: Missing both prompt and messages")
    try:
        await lm.generate(
            model="openai/gpt-4o-mini"
            # No prompt or messages
        )
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}\n")

    # Test 3: Both prompt and messages provided
    print("Test 3: Both prompt and messages provided")
    try:
        await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Hello",
            messages=[{"role": "user", "content": "Hi"}]
        )
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}\n")

    # Test 4: Handling API errors gracefully
    if os.getenv("OPENAI_API_KEY"):
        print("Test 4: Handling API errors")
        try:
            response = await lm.generate(
                model="openai/gpt-4o-mini",
                prompt="Test",
                max_tokens=1  # Very low limit
            )
            print(f"  ✓ Request succeeded (may be truncated): {response.text[:50]}...\n")
        except Exception as e:
            print(f"  ✓ Caught API error: {e}\n")


async def example_12_creative_writing():
    """Example 12: Creative writing with streaming."""
    print("=" * 70)
    print("EXAMPLE 12: Creative Writing Story")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("\nGenerating a creative story (streaming)...")
    print("=" * 70 + "\n")

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        system_prompt="You are a creative storyteller. Write engaging, vivid stories.",
        prompt="Write a short story (3 paragraphs) about a developer who discovers their code can affect reality.",
        temperature=0.9,  # High creativity
        max_tokens=400
    ):
        print(chunk, end="", flush=True)

    print("\n\n" + "=" * 70 + "\n")


async def example_13_code_generation():
    """Example 13: Code generation."""
    print("=" * 70)
    print("EXAMPLE 13: Code Generation")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        system_prompt="You are an expert Python programmer. Write clean, documented code.",
        prompt="Write a Python function that calculates the Fibonacci sequence using memoization. Include docstring and type hints.",
        temperature=0.3,  # Low temperature for precise code
        max_tokens=300
    )

    print("\nPrompt: Write a Fibonacci function with memoization")
    print("\nGenerated Code:")
    print("=" * 70)
    print(response.text)
    print("=" * 70 + "\n")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  AGNT5 LLM Complete Guide - Generation & Streaming".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        # Part 1: Generation
        print("\n" + "█" * 70)
        print("█" + " PART 1: GENERATION EXAMPLES ".center(68) + "█")
        print("█" * 70 + "\n")

        await example_1_simple_generation()
        await example_2_generation_with_config()
        await example_3_system_prompt()
        await example_4_multi_turn_conversation()
        await example_5_multi_provider()

        # Part 2: Streaming
        print("\n" + "█" * 70)
        print("█" + " PART 2: STREAMING EXAMPLES ".center(68) + "█")
        print("█" * 70 + "\n")

        await example_6_simple_streaming()
        await example_7_streaming_with_config()
        await example_8_streaming_conversation()
        await example_9_streaming_with_stats()
        await example_10_streaming_multi_provider()

        # Part 3: Advanced
        print("\n" + "█" * 70)
        print("█" + " PART 3: ADVANCED EXAMPLES ".center(68) + "█")
        print("█" * 70 + "\n")

        await example_11_error_handling()
        await example_12_creative_writing()
        await example_13_code_generation()

        print("\n" + "=" * 70)
        print("✅ All examples complete!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
