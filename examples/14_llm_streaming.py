"""
Streaming LLM Example

This example demonstrates streaming responses from LLMs with multiple providers.
Shows the new simplified API with real-time token streaming.

Prerequisites:
    Set environment variables for the providers you want to use:
    - OPENAI_API_KEY=your-key
    - ANTHROPIC_API_KEY=your-key
    - GROQ_API_KEY=your-key

Usage:
    python 14_llm_streaming.py
"""

import asyncio
import os
import sys
import time

from agnt5 import lm


async def example_basic_streaming():
    """Example: Basic streaming with OpenAI."""
    print("=== Example 1: Basic Streaming ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("Streaming response:")
    print("-" * 60)

    # Stream the response
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a short story about a robot learning to paint. Keep it to 3 paragraphs.",
        temperature=0.8,
        max_tokens=300
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60 + "\n")


async def example_streaming_with_indicator():
    """Example: Streaming with a typing indicator."""
    print("=== Example 2: Streaming with Typing Indicator ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("Assistant is typing", end="", flush=True)

    # Show typing indicator
    first_chunk = True
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Explain quantum computing in simple terms.",
        temperature=0.7,
        max_tokens=200
    ):
        if first_chunk:
            # Clear typing indicator
            print("\r" + " " * 30 + "\r", end="", flush=True)
            print("Assistant: ", end="", flush=True)
            first_chunk = False
        print(chunk, end="", flush=True)

    print("\n")


async def example_multi_provider_streaming():
    """Example: Compare streaming from different providers."""
    print("=== Example 3: Multi-Provider Streaming Comparison ===\n")

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

    question = "What are the benefits of streaming responses in AI applications?"

    for provider_name, model in providers:
        print(f"\n[{provider_name}]")
        print("-" * 60)

        start_time = time.time()
        async for chunk in lm.stream(
            model=model,
            prompt=question,
            temperature=0.7,
            max_tokens=150
        ):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start_time

        print(f"\n\n⏱️  Time: {elapsed:.2f}s")
        print("-" * 60)


async def example_streaming_with_stats():
    """Example: Track streaming statistics."""
    print("\n=== Example 4: Streaming with Statistics ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Track statistics
    start_time = time.time()
    chunk_count = 0
    total_chars = 0
    chunks_text = []

    print("Streaming haiku:")
    print("-" * 40)

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a haiku about code.",
        temperature=0.9,
        max_tokens=100
    ):
        print(chunk, end="", flush=True)
        chunk_count += 1
        total_chars += len(chunk)
        chunks_text.append(chunk)

    elapsed = time.time() - start_time

    print("\n" + "-" * 40)
    print("\nStatistics:")
    print(f"  Chunks received: {chunk_count}")
    print(f"  Total characters: {total_chars}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Average chunk size: {total_chars/chunk_count:.1f} chars")
    print(f"  Chars per second: {total_chars/elapsed:.1f}\n")


async def example_streaming_with_buffering():
    """Example: Buffer chunks and display by words."""
    print("=== Example 5: Buffered Word-by-Word Display ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("Response (word-by-word):")
    print("-" * 60)

    buffer = ""
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="List 5 tips for writing clean code.",
        temperature=0.7,
        max_tokens=200
    ):
        buffer += chunk

        # Display complete words
        while " " in buffer or "\n" in buffer:
            # Find the next word boundary
            space_idx = buffer.find(" ")
            newline_idx = buffer.find("\n")

            if space_idx == -1:
                split_idx = newline_idx
            elif newline_idx == -1:
                split_idx = space_idx
            else:
                split_idx = min(space_idx, newline_idx)

            # Extract and display word
            word = buffer[:split_idx + 1]
            print(word, end="", flush=True)
            buffer = buffer[split_idx + 1:]

            # Small delay for visual effect
            await asyncio.sleep(0.05)

    # Display remaining buffer
    if buffer:
        print(buffer, end="", flush=True)

    print("\n" + "-" * 60 + "\n")


async def example_streaming_cancellation():
    """Example: Cancel streaming mid-response."""
    print("=== Example 6: Streaming Cancellation ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    print("Streaming response (will stop after 200 chars):")
    print("-" * 60)

    char_count = 0
    max_chars = 200

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a long essay about the history of computers. Be very detailed.",
        temperature=0.7,
        max_tokens=500
    ):
        print(chunk, end="", flush=True)
        char_count += len(chunk)

        if char_count >= max_chars:
            print("\n\n[Cancelled after 200 characters]")
            break

    print("\n" + "-" * 60 + "\n")


async def example_parallel_streaming():
    """Example: Stream from multiple providers in parallel."""
    print("=== Example 7: Parallel Streaming ===\n")

    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI", "openai/gpt-4o-mini"))
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq", "groq/llama-3.3-70b-versatile"))

    if len(providers) < 2:
        print("⚠️  Skipped: Need at least 2 API keys set\n")
        return

    question = "What is AI?"

    async def stream_provider(name: str, model: str):
        """Stream from a single provider."""
        print(f"\n[{name}] Starting...")
        response_parts = []
        async for chunk in lm.stream(
            model=model,
            prompt=question,
            temperature=0.7,
            max_tokens=100
        ):
            response_parts.append(chunk)

        full_response = "".join(response_parts)
        print(f"\n[{name}] Complete ({len(full_response)} chars)")
        print(f"  {full_response[:100]}...")

    # Stream from all providers in parallel
    print("Streaming from multiple providers simultaneously...")
    tasks = [stream_provider(name, model) for name, model in providers]
    await asyncio.gather(*tasks)
    print()


async def main():
    print("=== AGNT5 Streaming LLM Examples ===")
    print("Real-time token streaming with simplified API\n")

    try:
        # Run all examples
        await example_basic_streaming()
        await example_streaming_with_indicator()
        await example_multi_provider_streaming()
        await example_streaming_with_stats()
        await example_streaming_with_buffering()
        await example_streaming_cancellation()
        await example_parallel_streaming()

        print("=== All Examples Complete ===")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
