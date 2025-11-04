"""Test script for OpenAI Responses API integration.

This script tests the new OpenAI Responses API provider implementation.
"""

import asyncio
import os
from agnt5 import lm


async def test_basic_generation():
    """Test basic text generation with Responses API."""
    print("Testing basic generation with OpenAI Responses API...")

    # Set API key from environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Say hello in exactly 5 words.",
            temperature=0.7,
            max_tokens=50
        )

        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")

        if response.usage:
            assert response.usage.total_tokens > 0, "Expected non-zero token usage"

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_with_system_prompt():
    """Test generation with system prompt."""
    print("\nTesting with system prompt...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant that speaks like a pirate.",
            prompt="What is the weather like?",
            temperature=0.7,
            max_tokens=100
        )

        print(f"✅ Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    print("\nTesting multi-turn conversation...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What's my name?"}
            ],
            temperature=0.7,
            max_tokens=50
        )

        print(f"✅ Response: {response.text}")
        assert "alice" in response.text.lower(), "Expected assistant to remember the name"
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_streaming():
    """Test streaming generation."""
    print("\nTesting streaming...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        print("Stream output: ", end="", flush=True)
        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o-mini",
            prompt="Count from 1 to 5.",
            temperature=0.7,
            max_tokens=50
        ):
            print(chunk, end="", flush=True)
            chunks.append(chunk)

        print()  # New line
        full_text = "".join(chunks)
        print(f"✅ Streamed {len(chunks)} chunks, total text length: {len(full_text)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenAI Responses API Provider Tests")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(await test_basic_generation())
    results.append(await test_with_system_prompt())
    results.append(await test_multi_turn_conversation())
    results.append(await test_streaming())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
