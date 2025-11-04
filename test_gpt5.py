"""Test script for GPT-5 models with OpenAI Responses API.

This script tests the new GPT-5 and GPT-5-mini models.
"""

import asyncio
import os
from agnt5 import lm


async def test_gpt5_basic():
    """Test basic generation with GPT-5."""
    print("Testing GPT-5 with Responses API...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-5",
            prompt="What are you? Answer in one sentence.",
            max_tokens=100
        )

        print(f"✅ Model: gpt-5")
        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")

        if response.usage:
            assert response.usage.total_tokens > 0, "Expected non-zero token usage"

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gpt5_mini_basic():
    """Test basic generation with GPT-5-mini."""
    print("\nTesting GPT-5-mini with Responses API...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-5-mini",
            prompt="What are you? Answer in one sentence.",
            max_tokens=100
        )

        print(f"✅ Model: gpt-5-mini")
        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")

        if response.usage:
            assert response.usage.total_tokens > 0, "Expected non-zero token usage"

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gpt5_reasoning():
    """Test GPT-5 with a reasoning task."""
    print("\nTesting GPT-5 with reasoning task...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-5",
            prompt="""Solve this logic puzzle:

You have 3 boxes. One contains only apples, one contains only oranges,
and one contains both. All boxes are labeled incorrectly. You can pick
one fruit from one box without looking inside. How do you label all boxes correctly?

Think step by step.""",
            max_tokens=500
        )

        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gpt5_mini_reasoning():
    """Test GPT-5-mini with a reasoning task."""
    print("\nTesting GPT-5-mini with reasoning task...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-5-mini",
            prompt="""What is 15 * 23? Show your work step by step.""",
            max_tokens=300
        )

        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")

        # Check if answer is correct
        if "345" in response.text:
            print("✅ Correct answer found in response!")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gpt5_conversation():
    """Test GPT-5 with multi-turn conversation."""
    print("\nTesting GPT-5 with conversation...")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False

    try:
        response = await lm.generate(
            model="openai/gpt-5",
            messages=[
                {"role": "user", "content": "I'm working on a complex distributed system."},
                {"role": "assistant", "content": "That sounds interesting! What kind of distributed system are you building?"},
                {"role": "user", "content": "An AI agent orchestration platform. What are the key challenges I should focus on?"}
            ],
            max_tokens=300
        )

        print(f"✅ Response: {response.text}")
        print(f"✅ Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all GPT-5 tests."""
    print("=" * 60)
    print("GPT-5 Models Test Suite (Responses API)")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(await test_gpt5_basic())
    results.append(await test_gpt5_mini_basic())
    results.append(await test_gpt5_reasoning())
    results.append(await test_gpt5_mini_reasoning())
    results.append(await test_gpt5_conversation())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All GPT-5 tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
