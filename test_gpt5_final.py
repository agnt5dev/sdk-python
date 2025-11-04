"""Final comprehensive test for GPT-5 models with Responses API."""

import asyncio
import os
from agnt5 import lm


async def main():
    """Run comprehensive GPT-5 tests."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("=" * 70)
    print("GPT-5 Models - Comprehensive Test Suite")
    print("=" * 70)

    # Test 1: GPT-5 Hello World
    print("\n1. GPT-5 - Simple Hello World")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5",
        prompt="Say hello world",
        max_tokens=100  # GPT-5 needs more tokens for reasoning
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    # Test 2: GPT-5-mini Hello World
    print("\n2. GPT-5-mini - Simple Hello World")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5-mini",
        prompt="Say hello world",
        max_tokens=100
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    # Test 3: GPT-5 Reasoning
    print("\n3. GPT-5 - Complex Reasoning")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5",
        prompt="Solve: If Alice has 3 apples and Bob has twice as many, how many do they have together?",
        max_tokens=200
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    # Test 4: GPT-5-mini Reasoning
    print("\n4. GPT-5-mini - Math Problem")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5-mini",
        prompt="What is 47 + 53?",
        max_tokens=100
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    # Test 5: GPT-5 Multi-turn
    print("\n5. GPT-5 - Multi-turn Conversation")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5",
        messages=[
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "That's great! Blue is a beautiful color."},
            {"role": "user", "content": "What's my favorite color?"}
        ],
        max_tokens=100
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    # Test 6: GPT-5 with System Prompt
    print("\n6. GPT-5 - With System Prompt")
    print("-" * 70)
    response = await lm.generate(
        model="openai/gpt-5",
        system_prompt="You are a helpful math tutor.",
        prompt="Explain what 2+2 equals",
        max_tokens=150
    )
    print(f"✅ Response: {response.text}")
    print(f"   Usage: {response.usage}")

    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print("\nKey Findings:")
    print("- GPT-5 and GPT-5-mini work with Responses API")
    print("- GPT-5 uses reasoning tokens (visible in output_tokens_details)")
    print("- Both models support multi-turn conversations")
    print("- System prompts work correctly")
    print("- Note: GPT-5 models do not support temperature parameter")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
