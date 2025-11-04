"""Simple GPT-5 test to check text output."""

import asyncio
import os
from agnt5 import lm


async def test_gpt5():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("Testing GPT-5...")
    response = await lm.generate(
        model="openai/gpt-5",
        prompt="Say exactly: Hello World",
        max_tokens=50
    )

    print(f"\nResponse object: {response}")
    print(f"\nText attribute: '{response.text}'")
    print(f"Text length: {len(response.text)}")
    print(f"Text repr: {repr(response.text)}")
    print(f"Usage: {response.usage}")


async def test_gpt5_mini():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("\n\nTesting GPT-5-mini...")
    response = await lm.generate(
        model="openai/gpt-5-mini",
        prompt="Say exactly: Hello World",
        max_tokens=50
    )

    print(f"\nResponse object: {response}")
    print(f"\nText attribute: '{response.text}'")
    print(f"Text length: {len(response.text)}")
    print(f"Text repr: {repr(response.text)}")
    print(f"Usage: {response.usage}")


async def main():
    await test_gpt5()
    await test_gpt5_mini()


if __name__ == "__main__":
    asyncio.run(main())
