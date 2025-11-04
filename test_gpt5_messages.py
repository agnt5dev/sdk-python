"""Test GPT-5 with messages format."""

import asyncio
import os
from agnt5 import lm


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("Testing GPT-5 with messages format...")
    response = await lm.generate(
        model="openai/gpt-5",
        messages=[
            {"role": "user", "content": "Say exactly: Hello World"}
        ],
        max_tokens=50
    )

    print(f"\nResponse: '{response.text}'")
    print(f"Text length: {len(response.text)}")
    print(f"Usage: {response.usage}")
    print(f"Finish reason: {response.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
