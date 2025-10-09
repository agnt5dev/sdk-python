"""Quick test for structured output implementation."""

import asyncio
from dataclasses import dataclass
from typing import List

from src.agnt5 import lm


@dataclass
class SimpleReview:
    """Simple code review structure."""
    summary: str
    score: int


async def test_simple():
    """Test basic structured output with dataclass."""
    print("Testing structured output...")

    try:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Review this code: def add(a, b): return a + b",
            response_format=SimpleReview,
            temperature=0.3,
        )

        print(f"✅ Response received!")
        print(f"   Text: {response.text[:100]}...")
        print(f"   Structured Output: {response.structured_output}")
        print(f"   Parsed: {response.parsed}")
        print(f"   Object: {response.object}")

        if response.structured_output:
            print(f"\n✅ Structured output works!")
            print(f"   Type: {type(response.structured_output)}")
            print(f"   Content: {response.structured_output}")
        else:
            print(f"\n⚠️  No structured output found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())
