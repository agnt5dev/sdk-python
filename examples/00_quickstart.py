"""
AGNT5 LLM Quickstart - Simple & Clean API

The simplest way to use LLMs with AGNT5. Works with OpenAI, Anthropic, Groq,
OpenRouter, and more. Provider auto-detected from model prefix.

Prerequisites:
    export OPENAI_API_KEY=your-key
    # or ANTHROPIC_API_KEY, GROQ_API_KEY, etc.

Usage:
    python 00_quickstart.py
"""

import asyncio
from agnt5 import lm


async def main():
    # ================================================================
    # 1. Simple Generation (one-shot prompt)
    # ================================================================

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Explain what AGNT5 is in one sentence.",
        temperature=0.7,
        max_tokens=100
    )

    print("=" * 60)
    print("SIMPLE GENERATION")
    print("=" * 60)
    print(f"Response: {response.text}")
    print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}\n")


    # ================================================================
    # 2. Multi-Turn Conversation (with messages)
    # ================================================================

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful Python tutor."},
            {"role": "user", "content": "What's a decorator?"},
            {"role": "assistant", "content": "A decorator is a function that modifies another function..."},
            {"role": "user", "content": "Show me an example"}
        ],
        temperature=0.7,
        max_tokens=200
    )

    print("=" * 60)
    print("MULTI-TURN CONVERSATION")
    print("=" * 60)
    print(f"Response: {response.text}\n")


    # ================================================================
    # 3. Streaming (real-time tokens)
    # ================================================================

    print("=" * 60)
    print("STREAMING RESPONSE")
    print("=" * 60)
    print("Response: ", end="", flush=True)

    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a haiku about coding",
        temperature=0.9,
        max_tokens=100
    ):
        print(chunk, end="", flush=True)

    print("\n")


    # ================================================================
    # 4. Different Providers (just change the prefix!)
    # ================================================================

    print("=" * 60)
    print("MULTI-PROVIDER SUPPORT")
    print("=" * 60)

    # OpenAI
    r1 = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="What is 2+2?"
    )
    print(f"OpenAI: {r1.text}")

    # Anthropic (if you have the API key)
    # r2 = await lm.generate(
    #     model="anthropic/claude-3-5-haiku-20241022",
    #     prompt="What is 2+2?"
    # )
    # print(f"Anthropic: {r2.text}")

    # Groq (if you have the API key)
    # r3 = await lm.generate(
    #     model="groq/llama-3.3-70b-versatile",
    #     prompt="What is 2+2?"
    # )
    # print(f"Groq: {r3.text}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
