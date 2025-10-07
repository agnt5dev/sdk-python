"""
Basic Streaming Example

This example demonstrates streaming responses from AGNT5 components.
Simulates LLM token generation by yielding words one at a time.

Usage:
    1. Start the dev server: just dev-server sdk/sdk-python/examples simple-python-service
    2. Start this worker: uv run python 11_streaming_basic.py
    3. In another terminal, test streaming: uv run python test_streaming_client.py
"""

import time
from agnt5 import function, Worker, Context


@function
async def generate_story(ctx: Context, prompt: str):
    """Stream text generation word by word (TRUE streaming with yield).

    This is an async generator function that yields chunks as they're produced.
    Each word is yielded as a separate chunk to simulate LLM token streaming.
    """
    import asyncio

    # Generate a response based on the prompt
    if "story" in prompt.lower():
        text = "Once upon a time, in a land far away, there lived a brave developer who built amazing AI applications with AGNT5."
    elif "poem" in prompt.lower():
        text = "Code flows like poetry, functions dance with grace, AGNT5 makes it easy, streaming in its place."
    else:
        text = f"Here's a response to your prompt: '{prompt}'. This text is being streamed token by token to demonstrate streaming capabilities."

    # Stream each word with a small delay (simulating LLM token generation)
    words = text.split()
    for i, word in enumerate(words):
        await asyncio.sleep(0.05)  # 50ms delay per word
        yield {
            "chunk": word + " ",
            "index": i,
            "total": len(words)
        }


@function
async def count_to_n(ctx: Context, n: int):
    """Count from 1 to n, streaming each number.

    Demonstrates streaming numerical data with incremental chunks.
    """
    import asyncio

    if n > 100:
        n = 100  # Limit to prevent abuse

    for i in range(1, n + 1):
        await asyncio.sleep(0.1)  # 100ms delay per number
        yield {
            "number": i,
            "total": n,
            "percentage": round((i / n) * 100, 1)
        }


if __name__ == "__main__":
    import asyncio
    import os

    # Use service name from environment or default to "default-service"
    service_name = os.getenv("AGNT5_SERVICE_NAME", "default-service")
    worker = Worker(service_name)

    print("🚀 Starting AGNT5 Worker with streaming functions...")
    print(f"📦 Service: {service_name}")
    print("📝 Registered functions:")
    print("  - generate_story: Simulate LLM text generation")
    print("  - count_to_n: Count from 1 to n")
    print()
    print("Waiting for invocations...")

    asyncio.run(worker.run())
