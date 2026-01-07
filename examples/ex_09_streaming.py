"""
Example: AGNT5 Streaming Functions

This example demonstrates streaming responses:
- Async generator functions that yield chunks
- Simulated LLM token streaming
- Progress streaming for long operations

Streaming enables real-time data delivery:
- Perfect for LLM token streaming
- Progress updates for long operations
- Large data transfer in chunks

Each function can be:
- Run standalone: python ex_09_streaming.py
- Registered with platform: imported by app.py
- Invoked via client: client.stream("stream_text", {"text": "..."})
"""

import asyncio

from agnt5 import function, FunctionContext


# =============================================================================
# TEXT STREAMING
# =============================================================================


@function
async def stream_text(ctx: FunctionContext, text: str, delay_ms: int = 50):
    """
    Stream text word by word.

    Simulates LLM token streaming by yielding words with delays.

    Args:
        text: The text to stream
        delay_ms: Delay in milliseconds between words (default: 50)

    Yields:
        Dict with chunk, word_index, and total_words

    Example:
        for chunk in client.stream("stream_text", {"text": "Hello world"}):
            print(chunk, end="", flush=True)
    """
    words = text.split()
    total = len(words)

    ctx.logger.info(f"[stream_text] Streaming {total} words...")

    for i, word in enumerate(words):
        await asyncio.sleep(delay_ms / 1000.0)
        yield word + " "


@function
async def stream_story(ctx: FunctionContext, prompt: str):
    """
    Generate and stream a story based on the prompt.

    Simulates creative text generation with streaming output.

    Args:
        prompt: The story prompt

    Yields:
        Story text word by word

    Example:
        for chunk in client.stream("stream_story", {"prompt": "dragon"}):
            print(chunk, end="", flush=True)
    """
    # Generate story based on prompt keywords
    if "dragon" in prompt.lower():
        story = "Once upon a time, in a land of mountains and magic, there lived a friendly dragon named Ember. Ember loved to help travelers find their way through the misty valleys."
    elif "robot" in prompt.lower():
        story = "In the year 3000, a small robot named Spark discovered something amazing. Hidden in the old data archives was a message from the first humans who built robots."
    elif "ocean" in prompt.lower():
        story = "Deep beneath the waves, where sunlight never reaches, an ancient city slept. Its crystal towers glowed with bioluminescent life."
    else:
        story = f"This is a story about: {prompt}. It begins with a mysterious event that would change everything. The adventure was about to begin."

    ctx.logger.info(f"[stream_story] Streaming story for prompt: {prompt[:30]}...")

    words = story.split()
    for word in words:
        await asyncio.sleep(0.05)  # 50ms per word
        yield word + " "


# =============================================================================
# PROGRESS STREAMING
# =============================================================================


@function
async def stream_counter(ctx: FunctionContext, count: int, delay_ms: int = 100):
    """
    Stream a count from 1 to n.

    Useful for testing streaming with structured data.

    Args:
        count: The number to count to (max 100)
        delay_ms: Delay in milliseconds between numbers

    Yields:
        Current number as string

    Example:
        for num in client.stream("stream_counter", {"count": 10}):
            print(num)
    """
    # Limit to prevent abuse
    count = min(count, 100)

    ctx.logger.info(f"[stream_counter] Counting to {count}...")

    for i in range(1, count + 1):
        await asyncio.sleep(delay_ms / 1000.0)
        yield str(i)


@function
async def stream_progress(ctx: FunctionContext, steps: int, step_duration_ms: int = 200):
    """
    Stream progress updates for a simulated long-running task.

    Args:
        steps: Number of steps in the process
        step_duration_ms: Duration of each step in milliseconds

    Yields:
        Progress messages

    Example:
        for update in client.stream("stream_progress", {"steps": 5}):
            print(update)
    """
    steps = min(steps, 20)  # Limit steps

    ctx.logger.info(f"[stream_progress] Starting {steps}-step process...")

    yield f"Starting {steps}-step process...\n"

    for i in range(1, steps + 1):
        await asyncio.sleep(step_duration_ms / 1000.0)
        percentage = int((i / steps) * 100)
        yield f"Step {i}/{steps} complete ({percentage}%)\n"

    yield "Process complete!\n"


# =============================================================================
# DATA STREAMING
# =============================================================================


@function
async def stream_json_chunks(ctx: FunctionContext, items: int):
    """
    Stream JSON data items one by one.

    Useful for streaming large datasets or API results.

    Args:
        items: Number of items to generate (max 50)

    Yields:
        JSON string for each item

    Example:
        for item in client.stream("stream_json_chunks", {"items": 5}):
            data = json.loads(item)
            print(data)
    """
    import json

    items = min(items, 50)

    ctx.logger.info(f"[stream_json_chunks] Streaming {items} JSON items...")

    for i in range(items):
        await asyncio.sleep(0.05)
        item = {
            "id": i + 1,
            "value": f"item-{i + 1}",
            "timestamp": asyncio.get_event_loop().time(),
        }
        yield json.dumps(item) + "\n"


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    print("=" * 60)
    print("AGNT5 Streaming Example")
    print("=" * 60)

    # Create a mock context for standalone execution
    from agnt5 import Context
    ctx = Context(
        run_id="streaming-demo",
        correlation_id="streaming-demo",
        parent_correlation_id="",
    )

    print("\n--- stream_text ---")
    print("Output: ", end="")
    async for chunk in stream_text(ctx, "Hello streaming world from AGNT5!", delay_ms=30):
        print(chunk, end="", flush=True)
    print()

    print("\n--- stream_story ---")
    print("Story: ", end="")
    async for chunk in stream_story(ctx, "Tell me about a dragon"):
        print(chunk, end="", flush=True)
    print()

    print("\n--- stream_counter ---")
    print("Count: ", end="")
    async for num in stream_counter(ctx, 10, delay_ms=50):
        print(num, end=" ", flush=True)
    print()

    print("\n--- stream_progress ---")
    async for update in stream_progress(ctx, 5, step_duration_ms=100):
        print(update, end="", flush=True)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
