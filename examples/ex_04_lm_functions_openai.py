"""
Example: AGNT5 Language Model Functions

This example demonstrates LLM integration via the `lm` module:
- lm.generate() - Non-streaming completion (waits for full response)
- lm.stream() - Streaming completion (yields tokens as they arrive)

These functions can be:
- Run standalone: python ex_02_lm_functions.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("lm_generate", {"prompt": "Hello", "model": "openai/gpt-4o-mini"})

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Or use other supported models (anthropic/claude-3-haiku, etc.)
"""

import asyncio

from agnt5 import function, FunctionContext, lm
from agnt5.lm import LMContentBlockDelta


# =============================================================================
# LANGUAGE MODEL FUNCTIONS
# =============================================================================


@function
async def lm_generate(ctx: FunctionContext, prompt: str, model: str = "openai/gpt-4o-mini") -> dict:
    """
    Generate text using LLM (non-streaming).

    Waits for the complete response before returning.

    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier (default: openai/gpt-4o-mini)

    Returns:
        Dict with prompt, response, and model info

    Example:
        result = client.run("lm_generate", {
            "prompt": "What is 2+2?",
            "model": "openai/gpt-4o-mini"
        })
        # Returns: {"prompt": "...", "response": "4", "model": "..."}
    """
    ctx.logger.info(f"[lm_generate] Calling {model} with prompt: {prompt[:50]}...")

    response = await lm.generate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.text
    ctx.logger.info(f"[lm_generate] Response: {result[:100]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model,
        "method": "generate",
    }


@function
async def lm_stream(ctx: FunctionContext, prompt: str, model: str = "openai/gpt-4o-mini") -> dict:
    """
    Generate text using LLM (streaming).

    Streams tokens as they arrive, collecting them into a full response.

    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier (default: openai/gpt-4o-mini)

    Returns:
        Dict with prompt, response, model info, and chunk count

    Example:
        result = client.run("lm_stream", {
            "prompt": "Count from 1 to 5",
            "model": "openai/gpt-4o-mini"
        })
        # Returns: {"prompt": "...", "response": "1, 2, 3, 4, 5", "chunk_count": 10, ...}
    """
    ctx.logger.info(f"[lm_stream] Calling {model} with prompt: {prompt[:50]}...")

    chunks = []
    chunk_count = 0

    # lm.stream() yields typed Event objects (LMContentBlockDelta, LMCompleted, etc.)
    async for event in lm.stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    ):
        chunk_count += 1
        if isinstance(event, LMContentBlockDelta):
            # LMContentBlockDelta has .content attribute with the text
            content = str(event.content) if event.content else ""
            chunks.append(content)
            ctx.logger.info(f"[lm_stream] chunk[{chunk_count}]: '{content}'")
        else:
            ctx.logger.info(f"[lm_stream] event[{chunk_count}]: {event.event_type}")

    result = "".join(chunks)
    ctx.logger.info(f"[lm_stream] Response ({chunk_count} events, {len(chunks)} text chunks): {result[:100]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model,
        "method": "stream",
        "chunk_count": chunk_count,
    }


@function
async def lm_summarize(ctx: FunctionContext, text: str, max_words: int = 50) -> dict:
    """
    Summarize text using LLM.

    Args:
        text: Text to summarize
        max_words: Maximum words in summary (default: 50)

    Returns:
        Dict with original text length and summary

    Example:
        result = client.run("lm_summarize", {
            "text": "Long article text here...",
            "max_words": 30
        })
    """
    ctx.logger.info(f"[lm_summarize] Summarizing {len(text)} chars to ~{max_words} words")

    prompt = f"Summarize the following text in {max_words} words or less:\n\n{text}"

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "original_length": len(text),
        "summary": response.text,
        "max_words": max_words,
    }


@function
async def lm_classify(ctx: FunctionContext, text: str, categories: list) -> dict:
    """
    Classify text into one of the given categories using LLM.

    Args:
        text: Text to classify
        categories: List of category options

    Returns:
        Dict with text, category, and confidence explanation

    Example:
        result = client.run("lm_classify", {
            "text": "I love this product!",
            "categories": ["positive", "negative", "neutral"]
        })
        # Returns: {"category": "positive", ...}
    """
    ctx.logger.info(f"[lm_classify] Classifying into categories: {categories}")

    categories_str = ", ".join(categories)
    prompt = f"""Classify the following text into exactly one of these categories: {categories_str}

Text: {text}

Respond with only the category name, nothing else."""

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    category = response.text.strip().lower()

    return {
        "text": text,
        "categories": categories,
        "category": category,
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging
    import os
    from agnt5 import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. LLM calls will fail.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return

    print("=" * 60)
    print("AGNT5 Language Model Functions Example")
    print("=" * 60)

    ctx = Context(
        run_id="lm-example-run",
        correlation_id="lm-example-run",
        parent_correlation_id="",
    )

    # Non-streaming generate
    print("\n--- lm.generate() ---")
    result = await lm_generate(ctx, "What is the capital of France? Answer in one word.")
    print(f"Response: {result['response']}")

    # Streaming
    print("\n--- lm.stream() ---")
    result = await lm_stream(ctx, "Count from 1 to 5, separated by commas.")
    print(f"Response: {result['response']} ({result['chunk_count']} chunks)")

    # Summarize
    print("\n--- lm_summarize() ---")
    long_text = """
    The Python programming language was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability with its notable use of significant indentation.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Python is dynamically typed and garbage-collected, making it accessible to beginners while still powerful for experts.
    """
    result = await lm_summarize(ctx, long_text, max_words=20)
    print(f"Summary: {result['summary']}")

    # Classify
    print("\n--- lm_classify() ---")
    result = await lm_classify(ctx, "This is the best product I've ever used!", ["positive", "negative", "neutral"])
    print(f"Category: {result['category']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
