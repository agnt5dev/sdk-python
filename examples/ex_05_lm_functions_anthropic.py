"""
Example: AGNT5 Language Model Functions (Anthropic)

This example demonstrates LLM integration with Anthropic Claude:
- lm.generate() - Non-streaming completion
- lm.stream() - Streaming completion

These functions can be:
- Run standalone: python ex_05_lm_functions_anthropic.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("claude_generate", {"prompt": "Hello"})

Prerequisites:
- Set ANTHROPIC_API_KEY environment variable
"""

import asyncio

from agnt5 import function, FunctionContext, lm
from agnt5.lm import LMContentBlockDelta


# Default model for Anthropic functions
DEFAULT_MODEL = "anthropic/claude-3-haiku-20240307"


# =============================================================================
# ANTHROPIC LM FUNCTIONS
# =============================================================================


@function
async def claude_generate(ctx: FunctionContext, prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Generate text using Claude (non-streaming).

    Waits for the complete response before returning.

    Args:
        prompt: The prompt to send to Claude
        model: Model identifier (default: anthropic/claude-3-haiku-20240307)

    Returns:
        Dict with prompt, response, and model info

    Example:
        result = client.run("claude_generate", {
            "prompt": "What is the meaning of life?",
        })
    """
    ctx.logger.info(f"[claude_generate] Calling {model} with prompt: {prompt[:50]}...")

    response = await lm.generate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.text
    ctx.logger.info(f"[claude_generate] Response: {result[:100]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model,
        "method": "generate",
        "provider": "anthropic",
    }


@function
async def claude_stream(ctx: FunctionContext, prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Generate text using Claude (streaming).

    Streams tokens as they arrive, collecting them into a full response.

    Args:
        prompt: The prompt to send to Claude
        model: Model identifier (default: anthropic/claude-3-haiku-20240307)

    Returns:
        Dict with prompt, response, model info, and chunk count

    Example:
        result = client.run("claude_stream", {
            "prompt": "Write a haiku about programming",
        })
    """
    ctx.logger.info(f"[claude_stream] Calling {model} with prompt: {prompt[:50]}...")

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
            ctx.logger.info(f"[claude_stream] chunk[{chunk_count}]: '{content}'")
        else:
            ctx.logger.info(f"[claude_stream] event[{chunk_count}]: {event.event_type}")

    result = "".join(chunks)
    ctx.logger.info(f"[claude_stream] Response ({chunk_count} events, {len(chunks)} text chunks): {result[:100]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model,
        "method": "stream",
        "chunk_count": chunk_count,
        "provider": "anthropic",
    }


@function
async def claude_analyze(ctx: FunctionContext, text: str, analysis_type: str = "sentiment") -> dict:
    """
    Analyze text using Claude.

    Args:
        text: Text to analyze
        analysis_type: Type of analysis (sentiment, summary, keywords)

    Returns:
        Dict with analysis results

    Example:
        result = client.run("claude_analyze", {
            "text": "I love this product!",
            "analysis_type": "sentiment"
        })
    """
    ctx.logger.info(f"[claude_analyze] Analyzing text with type: {analysis_type}")

    prompts = {
        "sentiment": f"Analyze the sentiment of this text and respond with only: positive, negative, or neutral.\n\nText: {text}",
        "summary": f"Summarize this text in one sentence:\n\n{text}",
        "keywords": f"Extract 3-5 keywords from this text, comma-separated:\n\n{text}",
    }

    prompt = prompts.get(analysis_type, prompts["sentiment"])

    response = await lm.generate(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "analysis_type": analysis_type,
        "result": response.text.strip(),
        "model": DEFAULT_MODEL,
    }


@function
async def claude_translate(ctx: FunctionContext, text: str, target_language: str = "Spanish") -> dict:
    """
    Translate text using Claude.

    Args:
        text: Text to translate
        target_language: Target language (default: Spanish)

    Returns:
        Dict with original and translated text

    Example:
        result = client.run("claude_translate", {
            "text": "Hello, how are you?",
            "target_language": "French"
        })
    """
    ctx.logger.info(f"[claude_translate] Translating to {target_language}")

    prompt = f"Translate the following text to {target_language}. Only respond with the translation, nothing else.\n\nText: {text}"

    response = await lm.generate(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "original": text,
        "translated": response.text.strip(),
        "target_language": target_language,
        "model": DEFAULT_MODEL,
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
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set. Claude calls will fail.")
        print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        return

    print("=" * 60)
    print("AGNT5 Anthropic LM Functions Example")
    print("=" * 60)

    ctx = Context(
        run_id="claude-example-run",
        correlation_id="claude-example-run",
        parent_correlation_id="",
    )

    # Non-streaming generate
    print("\n--- claude_generate() ---")
    result = await claude_generate(ctx, "What is 2+2? Reply with just the number.")
    print(f"Response: {result['response']}")

    # Streaming
    print("\n--- claude_stream() ---")
    result = await claude_stream(ctx, "Write a haiku about code.")
    print(f"Response: {result['response']} ({result['chunk_count']} chunks)")

    # Analyze
    print("\n--- claude_analyze() ---")
    result = await claude_analyze(ctx, "This product exceeded my expectations!", "sentiment")
    print(f"Sentiment: {result['result']}")

    # Translate
    print("\n--- claude_translate() ---")
    result = await claude_translate(ctx, "Hello, world!", "French")
    print(f"Translation: {result['translated']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
