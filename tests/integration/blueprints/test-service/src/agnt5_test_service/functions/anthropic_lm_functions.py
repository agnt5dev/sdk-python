"""
Anthropic Language Model Test Functions

Functions that test Anthropic Claude integration within AGNT5 platform.
All functions require ANTHROPIC_API_KEY to be set.
"""

from agnt5 import Context, function, lm


@function
async def generate_text_anthropic(ctx: Context, prompt: str) -> dict:
    """Generate text using Anthropic Claude.

    Tests Anthropic LM integration within a function.
    Requires ANTHROPIC_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="anthropic/claude-3-5-sonnet-20241022",
        prompt=prompt,
        max_tokens=150
    )

    ctx.logger.info(f"Anthropic Claude generated text - length: {len(response.text)}")

    return {
        "text": response.text,
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022"
    }


@function
async def summarize_with_anthropic(ctx: Context, text: str) -> dict:
    """Summarize text using Anthropic Claude.

    Tests Anthropic summarization capabilities.
    Requires ANTHROPIC_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="anthropic/claude-3-5-sonnet-20241022",
        prompt=f"Summarize the following text in one sentence: {text}",
        max_tokens=100
    )

    ctx.logger.info(f"Anthropic Claude summarized text")

    return {
        "summary": response.text,
        "provider": "anthropic"
    }


__all__ = [
    "generate_text_anthropic",
    "summarize_with_anthropic",
]
