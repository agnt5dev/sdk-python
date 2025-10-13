"""
OpenRouter Language Model Test Functions

Functions that test OpenRouter integration within AGNT5 platform.
OpenRouter provides access to multiple LM providers through a unified API.
All functions require OPENROUTER_API_KEY to be set.
"""

from agnt5 import Context, function, lm


@function
async def generate_text_openrouter(ctx: Context, prompt: str, model_name: str = "anthropic/claude-3-5-sonnet") -> dict:
    """Generate text using OpenRouter.

    Tests OpenRouter LM integration within a function.
    Requires OPENROUTER_API_KEY - will fail if not available.

    Args:
        prompt: The text prompt to generate from
        model_name: OpenRouter model to use (default: anthropic/claude-3-5-sonnet)
    """
    response = await lm.generate(
        model=f"openrouter/{model_name}",
        prompt=prompt,
        max_tokens=150
    )

    ctx.logger.info(f"OpenRouter generated text using {model_name} - length: {len(response.text)}")

    return {
        "text": response.text,
        "provider": "openrouter",
        "model": model_name
    }


@function
async def compare_models_openrouter(ctx: Context, prompt: str) -> dict:
    """Compare responses from different models via OpenRouter.

    Tests OpenRouter's multi-provider capabilities.
    Requires OPENROUTER_API_KEY - will fail if not available.
    """
    # Use a fast, cost-effective model through OpenRouter
    response = await lm.generate(
        model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
        prompt=prompt,
        max_tokens=100
    )

    ctx.logger.info(f"OpenRouter generated comparison text")

    return {
        "response": response.text,
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct:free"
    }


__all__ = [
    "generate_text_openrouter",
    "compare_models_openrouter",
]
