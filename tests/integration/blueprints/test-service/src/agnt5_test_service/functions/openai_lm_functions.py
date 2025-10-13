"""
OpenAI Language Model Test Functions

Functions that test OpenAI LM integration within AGNT5 platform.
All functions require OPENAI_API_KEY to be set.
"""

from dataclasses import dataclass
from typing import List
from agnt5 import Context, function, lm


@function
async def generate_greeting(ctx: Context, name: str, style: str) -> dict:
    """Generate a greeting using LM.

    Tests basic LM generation within a function.
    Requires OPENAI_API_KEY - will fail if not available.
    """
    prompt = f"Generate a {style} greeting for {name}. Just the greeting, no explanation."

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt=prompt,
        max_tokens=50
    )
    ctx.logger.info(f"LM generated greeting for {name} in {style} style - {response.text}")

    return {"greeting": response.text}


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result structure."""
    sentiment: str
    score: float
    keywords: List[str]


@function
async def analyze_sentiment(ctx: Context, text: str) -> dict:
    """Analyze sentiment using structured output.

    Tests LM structured output with dataclasses.
    Requires OPENAI_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="openai/gpt-4o",
        prompt=f"Analyze the sentiment of this text: {text}",
        response_format=SentimentAnalysis,
        max_tokens=100
    )

    ctx.logger.info(f"LM analyzed sentiment: {response.structured_output}")

    # Fail if no structured output - don't mask the problem
    if not response.structured_output:
        raise ValueError("LM response missing structured_output - this indicates a problem with structured output parsing")

    return response.structured_output


@function
async def generate_story(ctx: Context, topic: str) -> dict:
    """Generate a short story using streaming.

    Tests LM streaming within a function.
    Requires OPENAI_API_KEY - will fail if not available.
    """
    chunks = []
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt=f"Write a very short story (2-3 sentences) about: {topic}",
        max_tokens=100
    ):
        chunks.append(chunk)

    story = "".join(chunks)
    ctx.logger.info(f"LM generated story with {len(chunks)} chunks")
    return {"story": story}


@function
async def chat_with_context(ctx: Context, user_message: str, context: list) -> dict:
    """Chat with conversation context.

    Tests multi-turn conversations with LM.
    Requires OPENAI_API_KEY - will fail if not available.
    """
    # Build messages from context
    messages = context + [{"role": "user", "content": user_message}]

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )

    ctx.logger.info(f"LM chat response generated")
    return {"response": response.text}


@function
async def generate_with_invalid_model(ctx: Context, prompt: str) -> dict:
    """Test LM error handling with invalid model.

    This function is expected to fail - used to test error propagation.
    """
    # This should fail because model is invalid
    response = await lm.generate(
        model="invalid-provider/invalid-model",
        prompt=prompt
    )
    return {"result": response.text}


@function
async def generate_joke(ctx: Context, topic: str) -> dict:
    """Generate a joke about a topic.

    Tests live LM API integration.
    Requires OPENAI_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt=f"Tell a short joke about {topic}",
        max_tokens=100
    )
    ctx.logger.info(f"LM generated joke about {topic}")
    return {"joke": response.text}


__all__ = [
    "generate_greeting",
    "analyze_sentiment",
    "generate_story",
    "chat_with_context",
    "generate_with_invalid_model",
    "generate_joke",
    "SentimentAnalysis",
]
