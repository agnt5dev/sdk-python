"""
Anthropic API Compatibility Tests

Tests real Anthropic API connections to catch breaking changes.
Run weekly via CI to ensure SDK compatibility with Anthropic APIs.

Models tested:
- claude-3-haiku-20240307 (cheapest, fastest - primary test model)

Test categories:
- Basic generation (non-streaming)
- Streaming generation
- Multi-turn conversations
- Structured output
- Tool calling / Function calling

Run with:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/api_compat/test_anthropic.py -v
"""

import pytest
from dataclasses import dataclass

from agnt5 import lm
from agnt5.lm import (
    GenerateRequest,
    GenerationConfig,
    Message,
    ToolChoice,
    ToolDefinition,
    _LanguageModel,
)
from agnt5.events import EventType

from .conftest import skip_without_anthropic


# All tests in this module require Anthropic API key
pytestmark = [
    pytest.mark.api_compat,
    pytest.mark.anthropic,
    skip_without_anthropic,
]

# Use cheapest model for testing
MODEL = "anthropic/claude-3-haiku-20240307"


# =============================================================================
# Basic Generation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_basic(simple_prompt):
    """Test basic text generation."""
    response = await lm.generate(
        model=MODEL,
        prompt=simple_prompt,
    )

    assert response is not None
    assert response.text is not None
    assert "4" in response.text
    assert response.usage is not None
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_generate_with_system_prompt():
    """Test generation with system prompt."""
    response = await lm.generate(
        model=MODEL,
        prompt="What color is the sky?",
        system_prompt="You are a helpful assistant. Always respond in exactly one word.",
    )

    assert response is not None
    assert response.text is not None
    # Should be a short response due to system prompt
    assert len(response.text.split()) <= 5


@pytest.mark.asyncio
async def test_generate_with_temperature():
    """Test generation with temperature parameter."""
    response = await lm.generate(
        model=MODEL,
        prompt="Say hello",
        temperature=0.0,
    )

    assert response is not None
    assert response.text is not None


@pytest.mark.asyncio
async def test_generate_with_max_tokens():
    """Test generation with max_tokens limit."""
    response = await lm.generate(
        model=MODEL,
        prompt="Write a very long story about a dragon",
        max_tokens=10,
    )

    assert response is not None
    assert response.text is not None
    # Response should be truncated
    assert response.usage.completion_tokens <= 15  # Allow some margin


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_multi_turn(multi_turn_messages):
    """Test multi-turn conversation."""
    response = await lm.generate(
        model=MODEL,
        messages=multi_turn_messages,
    )

    assert response is not None
    assert response.text is not None
    # Should remember the name from earlier in conversation
    assert "alice" in response.text.lower()


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.asyncio
async def test_stream_basic(streaming_prompt):
    """Test basic streaming generation."""
    chunks = []
    full_text = ""

    async for event in lm.stream(
        model=MODEL,
        prompt=streaming_prompt,
    ):
        chunks.append(event)
        if event.event_type == EventType.LM_MESSAGE_DELTA:
            # event.data is the raw content string for delta events
            if event.data:
                full_text += event.data

    # Should receive multiple chunks
    assert len(chunks) > 0

    # Full text should contain the numbers
    assert "1" in full_text
    assert "2" in full_text
    assert "3" in full_text


@pytest.mark.asyncio
async def test_stream_with_system_prompt():
    """Test streaming with system prompt."""
    chunks = []

    async for event in lm.stream(
        model=MODEL,
        prompt="Say hello",
        system_prompt="You are a pirate. Always say 'Arrr' first.",
    ):
        chunks.append(event)

    assert len(chunks) > 0


# =============================================================================
# Structured Output Tests
# =============================================================================


@dataclass
class MathResult:
    """Structured output for math problems."""
    answer: int
    explanation: str


@pytest.mark.asyncio
async def test_generate_structured_output():
    """Test structured output with dataclass."""
    response = await lm.generate(
        model=MODEL,
        prompt="What is 15 + 27?",
        response_format=MathResult,
    )

    assert response is not None
    assert response.text is not None
    # Structured output should be parsed
    if response.structured_output:
        assert "answer" in response.structured_output or isinstance(
            response.structured_output, dict
        )


@dataclass
class SentimentResult:
    """Structured output for sentiment analysis."""
    sentiment: str
    confidence: float


@pytest.mark.asyncio
async def test_generate_structured_sentiment():
    """Test structured output for sentiment analysis."""
    response = await lm.generate(
        model=MODEL,
        prompt="Analyze the sentiment: 'I love this product!'",
        response_format=SentimentResult,
    )

    assert response is not None
    assert response.text is not None


# =============================================================================
# Claude-specific Features
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="Claude 3.5 Sonnet model requires additional API access - using Haiku for regular tests")
async def test_generate_sonnet():
    """Test with Claude Sonnet model.

    Note: This test is skipped because Claude 3.5 Sonnet requires
    additional API access or the model ID has changed. The core
    functionality is tested via Haiku model tests.
    """
    response = await lm.generate(
        model="anthropic/claude-3-5-sonnet-latest",
        prompt="What is 2+2? Reply with just the number.",
        max_tokens=10,
    )

    assert response is not None
    assert response.text is not None
    assert "4" in response.text


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_model_error():
    """Test error handling for invalid model."""
    with pytest.raises(Exception):
        await lm.generate(
            model="anthropic/not-a-real-model-12345",
            prompt="Hello",
        )


@pytest.mark.asyncio
async def test_empty_prompt_error():
    """Test error handling for empty prompt."""
    with pytest.raises(ValueError):
        await lm.generate(
            model=MODEL,
            prompt="",
        )


# =============================================================================
# Tool Calling / Function Calling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_function_calling_basic():
    """Test basic function calling with a simple tool."""
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    )

    model = _LanguageModel(provider="anthropic", default_model=None)
    request = GenerateRequest(
        model=MODEL,
        messages=[Message.user("What's the weather like in Paris?")],
        tools=[weather_tool],
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=100),
    )

    response = await model.generate(request)

    assert response is not None
    # Model should either call the tool or respond with text
    assert response.text is not None or response.tool_calls is not None


@pytest.mark.asyncio
async def test_function_calling_multiple_tools():
    """Test function calling with multiple tools available."""
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        ),
        ToolDefinition(
            name="get_time",
            description="Get current time for a timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        ),
    ]

    model = _LanguageModel(provider="anthropic", default_model=None)
    request = GenerateRequest(
        model=MODEL,
        messages=[Message.user("What time is it in Tokyo?")],
        tools=tools,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=100),
    )

    response = await model.generate(request)

    assert response is not None
    # Model should pick the appropriate tool or respond
    if response.tool_calls:
        assert response.tool_calls[0]["name"] == "get_time"
