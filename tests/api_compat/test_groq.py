"""
Groq API Compatibility Tests

Tests real Groq API connections to catch breaking changes.
Run weekly via CI to ensure SDK compatibility with Groq APIs.

Models tested:
- llama-3.1-8b-instant (fast, cheap - primary test model)
- llama-3.3-70b-versatile (larger model test)

Test categories:
- Basic generation (non-streaming)
- Streaming generation
- Multi-turn conversations
- Tool calling / Function calling

Run with:
    GROQ_API_KEY=gsk_... pytest tests/api_compat/test_groq.py -v
"""

import pytest

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

from .conftest import skip_without_groq


# All tests in this module require Groq API key
pytestmark = [
    pytest.mark.api_compat,
    pytest.mark.groq,
    skip_without_groq,
]

# Use fast/cheap model for testing
MODEL = "groq/llama-3.1-8b-instant"
MODEL_LARGE = "groq/llama-3.3-70b-versatile"


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
        system_prompt="You are a friendly assistant.",
    ):
        chunks.append(event)

    assert len(chunks) > 0


# =============================================================================
# Different Model Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_large_model():
    """Test with larger Llama model."""
    response = await lm.generate(
        model=MODEL_LARGE,
        prompt="What is 2+2? Reply with just the number.",
        max_tokens=10,
    )

    assert response is not None
    assert response.text is not None
    assert "4" in response.text


@pytest.mark.asyncio
async def test_generate_mixtral():
    """Test with Mixtral model."""
    response = await lm.generate(
        model="groq/mixtral-8x7b-32768",
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
            model="groq/not-a-real-model-12345",
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

    # Use the larger model for better tool calling support
    model = _LanguageModel(provider="groq", default_model=None)
    request = GenerateRequest(
        model=MODEL_LARGE,
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

    model = _LanguageModel(provider="groq", default_model=None)
    request = GenerateRequest(
        model=MODEL_LARGE,
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
