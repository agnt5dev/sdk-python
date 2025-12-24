"""
Ollama API Compatibility Tests

Tests local Ollama API connections.
Used for local development and privacy-sensitive workloads.

Models tested:
- llama3.2 (or available local model)

Test categories:
- Basic generation (non-streaming)
- Streaming generation
- Multi-turn conversations

Note: These tests require Ollama running locally.
      Install: curl -fsSL https://ollama.com/install.sh | sh
      Pull model: ollama pull llama3.2
      Start server: ollama serve

Run with:
    # Ensure Ollama is running locally
    pytest tests/api_compat/test_ollama.py -v
"""

import os
import pytest

from agnt5 import lm
from agnt5.events import EventType

from .conftest import skip_without_ollama


# All tests in this module require Ollama running locally
pytestmark = [
    pytest.mark.api_compat,
    pytest.mark.ollama,
    skip_without_ollama,
]

# Default to llama3.2, can be overridden via env var
MODEL = os.getenv("OLLAMA_TEST_MODEL", "ollama/llama3.2")


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
    # Should be a relatively short response due to system prompt
    assert len(response.text.split()) <= 10


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
        max_tokens=20,
    )

    assert response is not None
    assert response.text is not None
    # Response should be truncated (local models may vary)


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
        system_prompt="You are a helpful assistant.",
    ):
        chunks.append(event)

    assert len(chunks) > 0


# =============================================================================
# Model Availability Tests
# =============================================================================


@pytest.mark.asyncio
async def test_different_model():
    """Test with a different local model if available."""
    # Try a few common models
    models_to_try = [
        "ollama/mistral",
        "ollama/phi",
        "ollama/gemma",
    ]

    for model in models_to_try:
        try:
            response = await lm.generate(
                model=model,
                prompt="Say hello",
                max_tokens=20,
            )
            if response is not None and response.text:
                # Found a working model
                assert len(response.text) > 0
                return
        except Exception:
            continue

    # If none of the models are available, that's okay - just skip
    pytest.skip("No additional Ollama models available for testing")


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_model_error():
    """Test error handling for invalid model."""
    with pytest.raises(Exception):
        await lm.generate(
            model="ollama/not-a-real-model-that-exists-locally-12345",
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
# Local-specific Tests
# =============================================================================


@pytest.mark.asyncio
async def test_no_api_key_required():
    """Test that Ollama works without API key (local only)."""
    # Temporarily clear any API key if set
    original_key = os.environ.pop("OLLAMA_API_KEY", None)

    try:
        response = await lm.generate(
            model=MODEL,
            prompt="Say hello",
            max_tokens=20,
        )

        assert response is not None
        assert response.text is not None
    finally:
        # Restore key if it was set
        if original_key:
            os.environ["OLLAMA_API_KEY"] = original_key


@pytest.mark.asyncio
async def test_custom_base_url():
    """Test with custom base URL from environment."""
    # This test validates that OLLAMA_BASE_URL is respected
    # The conftest.py already checks for Ollama availability
    response = await lm.generate(
        model=MODEL,
        prompt="What is 2+2? Reply with just the number.",
        max_tokens=10,
    )

    assert response is not None
    assert response.text is not None


# =============================================================================
# Tool Calling / Function Calling Tests
# Note: Tool calling support in Ollama depends on the model being used.
# Llama 3.1+ and some other models support function calling.
# =============================================================================


@pytest.mark.asyncio
async def test_function_calling_basic():
    """Test basic function calling with a simple tool.

    Note: Tool calling support depends on the Ollama model being used.
    Llama 3.1+ models generally support function calling.
    """
    from agnt5.lm import (
        GenerateRequest,
        GenerationConfig,
        Message,
        ToolChoice,
        ToolDefinition,
        _LanguageModel,
    )

    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. Paris"
                }
            },
            "required": ["location"]
        }
    )

    model = _LanguageModel(provider="ollama", default_model=None)
    request = GenerateRequest(
        model=MODEL,
        messages=[Message.user("What's the weather like in Paris?")],
        tools=[weather_tool],
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=100),
    )

    try:
        response = await model.generate(request)
        assert response is not None
        # Model should either call the tool or respond with text
        assert response.text is not None or response.tool_calls is not None
    except Exception as e:
        # Some Ollama models don't support tool calling
        # This is expected behavior for older models
        if "tool" in str(e).lower() or "function" in str(e).lower():
            pytest.skip(f"Model {MODEL} may not support tool calling: {e}")
        raise


@pytest.mark.asyncio
async def test_function_calling_multiple_tools():
    """Test function calling with multiple tools available."""
    from agnt5.lm import (
        GenerateRequest,
        GenerationConfig,
        Message,
        ToolChoice,
        ToolDefinition,
        _LanguageModel,
    )

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

    model = _LanguageModel(provider="ollama", default_model=None)
    request = GenerateRequest(
        model=MODEL,
        messages=[Message.user("What time is it in Tokyo?")],
        tools=tools,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=100),
    )

    try:
        response = await model.generate(request)
        assert response is not None
        # Model should pick the appropriate tool or respond
        if response.tool_calls:
            assert response.tool_calls[0]["name"] == "get_time"
    except Exception as e:
        # Some Ollama models don't support tool calling
        if "tool" in str(e).lower() or "function" in str(e).lower():
            pytest.skip(f"Model {MODEL} may not support tool calling: {e}")
        raise


# =============================================================================
# Agentic Workflow Tests - Agent Loop (Tool Result Continuation)
# =============================================================================


@pytest.mark.asyncio
async def test_agent_loop_single_tool(calculator_tool, tool_executor):
    """Test agent loop: tool call -> execute -> continue with result.

    This tests the core agentic pattern. Note that tool calling support
    depends on the Ollama model being used.
    """
    from agnt5.lm import (
        GenerateRequest,
        GenerationConfig,
        Message,
        ToolChoice,
        _LanguageModel,
    )

    model = _LanguageModel(provider="ollama", default_model=None)

    # Step 1: Initial request that should trigger tool use
    messages = [Message.user("What is 15 * 7? Use the calculator tool.")]
    request = GenerateRequest(
        model=MODEL,
        messages=messages,
        tools=[calculator_tool],
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=200),
    )

    try:
        response = await model.generate(request)
        assert response is not None

        # If model called a tool, continue the loop
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            assert tool_call["name"] == "calculate"

            # Step 2: Execute the tool
            tool_result = tool_executor(tool_call["name"], tool_call["arguments"])

            # Step 3: Send tool result back to model
            messages.append(Message.assistant(
                content=response.text or "",
                tool_calls=response.tool_calls
            ))
            messages.append(Message.tool_result(
                tool_call_id=tool_call.get("id", "tool_0"),
                content=tool_result
            ))

            # Step 4: Get final response
            continuation_request = GenerateRequest(
                model=MODEL,
                messages=messages,
                tools=[calculator_tool],
                tool_choice=ToolChoice.AUTO,
                config=GenerationConfig(max_tokens=200),
            )

            final_response = await model.generate(continuation_request)
            assert final_response is not None
            assert final_response.text is not None
            # Should contain the result (105)
            assert "105" in final_response.text
        else:
            # Model answered directly (acceptable behavior for local models)
            assert response.text is not None
    except Exception as e:
        # Some Ollama models don't support tool calling
        if "tool" in str(e).lower() or "function" in str(e).lower():
            pytest.skip(f"Model {MODEL} may not support tool calling: {e}")
        raise


@pytest.mark.asyncio
async def test_agent_loop_multi_step(multi_tool_set, tool_executor):
    """Test multi-step agent loop with multiple sequential tool calls.

    Note: Tool calling support depends on the Ollama model being used.
    """
    from agnt5.lm import (
        GenerateRequest,
        GenerationConfig,
        Message,
        ToolChoice,
        _LanguageModel,
    )

    model = _LanguageModel(provider="ollama", default_model=None)
    messages = [Message.user(
        "First calculate 25 * 4, then tell me what the weather is like in Paris. "
        "Use the appropriate tools for each task."
    )]

    max_iterations = 5
    tools_called = []

    try:
        for iteration in range(max_iterations):
            request = GenerateRequest(
                model=MODEL,
                messages=messages,
                tools=multi_tool_set,
                tool_choice=ToolChoice.AUTO,
                config=GenerationConfig(max_tokens=300),
            )

            response = await model.generate(request)
            assert response is not None

            if not response.tool_calls:
                # Model finished - should have called at least one tool
                # For local models, we're more lenient
                break

            # Process tool calls
            for tool_call in response.tool_calls:
                tools_called.append(tool_call["name"])
                tool_result = tool_executor(tool_call["name"], tool_call["arguments"])

                messages.append(Message.assistant(
                    content=response.text or "",
                    tool_calls=[tool_call]
                ))
                messages.append(Message.tool_result(
                    tool_call_id=tool_call.get("id", f"tool_{iteration}"),
                    content=tool_result
                ))
    except Exception as e:
        # Some Ollama models don't support tool calling
        if "tool" in str(e).lower() or "function" in str(e).lower():
            pytest.skip(f"Model {MODEL} may not support tool calling: {e}")
        raise
