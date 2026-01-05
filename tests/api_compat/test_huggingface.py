"""
HuggingFace API Compatibility Tests

Tests real HuggingFace Inference Providers API connections to catch breaking changes.
Run weekly via CI to ensure SDK compatibility with HuggingFace APIs.

Models tested:
- meta-llama/Llama-3.1-8B-Instruct (fast, widely available - primary test model)
- meta-llama/Llama-3.3-70B-Instruct (larger model test)

Test categories:
- Basic generation (non-streaming)
- Streaming generation
- Multi-turn conversations
- Tool calling / Function calling

Run with:
    HUGGINGFACE_API_KEY=hf_... pytest tests/api_compat/test_huggingface.py -v
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
from typing import Callable
from .conftest import simulate_tool_execution

from .conftest import skip_without_huggingface


# All tests in this module require HuggingFace API key
pytestmark = [
    pytest.mark.api_compat,
    pytest.mark.huggingface,
    skip_without_huggingface,
]

# Two-model approach: different models for different capabilities
# Text generation model (fast, widely available)
TEXT_MODEL = "hf/meta-llama/Llama-3.3-70B-Instruct"

# Function calling model (supports tool use)
FUNCTION_MODEL = "hf/Qwen/Qwen2.5-7B-Instruct"

# Alias for backwards compatibility with basic tests
MODEL = TEXT_MODEL



# =============================================================================
# Basic Generation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_basic(simple_prompt: str):
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
    print(response)

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
async def test_generate_multi_turn(multi_turn_messages: list):
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
async def test_stream_basic(streaming_prompt: str):
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
        model=MODEL,
        prompt="What is 2+2? Reply with just the number.",
        max_tokens=10,
    )

    assert response is not None
    assert response.text is not None
    assert "4" in response.text


@pytest.mark.asyncio
async def test_generate_mistral():
    """Test with larger Llama model."""
    response = await lm.generate(
        model="hf/meta-llama/Llama-3.3-70B-Instruct",
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
            model="hf/not-a-real-model-12345",
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

    # Use function calling model (Qwen supports tool use)
    model = _LanguageModel(provider="huggingface", default_model=None)
    request = GenerateRequest(
        model=FUNCTION_MODEL,
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

    model = _LanguageModel(provider="huggingface", default_model=None)
    request = GenerateRequest(
        model=FUNCTION_MODEL,
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


# =============================================================================
# Agentic Workflow Tests - Agent Loop (Tool Result Continuation)
# =============================================================================


@pytest.mark.asyncio
async def test_agent_loop_single_tool(calculator_tool: ToolDefinition, tool_executor: Callable[..., str]):
    """Test agent loop: tool call -> execute -> continue with result.

    This tests the core agentic pattern where:
    1. Model decides to call a tool
    2. We execute the tool and get a result
    3. We send the result back to the model
    4. Model generates final response using the tool result
    """
    model = _LanguageModel(provider="huggingface", default_model=None)

    # Step 1: Initial request that should trigger tool use
    messages = [Message.user("What is 15 * 7? Use the calculator tool.")]
    request = GenerateRequest(
        model=FUNCTION_MODEL,
        messages=messages,
        tools=[calculator_tool],
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=200),
    )

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
            model=FUNCTION_MODEL,
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
        # Model answered directly (acceptable behavior)
        assert response.text is not None


@pytest.mark.asyncio
async def test_agent_loop_multi_step(multi_tool_set: list, tool_executor: Callable[..., str]):
    """Test multi-step agent loop with multiple sequential tool calls."""
    model = _LanguageModel(provider="huggingface", default_model=None)
    messages = [Message.user(
        "First calculate 25 * 4, then tell me what the weather is like in Paris. "
        "Use the appropriate tools for each task."
    )]

    max_iterations = 5
    tools_called = []

    for iteration in range(max_iterations):
        request = GenerateRequest(
            model=FUNCTION_MODEL,
            messages=messages,
            tools=multi_tool_set,
            tool_choice=ToolChoice.AUTO,
            config=GenerationConfig(max_tokens=300),
        )

        response = await model.generate(request)
        assert response is not None

        if not response.tool_calls:
            # Model finished - should have called at least one tool
            assert len(tools_called) >= 1, "Model should have used at least one tool"
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
    else:
        # Exceeded max iterations
        pytest.fail("Agent loop exceeded maximum iterations without completing")


# =============================================================================
# Parallel Tool Calls Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_tool_calls(multi_tool_set: list, tool_executor: Callable[..., str]):
    """Test model's ability to call multiple tools in parallel.

    HuggingFace supports parallel tool calls via the OpenAI-compatible API.
    """
    model = _LanguageModel(provider="huggingface", default_model=None)

    messages = [Message.user(
        "I need to know both: what is 10 + 20, AND what's the weather in Tokyo? "
        "Please use the calculator and weather tools."
    )]

    request = GenerateRequest(
        model=FUNCTION_MODEL,
        messages=messages,
        tools=multi_tool_set,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=300),
    )

    response = await model.generate(request)
    assert response is not None

    # Check if model made tool calls
    if response.tool_calls:
        # Process all tool results
        messages.append(Message.assistant(
            content=response.text or "",
            tool_calls=response.tool_calls
        ))

        for i, tool_call in enumerate(response.tool_calls):
            tool_result = tool_executor(tool_call["name"], tool_call["arguments"])
            messages.append(Message.tool_result(
                tool_call_id=tool_call.get("id", f"tool_{i}"),
                content=tool_result
            ))

        # Get final response
        continuation = GenerateRequest(
            model=FUNCTION_MODEL,
            messages=messages,
            tools=multi_tool_set,
            tool_choice=ToolChoice.AUTO,
            config=GenerationConfig(max_tokens=300),
        )

        final_response = await model.generate(continuation)
        assert final_response is not None
