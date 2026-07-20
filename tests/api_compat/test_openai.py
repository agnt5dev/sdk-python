"""
OpenAI API Compatibility Tests

Tests real OpenAI API connections to catch breaking changes.
Run weekly via CI to ensure SDK compatibility with OpenAI APIs.

Standard models tested:
- gpt-4o-mini (baseline, cheap)
- gpt-5.2, gpt-5.1, gpt-5 (GPT-5 family)
- gpt-5-mini, gpt-5-nano (GPT-5 fast/cheap variants)

O-series reasoning models tested:
- o3, o4-mini (different API: no streaming, no system prompts)

Test categories:
- Basic generation (non-streaming)
- Streaming generation (standard models only)
- Multi-turn conversations
- Structured output
- Tool calling / Function calling

Run with:
    OPENAI_API_KEY=sk-... pytest tests/api_compat/test_openai.py -v
"""

from dataclasses import dataclass

import pytest

from agnt5 import lm
from agnt5.events import EventType
from agnt5.lm import (
    BuiltInTool,
    GenerateRequest,
    GenerationConfig,
    Message,
    ToolChoice,
    ToolDefinition,
    _LanguageModel,
)

from .conftest import skip_without_openai

# All tests in this module require OpenAI API key
pytestmark = [
    pytest.mark.api_compat,
    pytest.mark.openai,
    skip_without_openai,
]

# Models to test - covering different model families and versions
# Standard models support streaming and system prompts
OPENAI_MODELS = [
    "openai/gpt-4o-mini",  # GPT-4o mini - baseline, cheap
    "openai/gpt-5.2",      # GPT-5.2 - newest
    "openai/gpt-5.1",      # GPT-5.1
    "openai/gpt-5",        # GPT-5 - flagship
    "openai/gpt-5-mini",   # GPT-5 mini - fast
    "openai/gpt-5-nano",   # GPT-5 nano - fastest/cheapest
]

# O-series reasoning models - different API behavior:
# - No streaming support
# - No system prompt support
# - No temperature parameter
OPENAI_O_SERIES_MODELS = [
    "openai/o3",           # O3 reasoning model
    "openai/o4-mini",      # O4 mini reasoning model
]

# All models for basic tests that work with both
OPENAI_ALL_MODELS = OPENAI_MODELS + OPENAI_O_SERIES_MODELS

# Models that reliably support tool_choice=REQUIRED
# Note: gpt-5, gpt-5-mini, gpt-5-nano, o3, o4-mini don't properly handle REQUIRED
OPENAI_TOOL_REQUIRED_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-5.2",
    "openai/gpt-5.1",
]

# Default model for non-parameterized tests
MODEL = OPENAI_MODELS[0]


# =============================================================================
# Basic Generation Tests (parameterized across all models including O-series)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_ALL_MODELS)
async def test_generate_basic(simple_prompt, model):
    """Test basic text generation across all OpenAI models including O-series."""
    response = await lm.generate(
        model=model,
        prompt=simple_prompt,
    )

    assert response is not None
    assert response.text is not None
    assert "4" in response.text
    assert response.usage is not None
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_MODELS)  # O-series doesn't support system prompts
async def test_generate_with_system_prompt(model):
    """Test generation with system prompt (standard models only, O-series doesn't support)."""
    response = await lm.generate(
        model=model,
        prompt="What color is the sky?",
        system_prompt="You are a helpful assistant. Always respond in exactly one word.",
    )

    assert response is not None
    assert response.text is not None
    # Should be a short response due to system prompt
    assert len(response.text.split()) <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_MODELS)  # O-series doesn't support temperature parameter
async def test_generate_with_temperature(model):
    """Test generation with temperature parameter (standard models only, O-series doesn't support)."""
    response = await lm.generate(
        model=model,
        prompt="Say hello",
        temperature=0.0,
    )

    assert response is not None
    assert response.text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_ALL_MODELS)
async def test_generate_with_max_tokens(model):
    """Test generation with max_tokens limit across all models."""
    response = await lm.generate(
        model=model,
        prompt="Write a very long story about a dragon",
        max_tokens=20,  # OpenAI Responses API requires minimum of 16
    )

    assert response is not None
    assert response.text is not None
    # Response should be truncated
    assert response.usage.completion_tokens <= 25  # Allow some margin


# =============================================================================
# Multi-turn Conversation Tests (parameterized)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_ALL_MODELS)
async def test_generate_multi_turn(multi_turn_messages, model):
    """Test multi-turn conversation across all models including O-series."""
    response = await lm.generate(
        model=model,
        messages=multi_turn_messages,
    )

    assert response is not None
    assert response.text is not None
    # Should remember the name from earlier in conversation
    assert "alice" in response.text.lower()


# =============================================================================
# Streaming Tests (standard models only - O-series doesn't support streaming)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_MODELS)  # O-series doesn't support streaming
async def test_stream_basic(streaming_prompt, model):
    """Test basic streaming generation (standard models only, O-series doesn't support)."""
    chunks = []
    full_text = ""

    async for event in lm.stream(
        model=model,
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
@pytest.mark.parametrize("model", OPENAI_MODELS)  # O-series doesn't support streaming or system prompts
async def test_stream_with_system_prompt(model):
    """Test streaming with system prompt (standard models only)."""
    chunks = []

    async for event in lm.stream(
        model=model,
        prompt="Say hello",
        system_prompt="You are a pirate. Always say 'Arrr' first.",
    ):
        chunks.append(event)

    assert len(chunks) > 0


# =============================================================================
# Structured Output Tests (parameterized - all models including O-series)
# =============================================================================


@dataclass
class MathResult:
    """Structured output for math problems."""
    answer: int
    explanation: str


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_ALL_MODELS)
async def test_generate_structured_output(model):
    """Test structured output with dataclass across all models including O-series.

    Uses text.format for the Responses API.
    """
    response = await lm.generate(
        model=model,
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
@pytest.mark.parametrize("model", OPENAI_ALL_MODELS)
async def test_generate_structured_sentiment(model):
    """Test structured output for sentiment analysis across all models including O-series.

    Uses text.format for the Responses API.
    """
    response = await lm.generate(
        model=model,
        prompt="Analyze the sentiment: 'I love this product!'",
        response_format=SentimentResult,
    )

    assert response is not None
    assert response.text is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_model_error():
    """Test error handling for invalid model."""
    with pytest.raises(Exception):
        await lm.generate(
            model="openai/not-a-real-model-12345",
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
# Tool Calling / Function Calling Tests (parameterized - all models including O-series)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_ALL_MODELS)
async def test_function_calling_basic(model_name):
    """Test basic function calling with a simple tool across all models including O-series."""
    # Define a simple weather tool
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

    # Create request with tool
    model = _LanguageModel(provider="openai", default_model=None)
    request = GenerateRequest(
        model=model_name,
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
@pytest.mark.parametrize("model_name", OPENAI_TOOL_REQUIRED_MODELS)
async def test_function_calling_required(model_name):
    """Test function calling with tool_choice=required.

    Forces the model to use a tool.
    Note: Only tested on models that reliably support REQUIRED (gpt-4o-mini, gpt-5.2, gpt-5.1).
    """
    calculator_tool = ToolDefinition(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )

    model = _LanguageModel(provider="openai", default_model=None)
    request = GenerateRequest(
        model=model_name,
        messages=[Message.user("What is 25 * 4?")],
        tools=[calculator_tool],
        tool_choice=ToolChoice.REQUIRED,
        config=GenerationConfig(max_tokens=100),
    )

    response = await model.generate(request)

    assert response is not None
    # With REQUIRED, model must call a tool
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "calculate"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_ALL_MODELS)
async def test_function_calling_multiple_tools(model_name):
    """Test function calling with multiple tools available across all models including O-series."""
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

    model = _LanguageModel(provider="openai", default_model=None)
    request = GenerateRequest(
        model=model_name,
        messages=[Message.user("What time is it in Tokyo?")],
        tools=tools,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=100),
    )

    response = await model.generate(request)

    assert response is not None
    # Model should pick the appropriate tool
    if response.tool_calls:
        assert response.tool_calls[0]["name"] == "get_time"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_MODELS)  # Web search may not work on O-series
async def test_built_in_tool_web_search(model_name):
    """Test OpenAI's built-in web search tool (standard models only)."""
    response = await lm.generate(
        model=model_name,
        prompt="What is the current price of Bitcoin? Just give a rough number.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
        max_tokens=100,
    )

    assert response is not None
    assert response.text is not None
    # Response should contain some price-related content
    # (we can't verify exact price but can check response exists)


# =============================================================================
# Agentic Workflow Tests - Agent Loop (Tool Result Continuation)
#
# These tests use previous_response_id for conversation continuation with
# the OpenAI Responses API. This is the recommended pattern:
# 1. First request gets a response with response_id
# 2. Continuation requests pass previous_response_id and only send new input
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_TOOL_REQUIRED_MODELS)
async def test_agent_loop_single_tool(calculator_tool, tool_executor, model_name):
    """Test agent loop: tool call -> execute -> continue with result.

    This tests the core agentic pattern where:
    1. Model decides to call a tool
    2. We execute the tool and get a result
    3. We send the result back using previous_response_id
    4. Model generates final response using the tool result
    """
    model = _LanguageModel(provider="openai", default_model=None)

    # Step 1: Initial request that should trigger tool use
    request = GenerateRequest(
        model=model_name,
        messages=[Message.user("What is 15 * 7? Use the calculator tool.")],
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

        # Step 3: Send tool result back using previous_response_id
        # With Responses API, we pass the response_id to continue the conversation
        continuation_request = GenerateRequest(
            model=model_name,
            messages=[Message.tool_result(
                tool_call_id=tool_call.get("id", "tool_0"),
                content=tool_result
            )],
            tools=[calculator_tool],
            tool_choice=ToolChoice.AUTO,
            config=GenerationConfig(
                max_tokens=200,
                previous_response_id=response.response_id,
            ),
        )

        final_response = await model.generate(continuation_request)
        assert final_response is not None
        assert final_response.text is not None
        # Should contain the result (105)
        assert "105" in final_response.text
    else:
        # Model answered directly (acceptable behavior for some models)
        assert response.text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_TOOL_REQUIRED_MODELS)
async def test_agent_loop_multi_step(multi_tool_set, tool_executor, model_name):
    """Test multi-step agent loop with multiple sequential tool calls.

    This tests a more complex agentic pattern where the model may need
    to call multiple tools in sequence to complete a task.
    Uses previous_response_id for conversation continuation.
    """
    model = _LanguageModel(provider="openai", default_model=None)

    # Initial request
    request = GenerateRequest(
        model=model_name,
        messages=[Message.user(
            "First calculate 25 * 4, then tell me what the weather is like in Paris. "
            "Use the appropriate tools for each task."
        )],
        tools=multi_tool_set,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=300),
    )

    response = await model.generate(request)
    assert response is not None

    max_iterations = 5
    tools_called = []
    current_response_id = response.response_id

    for iteration in range(max_iterations):
        if not response.tool_calls:
            # Model finished - no more tool calls
            break

        # Process ALL tool calls and collect results
        # (must send all results together when model makes multiple calls)
        tool_result_messages = []
        for tool_call in response.tool_calls:
            tools_called.append(tool_call["name"])
            tool_result = tool_executor(tool_call["name"], tool_call["arguments"])
            tool_result_messages.append(Message.tool_result(
                tool_call_id=tool_call.get("id", f"tool_{iteration}"),
                content=tool_result
            ))

        # Continue conversation with ALL tool results using previous_response_id
        continuation = GenerateRequest(
            model=model_name,
            messages=tool_result_messages,
            tools=multi_tool_set,
            tool_choice=ToolChoice.AUTO,
            config=GenerationConfig(
                max_tokens=300,
                previous_response_id=current_response_id,
            ),
        )

        response = await model.generate(continuation)
        assert response is not None
        current_response_id = response.response_id
    else:
        # Exceeded max iterations
        pytest.fail("Agent loop exceeded maximum iterations without completing")

    # Test validates SDK can handle multi-step tool loops
    # Number of tools called depends on model behavior


# =============================================================================
# Parallel Tool Calls Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", OPENAI_TOOL_REQUIRED_MODELS)
async def test_parallel_tool_calls(multi_tool_set, tool_executor, model_name):
    """Test model's ability to call multiple tools in parallel.

    Some models can request multiple tool calls in a single response.
    This tests that capability using previous_response_id for continuation.
    """
    model = _LanguageModel(provider="openai", default_model=None)

    request = GenerateRequest(
        model=model_name,
        messages=[Message.user(
            "I need to know both: what is 10 + 20, AND what's the weather in Tokyo? "
            "Please use the calculator and weather tools."
        )],
        tools=multi_tool_set,
        tool_choice=ToolChoice.AUTO,
        config=GenerationConfig(max_tokens=300),
    )

    response = await model.generate(request)
    assert response is not None

    # Check if model made tool calls
    if response.tool_calls:
        # Process all tool results and send them back
        tool_result_messages = []
        for tool_call in response.tool_calls:
            tool_result = tool_executor(tool_call["name"], tool_call["arguments"])
            tool_result_messages.append(Message.tool_result(
                tool_call_id=tool_call.get("id", f"tool_{len(tool_result_messages)}"),
                content=tool_result
            ))

        # Get final response with all tool results using previous_response_id
        continuation = GenerateRequest(
            model=model_name,
            messages=tool_result_messages,
            tools=multi_tool_set,
            tool_choice=ToolChoice.AUTO,
            config=GenerationConfig(
                max_tokens=300,
                previous_response_id=response.response_id,
            ),
        )

        final_response = await model.generate(continuation)
        assert final_response is not None
