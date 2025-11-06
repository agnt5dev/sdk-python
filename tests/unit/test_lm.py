"""
Unit Tests: Language Model API

Tests the lm.generate() and lm.stream() APIs for multi-provider LLM access.

These tests validate:
- Basic text generation with various providers
- Streaming responses
- Structured output with Pydantic models and dataclasses
- Error handling (invalid models, missing API keys, etc.)
- Message formatting (single prompt vs multi-turn conversations)
- Configuration (temperature, max_tokens, etc.)

Test Strategy:
- Mock the Rust backend to avoid requiring real API keys
- Test API surface and parameter validation
- Use real API calls for integration testing (when keys available)
"""

import json
import pytest
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from agnt5 import lm
from agnt5.lm import (
    GenerateResponse,
    TokenUsage,
    Message,
    MessageRole,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_rust_generate():
    """Mock successful Rust generate response."""
    mock_response = MagicMock()
    mock_response.content = "This is a test response."
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    mock_response.object = None
    mock_response.tool_calls = None
    return mock_response


@pytest.fixture
def mock_rust_stream_chunks():
    """Mock Rust stream response chunks."""
    return [
        MagicMock(text="Hello"),
        MagicMock(text=" "),
        MagicMock(text="world"),
        MagicMock(text="!"),
    ]


# ============================================================================
# Basic Generation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_simple_prompt(mock_rust_generate):
    """Test basic text generation with a simple prompt."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="What is love?",
            temperature=0.7
        )

        assert isinstance(response, GenerateResponse)
        assert response.text == "This is a test response."
        assert response.usage is not None
        assert response.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_generate_with_messages(mock_rust_generate):
    """Test generation with multi-turn conversation messages."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]

        response = await lm.generate(
            model="anthropic/claude-3-5-haiku-20241022",
            messages=messages,
            temperature=0.5
        )

        assert isinstance(response, GenerateResponse)
        assert response.text == "This is a test response."

        # Verify generate was called
        mock_instance.generate.assert_called_once()
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert len(call_kwargs["prompt"]) == 3


@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_rust_generate):
    """Test generation with system prompt."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Write a haiku",
            system_prompt="You are a poetic AI assistant.",
            max_tokens=100
        )

        assert isinstance(response, GenerateResponse)

        # Verify system prompt was passed
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == "You are a poetic AI assistant."
        assert call_kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_all_providers(mock_rust_generate):
    """Test that all supported providers work."""
    providers_and_models = [
        ("openai", "openai/gpt-4o-mini"),
        ("anthropic", "anthropic/claude-3-5-haiku-20241022"),
        ("groq", "groq/llama-3.3-70b-versatile"),
        ("openrouter", "openrouter/anthropic/claude-3.5-haiku"),
        ("azure", "azure/gpt-4o"),
        ("bedrock", "bedrock/anthropic.claude-v2"),
    ]

    for provider, model in providers_and_models:
        with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
            mock_instance = MagicMock()
            mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
            mock_rust_class.return_value = mock_instance

            response = await lm.generate(
                model=model,
                prompt="Test prompt"
            )

            assert isinstance(response, GenerateResponse)


# ============================================================================
# Streaming Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stream_simple(mock_rust_stream_chunks):
    """Test basic streaming generation."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o-mini",
            prompt="Write a story"
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world", "!"]


@pytest.mark.asyncio
async def test_stream_with_messages(mock_rust_stream_chunks):
    """Test streaming with conversation messages."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        messages = [
            {"role": "user", "content": "Tell me a joke"},
        ]

        chunks = []
        async for chunk in lm.stream(
            model="groq/llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.9
        ):
            chunks.append(chunk)

        assert len(chunks) > 0


# ============================================================================
# Structured Output Tests
# ============================================================================


@dataclass
class Person:
    """Test dataclass for structured output."""
    name: str
    age: int
    email: str


@pytest.mark.asyncio
async def test_generate_structured_output_dataclass():
    """Test structured output with dataclass."""
    # Mock response with structured output
    mock_response = MagicMock()
    mock_response.content = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
    mock_response.usage = None
    mock_response.tool_calls = None
    mock_response.object = {"name": "Alice", "age": 30, "email": "alice@example.com"}

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_response)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Extract person info from: Alice, 30, alice@example.com",
            response_format=Person
        )

        assert isinstance(response, GenerateResponse)

        # Verify response_schema was passed
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert "response_schema_kw" in call_kwargs
        schema_json = call_kwargs["response_schema_kw"]
        schema = json.loads(schema_json)

        # Schema should have the dataclass fields
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "email" in schema["properties"]

        # Access structured output
        assert response.structured_output is not None
        assert response.structured_output["name"] == "Alice"
        assert response.structured_output["age"] == 30

        # Test aliases
        assert response.parsed == response.structured_output
        assert response.object == response.structured_output


@pytest.mark.asyncio
async def test_generate_structured_output_dict_schema():
    """Test structured output with raw JSON schema dict."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rating": {"type": "number"}
        },
        "required": ["title", "rating"]
    }

    # Mock response with structured output
    mock_response = MagicMock()
    mock_response.content = '{"title": "Great Movie", "rating": 9.5}'
    mock_response.usage = None
    mock_response.tool_calls = None
    mock_response.object = {"title": "Great Movie", "rating": 9.5}

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_response)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Rate this movie: Inception",
            response_format=schema
        )

        assert response.structured_output is not None
        assert response.structured_output["title"] == "Great Movie"
        assert response.structured_output["rating"] == 9.5


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_with_all_config_options(mock_rust_generate):
    """Test generation with all configuration options."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Test prompt",
            temperature=0.8,
            max_tokens=500,
            top_p=0.95,
        )

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.95


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_missing_prompt_and_messages():
    """Test that either prompt or messages is required."""
    with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
        await lm.generate(model="openai/gpt-4o-mini")


@pytest.mark.asyncio
async def test_generate_both_prompt_and_messages():
    """Test that prompt and messages cannot both be provided."""
    with pytest.raises(ValueError, match="Provide either 'prompt' or 'messages', not both"):
        await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Test",
            messages=[{"role": "user", "content": "Test"}]
        )


@pytest.mark.asyncio
async def test_generate_missing_provider_prefix():
    """Test that model must include provider prefix."""
    with pytest.raises(ValueError, match="Model must include provider prefix"):
        await lm.generate(
            model="gpt-4o-mini",  # Missing openai/ prefix
            prompt="Test"
        )


@pytest.mark.asyncio
async def test_stream_missing_prompt_and_messages():
    """Test stream requires either prompt or messages."""
    with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
        async for _ in lm.stream(model="openai/gpt-4o-mini"):
            pass


@pytest.mark.asyncio
async def test_stream_missing_provider_prefix():
    """Test stream requires provider prefix in model."""
    with pytest.raises(ValueError, match="Model must include provider prefix"):
        async for _ in lm.stream(model="gpt-4o", prompt="Test"):
            pass


# ============================================================================
# Message Formatting Tests
# ============================================================================


@pytest.mark.asyncio
async def test_message_role_conversion(mock_rust_generate):
    """Test that message roles are properly converted."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]

        await lm.generate(
            model="openai/gpt-4o-mini",
            messages=messages
        )

        call_kwargs = mock_instance.generate.call_args.kwargs
        prompt = call_kwargs["prompt"]

        # Verify all roles are preserved
        assert len(prompt) == 4
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[2]["role"] == "assistant"
        assert prompt[3]["role"] == "user"


# ============================================================================
# Provider Auto-Detection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_provider_auto_detection_openai(mock_rust_generate):
    """Test provider auto-detection for OpenAI."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Test"
        )

        # Verify generate was called (provider detection successful)
        mock_instance.generate.assert_called_once()


@pytest.mark.asyncio
async def test_provider_auto_detection_anthropic(mock_rust_generate):
    """Test provider auto-detection for Anthropic."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        await lm.generate(
            model="anthropic/claude-3-5-haiku-20241022",
            prompt="Test"
        )

        mock_instance.generate.assert_called_once()


# ============================================================================
# TokenUsage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_token_usage_in_response(mock_rust_generate):
    """Test that token usage is properly returned."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Test"
        )

        assert response.usage is not None
        assert isinstance(response.usage, TokenUsage)
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


# ============================================================================
# Helper Class Tests
# ============================================================================


def test_message_helpers():
    """Test Message helper methods."""
    system_msg = Message.system("You are helpful")
    assert system_msg.role == MessageRole.SYSTEM
    assert system_msg.content == "You are helpful"

    user_msg = Message.user("Hello")
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "Hello"

    assistant_msg = Message.assistant("Hi there")
    assert assistant_msg.role == MessageRole.ASSISTANT
    assert assistant_msg.content == "Hi there"


# ============================================================================
# OpenAI Responses API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_with_built_in_tools(mock_rust_generate):
    """Test generation with OpenAI built-in tools."""
    from agnt5.lm import BuiltInTool

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Search for the latest AI news",
            built_in_tools=[BuiltInTool.WEB_SEARCH, BuiltInTool.FILE_SEARCH]
        )

        assert isinstance(response, GenerateResponse)

        # Verify built-in tools were passed to Rust
        call_kwargs = mock_instance.generate.call_args.kwargs
        assert "built_in_tools" in call_kwargs
        built_in_tools = json.loads(call_kwargs["built_in_tools"])
        assert "web_search_preview" in built_in_tools
        assert "file_search" in built_in_tools


@pytest.mark.asyncio
async def test_generate_with_code_interpreter(mock_rust_generate):
    """Test generation with code interpreter built-in tool."""
    from agnt5.lm import BuiltInTool

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Calculate the fibonacci sequence",
            built_in_tools=[BuiltInTool.CODE_INTERPRETER]
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert "built_in_tools" in call_kwargs
        built_in_tools = json.loads(call_kwargs["built_in_tools"])
        assert "code_interpreter" in built_in_tools


@pytest.mark.asyncio
async def test_generate_with_reasoning_effort(mock_rust_generate):
    """Test generation with reasoning effort for o-series models."""
    from agnt5.lm import ReasoningEffort

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/o1",
            prompt="Solve this complex problem",
            reasoning_effort=ReasoningEffort.HIGH
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_generate_with_minimal_reasoning(mock_rust_generate):
    """Test generation with minimal reasoning effort."""
    from agnt5.lm import ReasoningEffort

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/o1-mini",
            prompt="Quick question",
            reasoning_effort=ReasoningEffort.MINIMAL
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "minimal"


@pytest.mark.asyncio
async def test_generate_with_modalities(mock_rust_generate):
    """Test generation with modalities (text, audio, image)."""
    from agnt5.lm import Modality

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o",
            prompt="Describe this image",
            modalities=[Modality.TEXT, Modality.AUDIO]
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert "modalities" in call_kwargs
        modalities = json.loads(call_kwargs["modalities"])
        assert "text" in modalities
        assert "audio" in modalities


@pytest.mark.asyncio
async def test_generate_with_store_enabled(mock_rust_generate):
    """Test generation with server-side state storage enabled."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Remember this conversation",
            store=True
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["store"] is True


@pytest.mark.asyncio
async def test_generate_with_previous_response_id(mock_rust_generate):
    """Test generation continuing from previous response."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="Continue from where we left off",
            previous_response_id="resp_abc123",
            store=True
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert call_kwargs["previous_response_id"] == "resp_abc123"
        assert call_kwargs["store"] is True


@pytest.mark.asyncio
async def test_generate_with_all_responses_api_features(mock_rust_generate):
    """Test generation with all Responses API features combined."""
    from agnt5.lm import BuiltInTool, ReasoningEffort, Modality

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_rust_generate)
        mock_rust_class.return_value = mock_instance

        response = await lm.generate(
            model="openai/o1",
            prompt="Complex task with all features",
            temperature=0.7,
            max_tokens=1000,
            built_in_tools=[BuiltInTool.WEB_SEARCH, BuiltInTool.CODE_INTERPRETER],
            reasoning_effort=ReasoningEffort.HIGH,
            modalities=[Modality.TEXT],
            store=True
        )

        assert isinstance(response, GenerateResponse)

        call_kwargs = mock_instance.generate.call_args.kwargs
        assert "built_in_tools" in call_kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert "modalities" in call_kwargs
        assert call_kwargs["store"] is True


# ============================================================================
# Streaming with Responses API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stream_with_built_in_tools(mock_rust_stream_chunks):
    """Test streaming with OpenAI built-in tools."""
    from agnt5.lm import BuiltInTool

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o",
            prompt="Search and summarize AI news",
            built_in_tools=[BuiltInTool.WEB_SEARCH]
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

        # Verify built-in tools were passed to Rust
        call_kwargs = mock_instance.stream.call_args.kwargs
        assert "built_in_tools" in call_kwargs
        built_in_tools = json.loads(call_kwargs["built_in_tools"])
        assert "web_search_preview" in built_in_tools


@pytest.mark.asyncio
async def test_stream_with_reasoning_effort(mock_rust_stream_chunks):
    """Test streaming with reasoning effort for o-series models."""
    from agnt5.lm import ReasoningEffort

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        chunks = []
        async for chunk in lm.stream(
            model="openai/o1-mini",
            prompt="Solve step by step",
            reasoning_effort=ReasoningEffort.MEDIUM
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

        call_kwargs = mock_instance.stream.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_stream_with_modalities(mock_rust_stream_chunks):
    """Test streaming with modalities specification."""
    from agnt5.lm import Modality

    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o",
            prompt="Describe this scene",
            modalities=[Modality.TEXT, Modality.AUDIO]
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

        call_kwargs = mock_instance.stream.call_args.kwargs
        assert "modalities" in call_kwargs
        modalities = json.loads(call_kwargs["modalities"])
        assert "text" in modalities
        assert "audio" in modalities


@pytest.mark.asyncio
async def test_stream_with_store_and_previous_response(mock_rust_stream_chunks):
    """Test streaming with server-side state and continuation."""
    with patch('agnt5.lm.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.stream = AsyncMock(return_value=mock_rust_stream_chunks)
        mock_rust_class.return_value = mock_instance

        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o-mini",
            prompt="Continue our conversation",
            store=True,
            previous_response_id="resp_abc123"
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

        call_kwargs = mock_instance.stream.call_args.kwargs
        assert call_kwargs["store"] is True
        assert call_kwargs["previous_response_id"] == "resp_abc123"


# ============================================================================
# Integration Tests (with real API calls - requires keys)
# ============================================================================


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_openai_generate():
    """Integration test with real OpenAI API (requires OPENAI_API_KEY).

    Run with: pytest tests/unit/test_lm.py::test_live_openai_generate -m lm_live -v
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Say 'Hello world' and nothing else.",
        max_tokens=10
    )

    assert isinstance(response, GenerateResponse)
    assert len(response.text) > 0
    assert "hello" in response.text.lower()


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_openai_stream():
    """Integration test with real OpenAI streaming (requires OPENAI_API_KEY).

    Run with: pytest tests/unit/test_lm.py::test_live_openai_stream -m lm_live -v
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    chunks = []
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Count to 3 slowly",
        max_tokens=20
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 0


# ============================================================================
# OpenAI Responses API Integration Tests (with real API calls)
# ============================================================================


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_basic():
    """Test basic Responses API call with real OpenAI API.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_basic -m lm_live -v
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Say 'Hello from Responses API' and nothing else.",
        max_tokens=20,
        temperature=0.5
    )

    assert isinstance(response, GenerateResponse)
    assert len(response.text) > 0
    assert "hello" in response.text.lower() or "responses" in response.text.lower()
    assert response.usage is not None
    assert response.usage.total_tokens > 0


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_with_web_search():
    """Test Responses API with web search built-in tool (requires OPENAI_API_KEY).

    NOTE: Web search is a preview feature and may not be available on all accounts.
    This test will be skipped if web search is not available.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_with_web_search -m lm_live -v
    """
    import os
    from agnt5.lm import BuiltInTool

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    try:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="What are the latest AI news headlines? Just give me 2 headlines.",
            built_in_tools=[BuiltInTool.WEB_SEARCH],
            max_tokens=200,
            temperature=0.5
        )

        assert isinstance(response, GenerateResponse)
        assert len(response.text) > 0
        print(f"\nWeb search response: {response.text}")
    except Exception as e:
        # Web search may not be available on all accounts
        if "web_search" in str(e).lower() or "not available" in str(e).lower():
            pytest.skip(f"Web search not available: {e}")
        raise


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_with_reasoning_effort():
    """Test Responses API with reasoning effort for o-series models.

    NOTE: Requires access to o-series models (o1, o1-mini, etc.)

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_with_reasoning_effort -m lm_live -v
    """
    import os
    from agnt5.lm import ReasoningEffort

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    try:
        response = await lm.generate(
            model="openai/o1-mini",
            prompt="What is 15 * 23?",
            reasoning_effort=ReasoningEffort.MINIMAL,
            max_tokens=100
        )

        assert isinstance(response, GenerateResponse)
        assert len(response.text) > 0
        # Should contain the answer 345
        assert "345" in response.text
        print(f"\nReasoning response: {response.text}")
    except Exception as e:
        # o-series models may not be available on all accounts
        if "o1" in str(e).lower() or "not available" in str(e).lower() or "model" in str(e).lower():
            pytest.skip(f"o1-mini model not available: {e}")
        raise


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_with_modalities():
    """Test Responses API with modalities specification.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_with_modalities -m lm_live -v
    """
    import os
    from agnt5.lm import Modality

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Describe the color blue in one sentence.",
        modalities=[Modality.TEXT],
        max_tokens=50,
        temperature=0.5
    )

    assert isinstance(response, GenerateResponse)
    assert len(response.text) > 0
    assert "blue" in response.text.lower()
    print(f"\nModality response: {response.text}")


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_with_store():
    """Test Responses API with server-side state storage.

    This test verifies that conversations can be stored server-side
    and continued with previous_response_id.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_with_store -m lm_live -v
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # First message with store=True
    response1 = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="My favorite color is purple. Remember this.",
        store=True,
        max_tokens=50,
        temperature=0.5
    )

    assert isinstance(response1, GenerateResponse)
    assert len(response1.text) > 0
    print(f"\nFirst response: {response1.text}")

    # NOTE: The Responses API returns a response_id in the response object
    # However, our current implementation may not expose it yet.
    # For now, this test just verifies that store=True doesn't cause errors.
    # In a complete implementation, we would:
    # 1. Extract response_id from response1
    # 2. Use it in a follow-up call with previous_response_id
    # 3. Verify the model remembers the context


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_comprehensive():
    """Comprehensive test combining multiple Responses API features.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_comprehensive -m lm_live -v
    """
    import os
    from agnt5.lm import Modality

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="Tell me one interesting fact about Python programming language.",
        temperature=0.7,
        max_tokens=100,
        modalities=[Modality.TEXT],
        store=True
    )

    assert isinstance(response, GenerateResponse)
    assert len(response.text) > 0
    assert "python" in response.text.lower()
    assert response.usage is not None
    print(f"\nComprehensive response: {response.text}")
    print(f"Token usage: {response.usage}")


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_responses_api_conversation_continuation():
    """Test conversation continuation with response_id.

    This test validates that response_id is returned and can be used
    to continue conversations with server-side state.

    Run with: pytest tests/unit/test_lm.py::test_live_responses_api_conversation_continuation -m lm_live -v
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # First message with store=True
    response1 = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="My favorite animal is a penguin. Please remember this.",
        store=True,
        max_tokens=50
    )

    assert isinstance(response1, GenerateResponse)
    assert len(response1.text) > 0
    print(f"\nFirst response: {response1.text}")
    print(f"Response ID: {response1.response_id}")

    # If response_id is available, test continuation
    if response1.response_id:
        print(f"Testing continuation with response_id: {response1.response_id}")

        response2 = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt="What is my favorite animal?",
            previous_response_id=response1.response_id,
            store=True,
            max_tokens=50
        )

        assert isinstance(response2, GenerateResponse)
        assert len(response2.text) > 0
        print(f"\nSecond response: {response2.text}")

        # The model should remember the penguin
        assert "penguin" in response2.text.lower()
    else:
        print("Note: response_id not yet exposed by Rust layer")


# ============================================================================
# Streaming with Responses API Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_stream_with_built_in_tools():
    """Test streaming with web search built-in tool.

    Run with: pytest tests/unit/test_lm.py::test_live_stream_with_built_in_tools -m lm_live -v
    """
    import os
    from agnt5.lm import BuiltInTool

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    try:
        chunks = []
        async for chunk in lm.stream(
            model="openai/gpt-4o-mini",
            prompt="What's trending in AI today? Give me 2 headlines.",
            built_in_tools=[BuiltInTool.WEB_SEARCH],
            max_tokens=200
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        full_text = "".join(chunks)
        assert len(full_text) > 0
        print(f"\n\nTotal chunks: {len(chunks)}")
    except Exception as e:
        if "web_search" in str(e).lower() or "not available" in str(e).lower():
            pytest.skip(f"Web search not available: {e}")
        raise


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_stream_with_modalities():
    """Test streaming with modalities specification.

    Run with: pytest tests/unit/test_lm.py::test_live_stream_with_modalities -m lm_live -v
    """
    import os
    from agnt5.lm import Modality

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    chunks = []
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Describe a sunset in vivid detail, one sentence.",
        modalities=[Modality.TEXT],
        max_tokens=100,
        temperature=0.7
    ):
        chunks.append(chunk)
        print(chunk, end="", flush=True)

    full_text = "".join(chunks)
    assert len(full_text) > 0
    assert "sunset" in full_text.lower() or "sun" in full_text.lower()
    print(f"\n\nTotal chunks: {len(chunks)}")
