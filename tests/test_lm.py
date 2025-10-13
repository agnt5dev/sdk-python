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
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from agnt5 import lm
from agnt5.lm import (
    GenerateResponse,
    TokenUsage,
    Message,
    MessageRole,
    GenerationConfig,
    GenerateRequest,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_rust_lm():
    """Mock Rust language model backend."""
    with patch("agnt5.lm.RustLanguageModel") as mock_class:
        # Create mock instance
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # Mock generate() to return a response
        async def mock_generate(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.content = "This is a test response."
            mock_response.usage = MagicMock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
            mock_response.object = None  # For structured output
            return mock_response

        mock_instance.generate = AsyncMock(side_effect=mock_generate)

        # Mock stream() to return chunks
        async def mock_stream(*args, **kwargs):
            chunks = [
                MagicMock(text="Hello"),
                MagicMock(text=" "),
                MagicMock(text="world"),
                MagicMock(text="!"),
            ]
            return chunks

        mock_instance.stream = AsyncMock(side_effect=mock_stream)

        yield mock_instance


@pytest.fixture
def mock_rust_config():
    """Mock Rust config class."""
    with patch("agnt5.lm.RustLanguageModelConfig") as mock:
        yield mock


# ============================================================================
# Basic Generation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_simple_prompt(mock_rust_lm, mock_rust_config):
    """Test basic text generation with a simple prompt."""
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt="What is love?",
        temperature=0.7
    )

    assert isinstance(response, GenerateResponse)
    assert response.text == "This is a test response."
    assert response.usage is not None
    assert response.usage.total_tokens == 30

    # Verify Rust was called with correct parameters
    mock_rust_lm.generate.assert_called_once()
    call_kwargs = mock_rust_lm.generate.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_generate_with_messages(mock_rust_lm, mock_rust_config):
    """Test generation with multi-turn conversation messages."""
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

    # Verify messages were passed correctly
    call_kwargs = mock_rust_lm.generate.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-3-5-haiku-20241022"
    assert call_kwargs["provider"] == "anthropic"

    # Check prompt messages
    prompt = call_kwargs["prompt"]
    assert len(prompt) == 3
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == "What is Python?"


@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_rust_lm, mock_rust_config):
    """Test generation with system prompt."""
    response = await lm.generate(
        model="openai/gpt-4o",
        prompt="Write a haiku",
        system_prompt="You are a poetic AI assistant.",
        max_tokens=100
    )

    assert isinstance(response, GenerateResponse)

    # Verify system prompt was passed separately
    call_kwargs = mock_rust_lm.generate.call_args.kwargs
    assert call_kwargs["system_prompt"] == "You are a poetic AI assistant."
    assert call_kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_all_providers(mock_rust_lm, mock_rust_config):
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
        response = await lm.generate(
            model=model,
            prompt="Test prompt"
        )

        assert isinstance(response, GenerateResponse)
        call_kwargs = mock_rust_lm.generate.call_args.kwargs
        assert call_kwargs["provider"] == provider


# ============================================================================
# Streaming Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stream_simple(mock_rust_lm, mock_rust_config):
    """Test basic streaming generation."""
    chunks = []
    async for chunk in lm.stream(
        model="openai/gpt-4o-mini",
        prompt="Write a story"
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " ", "world", "!"]
    mock_rust_lm.stream.assert_called_once()


@pytest.mark.asyncio
async def test_stream_with_messages(mock_rust_lm, mock_rust_config):
    """Test streaming with conversation messages."""
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
    call_kwargs = mock_rust_lm.stream.call_args.kwargs
    assert call_kwargs["model"] == "groq/llama-3.3-70b-versatile"
    assert call_kwargs["provider"] == "groq"
    assert call_kwargs["temperature"] == 0.9


# ============================================================================
# Structured Output Tests
# ============================================================================


@dataclass
class Person:
    """Test dataclass for structured output."""
    name: str
    age: int
    email: str


@dataclass
class CodeReview:
    """Test dataclass for complex structured output."""
    issues: List[str]
    suggestions: List[str]
    overall_quality: int


@pytest.mark.asyncio
async def test_generate_structured_output_dataclass(mock_rust_lm, mock_rust_config):
    """Test structured output with dataclass."""
    # Mock response with structured output
    async def mock_generate_structured(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.content = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        mock_response.usage = None
        # Simulate parsed object
        mock_response.object = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        return mock_response

    mock_rust_lm.generate = AsyncMock(side_effect=mock_generate_structured)

    response = await lm.generate(
        model="openai/gpt-4o",
        prompt="Extract person info from: Alice, 30, alice@example.com",
        response_format=Person
    )

    assert isinstance(response, GenerateResponse)

    # Verify response_schema was passed
    call_kwargs = mock_rust_lm.generate.call_args.kwargs
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
async def test_generate_structured_output_dict_schema(mock_rust_lm, mock_rust_config):
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
    async def mock_generate_structured(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.content = '{"title": "Great Movie", "rating": 9.5}'
        mock_response.usage = None
        mock_response.object = {"title": "Great Movie", "rating": 9.5}
        return mock_response

    mock_rust_lm.generate = AsyncMock(side_effect=mock_generate_structured)

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
async def test_generate_with_all_config_options(mock_rust_lm, mock_rust_config):
    """Test generation with all configuration options."""
    response = await lm.generate(
        model="openai/gpt-4o",
        prompt="Test prompt",
        temperature=0.8,
        max_tokens=500,
        top_p=0.95,
    )

    call_kwargs = mock_rust_lm.generate.call_args.kwargs
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
async def test_message_role_conversion(mock_rust_lm, mock_rust_config):
    """Test that message roles are properly converted."""
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

    call_kwargs = mock_rust_lm.generate.call_args.kwargs
    prompt = call_kwargs["prompt"]

    # Verify all roles are preserved
    assert len(prompt) == 4
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert prompt[2]["role"] == "assistant"
    assert prompt[3]["role"] == "user"


# ============================================================================
# Provider-Specific Tests
# ============================================================================


@pytest.mark.asyncio
async def test_openrouter_with_nested_provider(mock_rust_lm, mock_rust_config):
    """Test OpenRouter with nested provider in model name."""
    response = await lm.generate(
        model="openrouter/anthropic/claude-3.5-haiku",
        prompt="Test"
    )

    call_kwargs = mock_rust_lm.generate.call_args.kwargs
    # Model should keep the nested provider path
    assert call_kwargs["model"] == "openrouter/anthropic/claude-3.5-haiku"
    # Provider should be openrouter
    assert call_kwargs["provider"] == "openrouter"


# ============================================================================
# TokenUsage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_token_usage_in_response(mock_rust_lm, mock_rust_config):
    """Test that token usage is properly returned."""
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
# Integration Tests (with real API calls - requires keys)
# ============================================================================


@pytest.mark.integration
@pytest.mark.lm_live
@pytest.mark.asyncio
async def test_live_openai_generate():
    """Integration test with real OpenAI API (requires OPENAI_API_KEY).

    Run with: pytest tests/test_lm.py::test_live_openai_generate --run-live-lm-tests -v
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

    Run with: pytest tests/test_lm.py::test_live_openai_stream --run-live-lm-tests -v
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
