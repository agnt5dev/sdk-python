"""
API Compatibility Test Configuration

Shared fixtures and configuration for API compatibility tests.
These tests validate real API connections to catch provider API breakages.

Usage:
    # Run all API compatibility tests (requires API keys)
    pytest tests/api_compat/ -v

    # Run specific provider tests
    pytest tests/api_compat/test_openai.py -v

    # Skip tests for providers without API keys
    pytest tests/api_compat/ -v  # Tests auto-skip if keys are missing

Environment Variables:
    OPENAI_API_KEY      - OpenAI API key
    ANTHROPIC_API_KEY   - Anthropic API key
    GROQ_API_KEY        - Groq API key
    OPENROUTER_API_KEY  - OpenRouter API key
"""

import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "api_compat: marks tests as API compatibility tests (may cost money)"
    )
    config.addinivalue_line(
        "markers", "openai: marks tests that require OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "anthropic: marks tests that require Anthropic API key"
    )
    config.addinivalue_line(
        "markers", "groq: marks tests that require Groq API key"
    )
    config.addinivalue_line(
        "markers", "openrouter: marks tests that require OpenRouter API key"
    )


# API Key availability checks
def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def has_groq_key() -> bool:
    """Check if Groq API key is available."""
    return bool(os.getenv("GROQ_API_KEY"))


def has_openrouter_key() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


# Skip conditions for markers
skip_without_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set"
)

skip_without_anthropic = pytest.mark.skipif(
    not has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set"
)

skip_without_groq = pytest.mark.skipif(
    not has_groq_key(),
    reason="GROQ_API_KEY not set"
)

skip_without_openrouter = pytest.mark.skipif(
    not has_openrouter_key(),
    reason="OPENROUTER_API_KEY not set"
)


@pytest.fixture
def simple_prompt() -> str:
    """Simple prompt for basic generation tests."""
    return "What is 2+2? Reply with just the number."


@pytest.fixture
def streaming_prompt() -> str:
    """Prompt for streaming tests."""
    return "Count from 1 to 5, separated by commas."


@pytest.fixture
def multi_turn_messages() -> list:
    """Messages for multi-turn conversation tests."""
    return [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]
