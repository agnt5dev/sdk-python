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
    GOOGLE_API_KEY      - Google Gemini API key (or GEMINI_API_KEY)
    DEEPSEEK_API_KEY    - DeepSeek API key
    XAI_API_KEY         - xAI (Grok) API key
    MISTRAL_API_KEY     - Mistral API key
    HUGGINGFACE_API_KEY - HuggingFace API key (or HF_TOKEN)
    OLLAMA_BASE_URL     - Ollama base URL (defaults to localhost:11434)
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
    config.addinivalue_line(
        "markers", "google: marks tests that require Google/Gemini API key"
    )
    config.addinivalue_line(
        "markers", "deepseek: marks tests that require DeepSeek API key"
    )
    config.addinivalue_line(
        "markers", "xai: marks tests that require xAI API key"
    )
    config.addinivalue_line(
        "markers", "mistral: marks tests that require Mistral API key"
    )
    config.addinivalue_line(
        "markers", "huggingface: marks tests that require HuggingFace API key"
    )
    config.addinivalue_line(
        "markers", "ollama: marks tests that require Ollama running locally"
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


def has_google_key() -> bool:
    """Check if Google/Gemini API key is available."""
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def has_deepseek_key() -> bool:
    """Check if DeepSeek API key is available."""
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def has_xai_key() -> bool:
    """Check if xAI API key is available."""
    return bool(os.getenv("XAI_API_KEY"))


def has_mistral_key() -> bool:
    """Check if Mistral API key is available."""
    return bool(os.getenv("MISTRAL_API_KEY"))


def has_huggingface_key() -> bool:
    """Check if HuggingFace API key is available."""
    return bool(os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"))


def has_ollama() -> bool:
    """Check if Ollama is available (check if server responds)."""
    import socket
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Extract host and port from URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 11434
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


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

skip_without_google = pytest.mark.skipif(
    not has_google_key(),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set"
)

skip_without_deepseek = pytest.mark.skipif(
    not has_deepseek_key(),
    reason="DEEPSEEK_API_KEY not set"
)

skip_without_xai = pytest.mark.skipif(
    not has_xai_key(),
    reason="XAI_API_KEY not set"
)

skip_without_mistral = pytest.mark.skipif(
    not has_mistral_key(),
    reason="MISTRAL_API_KEY not set"
)

skip_without_huggingface = pytest.mark.skipif(
    not has_huggingface_key(),
    reason="HUGGINGFACE_API_KEY or HF_TOKEN not set"
)

skip_without_ollama = pytest.mark.skipif(
    not has_ollama(),
    reason="Ollama not running at OLLAMA_BASE_URL (default: localhost:11434)"
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


# =============================================================================
# Agentic Workflow Test Fixtures
# =============================================================================


@pytest.fixture
def calculator_tool():
    """Calculator tool for testing function calling."""
    from agnt5.lm import ToolDefinition
    return ToolDefinition(
        name="calculate",
        description="Perform a mathematical calculation. Use this for any math operations.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    )


@pytest.fixture
def weather_tool():
    """Weather tool for testing function calling."""
    from agnt5.lm import ToolDefinition
    return ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris' or 'New York'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )


@pytest.fixture
def search_tool():
    """Search tool for testing function calling."""
    from agnt5.lm import ToolDefinition
    return ToolDefinition(
        name="search",
        description="Search for information on a topic",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    )


@pytest.fixture
def multi_tool_set(calculator_tool, weather_tool, search_tool):
    """Set of multiple tools for testing tool selection."""
    return [calculator_tool, weather_tool, search_tool]


def simulate_tool_execution(tool_name: str, arguments: dict | str) -> str:
    """Simulate tool execution for agent loop tests.

    Returns a realistic response that the model can use to continue the conversation.
    """
    import json
    # Handle string arguments (some providers return JSON string)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    if tool_name == "calculate":
        expr = arguments.get("expression", "0")
        try:
            # Safe eval for simple math
            result = eval(expr, {"__builtins__": {}}, {})
            return f"The result of {expr} is {result}"
        except Exception:
            return f"Error: Could not evaluate '{expr}'"

    elif tool_name == "get_weather":
        location = arguments.get("location", "Unknown")
        unit = arguments.get("unit", "celsius")
        temp = 22 if unit == "celsius" else 72
        return f"Weather in {location}: {temp}°{'C' if unit == 'celsius' else 'F'}, partly cloudy"

    elif tool_name == "search":
        query = arguments.get("query", "")
        return f"Search results for '{query}': Found 3 relevant articles about {query}."

    else:
        return f"Tool '{tool_name}' executed successfully."


@pytest.fixture
def tool_executor():
    """Fixture that provides the tool execution simulator."""
    return simulate_tool_execution
