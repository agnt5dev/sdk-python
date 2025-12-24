"""
API Compatibility Tests

Tests that validate real API connections to LLM providers.
These tests are designed to run weekly via CI to catch breaking API changes.

Supported Providers:
- OpenAI (test_openai.py)
- Anthropic (test_anthropic.py)
- Groq (test_groq.py)
- OpenRouter (test_openrouter.py)

Usage:
    # Run all tests (requires all API keys)
    pytest tests/api_compat/ -v

    # Run specific provider
    pytest tests/api_compat/test_openai.py -v

    # Run with specific markers
    pytest tests/api_compat/ -m "openai" -v
    pytest tests/api_compat/ -m "not anthropic" -v
"""
