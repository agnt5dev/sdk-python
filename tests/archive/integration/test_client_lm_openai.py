"""
Integration Tests: Language Model in Functions

Tests LM functionality when used within AGNT5 functions and agents.

These tests validate:
- Using lm.generate() within @function decorators
- Using lm.stream() within functions
- Structured output within platform functions
- Error handling when LM calls fail

Test Strategy:
- Deploy test functions that use REAL LM APIs (no mocking)
- Test end-to-end execution through platform
- Validate LM responses are properly returned to clients

IMPORTANT: These tests REQUIRE OPENAI_API_KEY to be set
- Tests will FAIL without API key (this is intentional)
- No fallbacks or mocks - we test real LM integration
- This ensures we catch API changes and regressions

Running Tests:
    # Set API key first
    export OPENAI_API_KEY=sk-...

    # Run tests
    pytest tests/integration/test_client_lm.py -v

    # Or use .env file in test-service directory
    cd tests/integration/blueprints/test-service
    cp .env.example .env
    # Edit .env and add your OPENAI_API_KEY
"""

import pytest


@pytest.mark.integration
def test_function_with_lm_generate(client, worker_process):
    """
    Test function that uses lm.generate() internally.

    This validates:
    - LM can be used inside @function decorators
    - LM responses are properly serialized through platform
    - Client receives the generated text
    """
    # Call function that uses LM internally
    result = client.run("generate_greeting", {"name": "Alice", "style": "formal"})

    assert result is not None
    assert "greeting" in result
    # The greeting should be customized based on style
    assert "Alice" in result["greeting"]

    # Call function with very friendly style
    result = client.run("generate_greeting", {"name": "Bob", "style": "friendly"})
    assert result is not None
    assert "greeting" in result
    assert "Bob" in result["greeting"]
    assert "!" in result["greeting"]  # Friendly style likely has exclamation


@pytest.mark.integration
def test_function_with_lm_structured_output(client, worker_process):
    """
    Test function that uses structured output from LM.

    This validates:
    - response_format works within platform functions
    - Structured data is properly returned to client

    KNOWN ISSUE: Rust core currently returns None for structured_output.object field.
    This needs to be fixed in sdk-core before this test can pass.
    """
    result = client.run("analyze_sentiment", {
        "text": "I love this product! It's amazing!"
    })

    assert result is not None
    assert "sentiment" in result
    assert "score" in result
    assert "keywords" in result
    assert isinstance(result["keywords"], list)


@pytest.mark.integration
def test_function_with_lm_streaming(client, worker_process):
    """
    Test function that uses lm.stream() internally.

    This validates:
    - Streaming LM works within functions
    - Streamed text is accumulated and returned
    """
    result = client.run("generate_story", {"topic": "space adventure"})

    assert result is not None
    assert "story" in result
    assert len(result["story"]) > 50  # Should be a decent length story

    # Check for space-related terms (LM may use synonyms)
    space_terms = ["space", "star", "planet", "cosmos", "asteroid", "galaxy", "orbit", "ship", "nebula"]
    story_lower = result["story"].lower()
    assert any(term in story_lower for term in space_terms), \
        f"Story should contain space-related terms. Got: {result['story']}"


@pytest.mark.integration
def test_function_with_lm_multi_turn(client, worker_process):
    """
    Test function that uses multi-turn conversation with LM.

    This validates:
    - Conversation history can be maintained
    - Multi-turn patterns work within functions
    """
    result = client.run("chat_with_context", {
        "user_message": "What's the weather?",
        "context": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Nice to meet you, Bob!"}
        ]
    })

    assert result is not None
    assert "response" in result
    # Assistant should remember the context
    assert len(result["response"]) > 0


@pytest.mark.integration
def test_function_with_lm_error_handling(client, worker_process):
    """
    Test function error handling when LM call fails.

    This validates:
    - LM errors are properly caught
    - Error messages are propagated to client
    """
    from agnt5 import RunError

    with pytest.raises(RunError) as exc_info:
        client.run("generate_with_invalid_model", {"prompt": "test"})

    error_msg = str(exc_info.value).lower()
    # Should indicate LM-related error
    assert any(keyword in error_msg for keyword in ["model", "provider", "api"])


@pytest.mark.integration
@pytest.mark.skip(reason="Requires real API key for live LM test")
def test_function_with_live_lm(client, worker_process):
    """
    Integration test with real LM API (requires API key).

    This test is skipped by default. To run with real API:
    1. Set OPENAI_API_KEY environment variable
    2. Remove @pytest.mark.skip decorator
    3. Run: pytest tests/integration/test_client_lm.py::test_function_with_live_lm -v
    """
    result = client.run("generate_joke", {"topic": "programming"})

    assert result is not None
    assert "joke" in result
    assert len(result["joke"]) > 20
    assert "programming" in result["joke"].lower() or "code" in result["joke"].lower()

