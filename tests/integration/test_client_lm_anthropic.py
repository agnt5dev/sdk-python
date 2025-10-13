"""
Integration Tests: Anthropic Claude Language Model

Tests Anthropic Claude integration when used within AGNT5 functions.

These tests validate:
- Using lm.generate() with Anthropic models
- Claude model responses are properly serialized
- Error handling for Anthropic-specific issues

Test Strategy:
- Deploy test functions that use REAL Anthropic API (no mocking)
- Test end-to-end execution through platform
- Validate responses are properly returned to clients

IMPORTANT: These tests REQUIRE ANTHROPIC_API_KEY to be set
- Tests will FAIL without API key (this is intentional)
- No fallbacks or mocks - we test real integration
- This ensures we catch API changes and regressions

Running Tests:
    # Set API key first
    export ANTHROPIC_API_KEY=sk-ant-...

    # Run tests
    pytest tests/integration/test_client_lm_anthropic.py -v

    # Or use .env file in test-service directory
    cd tests/integration/blueprints/test-service
    cp .env.example .env
    # Edit .env and add your ANTHROPIC_API_KEY
"""

import pytest


@pytest.mark.integration
def test_anthropic_generate_text(client, worker_process):
    """
    Test basic text generation with Anthropic Claude.

    This validates:
    - Anthropic models work within @function decorators
    - Responses are properly serialized through platform
    - Client receives the generated text
    """
    result = client.run("generate_text_anthropic", {
        "prompt": "Explain what a distributed system is in one sentence."
    })

    assert result is not None
    assert "text" in result
    assert "provider" in result
    assert result["provider"] == "anthropic"
    assert "model" in result

    # Should have meaningful content
    assert len(result["text"]) > 20
    assert any(term in result["text"].lower() for term in ["distributed", "system", "computer", "network"])


@pytest.mark.integration
def test_anthropic_summarization(client, worker_process):
    """
    Test text summarization with Anthropic Claude.

    This validates:
    - Claude can process and summarize text
    - Summarization preserves key information
    """
    long_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning models can now understand natural language, generate images,
    and even write code. These advances are transforming industries from healthcare
    to finance, enabling new capabilities that were previously impossible.
    However, challenges remain in areas like interpretability, fairness, and safety.
    """

    result = client.run("summarize_with_anthropic", {"text": long_text})

    assert result is not None
    assert "summary" in result
    assert "provider" in result
    assert result["provider"] == "anthropic"

    # Summary should be shorter than original
    assert len(result["summary"]) < len(long_text)

    # Should capture key concepts
    summary_lower = result["summary"].lower()
    # At least one key term should be present
    assert any(term in summary_lower for term in ["ai", "artificial intelligence", "machine learning", "ml"])


@pytest.mark.integration
def test_anthropic_creative_writing(client, worker_process):
    """
    Test Claude's creative writing capabilities.

    This validates:
    - Claude can generate creative content
    - Responses are coherent and on-topic
    """
    result = client.run("generate_text_anthropic", {
        "prompt": "Write a haiku about programming."
    })

    assert result is not None
    assert "text" in result
    assert result["provider"] == "anthropic"

    # Haiku should be relatively short
    assert len(result["text"]) < 200

    # Should have programming-related content
    text_lower = result["text"].lower()
    programming_terms = ["code", "program", "bug", "debug", "function", "computer", "software", "algorithm"]
    assert any(term in text_lower for term in programming_terms), \
        f"Haiku should mention programming. Got: {result['text']}"


@pytest.mark.integration
def test_anthropic_technical_explanation(client, worker_process):
    """
    Test Claude's ability to provide technical explanations.

    This validates:
    - Claude can handle technical topics
    - Explanations are accurate and detailed
    """
    result = client.run("generate_text_anthropic", {
        "prompt": "Explain what a REST API is and why it's useful."
    })

    assert result is not None
    assert "text" in result
    assert result["provider"] == "anthropic"

    # Should have substantial content
    assert len(result["text"]) > 50

    # Should mention key REST concepts
    text_lower = result["text"].lower()
    rest_terms = ["rest", "api", "http", "endpoint", "resource", "request", "response"]
    matches = sum(1 for term in rest_terms if term in text_lower)
    assert matches >= 3, f"Should mention multiple REST concepts. Got: {result['text']}"


@pytest.mark.integration
@pytest.mark.skip(reason="Only run when testing specific edge cases")
def test_anthropic_long_response(client, worker_process):
    """
    Test Claude with a request that generates a longer response.

    This test is skipped by default to avoid high token usage.
    Remove @pytest.mark.skip to run.
    """
    result = client.run("generate_text_anthropic", {
        "prompt": "List and briefly explain 5 important concepts in machine learning."
    })

    assert result is not None
    assert "text" in result

    # Should be a substantial response
    assert len(result["text"]) > 100

    # Should mention multiple ML concepts
    text_lower = result["text"].lower()
    ml_terms = ["supervised", "unsupervised", "neural", "training", "model", "data", "feature"]
    matches = sum(1 for term in ml_terms if term in text_lower)
    assert matches >= 4, f"Should mention multiple ML concepts. Got: {result['text']}"
