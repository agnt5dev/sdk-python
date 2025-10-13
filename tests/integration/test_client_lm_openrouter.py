"""
Integration Tests: OpenRouter Language Model

Tests OpenRouter multi-provider integration within AGNT5 functions.

OpenRouter provides unified access to multiple LM providers through a single API.
This allows testing different models without managing multiple API keys.

These tests validate:
- Using lm.generate() with OpenRouter models
- Multi-provider access through OpenRouter
- Free tier models for cost-effective testing

Test Strategy:
- Deploy test functions that use REAL OpenRouter API (no mocking)
- Test end-to-end execution through platform
- Validate responses are properly returned to clients
- Use free tier models where available to minimize costs

IMPORTANT: These tests REQUIRE OPENROUTER_API_KEY to be set
- Tests will FAIL without API key (this is intentional)
- No fallbacks or mocks - we test real integration
- This ensures we catch API changes and regressions

Running Tests:
    # Set API key first
    export OPENROUTER_API_KEY=sk-or-...

    # Run tests
    pytest tests/integration/test_client_lm_openrouter.py -v

    # Or use .env file in test-service directory
    cd tests/integration/blueprints/test-service
    cp .env.example .env
    # Edit .env and add your OPENROUTER_API_KEY

Note: OpenRouter offers free tier models that are ideal for testing.
These tests use free models where possible to minimize costs.
"""

import pytest


@pytest.mark.integration
def test_openrouter_generate_text_default(client, worker_process):
    """
    Test basic text generation with OpenRouter (default model).

    This validates:
    - OpenRouter works within @function decorators
    - Default model selection works correctly
    - Responses are properly serialized through platform
    """
    result = client.run("generate_text_openrouter", {
        "prompt": "What is quantum computing in one sentence?"
    })

    assert result is not None
    assert "text" in result
    assert "provider" in result
    assert result["provider"] == "openrouter"
    assert "model" in result

    # Should have meaningful content
    assert len(result["text"]) > 20
    assert any(term in result["text"].lower() for term in ["quantum", "computing", "computer", "qubit"])


@pytest.mark.integration
def test_openrouter_generate_text_custom_model(client, worker_process):
    """
    Test text generation with a specific model through OpenRouter.

    This validates:
    - Custom model selection works
    - Different models can be used through same function
    """
    result = client.run("generate_text_openrouter", {
        "prompt": "Explain machine learning briefly.",
        "model_name": "meta-llama/llama-3.1-8b-instruct:free"
    })

    assert result is not None
    assert "text" in result
    assert "provider" in result
    assert result["provider"] == "openrouter"
    assert "model" in result
    assert "llama" in result["model"].lower()

    # Should have content about ML
    assert len(result["text"]) > 20


@pytest.mark.integration
def test_openrouter_free_model(client, worker_process):
    """
    Test with OpenRouter's free tier model.

    This validates:
    - Free tier models work correctly
    - Cost-effective testing is possible
    """
    result = client.run("compare_models_openrouter", {
        "prompt": "What is Docker in simple terms?"
    })

    assert result is not None
    assert "response" in result
    assert "provider" in result
    assert result["provider"] == "openrouter"
    assert "model" in result

    # Should have meaningful content
    assert len(result["response"]) > 20
    assert "docker" in result["response"].lower()


@pytest.mark.integration
def test_openrouter_technical_query(client, worker_process):
    """
    Test OpenRouter with technical questions.

    This validates:
    - Models can handle technical topics
    - Responses are accurate and relevant
    """
    result = client.run("generate_text_openrouter", {
        "prompt": "What is the difference between synchronous and asynchronous programming?"
    })

    assert result is not None
    assert "text" in result
    assert result["provider"] == "openrouter"

    # Should have substantial content
    assert len(result["text"]) > 50

    # Should mention relevant concepts
    text_lower = result["text"].lower()
    async_terms = ["async", "synchronous", "thread", "wait", "parallel", "concurrent"]
    matches = sum(1 for term in async_terms if term in text_lower)
    assert matches >= 2, f"Should mention async concepts. Got: {result['text']}"


@pytest.mark.integration
def test_openrouter_creative_task(client, worker_process):
    """
    Test OpenRouter with creative writing task.

    This validates:
    - Models can handle creative tasks
    - Responses are coherent and on-topic
    """
    result = client.run("generate_text_openrouter", {
        "prompt": "Write a very short tagline for a cloud computing service (max 10 words)."
    })

    assert result is not None
    assert "text" in result
    assert result["provider"] == "openrouter"

    # Should be concise (tagline)
    word_count = len(result["text"].split())
    assert word_count <= 20, f"Tagline should be short. Got {word_count} words: {result['text']}"

    # Should be non-empty
    assert len(result["text"]) > 5


@pytest.mark.integration
def test_openrouter_code_explanation(client, worker_process):
    """
    Test OpenRouter with code-related explanation.

    This validates:
    - Models can explain technical concepts
    - Responses are accurate
    """
    result = client.run("compare_models_openrouter", {
        "prompt": "What is a Python decorator and what is it used for?"
    })

    assert result is not None
    assert "response" in result
    assert result["provider"] == "openrouter"

    # Should have meaningful explanation
    assert len(result["response"]) > 30

    # Should mention relevant Python concepts
    response_lower = result["response"].lower()
    python_terms = ["decorator", "function", "python", "@", "wrapper"]
    matches = sum(1 for term in python_terms if term in response_lower)
    assert matches >= 2, f"Should mention Python decorator concepts. Got: {result['response']}"


@pytest.mark.integration
@pytest.mark.skip(reason="Only run when testing provider diversity")
def test_openrouter_multiple_models(client, worker_process):
    """
    Test multiple models through OpenRouter to verify provider diversity.

    This test is skipped by default to minimize API calls.
    Remove @pytest.mark.skip to run.
    """
    models_to_test = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "anthropic/claude-3-5-sonnet",
    ]

    prompt = "Explain what a database index is."

    for model in models_to_test:
        result = client.run("generate_text_openrouter", {
            "prompt": prompt,
            "model_name": model
        })

        assert result is not None
        assert "text" in result
        assert result["provider"] == "openrouter"
        assert len(result["text"]) > 20

        # Should mention database/index concepts
        text_lower = result["text"].lower()
        assert any(term in text_lower for term in ["index", "database", "query", "search", "performance"])


@pytest.mark.integration
@pytest.mark.skip(reason="Only run for cost estimation")
def test_openrouter_response_length(client, worker_process):
    """
    Test response length to estimate token usage.

    This test is skipped by default.
    Remove @pytest.mark.skip to run for cost estimation.
    """
    result = client.run("generate_text_openrouter", {
        "prompt": "List 5 benefits of using microservices architecture."
    })

    assert result is not None
    assert "text" in result

    # Log response length for cost estimation
    response_length = len(result["text"])
    word_count = len(result["text"].split())

    print(f"\n=== Response Metrics ===")
    print(f"Characters: {response_length}")
    print(f"Words: {word_count}")
    print(f"Estimated tokens: ~{word_count * 1.3:.0f}")  # Rough estimate

    assert response_length > 100
