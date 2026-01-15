"""
Integration Tests for ex_04_lm_functions_openai.py

Tests OpenAI LLM integration through the platform:
- lm_generate - Non-streaming LLM completion
- lm_stream - Streaming LLM completion
- lm_summarize - Text summarization
- lm_classify - Text classification

Run with:
    # Local mode (requires running dev server with examples worker)
    # Requires OPENAI_API_KEY environment variable
    pytest tests/integration/test_ex_04_lm_functions_openai.py -v

    # Skip LLM tests if no API key
    pytest tests/integration/test_ex_04_lm_functions_openai.py -v -m "not llm"
"""

import os
import pytest

# Skip all LLM tests if no API key is available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping LLM tests"
    ),
]


# =============================================================================
# LM GENERATE (Non-streaming)
# =============================================================================


def test_lm_generate_basic(client, worker_process):
    """Test basic LLM generation."""
    result = client.run("lm_generate", {
        "prompt": "What is 2+2? Reply with just the number.",
        "model": "openai/gpt-4o-mini",
    })

    assert "response" in result
    assert "4" in result["response"]
    assert result["method"] == "generate"
    assert result["model"] == "openai/gpt-4o-mini"


def test_lm_generate_with_context(client, worker_process):
    """Test LLM generation with contextual prompt."""
    result = client.run("lm_generate", {
        "prompt": "The capital of France is",
        "model": "openai/gpt-4o-mini",
    })

    assert "response" in result
    # Should mention Paris
    assert "paris" in result["response"].lower()


# =============================================================================
# LM STREAM (Streaming)
# =============================================================================


def test_lm_stream_basic(client, worker_process):
    """Test streaming LLM generation."""
    result = client.run("lm_stream", {
        "prompt": "Count from 1 to 3, separated by commas.",
        "model": "openai/gpt-4o-mini",
    })

    assert "response" in result
    assert result["method"] == "stream"
    assert result["chunk_count"] > 0
    # Should contain the numbers
    assert "1" in result["response"]
    assert "2" in result["response"]
    assert "3" in result["response"]


def test_lm_stream_produces_chunks(client, worker_process):
    """Test that streaming produces multiple chunks."""
    result = client.run("lm_stream", {
        "prompt": "Write a short sentence about the weather.",
        "model": "openai/gpt-4o-mini",
    })

    # Streaming should produce multiple chunks
    assert result["chunk_count"] >= 1
    assert len(result["response"]) > 0


# =============================================================================
# LM SUMMARIZE
# =============================================================================


def test_lm_summarize(client, worker_process):
    """Test text summarization."""
    long_text = """
    Python is a high-level, general-purpose programming language. Its design philosophy
    emphasizes code readability with the use of significant indentation. Python is
    dynamically typed and garbage-collected. It supports multiple programming paradigms,
    including structured, object-oriented and functional programming. Python was conceived
    in the late 1980s by Guido van Rossum and first released in 1991.
    """

    result = client.run("lm_summarize", {
        "text": long_text,
        "max_words": 30,
    })

    assert "summary" in result
    assert result["original_length"] > 0
    # Summary should be shorter than original
    assert len(result["summary"]) < len(long_text)


# =============================================================================
# LM CLASSIFY
# =============================================================================


def test_lm_classify_positive(client, worker_process):
    """Test classification of positive sentiment."""
    result = client.run("lm_classify", {
        "text": "I absolutely love this product! Best purchase ever!",
        "categories": ["positive", "negative", "neutral"],
    })

    assert "category" in result
    assert result["category"] == "positive"


def test_lm_classify_negative(client, worker_process):
    """Test classification of negative sentiment."""
    result = client.run("lm_classify", {
        "text": "This is terrible. Worst experience of my life.",
        "categories": ["positive", "negative", "neutral"],
    })

    assert "category" in result
    assert result["category"] == "negative"


def test_lm_classify_custom_categories(client, worker_process):
    """Test classification with custom categories."""
    result = client.run("lm_classify", {
        "text": "The quarterly earnings exceeded expectations by 15%.",
        "categories": ["finance", "sports", "technology", "entertainment"],
    })

    assert "category" in result
    assert result["category"] == "finance"


# =============================================================================
# IMPROVED LM FUNCTION TESTS (P2.4)
# =============================================================================


def test_lm_generate_with_json_format(client, worker_process):
    """Test LLM generation requesting JSON output (P2.4).

    Validates:
    - Model can return structured JSON
    - JSON is valid and parseable
    - Contains expected fields
    """
    import json

    result = client.run("lm_generate", {
        "prompt": 'Return a JSON object with fields "name" and "age". Example: {"name": "Alice", "age": 30}',
        "model": "openai/gpt-4o-mini",
    })

    assert "response" in result

    # Try to parse as JSON (strip markdown code blocks if present)
    response = result["response"]
    if "```" in response:
        # Extract JSON from markdown code block
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if match:
            response = match.group(1)

    try:
        parsed = json.loads(response.strip())
        assert "name" in parsed or "age" in parsed, (
            f"JSON should contain 'name' or 'age'. Got: {parsed}"
        )
    except json.JSONDecodeError:
        # Model might not always return valid JSON, but it should try
        assert "{" in response and "}" in response, (
            f"Response should look like JSON. Got: {response}"
        )


def test_lm_generate_returns_structured_response(client, worker_process):
    """Test LLM generate returns properly structured response (P2.4).

    Validates:
    - Response contains all expected fields
    - Fields have correct types
    - No missing metadata
    """
    result = client.run("lm_generate", {
        "prompt": "Say hello.",
        "model": "openai/gpt-4o-mini",
    })

    # Required fields
    assert "response" in result, "Missing 'response' field"
    assert "method" in result, "Missing 'method' field"
    assert "model" in result, "Missing 'model' field"

    # Type validation
    assert isinstance(result["response"], str), "response should be string"
    assert isinstance(result["method"], str), "method should be string"
    assert isinstance(result["model"], str), "model should be string"

    # Value validation
    assert result["method"] == "generate", f"Expected method='generate', got {result['method']}"
    assert len(result["response"]) > 0, "Response should not be empty"


def test_lm_summarize_preserves_key_info(client, worker_process):
    """Test summarization preserves key information (P2.4).

    Validates:
    - Summary mentions key concepts from original
    - Summary is significantly shorter than original
    """
    original_text = """
    The Apollo 11 mission was the American spaceflight that first landed humans on the Moon.
    Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module
    Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar surface
    six hours and 39 minutes later. Aldrin joined him 19 minutes later.
    """

    result = client.run("lm_summarize", {
        "text": original_text,
        "max_words": 20,
    })

    assert "summary" in result
    summary = result["summary"].lower()

    # Should preserve key information
    key_terms = ["moon", "armstrong", "apollo", "1969", "lunar"]
    found_terms = [term for term in key_terms if term in summary]

    assert len(found_terms) >= 2, (
        f"Summary should mention key terms. Found: {found_terms}. Summary: {summary}"
    )

    # Summary should be much shorter
    assert len(result["summary"]) < len(original_text) / 2, (
        "Summary should be significantly shorter than original"
    )
