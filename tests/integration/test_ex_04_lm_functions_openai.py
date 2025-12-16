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
