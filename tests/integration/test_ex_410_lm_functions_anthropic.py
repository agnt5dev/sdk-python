"""
Integration Tests for ex_05_lm_functions_anthropic.py

Tests Anthropic Claude LLM integration through the platform:
- claude_generate - Non-streaming Claude completion
- claude_stream - Streaming Claude completion
- claude_analyze - Text analysis (sentiment, summary, keywords)
- claude_translate - Text translation

Run with:
    # Local mode (requires running dev server with examples worker)
    # Requires ANTHROPIC_API_KEY environment variable
    pytest tests/integration/test_ex_05_lm_functions_anthropic.py -v

    # Skip LLM tests if no API key
    pytest tests/integration/test_ex_05_lm_functions_anthropic.py -v -m "not llm"
"""

import os
import pytest

# Skip all LLM tests if no API key is available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set - skipping LLM tests"
    ),
]


# =============================================================================
# CLAUDE GENERATE (Non-streaming)
# =============================================================================


def test_claude_generate_basic(client, worker_process):
    """Test basic Claude generation."""
    result = client.run("claude_generate", {
        "prompt": "What is 2+2? Reply with just the number.",
    })

    assert "response" in result
    assert "4" in result["response"]
    assert result["method"] == "generate"
    assert result["provider"] == "anthropic"


def test_claude_generate_with_context(client, worker_process):
    """Test Claude generation with contextual prompt."""
    result = client.run("claude_generate", {
        "prompt": "The capital of Japan is",
    })

    assert "response" in result
    assert "tokyo" in result["response"].lower()


def test_claude_generate_custom_model(client, worker_process):
    """Test Claude generation with explicit model."""
    result = client.run("claude_generate", {
        "prompt": "Say hello",
        "model": "anthropic/claude-3-haiku-20240307",
    })

    assert result["model"] == "anthropic/claude-3-haiku-20240307"
    assert "response" in result


# =============================================================================
# CLAUDE STREAM (Streaming)
# =============================================================================


@pytest.mark.xfail(
    reason="Rust SDK core doesn't handle Anthropic 'ping' events - see sdk-core/src/lm/anthropic.rs",
    strict=False,
)
def test_claude_stream_basic(client, worker_process):
    """Test streaming Claude generation."""
    result = client.run("claude_stream", {
        "prompt": "Count from 1 to 3, separated by commas.",
    })

    assert "response" in result
    assert result["method"] == "stream"
    assert result["chunk_count"] > 0
    assert result["provider"] == "anthropic"
    # Should contain the numbers
    assert "1" in result["response"]
    assert "2" in result["response"]
    assert "3" in result["response"]


@pytest.mark.xfail(
    reason="Rust SDK core doesn't handle Anthropic 'ping' events - see sdk-core/src/lm/anthropic.rs",
    strict=False,
)
def test_claude_stream_produces_chunks(client, worker_process):
    """Test that streaming produces multiple chunks."""
    result = client.run("claude_stream", {
        "prompt": "Write a short sentence about programming.",
    })

    assert result["chunk_count"] >= 1
    assert len(result["response"]) > 0


# =============================================================================
# CLAUDE ANALYZE
# =============================================================================


def test_claude_analyze_sentiment_positive(client, worker_process):
    """Test sentiment analysis - positive."""
    result = client.run("claude_analyze", {
        "text": "I absolutely love this product! Best purchase ever!",
        "analysis_type": "sentiment",
    })

    assert "result" in result
    assert result["analysis_type"] == "sentiment"
    assert "positive" in result["result"].lower()


def test_claude_analyze_sentiment_negative(client, worker_process):
    """Test sentiment analysis - negative."""
    result = client.run("claude_analyze", {
        "text": "This is terrible. Worst experience of my life.",
        "analysis_type": "sentiment",
    })

    assert "result" in result
    assert "negative" in result["result"].lower()


def test_claude_analyze_summary(client, worker_process):
    """Test text summarization."""
    long_text = """
    Python is a high-level, general-purpose programming language. Its design philosophy
    emphasizes code readability with the use of significant indentation. Python is
    dynamically typed and garbage-collected. It supports multiple programming paradigms,
    including structured, object-oriented and functional programming.
    """

    result = client.run("claude_analyze", {
        "text": long_text,
        "analysis_type": "summary",
    })

    assert "result" in result
    assert result["analysis_type"] == "summary"
    # Summary should be shorter than original
    assert len(result["result"]) < len(long_text)


def test_claude_analyze_keywords(client, worker_process):
    """Test keyword extraction."""
    result = client.run("claude_analyze", {
        "text": "Machine learning and artificial intelligence are transforming data science and analytics.",
        "analysis_type": "keywords",
    })

    assert "result" in result
    assert result["analysis_type"] == "keywords"
    # Should extract relevant keywords
    result_lower = result["result"].lower()
    assert any(kw in result_lower for kw in ["machine learning", "artificial intelligence", "data", "analytics"])


def test_claude_analyze_default_type(client, worker_process):
    """Test default analysis type is sentiment."""
    result = client.run("claude_analyze", {
        "text": "This is great!",
    })

    assert result["analysis_type"] == "sentiment"


# =============================================================================
# CLAUDE TRANSLATE
# =============================================================================


def test_claude_translate_to_spanish(client, worker_process):
    """Test translation to Spanish."""
    result = client.run("claude_translate", {
        "text": "Hello, world!",
        "target_language": "Spanish",
    })

    assert "translated" in result
    assert result["target_language"] == "Spanish"
    assert result["original"] == "Hello, world!"
    # Should contain Spanish translation
    assert "hola" in result["translated"].lower()


def test_claude_translate_to_french(client, worker_process):
    """Test translation to French."""
    result = client.run("claude_translate", {
        "text": "Good morning",
        "target_language": "French",
    })

    assert "translated" in result
    assert result["target_language"] == "French"
    # Should contain French translation
    translated_lower = result["translated"].lower()
    assert "bonjour" in translated_lower or "matin" in translated_lower


def test_claude_translate_to_german(client, worker_process):
    """Test translation to German."""
    result = client.run("claude_translate", {
        "text": "Thank you",
        "target_language": "German",
    })

    assert "translated" in result
    assert result["target_language"] == "German"
    # Accept "danke" or "dank" (e.g., "Vielen Dank")
    assert "dank" in result["translated"].lower()


def test_claude_translate_default_language(client, worker_process):
    """Test default target language is Spanish."""
    result = client.run("claude_translate", {
        "text": "Hello",
    })

    assert result["target_language"] == "Spanish"


def test_claude_translate_longer_text(client, worker_process):
    """Test translation of longer text."""
    result = client.run("claude_translate", {
        "text": "The quick brown fox jumps over the lazy dog.",
        "target_language": "Japanese",
    })

    assert "translated" in result
    assert len(result["translated"]) > 0
    # Japanese uses different characters
    assert result["translated"] != result["original"]
