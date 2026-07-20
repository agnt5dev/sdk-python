"""
Integration Tests for Context Propagation Workflows

Tests context propagation patterns:
- Sequential context: Pass accumulated context through steps
- Parallel context: Branch context to parallel steps, merge results

Run with:
    # Local mode (requires running dev server with examples worker)
    pytest tests/integration/test_ex_700_context_propagation.py -v

    # Specific test
    pytest tests/integration/test_ex_700_context_propagation.py::test_sequential_context_basic -v
"""

import pytest

# =============================================================================
# SEQUENTIAL CONTEXT WORKFLOWS (wf_42)
# =============================================================================


@pytest.mark.integration
def test_sequential_context_basic(client, worker_process):
    """Test basic sequential context propagation."""
    result = client.run(
        "sequential_context_workflow",
        {"input_data": "Hello World"},
        component_type="workflow"
    )

    assert result["input"] == "Hello World"

    # Verify parsed step
    assert result["parsed"]["words"] == ["Hello", "World"]
    assert result["parsed"]["length"] == 11
    assert result["parsed"]["uppercase"] == "HELLO WORLD"

    # Verify analysis step (uses parsed data)
    assert result["analysis"]["word_count"] == 2
    assert result["analysis"]["char_count"] == 11
    assert result["analysis"]["has_uppercase"] is True

    # Verify enriched step (uses both parsed and analysis)
    assert result["enriched"]["summary"] == "2 words, 11 chars"
    assert result["enriched"]["transformed"] == "HELLO WORLD"


@pytest.mark.integration
def test_sequential_context_empty_input(client, worker_process):
    """Test sequential context with empty input."""
    result = client.run(
        "sequential_context_workflow",
        {"input_data": ""},
        component_type="workflow"
    )

    assert result["input"] == ""
    assert result["parsed"]["words"] == []
    assert result["parsed"]["length"] == 0
    assert result["analysis"]["word_count"] == 0


@pytest.mark.integration
def test_sequential_context_special_characters(client, worker_process):
    """Test sequential context with special characters."""
    result = client.run(
        "sequential_context_workflow",
        {"input_data": "Hello, World! How are you?"},
        component_type="workflow"
    )

    assert result["input"] == "Hello, World! How are you?"
    assert result["analysis"]["word_count"] == 5
    assert result["analysis"]["has_uppercase"] is True


@pytest.mark.integration
def test_sequential_context_single_word(client, worker_process):
    """Test sequential context with single word."""
    result = client.run(
        "sequential_context_workflow",
        {"input_data": "AGNT5"},
        component_type="workflow"
    )

    assert result["analysis"]["word_count"] == 1
    assert result["parsed"]["uppercase"] == "AGNT5"


# =============================================================================
# PARALLEL CONTEXT WORKFLOWS (wf_43)
# =============================================================================


@pytest.mark.integration
def test_parallel_context_basic(client, worker_process):
    """Test parallel context branching and merging."""
    result = client.run(
        "parallel_context_workflow",
        {"text": "The quick brown fox jumps over the lazy dog."},
        component_type="workflow"
    )

    assert result["input"] == "The quick brown fox jumps over the lazy dog."

    # Verify parallel branches ran
    parallel = result["parallel_results"]
    assert parallel["parallel_branches"] == 3

    # Verify length analysis
    assert parallel["length_analysis"]["char_count"] > 0
    assert parallel["length_analysis"]["word_count"] == 9

    # Verify content analysis
    assert parallel["content_analysis"]["unique_words"] > 0
    assert parallel["content_analysis"]["avg_word_length"] > 0

    # Verify structure analysis
    assert parallel["structure_analysis"]["sentences"] == 1
    assert parallel["structure_analysis"]["has_punctuation"] is True


@pytest.mark.integration
def test_parallel_context_no_punctuation(client, worker_process):
    """Test parallel context with no punctuation."""
    result = client.run(
        "parallel_context_workflow",
        {"text": "Hello world how are you"},
        component_type="workflow"
    )

    parallel = result["parallel_results"]
    assert parallel["structure_analysis"]["sentences"] == 0
    assert parallel["structure_analysis"]["has_punctuation"] is False


@pytest.mark.integration
def test_parallel_context_multiple_sentences(client, worker_process):
    """Test parallel context with multiple sentences."""
    result = client.run(
        "parallel_context_workflow",
        {"text": "First sentence. Second sentence! Third sentence?"},
        component_type="workflow"
    )

    parallel = result["parallel_results"]
    assert parallel["structure_analysis"]["sentences"] == 3
    assert parallel["structure_analysis"]["has_punctuation"] is True


@pytest.mark.integration
def test_parallel_context_repeated_words(client, worker_process):
    """Test parallel context counts unique words correctly."""
    result = client.run(
        "parallel_context_workflow",
        {"text": "the the the cat cat sat"},
        component_type="workflow"
    )

    parallel = result["parallel_results"]
    assert parallel["length_analysis"]["word_count"] == 6
    assert parallel["content_analysis"]["unique_words"] == 3  # the, cat, sat


# =============================================================================
# UNICODE / INTERNATIONALIZATION (P1.5)
# =============================================================================


@pytest.mark.integration
def test_sequential_context_unicode_input(client, worker_process):
    """Test sequential context with Unicode input (P1.5).

    Validates:
    - CJK characters propagate through workflow steps
    - Mixed scripts are handled correctly
    - Unicode is preserved in all transformation steps
    """
    # Test with mixed Unicode content
    unicode_input = "Hello 世界 café"

    result = client.run(
        "sequential_context_workflow",
        {"input_data": unicode_input},
        component_type="workflow"
    )

    # Verify input preserved
    assert result["input"] == unicode_input

    # Verify parsed step handles Unicode
    assert result["parsed"]["uppercase"] == unicode_input.upper()
    assert result["parsed"]["length"] == len(unicode_input)

    # Words should be split correctly (space-separated)
    assert len(result["parsed"]["words"]) == 3  # "Hello", "世界", "café"

    # Verify analysis uses correct counts
    assert result["analysis"]["char_count"] == len(unicode_input)
    assert result["analysis"]["word_count"] == 3


@pytest.mark.integration
def test_parallel_context_unicode_input(client, worker_process):
    """Test parallel context with Unicode input (P1.5).

    Validates:
    - Unicode text is correctly analyzed in parallel branches
    - Character and word counts handle multi-byte characters
    """
    # Japanese text with punctuation
    unicode_input = "こんにちは世界。今日は良い天気です。"

    result = client.run(
        "parallel_context_workflow",
        {"text": unicode_input},
        component_type="workflow"
    )

    # Verify input preserved
    assert result["input"] == unicode_input

    # Verify parallel results exist
    parallel = result["parallel_results"]
    assert "length_analysis" in parallel
    assert "content_analysis" in parallel
    assert "structure_analysis" in parallel

    # Character count should handle Unicode correctly
    assert parallel["length_analysis"]["char_count"] == len(unicode_input)

    # Should detect punctuation
    assert parallel["structure_analysis"]["has_punctuation"] is True
