"""
Integration Tests: stream() - Streaming Function Invocation

Tests streaming function execution using client.stream():
- Returns an iterator that yields chunks as they arrive
- Uses Server-Sent Events (SSE) for real-time delivery
- Perfect for LLM token streaming and progress updates

Run with:
    pytest tests/integration/test_ex_120_stream_functions.py -v
"""

import json
import pytest


def decode_chunk(chunk):
    """Decode a streamed chunk.

    Chunks may come as JSON-encoded strings, so we try to decode them.
    """
    if isinstance(chunk, str):
        try:
            decoded = json.loads(chunk)
            # If it's a string, return it; if it's a dict/list, return as-is
            return decoded
        except (json.JSONDecodeError, TypeError):
            return chunk
    return chunk


# =============================================================================
# BASIC STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_text_basic(client, worker_process, platform):
    """Test basic text streaming."""
    chunks = [decode_chunk(c) for c in client.stream("stream_text", {"text": "Hello world test", "delay_ms": 10})]

    # Should receive multiple chunks
    assert len(chunks) >= 1

    # Reconstructed text should match (with extra spaces)
    full_text = "".join(chunks)
    assert "Hello" in full_text
    assert "world" in full_text
    assert "test" in full_text


@pytest.mark.integration
def test_stream_text_empty(client, worker_process):
    """Test streaming empty text."""
    chunks = [decode_chunk(c) for c in client.stream("stream_text", {"text": "", "delay_ms": 10})]

    # Empty text should yield no chunks or empty chunks
    assert len(chunks) == 0 or all(str(c).strip() == "" for c in chunks)


@pytest.mark.integration
def test_stream_text_single_word(client, worker_process):
    """Test streaming a single word."""
    chunks = [decode_chunk(c) for c in client.stream("stream_text", {"text": "Hello", "delay_ms": 10})]

    assert len(chunks) >= 1
    full_text = "".join(chunks).strip()
    assert full_text == "Hello"


@pytest.mark.integration
def test_stream_text_long(client, worker_process):
    """Test streaming longer text."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 5
    chunks = [decode_chunk(c) for c in client.stream("stream_text", {"text": long_text, "delay_ms": 5})]

    # Should get multiple chunks
    assert len(chunks) > 1

    # Full text should be reconstructed
    full_text = "".join(chunks)
    assert "quick" in full_text
    assert "brown" in full_text
    assert "fox" in full_text


# =============================================================================
# STORY STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_story_dragon(client, worker_process):
    """Test streaming a dragon story."""
    chunks = [decode_chunk(c) for c in client.stream("stream_story", {"prompt": "dragon"})]

    assert len(chunks) > 1
    full_story = "".join(chunks)

    # Should contain dragon-related content
    assert "dragon" in full_story.lower() or "ember" in full_story.lower()


@pytest.mark.integration
def test_stream_story_robot(client, worker_process):
    """Test streaming a robot story."""
    chunks = [decode_chunk(c) for c in client.stream("stream_story", {"prompt": "robot"})]

    assert len(chunks) > 1
    full_story = "".join(chunks)

    # Should contain robot-related content
    assert "robot" in full_story.lower() or "spark" in full_story.lower()


@pytest.mark.integration
def test_stream_story_generic(client, worker_process):
    """Test streaming with generic prompt."""
    chunks = [decode_chunk(c) for c in client.stream("stream_story", {"prompt": "adventure"})]

    assert len(chunks) > 1
    full_story = "".join(chunks)

    # Should contain the prompt
    assert "adventure" in full_story.lower() or "story" in full_story.lower()


# =============================================================================
# COUNTER STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_counter_small(client, worker_process):
    """Test streaming counter with small count."""
    chunks = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 5, "delay_ms": 10})]

    # Should get 5 chunks
    assert len(chunks) == 5

    # Should be numbers 1-5
    numbers = [int(c) for c in chunks]
    assert numbers == [1, 2, 3, 4, 5]


@pytest.mark.integration
def test_stream_counter_ten(client, worker_process):
    """Test streaming counter to 10."""
    chunks = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 10, "delay_ms": 10})]

    assert len(chunks) == 10
    numbers = [int(c) for c in chunks]
    assert numbers == list(range(1, 11))


@pytest.mark.integration
def test_stream_counter_limited(client, worker_process):
    """Test counter is limited to 100."""
    # Request 200, should be limited to 100
    chunks = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 200, "delay_ms": 5})]

    assert len(chunks) == 100


# =============================================================================
# PROGRESS STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_progress_basic(client, worker_process):
    """Test streaming progress updates."""
    chunks = [decode_chunk(c) for c in client.stream("stream_progress", {"steps": 3, "step_duration_ms": 20})]

    # Should have: starting message + 3 step updates + complete message = 5
    assert len(chunks) >= 4

    full_output = "".join(chunks)
    assert "Starting" in full_output
    assert "Step 1" in full_output
    assert "Step 2" in full_output
    assert "Step 3" in full_output
    assert "complete" in full_output.lower()


@pytest.mark.integration
def test_stream_progress_single_step(client, worker_process):
    """Test progress with single step."""
    chunks = [decode_chunk(c) for c in client.stream("stream_progress", {"steps": 1, "step_duration_ms": 20})]

    full_output = "".join(chunks)
    assert "Step 1" in full_output
    assert "100%" in full_output


@pytest.mark.integration
def test_stream_progress_limited(client, worker_process):
    """Test progress is limited to 20 steps."""
    # Request 50, should be limited to 20
    chunks = [decode_chunk(c) for c in client.stream("stream_progress", {"steps": 50, "step_duration_ms": 10})]

    full_output = "".join(chunks)
    assert "Step 20" in full_output
    # Should not have step 21
    assert "Step 21" not in full_output


# =============================================================================
# JSON STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_json_chunks_basic(client, worker_process):
    """Test streaming JSON data."""
    chunks = [decode_chunk(c) for c in client.stream("stream_json_chunks", {"items": 3})]

    assert len(chunks) == 3

    # Each chunk should be valid JSON - decode_chunk returns the parsed object
    for chunk in chunks:
        # chunk might be a string (JSON line) or already parsed dict
        if isinstance(chunk, str):
            data = json.loads(chunk.strip())
        else:
            data = chunk
        assert "id" in data
        assert "value" in data
        assert "timestamp" in data


@pytest.mark.integration
def test_stream_json_chunks_sequential_ids(client, worker_process):
    """Test JSON chunks have sequential IDs."""
    chunks = [decode_chunk(c) for c in client.stream("stream_json_chunks", {"items": 5})]

    ids = []
    for chunk in chunks:
        if isinstance(chunk, str):
            data = json.loads(chunk.strip())
        else:
            data = chunk
        ids.append(data["id"])
    assert ids == [1, 2, 3, 4, 5]


@pytest.mark.integration
def test_stream_json_chunks_limited(client, worker_process):
    """Test JSON chunks are limited to 50."""
    # Request 100, should be limited to 50
    chunks = list(client.stream("stream_json_chunks", {"items": 100}))

    assert len(chunks) == 50


# =============================================================================
# STREAM CONSUMPTION PATTERNS
# =============================================================================


@pytest.mark.integration
def test_stream_partial_consumption(client, worker_process):
    """Test consuming only part of a stream."""
    count = 0
    for chunk in client.stream("stream_counter", {"count": 100, "delay_ms": 5}):
        count += 1
        if count >= 5:
            break

    # Should have consumed at least 5 chunks
    assert count >= 5


@pytest.mark.integration
def test_stream_collect_vs_iterate(client, worker_process):
    """Test collecting stream vs iterating."""
    # Collect all at once
    chunks_list = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 5, "delay_ms": 10})]

    # Iterate one by one
    chunks_iter = []
    for chunk in client.stream("stream_counter", {"count": 5, "delay_ms": 10}):
        chunks_iter.append(decode_chunk(chunk))

    # Should have same content
    assert [int(c) for c in chunks_list] == [int(c) for c in chunks_iter]


@pytest.mark.integration
def test_stream_multiple_sequential(client, worker_process):
    """Test multiple sequential stream calls."""
    # First stream
    chunks1 = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 3, "delay_ms": 10})]
    assert len(chunks1) == 3

    # Second stream
    chunks2 = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 5, "delay_ms": 10})]
    assert len(chunks2) == 5

    # Third stream
    chunks3 = [decode_chunk(c) for c in client.stream("stream_text", {"text": "Hello", "delay_ms": 10})]
    assert len(chunks3) >= 1


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_stream_nonexistent_function(client, worker_process):
    """Test streaming a function that doesn't exist."""
    from agnt5.client import RunError

    with pytest.raises((RunError, Exception)):
        list(client.stream("nonexistent_stream_function", {}))


@pytest.mark.integration
def test_stream_after_error(client, worker_process):
    """Test that streaming works after an error."""
    from agnt5.client import RunError

    # Try to stream a nonexistent function
    try:
        list(client.stream("nonexistent_function", {}))
    except (RunError, Exception):
        pass

    # Streaming should still work
    chunks = [decode_chunk(c) for c in client.stream("stream_counter", {"count": 3, "delay_ms": 10})]
    assert len(chunks) == 3
