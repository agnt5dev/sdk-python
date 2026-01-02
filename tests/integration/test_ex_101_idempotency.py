"""
Integration Tests: Idempotency (Exactly-Once Semantics)

Tests the Idempotency-Key header behavior:
- Same key returns same result (no re-execution)
- Different keys execute independently
- Expired keys allow re-execution
- Concurrent requests with same key deduplicate

Usage:
    pytest tests/integration/test_idempotency.py -v
"""

import concurrent.futures
import uuid

import httpx
import pytest


# =============================================================================
# BASIC IDEMPOTENCY
# =============================================================================


@pytest.mark.integration
def test_idempotency_returns_same_result(platform, worker_process):
    """Same idempotency key returns identical result without re-execution."""
    idempotency_key = f"test-idemp-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    # First request - should execute
    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result1 = resp1.json()

    assert resp1.status_code == 200
    assert result1["status"] == "completed"
    run_id_1 = result1["runId"]

    # Second request with SAME key - should return cached result
    with httpx.Client(timeout=30) as http_client:
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    assert resp2.status_code == 200
    assert result2["runId"] == run_id_1  # Same run ID


@pytest.mark.integration
def test_different_idempotency_keys_execute_independently(platform, worker_process):
    """Different idempotency keys execute as separate requests."""
    gateway_url = platform["gateway_url"]
    key1 = f"test-idemp-{uuid.uuid4()}"
    key2 = f"test-idemp-{uuid.uuid4()}"

    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={"Content-Type": "application/json", "Idempotency-Key": key1},
        )
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Bob"},
            headers={"Content-Type": "application/json", "Idempotency-Key": key2},
        )

    result1 = resp1.json()
    result2 = resp2.json()

    # Different run IDs
    assert result1["runId"] != result2["runId"]


@pytest.mark.integration
def test_no_idempotency_key_executes_every_time(platform, worker_process):
    """Without idempotency key, each request executes independently."""
    gateway_url = platform["gateway_url"]

    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={"Content-Type": "application/json"},
            # No Idempotency-Key header
        )
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={"Content-Type": "application/json"},
            # No Idempotency-Key header
        )

    result1 = resp1.json()
    result2 = resp2.json()

    # Different run IDs - both executed
    assert result1["runId"] != result2["runId"]


# =============================================================================
# IDEMPOTENCY WITH FAILURES
# =============================================================================


@pytest.mark.integration
def test_idempotency_with_failed_execution(platform, worker_process):
    """Failed executions allow retry with same idempotency key.

    Unlike successful requests which return cached responses,
    failed requests are allowed to be retried. This enables
    recovery from transient failures without requiring new keys.
    """
    idempotency_key = f"test-idemp-fail-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    with httpx.Client(timeout=30) as http_client:
        # First request - should fail
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/failing_function",
            json={"should_fail": True, "error_type": "ValueError"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result1 = resp1.json()

    assert result1["status"] == "failed"
    run_id_1 = result1["runId"]

    with httpx.Client(timeout=30) as http_client:
        # Second request with SAME key - should execute again (retry allowed)
        # Because the first request failed, the platform allows retry
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/failing_function",
            json={"should_fail": True, "error_type": "ValueError"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    # Retry should execute with NEW run ID (not cached)
    assert result2["runId"] != run_id_1  # Different run ID - this is a retry
    assert result2["status"] == "failed"  # Still fails (same input)


# =============================================================================
# CONCURRENT IDEMPOTENCY
# =============================================================================


@pytest.mark.integration
def test_concurrent_requests_with_same_key(platform, worker_process):
    """Concurrent requests with same key should deduplicate.

    The platform uses database-level unique constraints on idempotency keys
    to ensure only one request wins. Other concurrent requests receive:
    - 409 Conflict if the first request is still in progress
    - Cached response if the first request completed
    """
    idempotency_key = f"test-idemp-concurrent-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    def make_request():
        with httpx.Client(timeout=30) as http_client:
            resp = http_client.post(
                f"{gateway_url}/v1/run/function/greet",
                json={"name": "Concurrent"},
                headers={
                    "Content-Type": "application/json",
                    "Idempotency-Key": idempotency_key,
                },
            )
            return {"status_code": resp.status_code, "body": resp.json()}

    # Fire 5 concurrent requests with same idempotency key
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Categorize responses
    successful = [r for r in results if r["status_code"] == 200]
    conflicts = [r for r in results if r["status_code"] == 409]

    # At least one request should succeed
    assert len(successful) >= 1, f"Expected at least 1 successful request, got {len(successful)}"

    # All successful requests should have the same run ID (only one execution)
    run_ids = {r["body"]["runId"] for r in successful}
    assert len(run_ids) == 1, f"Expected 1 unique run ID, got {len(run_ids)}: {run_ids}"

    # Total should be 5 (either succeeded or got conflict)
    assert len(successful) + len(conflicts) == 5, (
        f"Expected 5 total responses, got {len(successful)} successful + {len(conflicts)} conflicts"
    )


# =============================================================================
# IDEMPOTENCY EDGE CASES (P1.2)
# =============================================================================


@pytest.mark.integration
def test_idempotency_same_key_different_payload(platform, worker_process):
    """Same idempotency key with different payload should return cached result.

    When the same idempotency key is used with a different payload,
    the platform should return the cached result from the first request,
    NOT execute a new request with the different payload.

    This is the expected behavior for idempotency - the key locks the
    first request's result regardless of subsequent payloads.
    """
    idempotency_key = f"test-idemp-diff-payload-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    # First request with name="Alice"
    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Alice"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result1 = resp1.json()

    assert resp1.status_code == 200
    assert result1["status"] == "completed"
    assert "Alice" in result1.get("output", "")
    run_id_1 = result1["runId"]

    # Second request with SAME key but name="Bob"
    with httpx.Client(timeout=30) as http_client:
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Bob"},  # Different payload!
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    # Should return the cached result (Alice, not Bob)
    assert resp2.status_code == 200
    assert result2["runId"] == run_id_1, (
        "Expected same run ID for idempotent request with different payload"
    )
    # The output should still be "Alice" (cached), not "Bob"
    if "output" in result2:
        assert "Alice" in result2["output"], (
            "Expected cached result with 'Alice', not new execution with 'Bob'"
        )


@pytest.mark.integration
def test_idempotency_validates_response_output(platform, worker_process):
    """Validate full response structure for idempotent requests.

    This test ensures that idempotent responses include all expected fields,
    not just the runId.
    """
    idempotency_key = f"test-idemp-full-response-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    # First request
    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/add",
            json={"a": 10, "b": 20},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result1 = resp1.json()

    assert resp1.status_code == 200
    assert result1["status"] == "completed"
    assert result1["output"] == 30  # 10 + 20

    # Second request (should return cached)
    with httpx.Client(timeout=30) as http_client:
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/add",
            json={"a": 10, "b": 20},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    # Verify full response structure, not just runId
    assert resp2.status_code == 200
    assert result2["runId"] == result1["runId"], "Run IDs should match"
    assert "status" in result2, "Idempotent response should include status"
    assert result2["status"] == "completed", "Cached response should show completed status"

    # Output should also be preserved in cached response
    if "output" in result1:
        assert "output" in result2, "Idempotent response should include output"
        assert result2["output"] == result1["output"], (
            f"Cached output should match original. "
            f"Expected {result1['output']}, got {result2.get('output')}"
        )


@pytest.mark.integration
@pytest.mark.skip(reason="Idempotency key TTL not yet implemented in platform")
def test_idempotency_key_expiry(platform, worker_process):
    """Test that idempotency keys expire after TTL.

    This test verifies that after the TTL expires, the same idempotency key
    will execute a new request instead of returning cached results.

    Note: This test requires the platform to implement idempotency key TTL,
    which is not yet available. Skipped until implemented.
    """
    import time

    idempotency_key = f"test-idemp-expiry-{uuid.uuid4()}"
    gateway_url = platform["gateway_url"]

    # First request
    with httpx.Client(timeout=30) as http_client:
        resp1 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "First"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result1 = resp1.json()

    run_id_1 = result1["runId"]

    # Wait for TTL to expire (assuming 60 second TTL for example)
    # Note: Adjust sleep time based on actual TTL configuration
    time.sleep(65)

    # Second request after TTL expiry
    with httpx.Client(timeout=30) as http_client:
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/greet",
            json={"name": "Second"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    # After TTL expiry, should get a NEW run ID
    assert result2["runId"] != run_id_1, (
        "After TTL expiry, same idempotency key should execute new request"
    )
    assert "Second" in result2.get("output", ""), (
        "New execution should use the new payload"
    )
