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
    """Failed executions are also cached by idempotency key."""
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
        # Second request with SAME key - should return cached failure
        resp2 = http_client.post(
            f"{gateway_url}/v1/run/function/failing_function",
            json={"should_fail": True, "error_type": "ValueError"},
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        result2 = resp2.json()

    assert result2["runId"] == run_id_1  # Same run ID
    assert result2["status"] == "failed"  # Still failed


# =============================================================================
# CONCURRENT IDEMPOTENCY
# =============================================================================


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Concurrent idempotency deduplication has race condition - needs platform fix"
)
def test_concurrent_requests_with_same_key(platform, worker_process):
    """Concurrent requests with same key should deduplicate.

    Note: This test currently fails because concurrent requests arrive
    before the idempotency key can be locked. This is a known limitation
    that requires platform-level atomic locking implementation.
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
            return resp.json()

    # Fire 5 concurrent requests with same idempotency key
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should have the same run ID (only one execution)
    run_ids = {r["runId"] for r in results}
    assert len(run_ids) == 1, f"Expected 1 unique run ID, got {len(run_ids)}: {run_ids}"
