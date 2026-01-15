"""
Test helper utilities for integration tests.

Provides common functions for:
- Journal event verification via API
- Event assertions
"""

import time
from typing import Dict, List, Optional, Tuple

import requests


# ============================================================================
# API-Based Helper Functions (Recommended)
# ============================================================================


def get_latest_run_id_via_api(gateway_url: str) -> Optional[str]:
    """
    Get the most recent run ID via platform API.

    Args:
        gateway_url: Platform gateway URL (from platform fixture)

    Returns:
        Run ID of most recent run, or None if no runs found

    Raises:
        requests.HTTPError: If API request fails
    """
    response = requests.get(f"{gateway_url}/v1/runs", params={"limit": 1}, timeout=5)
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    return items[0]["id"] if items else None


def fetch_run_events_via_api(gateway_url: str, run_id: str, max_retries: int = 3) -> List[Dict]:
    """
    Fetch journal events for a run via platform API.

    Retries with backoff to allow events to be persisted.

    Args:
        gateway_url: Platform gateway URL (from platform fixture)
        run_id: Run ID to fetch events for
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        List of event dictionaries with fields:
        - id, sequence_num, event_type, run_id, step_key,
          input_data, output_data, metadata, source_timestamp_ns, created_at

    Raises:
        requests.HTTPError: If API request fails after all retries
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Try the main endpoint first
            response = requests.get(f"{gateway_url}/v1/runs/{run_id}/events", timeout=5)
            response.raise_for_status()
            data = response.json()

            # Handle both response formats
            if isinstance(data, dict):
                # Paginated response: {"count": N, "items": [...]}
                events = data.get("items", [])
            elif isinstance(data, list):
                # Direct list response
                events = data
            else:
                events = []

            # If we got events, return them
            if events:
                return events

            # If empty and this isn't the last attempt, retry
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff: 0.1s, 0.2s, 0.3s
                continue

            return events  # Return empty list on last attempt

        except requests.exceptions.HTTPError as e:
            last_error = e
            # If 500/501/503 error, try observability endpoint as fallback
            if e.response.status_code in (500, 501, 503):
                try:
                    response = requests.get(f"{gateway_url}/v1/_obs/runs/{run_id}/events", timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    # Observability endpoint might return {"items": [...]} format
                    if isinstance(data, dict) and "items" in data:
                        return data["items"]
                    return data if isinstance(data, list) else []
                except requests.exceptions.HTTPError:
                    pass  # Continue to retry

            # If this isn't the last attempt, wait and retry
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise

    # If we get here, raise the last error
    if last_error:
        raise last_error
    return []




def verify_journal_events(
    platform: Dict,
    expected_events: List[str],
    match_order: bool = True,
    allow_extra: bool = False,
) -> Tuple[str, List[str]]:
    """
    Verify that journal events were recorded via platform API.

    Uses /v1/runs and /v1/runs/{runId}/events endpoints. Works for both embedded and hosted modes.

    Args:
        platform: Platform fixture containing gateway_url
        expected_events: List of expected event types in order
        match_order: If True, verify events appear in order (default: True)
        allow_extra: If True, allow extra events in actual (default: False)

    Returns:
        Tuple of (run_id, event_types)

    Raises:
        AssertionError: If events are missing or incorrect
        requests.HTTPError: If API request fails

    Example:
        # Test successful function execution
        verify_journal_events(platform, [
            "run.enqueued",
            "run.started",
            "function.started",
            "function.completed",
            "run.completed",
        ])

        # Test function that fails (allow extra events)
        verify_journal_events(platform, [
            "run.started",
            "function.failed",
            "run.failed",
        ], allow_extra=True)
    """
    gateway_url = platform["gateway_url"]

    # 1. Get latest run ID from API
    run_id = get_latest_run_id_via_api(gateway_url)
    assert run_id is not None, "No runs found via API"

    # 2. Fetch events via API
    events = fetch_run_events_via_api(gateway_url, run_id)
    assert len(events) > 0, f"No events found for run_id {run_id}"

    # 3. Extract event types
    event_types = [event["event_type"] for event in events]

    # 4. Assert against expected_events
    assert_events_match(event_types, expected_events, match_order, allow_extra)

    # 5. Return run_id and event_types
    return run_id, event_types


def assert_events_match(actual_events: List[str], expected_events: List[str], match_order: bool = False, allow_extra: bool = True):
    """
    Assert that actual events match expected events.

    Args:
        actual_events: Actual event types from platform API
        expected_events: Expected event types
        match_order: If True, verify events appear in order (default: False)
        allow_extra: If True, allow extra events in actual (default: True)
                    Only applies when match_order=True

    Raises:
        AssertionError if events don't match

    Example:
        # Just check presence (order doesn't matter)
        assert_events_match(actual, expected)

        # Check events appear in order (allows extra events)
        assert_events_match(actual, expected, match_order=True)

        # Check exact match (no extra events allowed)
        assert_events_match(actual, expected, match_order=True, allow_extra=False)
    """
    if match_order:
        if allow_extra:
            # Verify expected events appear in order (subsequence check)
            expected_idx = 0
            for actual_event in actual_events:
                if expected_idx < len(expected_events) and actual_event == expected_events[expected_idx]:
                    expected_idx += 1

            if expected_idx != len(expected_events):
                raise AssertionError(
                    f"Expected events not found in order:\n"
                    f"Expected: {expected_events}\n"
                    f"Actual:   {actual_events}\n"
                    f"Found only: {expected_events[:expected_idx]}"
                )
        else:
            # Verify exact order and no extra events
            if actual_events != expected_events:
                raise AssertionError(
                    f"Event order mismatch:\n"
                    f"Expected: {expected_events}\n"
                    f"Actual:   {actual_events}"
                )
    else:
        # Just verify all expected events are present
        for expected in expected_events:
            if expected not in actual_events:
                raise AssertionError(
                    f"Missing event: {expected}\n"
                    f"Expected: {expected_events}\n"
                    f"Actual:   {actual_events}"
                )


def print_journal_events(run_id: str, event_types: List[str]):
    """
    Pretty-print journal events for debugging.

    Args:
        run_id: Run ID
        event_types: List of event type strings
    """
    print(f"\n✅ Journal events verified for run {run_id}:")
    for event_type in event_types:
        print(f"   • {event_type}")


