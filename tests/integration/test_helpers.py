"""
Test helper utilities for integration tests.

Provides common functions for:
- Journal event verification via SDK client
- Event assertions
"""

import time
from typing import List, Tuple

from agnt5 import Client

# ============================================================================
# SDK Client-Based Helper Functions
# ============================================================================


def fetch_run_events(client: Client, run_id: str, max_retries: int = 3) -> List[str]:
    """
    Fetch journal events for a run using SDK client.

    Retries with backoff to allow events to be persisted. The gateway returns
    events in journal-offset order, which is authoritative across processes;
    worker wall-clock timestamps may be skewed relative to the gateway.

    Args:
        client: AGNT5 SDK client
        run_id: Run ID to fetch events for
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        List of event type strings in chronological order

    Raises:
        Exception: If API request fails after all retries
    """
    for attempt in range(max_retries):
        events_response = client.get_events(run_id)

        if len(events_response) > 0:
            return [event.event_type for event in events_response.items]

        # If empty and this isn't the last attempt, retry
        if attempt < max_retries - 1:
            time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    return []


def verify_journal_events(
    client: Client,
    run_id: str,
    expected_events: List[str],
    match_order: bool = True,
    allow_extra: bool = False,
) -> Tuple[str, List[str]]:
    """
    Verify that journal events were recorded for a run.

    Uses SDK client's get_events() method.

    Args:
        client: AGNT5 SDK client
        run_id: Run ID to verify events for
        expected_events: List of expected event types in order
        match_order: If True, verify events appear in order (default: True)
        allow_extra: If True, allow extra events in actual (default: False)

    Returns:
        Tuple of (run_id, event_types)

    Raises:
        AssertionError: If events are missing or incorrect

    Example:
        response = client.run("add", {"a": 5, "b": 3})
        verify_journal_events(client, response.run_id, [
            "run.queued",
            "run.started",
            "function.started",
            "function.completed",
            "run.completed",
        ])
    """
    # Fetch events via SDK client
    event_types = fetch_run_events(client, run_id)
    assert len(event_types) > 0, f"No events found for run_id {run_id}"

    # Assert against expected_events
    assert_events_match(event_types, expected_events, match_order, allow_extra)

    return run_id, event_types


def assert_events_match(
    actual_events: List[str],
    expected_events: List[str],
    match_order: bool = False,
    allow_extra: bool = True,
):
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
                if (
                    expected_idx < len(expected_events)
                    and actual_event == expected_events[expected_idx]
                ):
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
                    f"Event order mismatch:\nExpected: {expected_events}\nActual:   {actual_events}"
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
