#!/usr/bin/env python3
"""
MCP Verification Helper

This module provides utilities for verifying AGNT5 observability via MCP tools:
- Logs (list_run_logs)
- Traces (list_run_traces, get_trace_by_id)
- Journal Events (list_run_events)
- Entities (get_entity)

Usage:
    from verify_mcp import MCPVerifier

    verifier = MCPVerifier()
    verifier.verify_all(run_id, expected={
        "logs_min_count": 5,
        "traces_min_count": 2,
        "events_min_count": 2,
        "entity_exists": True,
    })
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str


class MCPVerifier:
    """
    MCP Verification Helper for testing AGNT5 observability.

    This class provides methods to verify that logs, traces, events, and entities
    are correctly created and stored via MCP tools.
    """

    def __init__(self):
        """Initialize MCP verifier."""
        self.results: List[VerificationResult] = []

    def verify_logs(
        self, run_id: str, min_count: int = 1, expected_patterns: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify that logs exist for a run.

        Args:
            run_id: The run ID to check logs for
            min_count: Minimum number of log entries expected
            expected_patterns: List of patterns that should appear in logs

        Returns:
            VerificationResult with pass/fail status
        """
        try:
            # NOTE: This would use the actual MCP client
            # For now, this is a template showing the verification pattern
            # In real usage, you would call: mcp_client.list_run_logs(run_id)

            # Placeholder for actual MCP call
            # logs = mcp_client.list_run_logs(run_id)
            logs = []  # Replace with actual MCP call

            # Check minimum count
            if len(logs) < min_count:
                result = VerificationResult(
                    check_name="logs_count",
                    passed=False,
                    expected=f">= {min_count}",
                    actual=len(logs),
                    message=f"Expected at least {min_count} logs, found {len(logs)}",
                )
                self.results.append(result)
                return result

            # Check for expected patterns
            if expected_patterns:
                log_bodies = [log.get("body", "") for log in logs]
                for pattern in expected_patterns:
                    found = any(pattern in body for body in log_bodies)
                    if not found:
                        result = VerificationResult(
                            check_name=f"logs_pattern_{pattern}",
                            passed=False,
                            expected=f"Pattern '{pattern}' in logs",
                            actual="Pattern not found",
                            message=f"Expected pattern '{pattern}' not found in logs",
                        )
                        self.results.append(result)
                        return result

            result = VerificationResult(
                check_name="logs_verification",
                passed=True,
                expected=f">= {min_count} logs",
                actual=f"{len(logs)} logs",
                message=f"✓ Found {len(logs)} logs for run {run_id}",
            )
            self.results.append(result)
            return result

        except Exception as e:
            result = VerificationResult(
                check_name="logs_verification",
                passed=False,
                expected="Logs retrieval",
                actual=f"Error: {e}",
                message=f"Failed to verify logs: {e}",
            )
            self.results.append(result)
            return result

    def verify_traces(
        self, run_id: str, min_span_count: int = 1, expected_span_names: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify that traces exist for a run.

        Args:
            run_id: The run ID to check traces for
            min_span_count: Minimum number of spans expected
            expected_span_names: List of span names that should exist

        Returns:
            VerificationResult with pass/fail status
        """
        try:
            # NOTE: This would use the actual MCP client
            # traces = mcp_client.list_run_traces(run_id)
            traces = []  # Replace with actual MCP call

            if len(traces) == 0:
                result = VerificationResult(
                    check_name="traces_exist",
                    passed=False,
                    expected="At least 1 trace",
                    actual="0 traces",
                    message=f"No traces found for run {run_id}",
                )
                self.results.append(result)
                return result

            # Get all spans from traces
            all_spans = []
            for trace in traces:
                # Each trace should have a trace_id, we'd fetch full details
                # full_trace = mcp_client.get_trace_by_id(trace['trace_id'])
                # all_spans.extend(full_trace.get('spans', []))
                pass

            # Check minimum span count
            if len(all_spans) < min_span_count:
                result = VerificationResult(
                    check_name="traces_span_count",
                    passed=False,
                    expected=f">= {min_span_count} spans",
                    actual=f"{len(all_spans)} spans",
                    message=f"Expected at least {min_span_count} spans, found {len(all_spans)}",
                )
                self.results.append(result)
                return result

            # Check for expected span names
            if expected_span_names:
                span_names = [span.get("name", "") for span in all_spans]
                for expected_name in expected_span_names:
                    if expected_name not in span_names:
                        result = VerificationResult(
                            check_name=f"traces_span_{expected_name}",
                            passed=False,
                            expected=f"Span '{expected_name}'",
                            actual="Span not found",
                            message=f"Expected span '{expected_name}' not found in traces",
                        )
                        self.results.append(result)
                        return result

            result = VerificationResult(
                check_name="traces_verification",
                passed=True,
                expected=f">= {min_span_count} spans",
                actual=f"{len(all_spans)} spans",
                message=f"✓ Found {len(all_spans)} spans for run {run_id}",
            )
            self.results.append(result)
            return result

        except Exception as e:
            result = VerificationResult(
                check_name="traces_verification",
                passed=False,
                expected="Traces retrieval",
                actual=f"Error: {e}",
                message=f"Failed to verify traces: {e}",
            )
            self.results.append(result)
            return result

    def verify_events(
        self, run_id: str, min_count: int = 2, expected_event_types: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify that journal events exist for a run.

        Args:
            run_id: The run ID to check events for
            min_count: Minimum number of events expected
            expected_event_types: List of event types that should exist

        Returns:
            VerificationResult with pass/fail status
        """
        try:
            # NOTE: This would use the actual MCP client
            # events = mcp_client.list_run_events(run_id)
            events = []  # Replace with actual MCP call

            # Check minimum count
            if len(events) < min_count:
                result = VerificationResult(
                    check_name="events_count",
                    passed=False,
                    expected=f">= {min_count} events",
                    actual=f"{len(events)} events",
                    message=f"Expected at least {min_count} events, found {len(events)}",
                )
                self.results.append(result)
                return result

            # Check for expected event types
            if expected_event_types:
                event_types = [event.get("event_type", "") for event in events]
                for expected_type in expected_event_types:
                    if expected_type not in event_types:
                        result = VerificationResult(
                            check_name=f"events_type_{expected_type}",
                            passed=False,
                            expected=f"Event type '{expected_type}'",
                            actual="Event type not found",
                            message=f"Expected event type '{expected_type}' not found",
                        )
                        self.results.append(result)
                        return result

            result = VerificationResult(
                check_name="events_verification",
                passed=True,
                expected=f">= {min_count} events",
                actual=f"{len(events)} events",
                message=f"✓ Found {len(events)} events for run {run_id}",
            )
            self.results.append(result)
            return result

        except Exception as e:
            result = VerificationResult(
                check_name="events_verification",
                passed=False,
                expected="Events retrieval",
                actual=f"Error: {e}",
                message=f"Failed to verify events: {e}",
            )
            self.results.append(result)
            return result

    def verify_entity(self, entity_key: str, should_exist: bool = True) -> VerificationResult:
        """
        Verify that an entity exists (or doesn't exist).

        Args:
            entity_key: The entity key to check (e.g., "workflow:run-123")
            should_exist: Whether the entity should exist

        Returns:
            VerificationResult with pass/fail status
        """
        try:
            # NOTE: This would use the actual MCP client
            # entity = mcp_client.get_entity(entity_key)
            entity = None  # Replace with actual MCP call

            exists = entity is not None

            if exists != should_exist:
                result = VerificationResult(
                    check_name="entity_existence",
                    passed=False,
                    expected=f"Entity {'exists' if should_exist else 'does not exist'}",
                    actual=f"Entity {'exists' if exists else 'does not exist'}",
                    message=f"Entity {entity_key}: expected {'to exist' if should_exist else 'not to exist'}",
                )
                self.results.append(result)
                return result

            result = VerificationResult(
                check_name="entity_verification",
                passed=True,
                expected=f"Entity {'exists' if should_exist else 'does not exist'}",
                actual=f"Entity {'exists' if exists else 'does not exist'}",
                message=f"✓ Entity {entity_key} verification passed",
            )
            self.results.append(result)
            return result

        except Exception as e:
            result = VerificationResult(
                check_name="entity_verification",
                passed=False,
                expected="Entity retrieval",
                actual=f"Error: {e}",
                message=f"Failed to verify entity: {e}",
            )
            self.results.append(result)
            return result

    def verify_all(
        self,
        run_id: str,
        expected: Dict[str, Any],
    ) -> bool:
        """
        Verify all observability aspects for a run.

        Args:
            run_id: The run ID to verify
            expected: Dictionary with expected values:
                - logs_min_count: Minimum log entries
                - logs_patterns: List of patterns in logs
                - traces_min_spans: Minimum span count
                - traces_span_names: List of expected span names
                - events_min_count: Minimum event count
                - events_types: List of expected event types
                - entity_exists: Whether entity should exist

        Returns:
            True if all verifications passed, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"MCP Verification for run_id: {run_id}")
        print(f"{'='*60}\n")

        all_passed = True

        # Verify logs
        if "logs_min_count" in expected:
            result = self.verify_logs(
                run_id,
                min_count=expected["logs_min_count"],
                expected_patterns=expected.get("logs_patterns"),
            )
            print(f"  {result.message}")
            all_passed = all_passed and result.passed

        # Verify traces
        if "traces_min_spans" in expected:
            result = self.verify_traces(
                run_id,
                min_span_count=expected["traces_min_spans"],
                expected_span_names=expected.get("traces_span_names"),
            )
            print(f"  {result.message}")
            all_passed = all_passed and result.passed

        # Verify events
        if "events_min_count" in expected:
            result = self.verify_events(
                run_id,
                min_count=expected["events_min_count"],
                expected_event_types=expected.get("events_types"),
            )
            print(f"  {result.message}")
            all_passed = all_passed and result.passed

        # Verify entity
        if "entity_exists" in expected:
            entity_key = f"workflow:{run_id}"
            result = self.verify_entity(entity_key, should_exist=expected["entity_exists"])
            print(f"  {result.message}")
            all_passed = all_passed and result.passed

        print(f"\n{'='*60}")
        if all_passed:
            print("✅ ALL VERIFICATIONS PASSED")
        else:
            print("❌ SOME VERIFICATIONS FAILED")
        print(f"{'='*60}\n")

        return all_passed

    def print_summary(self):
        """Print summary of all verification results."""
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.check_name}: {result.message}")

        print(f"\nTotal: {passed_count}/{total_count} checks passed")
        print("=" * 60 + "\n")

        return passed_count == total_count


def print_run_info(run_id: str, result: Dict[str, Any]):
    """
    Print run information in a formatted way.

    Args:
        run_id: The run ID
        result: The workflow result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    print(f"Scenario: {result.get('scenario', 'unknown')}")
    print(f"Output: {result.get('output', '')[:100]}...")
    print(f"Tool calls: {result.get('tool_calls_count', 0)}")

    if "tools_used" in result:
        print(f"Tools used: {result['tools_used']}")

    if "agents_in_workflow" in result:
        print(f"Agents: {result['agents_in_workflow']}")

    if "agent_delegations" in result:
        print(f"Agent delegations: {result['agent_delegations']}")

    if "hitl_interactions" in result:
        print(f"HITL interactions: {result['hitl_interactions']}")

    print(f"{'='*60}\n")


# Usage example
if __name__ == "__main__":
    print("MCP Verifier - Usage Example")
    print("\nThis is a template. To use with actual MCP:")
    print("1. Import the appropriate MCP client")
    print("2. Pass run_id from workflow execution")
    print("3. Call verify_all() with expected values")
    print("\nExample:")
    print("""
    from agnt5 import Client
    from verify_mcp import MCPVerifier

    client = Client("http://localhost:34181")
    result = client.run("test_agent_basic", {"task": "..."}, component_type="workflow")

    verifier = MCPVerifier()
    passed = verifier.verify_all(result['run_id'], expected={
        "logs_min_count": 5,
        "traces_min_spans": 2,
        "events_min_count": 2,
        "entity_exists": True,
    })
    """)
