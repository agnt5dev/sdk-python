#!/usr/bin/env python3
"""
Comprehensive Testing Script for AGNT5

This script tests all 6 comprehensive test scenarios and verifies observability
via MCP tools (logs, traces, events, entities).

Usage:
    python test_comprehensive.py                    # Run all tests
    python test_comprehensive.py --scenario 1       # Run specific scenario
    python test_comprehensive.py --verify-mcp       # Include MCP verification

Test Scenarios:
    1. test_agent_basic - Workflow + Agent
    2. test_agent_with_simple_tool - Workflow + Agent + Simple Tool
    3. test_agent_with_multiple_tools - Workflow + Agent + Multiple Tools
    4. test_agent_with_agent_as_tool - Workflow + Agent + Tool + Agent as Tool
    5. test_agent_with_hitl - Workflow + Agent + HITL
    6. test_comprehensive_scenario - All features combined
"""

import sys
import argparse
import json
import os
from typing import Dict, List, Any
from agnt5 import Client
from verify_mcp import MCPVerifier, print_run_info

GATEWAY_URL = "http://localhost:34181"
DEFAULT_TENANT_ID = "00000000-0000-0000-0000-00001f718175"
DEFAULT_DEPLOYMENT_ID = "00000000-0000-0000-0000-00001f718175"


# ============================================================================
# TEST SCENARIO DEFINITIONS
# ============================================================================


SCENARIOS = {
    "1": {
        "name": "test_agent_basic",
        "description": "Workflow + Agent (Basic)",
        "task": "Explain what a REST API is in simple terms (2-3 sentences).",
        "expected_mcp": {
            "logs_min_count": 3,
            "logs_patterns": ["BasicAgent", "Executing agent"],
            "traces_min_spans": 2,  # workflow + agent.run
            "traces_span_names": ["workflow.execute", "agent.run"],
            "events_min_count": 2,  # run.started, run.completed
            "events_types": ["run.started", "run.completed"],
            "entity_exists": True,
        },
    },
    "2": {
        "name": "test_agent_with_simple_tool",
        "description": "Workflow + Agent + Simple Tool",
        "task": "Calculate the result of 25 * 4 + 10",
        "expected_mcp": {
            "logs_min_count": 4,
            "logs_patterns": ["CalculatorAgent", "simple_calculator", "evaluating"],
            "traces_min_spans": 3,  # workflow + agent.run + tool.invoke
            "events_min_count": 2,
            "entity_exists": True,
        },
    },
    "3": {
        "name": "test_agent_with_multiple_tools",
        "description": "Workflow + Agent + Multiple Tools",
        "task": "Search for information about Python programming, calculate 15 * 8, and tell me the weather in San Francisco",
        "expected_mcp": {
            "logs_min_count": 6,
            "logs_patterns": ["MultiToolAgent", "search_web", "simple_calculator", "get_weather"],
            "traces_min_spans": 5,  # workflow + agent.run + 3 tools
            "events_min_count": 2,
            "entity_exists": True,
        },
    },
    "4": {
        "name": "test_agent_with_agent_as_tool",
        "description": "Workflow + Agent + Tool + Agent as Tool",
        "task": "Search for information about AI trends and then perform a detailed domain-specific analysis of the findings",
        "expected_mcp": {
            "logs_min_count": 6,
            "logs_patterns": ["Coordinator", "DomainSpecialist", "search_web", "domain_specific_analysis"],
            "traces_min_spans": 5,  # workflow + coordinator + specialist + tools
            "events_min_count": 2,
            "entity_exists": True,
        },
    },
    "5": {
        "name": "test_agent_with_hitl",
        "description": "Workflow + Agent + HITL",
        "task": "Help me plan a project. Ask me what project I'm working on, then suggest next steps.",
        "expected_mcp": {
            "logs_min_count": 4,
            "logs_patterns": ["HITLAgent"],
            "traces_min_spans": 3,
            "events_min_count": 2,
            "entity_exists": True,
        },
    },
    "6": {
        "name": "test_comprehensive_scenario",
        "description": "Complete Integration Test (All Features)",
        "task": "Research current weather in New York, calculate travel time if driving 60 mph for 3.5 hours, delegate detailed analysis to a specialist, and request approval for the final plan.",
        "expected_mcp": {
            "logs_min_count": 8,
            "logs_patterns": ["ComprehensiveCoordinator", "ResearchSpecialist"],
            "traces_min_spans": 7,
            "events_min_count": 2,
            "entity_exists": True,
        },
    },
}


# ============================================================================
# TEST EXECUTION
# ============================================================================


def run_scenario(scenario_id: str, verify_mcp: bool = False) -> Dict[str, Any]:
    """
    Run a single test scenario.

    Args:
        scenario_id: The scenario ID (1-6)
        verify_mcp: Whether to verify via MCP tools

    Returns:
        Test results dictionary
    """
    scenario = SCENARIOS[scenario_id]

    print("\n" + "=" * 70)
    print(f"TEST SCENARIO {scenario_id}: {scenario['description']}")
    print("=" * 70)
    print(f"Workflow: {scenario['name']}")
    print(f"Task: {scenario['task']}")
    print("=" * 70 + "\n")

    # Get tenant/deployment IDs from environment or use defaults
    tenant_id = os.getenv("AGNT5_TENANT_ID", DEFAULT_TENANT_ID)
    deployment_id = os.getenv("AGNT5_DEPLOYMENT_ID", DEFAULT_DEPLOYMENT_ID)

    # Create client with custom headers
    # Timeout matches gateway's ExecutionTimeoutSecs (300s) + buffer for network overhead
    import httpx
    client_http = httpx.Client(timeout=330.0)

    # Build request
    url = f"{GATEWAY_URL}/v1/run/workflow/{scenario['name']}"
    headers = {
        "Content-Type": "application/json",
        "X-TENANT-ID": tenant_id,
        "X-DEPLOYMENT-ID": deployment_id,
    }
    payload = {"task": scenario["task"]}

    # Execute workflow
    print("🚀 Executing workflow...\n")
    try:
        response = client_http.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Print result info
        print_run_info(result["run_id"], result)

        # Verify via MCP if requested
        if verify_mcp:
            print("\n📊 Verifying observability via MCP...\n")
            verifier = MCPVerifier()
            mcp_passed = verifier.verify_all(
                result["run_id"],
                scenario["expected_mcp"],
            )

            return {
                "scenario_id": scenario_id,
                "scenario_name": scenario["name"],
                "passed": True,
                "mcp_verified": mcp_passed,
                "result": result,
            }
        else:
            return {
                "scenario_id": scenario_id,
                "scenario_name": scenario["name"],
                "passed": True,
                "mcp_verified": None,
                "result": result,
            }

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()

        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario["name"],
            "passed": False,
            "mcp_verified": False,
            "error": str(e),
        }


def run_all_scenarios(verify_mcp: bool = False) -> List[Dict[str, Any]]:
    """
    Run all test scenarios.

    Args:
        verify_mcp: Whether to verify via MCP tools

    Returns:
        List of test results
    """
    results = []

    for scenario_id in SCENARIOS.keys():
        result = run_scenario(scenario_id, verify_mcp)
        results.append(result)

    return results


def print_summary(results: List[Dict[str, Any]]):
    """
    Print summary of all test results.

    Args:
        results: List of test results
    """
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70 + "\n")

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    for result in results:
        status = "✅" if result["passed"] else "❌"
        scenario_id = result["scenario_id"]
        scenario_name = result["scenario_name"]

        mcp_status = ""
        if result.get("mcp_verified") is not None:
            mcp_status = " (MCP: ✅)" if result["mcp_verified"] else " (MCP: ❌)"

        print(f"{status} Scenario {scenario_id}: {scenario_name}{mcp_status}")

    print(f"\n{'=' * 70}")
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")

    print("=" * 70 + "\n")

    return passed_count == total_count


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive testing for AGNT5 workflows and agents"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run specific scenario (1-6)",
        choices=list(SCENARIOS.keys()),
    )
    parser.add_argument(
        "--verify-mcp",
        action="store_true",
        help="Verify observability via MCP tools",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios",
    )

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        print("\nAvailable Test Scenarios:")
        print("=" * 70)
        for scenario_id, scenario in SCENARIOS.items():
            print(f"\n{scenario_id}. {scenario['description']}")
            print(f"   Workflow: {scenario['name']}")
            print(f"   Task: {scenario['task']}")
        print("\n" + "=" * 70 + "\n")
        return 0

    # Run specific scenario
    if args.scenario:
        result = run_scenario(args.scenario, args.verify_mcp)
        success = result["passed"]
        if args.verify_mcp:
            success = success and result.get("mcp_verified", False)
        return 0 if success else 1

    # Run all scenarios
    print("\n" + "=" * 70)
    print("COMPREHENSIVE AGNT5 TESTING")
    print("=" * 70)
    print("Running all 6 test scenarios...")
    print("=" * 70 + "\n")

    results = run_all_scenarios(args.verify_mcp)
    all_passed = print_summary(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
