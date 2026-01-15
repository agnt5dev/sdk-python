"""
Integration Tests for HITL (Human-in-the-Loop) Workflows

Tests Human-in-the-Loop workflows through the platform:
- hitl_conditional_approval - Conditional approval based on threshold
- hitl_text_input - Text input collection
- hitl_approval - Binary approval decisions
- hitl_choice - Multiple choice selection
- hitl_onboarding - Multi-step interaction

Key test patterns:
1. Auto-approve path: Amount under threshold completes immediately
2. Pause verification: Amount at/above threshold pauses with "awaiting_user_input" status
3. Resume flow: Paused workflow can be resumed with user input

Run with:
    pytest tests/integration/test_ex_600_hitl.py -v
"""

import time
from typing import Any, Dict, Optional, Set

import httpx
import pytest


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def wait_for_status(
    client,
    run_id: str,
    expected_statuses: Set[str],
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Poll until workflow reaches expected status or timeout.

    Use this instead of fixed time.sleep() for HITL tests.

    Args:
        client: AGNT5 client instance
        run_id: Workflow run ID
        expected_statuses: Set of status values to wait for
        timeout: Maximum wait time in seconds
        poll_interval: Time between polls in seconds

    Returns:
        Final status dict, or None if timeout
    """
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get_status(run_id)
        if status.get("status") in expected_statuses:
            return status
        time.sleep(poll_interval)
    return status


def resume_workflow(gateway_url: str, run_id: str, user_response: str) -> Dict[str, Any]:
    """Resume a paused workflow with user input.

    The SDK client doesn't have a resume() method yet, so we call the
    gateway endpoint directly: POST /workflows/resume/{runId}

    Args:
        gateway_url: Gateway base URL
        run_id: The run ID to resume
        user_response: The user's response (approval, text input, etc.)

    Returns:
        Response dict from gateway
    """
    with httpx.Client(timeout=30) as http_client:
        response = http_client.post(
            f"{gateway_url}/v1/workflows/resume/{run_id}",
            json={"user_response": user_response},
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# AUTO-APPROVE PATH TESTS
# These workflows complete immediately (amount under threshold)
# =============================================================================


@pytest.mark.integration
def test_conditional_approval_auto_approve(client, worker_process, platform):
    """Test conditional approval auto-approves small amounts."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": 500.0,
            "description": "Office supplies",
        },
        component_type="workflow",
    )

    assert result["status"] == "completed", f"Expected completed, got: {result}"
    assert result["approval_status"] == "auto_approved"
    assert result["required_approval"] is False
    assert result["amount"] == 500.0


@pytest.mark.integration
def test_conditional_approval_under_threshold(client, worker_process):
    """Test amounts just under threshold are auto-approved."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": 999.99,
            "description": "Software subscription",
        },
        component_type="workflow",
    )

    assert result["status"] == "completed", f"Expected completed, got: {result}"
    assert result["approval_status"] == "auto_approved"


@pytest.mark.integration
def test_conditional_approval_zero_amount(client, worker_process):
    """Test zero amount is auto-approved."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": 0.0,
            "description": "Free trial",
        },
        component_type="workflow",
    )

    assert result["status"] == "completed", f"Expected completed, got: {result}"
    assert result["approval_status"] == "auto_approved"


@pytest.mark.integration
def test_conditional_approval_negative_amount(client, worker_process):
    """Test conditional approval handles negative amount (refund)."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": -100.0,
            "description": "Refund",
        },
        component_type="workflow",
    )

    # Negative amounts are below threshold, so auto-approved
    assert result["status"] == "completed", f"Expected completed, got: {result}"
    assert result["approval_status"] == "auto_approved"


# =============================================================================
# PAUSE VERIFICATION TESTS
# These workflows should pause with status "awaiting_user_input"
# =============================================================================


@pytest.mark.integration
def test_conditional_approval_at_threshold_pauses(client, worker_process, platform):
    """Test that amount at threshold pauses workflow with awaiting_user_input status."""
    # Submit workflow (don't use run() which blocks until completion)
    run_id = client.submit(
        "hitl_conditional_approval",
        {
            "amount": 1000.0,
            "description": "Equipment purchase",
        },
        component_type="workflow",
    )

    # Poll for paused state
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    # Assert workflow paused correctly
    assert status is not None, "Workflow should reach paused state within timeout"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


@pytest.mark.integration
def test_conditional_approval_large_amount_pauses(client, worker_process):
    """Test that large amount pauses workflow for approval."""
    run_id = client.submit(
        "hitl_conditional_approval",
        {
            "amount": 1000000.0,
            "description": "Major capital expenditure",
        },
        component_type="workflow",
    )

    # Poll for paused state
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    assert status is not None, "Workflow should reach paused state"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


@pytest.mark.integration
def test_hitl_text_input_pauses_for_input(client, worker_process):
    """Test hitl_text_input pauses with awaiting_user_input status."""
    run_id = client.submit(
        "hitl_text_input",
        {"prompt": "Enter your name"},
        component_type="workflow",
    )

    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    assert status is not None, "Workflow should pause for text input"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


@pytest.mark.integration
def test_hitl_approval_pauses_for_decision(client, worker_process):
    """Test hitl_approval pauses with awaiting_user_input status."""
    run_id = client.submit(
        "hitl_approval",
        {"action": "delete", "resource": "production-database"},
        component_type="workflow",
    )

    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    assert status is not None, "Workflow should pause for approval"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


@pytest.mark.integration
def test_hitl_choice_pauses_for_selection(client, worker_process):
    """Test hitl_choice pauses with awaiting_user_input status."""
    run_id = client.submit(
        "hitl_choice",
        {
            "question": "Select deployment environment",
            "options": ["development", "staging", "production"],
        },
        component_type="workflow",
    )

    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    assert status is not None, "Workflow should pause for choice selection"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


@pytest.mark.integration
def test_hitl_onboarding_pauses_for_name(client, worker_process):
    """Test hitl_onboarding pauses at first step for name input."""
    run_id = client.submit(
        "hitl_onboarding",
        {"welcome_message": "Welcome to the platform!"},
        component_type="workflow",
    )

    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )

    assert status is not None, "Workflow should pause for name input"
    assert status["status"] == "awaiting_user_input", (
        f"Expected 'awaiting_user_input', got: {status['status']}"
    )


# =============================================================================
# RESUME FLOW TESTS
# These test the full pause → resume → completion flow
# =============================================================================


@pytest.mark.integration
def test_conditional_approval_resume_with_approval(client, worker_process, platform):
    """Test paused workflow can be resumed with approval and completes."""
    # 1. Submit workflow that requires approval
    run_id = client.submit(
        "hitl_conditional_approval",
        {
            "amount": 2000.0,
            "description": "Server hardware purchase",
        },
        component_type="workflow",
    )

    # 2. Wait for workflow to pause
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for approval"
    assert status["status"] == "awaiting_user_input"

    # 3. Resume with approval
    gateway_url = platform["gateway_url"]
    resume_response = resume_workflow(gateway_url, run_id, user_response="approved")

    # Validate resume response
    assert "runId" in resume_response or "status" in resume_response, (
        f"Resume response should contain runId or status, got: {resume_response}"
    )

    # 4. Wait for completion
    final_status = wait_for_status(
        client,
        run_id,
        expected_statuses={"completed", "failed"},
        timeout=15,
    )

    assert final_status is not None, "Workflow should complete after resume"
    assert final_status["status"] == "completed", (
        f"Expected 'completed' after resume, got: {final_status['status']}"
    )


@pytest.mark.integration
def test_hitl_text_input_resume_with_response(client, worker_process, platform):
    """Test text input workflow can be resumed with user response."""
    # 1. Submit workflow
    run_id = client.submit(
        "hitl_text_input",
        {"prompt": "What is your favorite color?"},
        component_type="workflow",
    )

    # 2. Wait for pause
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for input"
    assert status["status"] == "awaiting_user_input"

    # 3. Resume with text response
    gateway_url = platform["gateway_url"]
    resume_response = resume_workflow(gateway_url, run_id, user_response="Blue")

    # Validate resume response
    assert "runId" in resume_response or "status" in resume_response, (
        f"Resume response should contain runId or status, got: {resume_response}"
    )

    # 4. Wait for completion
    final_status = wait_for_status(
        client,
        run_id,
        expected_statuses={"completed", "failed"},
        timeout=15,
    )

    assert final_status is not None, "Workflow should complete after resume"
    assert final_status["status"] == "completed", (
        f"Expected 'completed' after resume, got: {final_status['status']}"
    )


@pytest.mark.integration
def test_hitl_choice_resume_with_selection(client, worker_process, platform):
    """Test choice workflow can be resumed with selected option."""
    # 1. Submit workflow
    run_id = client.submit(
        "hitl_choice",
        {
            "question": "Select deployment environment",
            "options": ["development", "staging", "production"],
        },
        component_type="workflow",
    )

    # 2. Wait for pause
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for choice selection"
    assert status["status"] == "awaiting_user_input"

    # 3. Resume with selection
    gateway_url = platform["gateway_url"]
    resume_response = resume_workflow(gateway_url, run_id, user_response="production")

    # Validate resume response
    assert "runId" in resume_response or "status" in resume_response, (
        f"Resume response should contain runId or status, got: {resume_response}"
    )

    # 4. Wait for completion
    final_status = wait_for_status(
        client,
        run_id,
        expected_statuses={"completed", "failed"},
        timeout=15,
    )

    assert final_status is not None, "Workflow should complete after resume"
    assert final_status["status"] == "completed", (
        f"Expected 'completed' after resume, got: {final_status['status']}"
    )


@pytest.mark.integration
def test_hitl_onboarding_resume_full_flow(client, worker_process, platform):
    """Test onboarding workflow can complete all 3 steps with resume."""
    # 1. Submit workflow
    run_id = client.submit(
        "hitl_onboarding",
        {"welcome_message": "Welcome to the platform!"},
        component_type="workflow",
    )

    gateway_url = platform["gateway_url"]

    # Step 1: Wait for name input pause, then provide name
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for name input"
    assert status["status"] == "awaiting_user_input"

    resume_response = resume_workflow(gateway_url, run_id, user_response="Alice")
    assert "runId" in resume_response or "status" in resume_response

    # Step 2: Wait for preference selection pause, then select
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for preference selection"
    assert status["status"] == "awaiting_user_input"

    resume_response = resume_workflow(gateway_url, run_id, user_response="tech")
    assert "runId" in resume_response or "status" in resume_response

    # Step 3: Wait for confirmation pause, then confirm
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input"},
        timeout=10,
    )
    assert status is not None, "Workflow should pause for confirmation"
    assert status["status"] == "awaiting_user_input"

    resume_response = resume_workflow(gateway_url, run_id, user_response="confirm")
    assert "runId" in resume_response or "status" in resume_response

    # 4. Wait for final completion
    final_status = wait_for_status(
        client,
        run_id,
        expected_statuses={"completed", "failed"},
        timeout=15,
    )

    assert final_status is not None, "Workflow should complete after all steps"
    assert final_status["status"] == "completed", (
        f"Expected 'completed' after full onboarding, got: {final_status['status']}"
    )


# =============================================================================
# AGENT WITH QUESTIONS (HITL via Agent Tools)
# Requires OPENAI_API_KEY
# =============================================================================


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(
    not __import__("os").getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_hitl_agent_questions_pauses(client, worker_process):
    """Test hitl_agent_questions workflow pauses for user input."""
    run_id = client.submit(
        "hitl_agent_questions",
        {"task": "Plan a complex multi-step project"},
        component_type="workflow",
    )

    # Agent workflows may complete quickly or pause depending on the task
    # We check for either paused or completed status
    status = wait_for_status(
        client,
        run_id,
        expected_statuses={"awaiting_user_input", "completed", "failed"},
        timeout=30,  # LLM calls can be slow
    )

    assert status is not None, "Workflow should reach terminal or paused state"
    # The workflow either paused for questions or completed without needing input
    assert status["status"] in {"awaiting_user_input", "completed"}, (
        f"Expected 'awaiting_user_input' or 'completed', got: {status['status']}"
    )
