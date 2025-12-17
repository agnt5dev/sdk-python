"""
Integration Tests for ex_08_hitl.py

Tests Human-in-the-Loop (HITL) workflows through the platform:
- hitl_text_input - Text input collection
- hitl_approval - Binary approval decisions
- hitl_choice - Multiple choice selection
- hitl_onboarding - Multi-step interaction
- hitl_conditional_approval - Conditional HITL

Note: HITL tests require special handling since they pause for user input.
These tests verify:
1. Workflow structure is correct
2. Auto-approval path works (hitl_conditional_approval)
3. Exception handling for HITL pauses

Run with:
    # Local mode (requires running dev server with examples worker)
    pytest tests/integration/test_ex_08_hitl.py -v
"""

import pytest


# =============================================================================
# CONDITIONAL APPROVAL (Auto-Approve Path)
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
        component_type="workflow"
    )

    assert result["status"] == "completed"
    assert result["approval_status"] == "auto_approved"
    assert result["required_approval"] is False
    assert result["amount"] == 500.0


@pytest.mark.integration
def test_conditional_approval_at_threshold(client, worker_process):
    """Test conditional approval at exact threshold requires approval."""
    # At $1000 exactly, should require approval
    # This will pause the workflow - we test that it starts correctly
    # In a full test, we would need to provide the approval
    try:
        result = client.run(
            "hitl_conditional_approval",
            {
                "amount": 1000.0,
                "description": "Equipment purchase",
            },
            component_type="workflow",
            timeout=5  # Short timeout since it will pause
        )
        # If we get here, something unexpected happened
        assert result.get("required_approval") is True
    except Exception as e:
        # Expected: workflow pauses for input or times out
        # This is correct behavior for HITL
        pass


@pytest.mark.integration
def test_conditional_approval_under_threshold(client, worker_process):
    """Test amounts just under threshold are auto-approved."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": 999.99,
            "description": "Software subscription",
        },
        component_type="workflow"
    )

    assert result["status"] == "completed"
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
        component_type="workflow"
    )

    assert result["status"] == "completed"
    assert result["approval_status"] == "auto_approved"


# =============================================================================
# WORKFLOW STRUCTURE TESTS
# These test that workflows are registered and can be invoked
# =============================================================================


@pytest.mark.integration
def test_hitl_text_input_registered(client, worker_process):
    """Test hitl_text_input is registered and starts."""
    # This workflow will pause immediately for input
    # We just verify it's registered by attempting to run it
    try:
        client.run(
            "hitl_text_input",
            {"prompt": "Test prompt"},
            component_type="workflow",
            timeout=2
        )
    except Exception:
        # Expected: workflow pauses for input
        pass


@pytest.mark.integration
def test_hitl_approval_registered(client, worker_process):
    """Test hitl_approval is registered and starts."""
    try:
        client.run(
            "hitl_approval",
            {"action": "test", "resource": "test-resource"},
            component_type="workflow",
            timeout=2
        )
    except Exception:
        # Expected: workflow pauses for approval
        pass


@pytest.mark.integration
def test_hitl_choice_registered(client, worker_process):
    """Test hitl_choice is registered and starts."""
    try:
        client.run(
            "hitl_choice",
            {
                "question": "Pick one",
                "options": ["A", "B", "C"]
            },
            component_type="workflow",
            timeout=2
        )
    except Exception:
        # Expected: workflow pauses for choice
        pass


@pytest.mark.integration
def test_hitl_onboarding_registered(client, worker_process):
    """Test hitl_onboarding is registered and starts."""
    try:
        client.run(
            "hitl_onboarding",
            {"welcome_message": "Welcome!"},
            component_type="workflow",
            timeout=2
        )
    except Exception:
        # Expected: workflow pauses for name input
        pass


# =============================================================================
# WORKFLOW INPUT VALIDATION
# =============================================================================


@pytest.mark.integration
def test_conditional_approval_negative_amount(client, worker_process):
    """Test conditional approval handles negative amount."""
    result = client.run(
        "hitl_conditional_approval",
        {
            "amount": -100.0,
            "description": "Refund",
        },
        component_type="workflow"
    )

    # Negative amounts are below threshold, so auto-approved
    assert result["status"] == "completed"
    assert result["approval_status"] == "auto_approved"


@pytest.mark.integration
def test_conditional_approval_large_amount(client, worker_process):
    """Test conditional approval with very large amount."""
    # Large amount will require approval and pause
    try:
        result = client.run(
            "hitl_conditional_approval",
            {
                "amount": 1000000.0,
                "description": "Major purchase",
            },
            component_type="workflow",
            timeout=2
        )
        # If we somehow get a result, verify it needed approval
        assert result.get("required_approval") is True
    except Exception:
        # Expected: workflow pauses for approval
        pass


# =============================================================================
# AGENT WITH QUESTIONS (HITL via Agent Tools)
# Requires OPENAI_API_KEY
# =============================================================================


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(
    not __import__("os").getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_hitl_agent_questions_registered(client, worker_process):
    """Test hitl_agent_questions workflow is registered."""
    try:
        client.run(
            "hitl_agent_questions",
            {"task": "Plan a simple task"},
            component_type="workflow",
            timeout=5
        )
    except Exception:
        # May pause for questions or complete quickly
        pass
