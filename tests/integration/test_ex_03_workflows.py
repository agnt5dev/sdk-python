"""
Integration Tests for ex_03_workflows.py

Tests workflow features through the platform:
- data_pipeline - Basic data processing workflow
- order_fulfillment - Multi-stage order processing
- etl_workflow - Extract-Transform-Load pattern
- approval_workflow - Request approval pattern

Run with:
    # Local mode (requires running dev server with examples worker)
    pytest tests/integration/test_ex_03_workflows.py -v

    # Subprocess mode (for CI)
    pytest tests/integration/test_ex_03_workflows.py -v --use-subprocess
"""

import pytest


# =============================================================================
# DATA PIPELINE WORKFLOW
# =============================================================================


@pytest.mark.integration
def test_data_pipeline_basic(client, worker_process, platform):
    """Test basic data pipeline workflow."""
    result = client.run(
        "data_pipeline",
        {"source": "database"},
        component_type="workflow"
    )

    assert result["source"] == "database"
    assert result["original_count"] == 5  # [1,2,3,4,5]
    assert result["transformed"] == [2, 4, 6, 8, 10]  # doubled
    assert result["valid"] is True


@pytest.mark.integration
def test_data_pipeline_different_source(client, worker_process):
    """Test data pipeline with different source."""
    result = client.run(
        "data_pipeline",
        {"source": "api"},
        component_type="workflow"
    )

    assert result["source"] == "api"
    assert result["valid"] is True


# =============================================================================
# ORDER FULFILLMENT WORKFLOW
# =============================================================================


@pytest.mark.integration
def test_order_fulfillment_basic(client, worker_process):
    """Test basic order fulfillment workflow."""
    result = client.run(
        "order_fulfillment",
        {
            "order_id": "ORD-123",
            "items": [
                {"sku": "WIDGET", "qty": 2},
                {"sku": "GADGET", "qty": 1},
            ]
        },
        component_type="workflow"
    )

    assert result["order_id"] == "ORD-123"
    assert result["status"] == "fulfilled"
    assert result["items_count"] == 2
    assert result["transaction_id"] == "TXN-ORD-123"
    assert result["tracking"] == "TRACK-ORD-123"


@pytest.mark.integration
def test_order_fulfillment_single_item(client, worker_process):
    """Test order fulfillment with single item."""
    result = client.run(
        "order_fulfillment",
        {
            "order_id": "ORD-456",
            "items": [{"sku": "SINGLE", "qty": 5}]
        },
        component_type="workflow"
    )

    assert result["status"] == "fulfilled"
    assert result["items_count"] == 1


@pytest.mark.integration
def test_order_fulfillment_multiple_orders(client, worker_process):
    """Test multiple orders are processed independently."""
    result1 = client.run(
        "order_fulfillment",
        {"order_id": "MULTI-1", "items": [{"sku": "A", "qty": 1}]},
        component_type="workflow"
    )
    result2 = client.run(
        "order_fulfillment",
        {"order_id": "MULTI-2", "items": [{"sku": "B", "qty": 2}]},
        component_type="workflow"
    )

    assert result1["order_id"] == "MULTI-1"
    assert result2["order_id"] == "MULTI-2"
    assert result1["tracking"] != result2["tracking"]


# =============================================================================
# ETL WORKFLOW
# =============================================================================


@pytest.mark.integration
def test_etl_workflow_basic(client, worker_process):
    """Test basic ETL workflow."""
    result = client.run(
        "etl_workflow",
        {
            "dataset": "users",
            "destination": "warehouse"
        },
        component_type="workflow"
    )

    assert result["dataset"] == "users"
    assert result["destination"] == "warehouse"
    assert result["rows_processed"] == 1000
    assert result["status"] == "completed"


@pytest.mark.integration
def test_etl_workflow_different_datasets(client, worker_process):
    """Test ETL with different dataset/destination combinations."""
    test_cases = [
        {"dataset": "orders", "destination": "analytics"},
        {"dataset": "products", "destination": "reporting"},
        {"dataset": "logs", "destination": "archive"},
    ]

    for case in test_cases:
        result = client.run(
            "etl_workflow",
            case,
            component_type="workflow"
        )
        assert result["dataset"] == case["dataset"]
        assert result["destination"] == case["destination"]
        assert result["status"] == "completed"


# =============================================================================
# APPROVAL WORKFLOW
# =============================================================================


@pytest.mark.integration
def test_approval_workflow_approved(client, worker_process):
    """Test approval workflow (auto-approved)."""
    result = client.run(
        "approval_workflow",
        {
            "request_id": "REQ-001",
            "requester": "alice@example.com"
        },
        component_type="workflow"
    )

    assert result["request_id"] == "REQ-001"
    assert result["status"] == "approved"
    assert result["approver"] == "system"
    assert result["executed"] is True


@pytest.mark.integration
def test_approval_workflow_different_requesters(client, worker_process):
    """Test approval workflow with different requesters."""
    requesters = [
        "alice@example.com",
        "bob@example.com",
        "admin@example.com",
    ]

    for i, requester in enumerate(requesters):
        result = client.run(
            "approval_workflow",
            {
                "request_id": f"REQ-{i:03d}",
                "requester": requester
            },
            component_type="workflow"
        )
        assert result["status"] == "approved"


# =============================================================================
# WORKFLOW EXECUTION PROPERTIES
# =============================================================================


@pytest.mark.integration
def test_workflow_returns_correct_types(client, worker_process):
    """Test that workflows return correct data types."""
    result = client.run(
        "data_pipeline",
        {"source": "test"},
        component_type="workflow"
    )

    # Check types
    assert isinstance(result["source"], str)
    assert isinstance(result["original_count"], int)
    assert isinstance(result["transformed"], list)
    assert isinstance(result["valid"], bool)


@pytest.mark.integration
def test_workflow_handles_special_characters(client, worker_process):
    """Test workflow handles special characters in input."""
    result = client.run(
        "data_pipeline",
        {"source": "db-prod/users@2024"},
        component_type="workflow"
    )

    assert result["source"] == "db-prod/users@2024"
