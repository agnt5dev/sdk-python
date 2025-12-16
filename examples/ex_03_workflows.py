"""
Example: AGNT5 Workflows

This example demonstrates workflow features:
- Creating workflows with @workflow decorator
- Durable steps with ctx.step() for crash recovery
- State management with ctx.state
- Sequential and checkpoint-based execution

Workflows are multi-step orchestrations that:
- Can suspend, resume, and recover from failures
- Use ctx.step() to checkpoint progress
- Persist state across restarts

Each workflow can be:
- Run standalone: python ex_03_workflows.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("data_pipeline", {"source": "db"}, component_type="workflow")
"""

import asyncio

from agnt5 import workflow, WorkflowContext


# =============================================================================
# BASIC WORKFLOW
# =============================================================================


@workflow
async def data_pipeline(ctx: WorkflowContext, source: str) -> dict:
    """
    Simple data processing pipeline with checkpointed steps.

    Each ctx.step() creates a checkpoint. If the workflow crashes,
    completed steps are skipped on restart.

    Args:
        source: Data source identifier

    Returns:
        Dict with processing results

    Example:
        result = client.run("data_pipeline", {"source": "database"}, component_type="workflow")
    """
    ctx.logger.info(f"Starting data pipeline for source: {source}")

    # Step 1: Fetch data (checkpointed)
    data = await ctx.step("fetch", lambda: {"source": source, "records": [1, 2, 3, 4, 5]})
    ctx.logger.info(f"Fetched {len(data['records'])} records from {source}")

    # Step 2: Transform data (checkpointed)
    transformed = await ctx.step("transform", lambda: {
        "source": data["source"],
        "records": [r * 2 for r in data["records"]],
    })
    ctx.logger.info(f"Transformed records: {transformed['records']}")

    # Step 3: Validate (checkpointed)
    valid = await ctx.step("validate", lambda: len(transformed["records"]) > 0)
    ctx.logger.info(f"Validation result: {valid}")

    return {
        "source": source,
        "original_count": len(data["records"]),
        "transformed": transformed["records"],
        "valid": valid,
    }


# =============================================================================
# ORDER FULFILLMENT WORKFLOW
# =============================================================================


@workflow
async def order_fulfillment(ctx: WorkflowContext, order_id: str, items: list) -> dict:
    """
    Order fulfillment workflow with multiple stages.

    Demonstrates:
    - Multi-step processing
    - State tracking
    - Conditional logic

    Args:
        order_id: Order identifier
        items: List of items to fulfill

    Returns:
        Dict with fulfillment status

    Example:
        result = client.run("order_fulfillment", {
            "order_id": "ORD-123",
            "items": [{"sku": "WIDGET", "qty": 2}]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting fulfillment for order {order_id}")

    # Step 1: Validate order
    validation = await ctx.step("validate", lambda: {
        "valid": True,
        "item_count": len(items),
        "total_qty": sum(item.get("qty", 1) for item in items),
    })

    if not validation["valid"]:
        return {"order_id": order_id, "status": "rejected", "reason": "validation_failed"}

    # Step 2: Reserve inventory
    inventory = await ctx.step("reserve_inventory", lambda: {
        "reserved": True,
        "items": items,
    })
    ctx.logger.info(f"Inventory reserved: {inventory['reserved']}")

    # Step 3: Process payment (simulated)
    payment = await ctx.step("process_payment", lambda: {
        "status": "completed",
        "transaction_id": f"TXN-{order_id}",
    })
    ctx.logger.info(f"Payment: {payment['status']}")

    # Step 4: Create shipment
    shipment = await ctx.step("create_shipment", lambda: {
        "tracking": f"TRACK-{order_id}",
        "carrier": "FastShip",
        "estimated_days": 3,
    })
    ctx.logger.info(f"Shipment created: {shipment['tracking']}")

    return {
        "order_id": order_id,
        "status": "fulfilled",
        "items_count": validation["item_count"],
        "transaction_id": payment["transaction_id"],
        "tracking": shipment["tracking"],
    }


# =============================================================================
# ETL WORKFLOW
# =============================================================================


@workflow
async def etl_workflow(ctx: WorkflowContext, dataset: str, destination: str) -> dict:
    """
    Extract-Transform-Load workflow.

    Demonstrates:
    - Classic ETL pattern
    - Progress tracking
    - Error handling in steps

    Args:
        dataset: Source dataset name
        destination: Target destination

    Returns:
        Dict with ETL results

    Example:
        result = client.run("etl_workflow", {
            "dataset": "users",
            "destination": "warehouse"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting ETL: {dataset} -> {destination}")

    # Extract
    extracted = await ctx.step("extract", lambda: {
        "dataset": dataset,
        "rows": 1000,
        "columns": ["id", "name", "email", "created_at"],
    })
    ctx.logger.info(f"Extracted {extracted['rows']} rows from {dataset}")

    # Transform
    transformed = await ctx.step("transform", lambda: {
        "rows": extracted["rows"],
        "transformations": ["normalize_email", "parse_dates", "add_metadata"],
        "errors": 0,
    })
    ctx.logger.info(f"Applied {len(transformed['transformations'])} transformations")

    # Load
    loaded = await ctx.step("load", lambda: {
        "destination": destination,
        "rows_loaded": transformed["rows"],
        "duration_ms": 250,
    })
    ctx.logger.info(f"Loaded {loaded['rows_loaded']} rows to {destination}")

    return {
        "dataset": dataset,
        "destination": destination,
        "rows_processed": loaded["rows_loaded"],
        "status": "completed",
    }


# =============================================================================
# APPROVAL WORKFLOW (with simulated wait)
# =============================================================================


@workflow
async def approval_workflow(ctx: WorkflowContext, request_id: str, requester: str) -> dict:
    """
    Approval workflow that processes a request.

    In production, this would wait for external signals.
    Here we simulate automatic approval.

    Args:
        request_id: Request identifier
        requester: Person who made the request

    Returns:
        Dict with approval status

    Example:
        result = client.run("approval_workflow", {
            "request_id": "REQ-001",
            "requester": "alice@example.com"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Processing approval request {request_id} from {requester}")

    # Step 1: Validate request
    validation = await ctx.step("validate_request", lambda: {
        "valid": True,
        "request_id": request_id,
        "requester": requester,
    })

    if not validation["valid"]:
        return {"request_id": request_id, "status": "invalid"}

    # Step 2: Auto-approve (in production, would wait for signal)
    approval = await ctx.step("get_approval", lambda: {
        "approved": True,
        "approver": "system",
        "reason": "auto-approved",
    })

    # Step 3: Execute action based on approval
    if approval["approved"]:
        result = await ctx.step("execute", lambda: {
            "executed": True,
            "action": "request_processed",
        })
        return {
            "request_id": request_id,
            "status": "approved",
            "approver": approval["approver"],
            "executed": result["executed"],
        }
    else:
        return {
            "request_id": request_id,
            "status": "rejected",
            "reason": approval.get("reason", "unknown"),
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    from agnt5 import Context
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Workflows Example")
    print("=" * 60)

    ctx = Context(run_id="workflow-demo")

    # Data pipeline
    print("\n--- Data Pipeline Workflow ---")
    result = await data_pipeline(ctx, source="database")
    print(f"Result: {result}")

    # Order fulfillment
    print("\n--- Order Fulfillment Workflow ---")
    result = await order_fulfillment(ctx, order_id="ORD-123", items=[
        {"sku": "WIDGET", "qty": 2},
        {"sku": "GADGET", "qty": 1},
    ])
    print(f"Result: {result}")

    # ETL
    print("\n--- ETL Workflow ---")
    result = await etl_workflow(ctx, dataset="users", destination="warehouse")
    print(f"Result: {result}")

    # Approval
    print("\n--- Approval Workflow ---")
    result = await approval_workflow(ctx, request_id="REQ-001", requester="alice@example.com")
    print(f"Result: {result}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
