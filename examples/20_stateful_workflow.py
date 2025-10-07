"""
Example: Stateful Workflow with ctx.state

This example demonstrates:
- Using ctx.state.get/set for durable workflow state
- Step-by-step state tracking
- Recording workflow progress (Phase 6A)
- Future: State persists across crashes and enables replay (Phase 6B)
"""

import asyncio
from typing import Dict

from agnt5 import Context, function, workflow


# Define functions to be called by workflow
@function
async def validate_order(ctx: Context, order_id: str) -> Dict:
    """Validate an order."""
    ctx.logger.info(f"Validating order {order_id}...")
    await asyncio.sleep(0.1)  # Simulate validation

    # Validation logic
    is_valid = len(order_id) > 5  # Simple validation

    return {
        "order_id": order_id,
        "valid": is_valid,
        "total": 99.99 if is_valid else 0,
        "items": ["Widget A", "Widget B"] if is_valid else []
    }


@function
async def process_payment(ctx: Context, payment_data: Dict) -> Dict:
    """Process payment for an order."""
    ctx.logger.info(f"Processing payment of ${payment_data['amount']}...")
    await asyncio.sleep(0.15)  # Simulate payment processing

    return {
        "payment_id": f"pay_{payment_data['order_id']}",
        "amount": payment_data["amount"],
        "status": "completed",
        "transaction_id": "txn_abc123"
    }


@function
async def ship_order(ctx: Context, shipment_data: Dict) -> Dict:
    """Ship an order."""
    ctx.logger.info(f"Shipping order {shipment_data['order_id']}...")
    await asyncio.sleep(0.1)  # Simulate shipping

    return {
        "order_id": shipment_data["order_id"],
        "tracking_number": "TRACK123456",
        "carrier": "FastShip",
        "status": "shipped"
    }


# Example 1: Simple Stateful Workflow
@workflow
async def order_fulfillment(ctx: Context, order_id: str) -> Dict:
    """
    Multi-step order processing with durable state tracking.

    State is recorded at each step for:
    - Progress monitoring
    - Retry logic
    - Debugging
    - Future: Crash recovery (Phase 6B)
    """
    ctx.logger.info(f"Starting order fulfillment for {order_id}")

    # Initialize workflow state
    await ctx.state.set("order_id", order_id)
    await ctx.state.set("status", "pending")
    await ctx.state.set("retry_count", 0)
    await ctx.state.set("started_at", asyncio.get_event_loop().time())

    # Step 1: Validate order
    await ctx.state.set("status", "validating")
    ctx.logger.info("Step 1: Validating order...")

    order = await ctx.task("orders", "validate_order", input=order_id)

    if not order["valid"]:
        await ctx.state.set("status", "failed")
        await ctx.state.set("failure_reason", "Invalid order")
        return {
            "order_id": order_id,
            "status": "failed",
            "reason": "Order validation failed"
        }

    # Save order details in state
    await ctx.state.set("order_total", order["total"])
    await ctx.state.set("order_items", order["items"])

    # Step 2: Process payment
    await ctx.state.set("status", "processing_payment")
    ctx.logger.info("Step 2: Processing payment...")

    # Get retry count (for retry logic demonstration)
    retry_count = await ctx.state.get("retry_count", 0)
    ctx.logger.info(f"Payment attempt #{retry_count + 1}")

    payment = await ctx.task(
        "payments",
        "process_payment",
        input={
            "order_id": order_id,
            "amount": order["total"]
        }
    )

    # Save payment details in state
    await ctx.state.set("payment_id", payment["payment_id"])
    await ctx.state.set("transaction_id", payment["transaction_id"])

    # Step 3: Ship order
    await ctx.state.set("status", "shipping")
    ctx.logger.info("Step 3: Shipping order...")

    shipment = await ctx.task(
        "shipping",
        "ship_order",
        input={
            "order_id": order_id,
            "items": order["items"]
        }
    )

    # Save shipment details in state
    await ctx.state.set("tracking_number", shipment["tracking_number"])
    await ctx.state.set("carrier", shipment["carrier"])

    # Complete workflow
    await ctx.state.set("status", "completed")
    await ctx.state.set("completed_at", asyncio.get_event_loop().time())

    ctx.logger.info(f"Order fulfillment completed for {order_id}")

    # Return final result (state is persisted separately)
    return {
        "order_id": await ctx.state.get("order_id"),
        "status": "completed",
        "order_total": await ctx.state.get("order_total"),
        "payment_id": await ctx.state.get("payment_id"),
        "tracking_number": await ctx.state.get("tracking_number"),
        "carrier": await ctx.state.get("carrier")
    }


# Example 2: Workflow with Retry Logic
@workflow
async def data_processing_with_retries(ctx: Context, data_source: str) -> Dict:
    """
    Data processing workflow with retry tracking using state.

    Demonstrates how ctx.state enables retry logic and progress tracking.
    """
    ctx.logger.info(f"Processing data from {data_source}")

    # Initialize state
    await ctx.state.set("source", data_source)
    await ctx.state.set("status", "fetching")
    await ctx.state.set("retry_count", 0)
    await ctx.state.set("max_retries", 3)

    # Fetch data (with retry tracking)
    retry_count = await ctx.state.get("retry_count", 0)
    max_retries = await ctx.state.get("max_retries", 3)

    ctx.logger.info(f"Fetch attempt {retry_count + 1}/{max_retries}")

    # In real scenario, this would handle failures and retry
    # For now, just track the attempt
    await ctx.state.set("status", "processing")

    # Simulate processing
    await asyncio.sleep(0.1)

    await ctx.state.set("status", "completed")
    await ctx.state.set("records_processed", 150)

    return {
        "source": data_source,
        "status": "completed",
        "records": await ctx.state.get("records_processed"),
        "retries": retry_count
    }


# Example 3: Multi-Stage Workflow with Progress Tracking
@workflow
async def multi_stage_pipeline(ctx: Context, job_id: str) -> Dict:
    """
    Complex multi-stage workflow with detailed progress tracking.

    Each stage updates status, enabling real-time monitoring.
    """
    stages = ["prepare", "extract", "transform", "load", "validate"]

    # Initialize
    await ctx.state.set("job_id", job_id)
    await ctx.state.set("total_stages", len(stages))
    await ctx.state.set("completed_stages", 0)
    await ctx.state.set("current_stage", "starting")

    ctx.logger.info(f"Starting {len(stages)}-stage pipeline for job {job_id}")

    results = {}

    for i, stage in enumerate(stages):
        # Update progress
        await ctx.state.set("current_stage", stage)
        await ctx.state.set("stage_index", i)
        await ctx.state.set("progress_percent", int((i / len(stages)) * 100))

        ctx.logger.info(f"Stage {i+1}/{len(stages)}: {stage}")

        # Simulate stage processing
        await asyncio.sleep(0.05)
        results[stage] = f"{stage}_completed"

        # Update completed count
        completed = await ctx.state.get("completed_stages", 0)
        await ctx.state.set("completed_stages", completed + 1)

    # Mark complete
    await ctx.state.set("current_stage", "completed")
    await ctx.state.set("progress_percent", 100)

    return {
        "job_id": job_id,
        "status": "completed",
        "total_stages": len(stages),
        "completed_stages": await ctx.state.get("completed_stages"),
        "results": results
    }


async def main():
    print("=== Stateful Workflow Examples (Phase 6A) ===\n")

    # Example 1: Order fulfillment with state tracking
    print("1. Order Fulfillment with State Tracking:\n")
    result = await order_fulfillment(order_id="order-12345")
    print(f"   Result: {result}")
    print(f"   ✅ State tracked: order ID, status, payment ID, tracking number")
    print()

    # Example 2: Data processing with retries
    print("2. Data Processing with Retry Logic:\n")
    result = await data_processing_with_retries(data_source="customer_db")
    print(f"   Result: {result}")
    print(f"   ✅ State tracked: retry count, status, records processed")
    print()

    # Example 3: Multi-stage pipeline
    print("3. Multi-Stage Pipeline with Progress:\n")
    result = await multi_stage_pipeline(job_id="pipeline-789")
    print(f"   Result: {result}")
    print(f"   ✅ State tracked: current stage, progress percent, completed stages")
    print()

    print("═" * 60)
    print("Phase 6A: Stateful Workflows")
    print("═" * 60)
    print("✅ Steps recorded: Each ctx.task() call records result")
    print("✅ State tracked: ctx.state.set/get for workflow state")
    print("✅ Events published: workflow.step.completed + workflow.state.changed")
    print("✅ Persisted: Events → Projectors → Entity table")
    print()
    print("📊 Phase 6B (Future): Replay from recorded steps on crash")
    print()


if __name__ == "__main__":
    asyncio.run(main())
