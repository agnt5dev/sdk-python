"""
Example: Basic Workflow Usage

This example demonstrates:
- Creating workflows with @workflow decorator
- Sequential execution with await
- Parallel execution with ctx.parallel() and ctx.gather()
- Calling functions from workflows with ctx.task()
- Using signals for coordination
"""

import asyncio
from typing import Dict, List

from agnt5 import Context, function, workflow


# Define some functions to be called by workflows
@function
async def fetch_data(ctx: Context, source: str) -> Dict:
    """Simulate fetching data from a source."""
    ctx.logger.info(f"Fetching data from {source}...")
    await asyncio.sleep(0.1)  # Simulate I/O
    return {"source": source, "data": [1, 2, 3, 4, 5]}


@function
async def process_data(ctx: Context, data: Dict) -> Dict:
    """Process fetched data."""
    ctx.logger.info(f"Processing data from {data['source']}...")
    await asyncio.sleep(0.1)  # Simulate processing
    total = sum(data["data"])
    return {"source": data["source"], "total": total, "count": len(data["data"])}


@function
async def validate_order(ctx: Context, order_id: str) -> Dict:
    """Validate an order."""
    ctx.logger.info(f"Validating order {order_id}...")
    await asyncio.sleep(0.05)
    return {"order_id": order_id, "valid": True, "total": 99.99}


@function
async def process_payment(ctx: Context, amount: float) -> Dict:
    """Process payment."""
    ctx.logger.info(f"Processing payment of ${amount}...")
    await asyncio.sleep(0.05)
    return {"amount": amount, "status": "completed", "transaction_id": "txn_12345"}


@function
async def ship_order(ctx: Context, order_id: str) -> Dict:
    """Ship an order."""
    ctx.logger.info(f"Shipping order {order_id}...")
    await asyncio.sleep(0.05)
    return {"order_id": order_id, "tracking": "TRACK123", "status": "shipped"}


# Basic sequential workflow
@workflow
async def data_pipeline(ctx: Context, source: str) -> Dict:
    """Simple sequential data processing pipeline.

    Phase 1: Each step runs sequentially using local function calls.
    Phase 2: Will support distributed execution and durable checkpoints.
    """
    ctx.logger.info(f"Starting data pipeline for {source}")

    # Step 1: Fetch data
    data = await ctx.task("data-service", "fetch_data", input=source)

    # Step 2: Process data
    result = await ctx.task("data-service", "process_data", input=data)

    ctx.logger.info(f"Pipeline completed: {result}")
    return result


# Parallel execution workflow
@workflow
async def multi_source_pipeline(ctx: Context) -> Dict:
    """Process multiple data sources in parallel.

    Demonstrates ctx.parallel() for concurrent execution.
    """
    ctx.logger.info("Starting multi-source pipeline")

    # Fetch from multiple sources in parallel
    results = await ctx.parallel(
        ctx.task("data-service", "fetch_data", input="database"),
        ctx.task("data-service", "fetch_data", input="api"),
        ctx.task("data-service", "fetch_data", input="file"),
    )

    ctx.logger.info(f"Fetched from {len(results)} sources")

    # Process all in parallel
    processed = await ctx.parallel(
        ctx.task("data-service", "process_data", input=results[0]),
        ctx.task("data-service", "process_data", input=results[1]),
        ctx.task("data-service", "process_data", input=results[2]),
    )

    # Aggregate results
    total_sum = sum(p["total"] for p in processed)
    total_count = sum(p["count"] for p in processed)

    return {
        "sources": len(results),
        "total_sum": total_sum,
        "total_count": total_count,
        "average": total_sum / total_count if total_count > 0 else 0,
    }


# Named parallel execution with ctx.gather()
@workflow
async def order_processing(ctx: Context, order_id: str) -> Dict:
    """Process an order with sequential and parallel steps.

    Demonstrates:
    - Sequential execution (validate first)
    - Named parallel execution with ctx.gather()
    - Conditional logic
    """
    ctx.logger.info(f"Processing order {order_id}")

    # Step 1: Validate order (sequential)
    order = await ctx.task("orders", "validate_order", input=order_id)

    if not order["valid"]:
        return {"status": "failed", "reason": "Invalid order"}

    # Step 2: Payment and fulfillment preparation in parallel
    results = await ctx.gather(
        payment=ctx.task("payments", "process_payment", input=order["total"]),
        shipment=ctx.task("fulfillment", "ship_order", input=order_id),
    )

    # Step 3: Return aggregated results
    return {
        "order_id": order_id,
        "status": "completed",
        "payment": results["payment"],
        "shipment": results["shipment"],
    }


# Workflow with signal coordination
@workflow
async def approval_workflow(ctx: Context, request_id: str) -> Dict:
    """Workflow that waits for external approval signal.

    Phase 1: Uses asyncio.Event for in-process signaling.
    Phase 2: Will support durable cross-process signals.
    """
    ctx.logger.info(f"Starting approval workflow for request {request_id}")

    # Prepare request
    ctx.set("request_id", request_id)
    ctx.set("status", "pending")

    # In a real scenario, this would trigger an external notification
    # For demo, we'll send signal after a short delay
    async def send_approval():
        await asyncio.sleep(0.2)
        ctx.signal_send("approval", {"approved": True, "reviewer": "admin"})

    # Start background task to send signal
    asyncio.create_task(send_approval())

    # Wait for approval signal (with timeout)
    ctx.logger.info("Waiting for approval signal...")
    approval = await ctx.signal("approval", timeout_ms=5000)

    if approval and approval.get("approved"):
        ctx.set("status", "approved")
        return {
            "request_id": request_id,
            "status": "approved",
            "reviewer": approval.get("reviewer"),
        }
    else:
        ctx.set("status", "rejected")
        return {"request_id": request_id, "status": "rejected"}


# Workflow with delays
@workflow
async def scheduled_workflow(ctx: Context, task_name: str) -> Dict:
    """Workflow with delays using ctx.sleep() and ctx.timer().

    Phase 1: Uses asyncio.sleep for delays.
    Phase 2: Will support durable sleep that survives restarts.
    """
    ctx.logger.info(f"Starting scheduled workflow: {task_name}")

    # Initial delay
    ctx.logger.info("Waiting 0.1 seconds...")
    await ctx.sleep(0.1)

    # Use timer with milliseconds
    ctx.logger.info("Waiting 100ms...")
    await ctx.timer(delay_ms=100)

    ctx.logger.info("Scheduled workflow completed")
    return {"task": task_name, "status": "completed"}


async def main():
    print("=== Basic Workflow Examples ===\n")

    # 1. Sequential workflow
    print("1. Sequential Data Pipeline:\n")
    result = await data_pipeline(source="database")
    print(f"   Result: {result}")
    print()

    # 2. Parallel execution
    print("2. Multi-Source Pipeline (Parallel):\n")
    result = await multi_source_pipeline()
    print(f"   Sources processed: {result['sources']}")
    print(f"   Total sum: {result['total_sum']}")
    print(f"   Average: {result['average']:.2f}")
    print()

    # 3. Named parallel with gather
    print("3. Order Processing (Sequential + Parallel):\n")
    result = await order_processing(order_id="order-123")
    print(f"   Order: {result['order_id']}")
    print(f"   Status: {result['status']}")
    print(f"   Payment: {result['payment']['transaction_id']}")
    print(f"   Tracking: {result['shipment']['tracking']}")
    print()

    # 4. Signal coordination
    print("4. Approval Workflow (Signals):\n")
    result = await approval_workflow(request_id="req-456")
    print(f"   Request: {result['request_id']}")
    print(f"   Status: {result['status']}")
    if result["status"] == "approved":
        print(f"   Reviewer: {result['reviewer']}")
    print()

    # 5. Scheduled workflow with delays
    print("5. Scheduled Workflow (Delays):\n")
    result = await scheduled_workflow(task_name="cleanup")
    print(f"   Task: {result['task']}")
    print(f"   Status: {result['status']}")
    print()

    # 6. Direct function call (without workflow)
    print("6. Direct Function Call (No Workflow):\n")
    direct_result = await fetch_data(source="cache")
    print(f"   Direct call result: {direct_result}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
