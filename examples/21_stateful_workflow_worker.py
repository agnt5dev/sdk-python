"""
Worker for Stateful Workflow Testing

This worker exposes the stateful workflows for platform testing.
Run this to test workflows through the platform HTTP API.
"""

import asyncio
from typing import Dict

from agnt5 import Context, Worker, function, workflow

# Initialize worker
worker = Worker(service_name="default-service")


# Functions called by workflows
@function
async def validate_order(ctx: Context, order_id: str) -> Dict:
    """Validate an order."""
    ctx.logger.info(f"Validating order {order_id}...")
    await asyncio.sleep(0.1)

    is_valid = len(order_id) > 5
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
    await asyncio.sleep(0.15)

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
    await asyncio.sleep(0.1)

    return {
        "order_id": shipment_data["order_id"],
        "tracking_number": "TRACK123456",
        "carrier": "FastShip",
        "status": "shipped"
    }


# Stateful Workflow
@workflow
async def order_fulfillment(ctx: Context, order_id: str) -> Dict:
    """
    Multi-step order processing with durable state tracking.
    """
    ctx.logger.info(f"🚀 Starting order fulfillment for {order_id}")

    # Initialize workflow state
    await ctx.state.set("order_id", order_id)
    await ctx.state.set("status", "pending")
    await ctx.state.set("steps_completed", 0)

    # Step 1: Validate order
    await ctx.state.set("status", "validating")
    ctx.logger.info("📋 Step 1/3: Validating order...")

    order = await ctx.task("default-service", "validate_order", input=order_id)

    if not order["valid"]:
        await ctx.state.set("status", "failed")
        ctx.logger.error(f"❌ Order validation failed for {order_id}")
        return {
            "order_id": order_id,
            "status": "failed",
            "reason": "Order validation failed"
        }

    await ctx.state.set("steps_completed", 1)
    await ctx.state.set("order_total", order["total"])
    ctx.logger.info(f"✅ Order validated: ${order['total']}")

    # Step 2: Process payment
    await ctx.state.set("status", "processing_payment")
    ctx.logger.info("💳 Step 2/3: Processing payment...")

    payment = await ctx.task(
        "default-service",
        "process_payment",
        input={
            "order_id": order_id,
            "amount": order["total"]
        }
    )

    await ctx.state.set("steps_completed", 2)
    await ctx.state.set("payment_id", payment["payment_id"])
    ctx.logger.info(f"✅ Payment processed: {payment['payment_id']}")

    # Step 3: Ship order
    await ctx.state.set("status", "shipping")
    ctx.logger.info("📦 Step 3/3: Shipping order...")

    shipment = await ctx.task(
        "default-service",
        "ship_order",
        input={
            "order_id": order_id,
            "items": order["items"]
        }
    )

    await ctx.state.set("steps_completed", 3)
    await ctx.state.set("tracking_number", shipment["tracking_number"])
    ctx.logger.info(f"✅ Order shipped: {shipment['tracking_number']}")

    # Complete workflow
    await ctx.state.set("status", "completed")

    ctx.logger.info(f"🎉 Order fulfillment completed for {order_id}")

    return {
        "order_id": order_id,
        "status": "completed",
        "order_total": await ctx.state.get("order_total"),
        "payment_id": await ctx.state.get("payment_id"),
        "tracking_number": await ctx.state.get("tracking_number"),
        "steps_completed": await ctx.state.get("steps_completed")
    }


@workflow
async def multi_stage_pipeline(ctx: Context, job_id: str) -> Dict:
    """Complex multi-stage workflow with progress tracking."""
    stages = ["prepare", "extract", "transform", "load", "validate"]

    await ctx.state.set("job_id", job_id)
    await ctx.state.set("total_stages", len(stages))
    await ctx.state.set("completed_stages", 0)

    ctx.logger.info(f"🔄 Starting {len(stages)}-stage pipeline for {job_id}")

    for i, stage in enumerate(stages):
        await ctx.state.set("current_stage", stage)
        await ctx.state.set("progress_percent", int((i / len(stages)) * 100))

        ctx.logger.info(f"📊 Stage {i+1}/{len(stages)}: {stage}")
        await asyncio.sleep(0.05)

        completed = await ctx.state.get("completed_stages", 0)
        await ctx.state.set("completed_stages", completed + 1)

    await ctx.state.set("progress_percent", 100)
    ctx.logger.info(f"✅ Pipeline completed for {job_id}")

    return {
        "job_id": job_id,
        "status": "completed",
        "total_stages": len(stages),
        "completed_stages": await ctx.state.get("completed_stages")
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Stateful Workflow Worker Starting")
    print("=" * 60)
    print("Service: default-service")
    print("Components:")
    print("  • Functions: validate_order, process_payment, ship_order")
    print("  • Workflows: order_fulfillment, multi_stage_pipeline")
    print("")
    print("Test with:")
    print("  curl -X POST http://localhost:34181/run/order_fulfillment \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"order_id\": \"test-123\"}'")
    print("=" * 60)
    print("")

    asyncio.run(worker.run())
