"""
Test Workflows for Integration Testing

Provides multi-step workflows for testing durability and state management.
"""

import asyncio
from agnt5 import Context, workflow


@workflow
async def order_fulfillment(ctx: Context, order_id: str, items: list) -> dict:
    """
    Multi-step order processing workflow.

    Tests:
    - Multi-step workflow execution
    - State persistence across steps
    - Error handling in workflows
    """
    ctx.logger.info(f"Starting order fulfillment for order {order_id}")

    # Step 1: Validate order
    ctx.logger.info("Step 1: Validating order")
    await asyncio.sleep(0.1)  # Simulate validation

    # Step 2: Process payment
    ctx.logger.info("Step 2: Processing payment")
    await asyncio.sleep(0.1)  # Simulate payment processing

    # Step 3: Reserve inventory
    ctx.logger.info("Step 3: Reserving inventory")
    await asyncio.sleep(0.1)  # Simulate inventory check

    # Step 4: Ship order
    ctx.logger.info("Step 4: Shipping order")
    await asyncio.sleep(0.1)  # Simulate shipping

    return {
        "order_id": order_id,
        "status": "completed",
        "items_count": len(items),
        "steps_completed": 4
    }


@workflow
async def long_workflow(ctx: Context, steps: int) -> dict:
    """
    Multi-step workflow for testing crash recovery.

    Tests:
    - Workflow state survives worker crashes
    - Workflow can resume from last completed step
    - Long-running workflow durability
    """
    ctx.logger.info(f"Starting long workflow with {steps} steps")

    results = []
    for i in range(steps):
        ctx.logger.info(f"Executing step {i + 1}/{steps}")
        await asyncio.sleep(0.05)  # Simulate step processing
        results.append(f"step_{i + 1}_completed")

    return {
        "status": "completed",
        "total_steps": steps,
        "results": results
    }


@workflow
async def data_pipeline(ctx: Context, dataset_id: str, transform: str) -> dict:
    """
    Data processing pipeline workflow.

    Tests:
    - Workflow with conditional logic
    - Data transformation steps
    - Pipeline state management
    """
    ctx.logger.info(f"Starting data pipeline for dataset {dataset_id}")

    # Step 1: Load data
    ctx.logger.info("Loading dataset")
    await asyncio.sleep(0.1)
    record_count = 100  # Simulated

    # Step 2: Transform data
    ctx.logger.info(f"Applying transformation: {transform}")
    await asyncio.sleep(0.1)

    # Step 3: Validate output
    ctx.logger.info("Validating transformed data")
    await asyncio.sleep(0.1)

    # Step 4: Store results
    ctx.logger.info("Storing results")
    await asyncio.sleep(0.1)

    return {
        "dataset_id": dataset_id,
        "transform": transform,
        "records_processed": record_count,
        "status": "completed"
    }


__all__ = ["order_fulfillment", "long_workflow", "data_pipeline"]
