"""
Phase 6B Test: Crash Recovery with Replay

This example demonstrates workflow crash recovery using cached steps.
The workflow simulates a crash after completing some steps, then resumes
from cached results on retry.
"""

import asyncio
from typing import Dict

from agnt5 import Context, Worker, function, workflow


# Initialize worker
worker = Worker(service_name="default-service")


# Helper functions for the workflow
@function
async def step1_fetch_data(ctx: Context, query: str) -> Dict:
    """Simulate expensive data fetching (e.g., API call, LLM)."""
    ctx.logger.info(f"💰 EXPENSIVE: Fetching data for '{query}'...")
    await asyncio.sleep(0.2)  # Simulate slow operation

    return {
        "data": f"Result for {query}",
        "timestamp": "2025-10-07T00:00:00Z",
        "cost": 0.50  # Simulated cost
    }


@function
async def step2_process_data(ctx: Context, data: Dict) -> Dict:
    """Simulate data processing."""
    ctx.logger.info(f"🔧 Processing data: {data['data']}")
    await asyncio.sleep(0.1)

    return {
        "processed": data["data"].upper(),
        "metadata": {"source": data.get("timestamp")}
    }


@function
async def step3_save_result(ctx: Context, result: Dict) -> Dict:
    """Simulate saving results (e.g., database write)."""
    ctx.logger.info(f"💾 Saving result: {result['processed']}")
    await asyncio.sleep(0.1)

    return {
        "saved": True,
        "id": "result-123",
        "data": result["processed"]
    }


# Example 1: Workflow that crashes and recovers
@workflow
async def crash_recovery_workflow(ctx: Context, query: str) -> Dict:
    """
    Workflow that crashes after step 2, then recovers on retry.

    On first attempt:
      - Step 1: Executes (expensive)
      - Step 2: Executes
      - Step 3: CRASHES 💥

    On retry:
      - Step 1: Cached ✅ (free, instant)
      - Step 2: Cached ✅ (free, instant)
      - Step 3: Executes 🔄 (only this runs)
    """
    ctx.logger.info(f"🚀 Starting crash recovery workflow for '{query}'")

    # Track attempt in state
    await ctx.state.set("query", query)
    attempt = await ctx.state.get("attempt", 0)
    await ctx.state.set("attempt", attempt + 1)

    ctx.logger.info(f"📊 Attempt #{attempt + 1}")

    # Step 1: Fetch data (expensive - we don't want to re-run this)
    await ctx.state.set("status", "fetching")
    data = await ctx.task("default-service", "step1_fetch_data", input=query)
    await ctx.state.set("fetched_at", data.get("timestamp"))
    ctx.logger.info(f"✅ Step 1 complete: Got data")

    # Step 2: Process data
    await ctx.state.set("status", "processing")
    processed = await ctx.task("default-service", "step2_process_data", input=data)
    await ctx.state.set("processed", True)
    ctx.logger.info(f"✅ Step 2 complete: Processed data")

    # Step 3: Save result (THIS IS WHERE THE CRASH HAPPENS)
    await ctx.state.set("status", "saving")

    # Simulate crash on first attempt
    if attempt == 0:
        ctx.logger.error("💥 CRASH SIMULATION: Worker crashed before step 3!")
        raise Exception("Simulated crash - worker died before completing step 3")

    # On retry, this step will execute
    result = await ctx.task("default-service", "step3_save_result", input=processed)
    await ctx.state.set("saved_id", result.get("id"))
    ctx.logger.info(f"✅ Step 3 complete: Saved with ID {result.get('id')}")

    # Workflow complete
    await ctx.state.set("status", "completed")
    ctx.logger.info(f"🎉 Workflow completed successfully on attempt #{attempt + 1}")

    return {
        "status": "completed",
        "attempt": attempt + 1,
        "result_id": result.get("id"),
        "message": f"Recovered after {attempt} failures"
    }


# Example 2: Workflow with multiple failure points
@workflow
async def multi_step_recovery(ctx: Context, job_id: str) -> Dict:
    """
    Workflow with 5 steps that can fail at any point.
    Uses ctx.state to track which step to fail on.
    """
    ctx.logger.info(f"🔄 Multi-step recovery workflow: {job_id}")

    # Initialize state
    await ctx.state.set("job_id", job_id)
    attempt = await ctx.state.get("attempt", 0)
    await ctx.state.set("attempt", attempt + 1)
    fail_at_step = await ctx.state.get("fail_at_step", 3)  # Default: fail at step 3

    ctx.logger.info(f"📊 Attempt #{attempt + 1}, will fail at step {fail_at_step}")

    steps = ["initialize", "validate", "process", "finalize", "cleanup"]
    results = []

    for i, step_name in enumerate(steps):
        step_num = i + 1
        ctx.logger.info(f"▶️  Executing step {step_num}: {step_name}")

        # Execute step (simulated with async sleep)
        await asyncio.sleep(0.05)
        result = f"{step_name}_complete"
        results.append(result)

        await ctx.state.set(f"step_{step_num}_result", result)
        await ctx.state.set("last_completed_step", step_num)

        # Simulate crash at specific step on first attempt
        if attempt == 0 and step_num == fail_at_step:
            ctx.logger.error(f"💥 CRASH at step {step_num}: {step_name}")
            raise Exception(f"Simulated crash at step {step_num}")

        ctx.logger.info(f"✅ Step {step_num} complete: {step_name}")

    await ctx.state.set("status", "completed")
    ctx.logger.info(f"🎉 All {len(steps)} steps completed on attempt #{attempt + 1}")

    return {
        "job_id": job_id,
        "status": "completed",
        "attempt": attempt + 1,
        "steps_completed": len(results),
        "results": results
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 6B: Crash Recovery Test Workflows")
    print("=" * 70)
    print("Service: default-service")
    print()
    print("Test 1: crash_recovery_workflow")
    print("  - Crashes after step 2 on first attempt")
    print("  - Recovers on retry using cached steps 1 & 2")
    print()
    print("Test 2: multi_step_recovery")
    print("  - 5-step workflow that can fail at any step")
    print("  - Configure fail_at_step via ctx.state")
    print()
    print("Test with:")
    print("  # First attempt (will fail):")
    print("  curl -X POST http://localhost:34181/run-workflow/crash_recovery_workflow \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"test data\"}'")
    print()
    print("  # Manual retry (will succeed):")
    print("  curl -X POST http://localhost:34181/run-workflow/crash_recovery_workflow \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"test data\"}'")
    print("=" * 70)
    print()

    asyncio.run(worker.run())
