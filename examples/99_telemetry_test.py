"""
Test OpenTelemetry Logging Integration

This example demonstrates that ctx.logger logs are properly:
1. Sent to OpenTelemetry OTLP exporter
2. Printed to console with invocation.id
3. Integrated with distributed tracing

Prerequisites:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export AGNT5_COORDINATOR_ENDPOINT=http://localhost:34186

Usage:
    python examples/99_telemetry_test.py
"""

import asyncio
from agnt5 import Context, Worker, function, workflow


@function
async def test_logging(ctx: Context, message: str) -> dict:
    """Test function that uses ctx.logger at different levels."""
    ctx.logger.debug(f"DEBUG: Processing message: {message}")
    ctx.logger.info(f"INFO: Processing message: {message}")
    ctx.logger.warning(f"WARNING: This is a warning for: {message}")
    ctx.logger.error(f"ERROR: This is an error for: {message}")

    # Simulate some work
    await asyncio.sleep(0.1)

    ctx.logger.info(f"✅ Completed processing: {message}")

    return {
        "status": "success",
        "message": f"Processed: {message}",
        "run_id": ctx.run_id
    }


@workflow
async def test_workflow_logging(ctx: Context, job_id: str) -> dict:
    """Test workflow that demonstrates logging across multiple steps."""
    ctx.logger.info(f"🚀 Starting workflow for job: {job_id}")

    # Step 1
    ctx.logger.info("📋 Step 1: Validation")
    result1 = await ctx.task("default-service", "test_logging", input={"message": "Step 1"})
    ctx.logger.info(f"Step 1 result: {result1}")

    # Step 2
    ctx.logger.info("📊 Step 2: Processing")
    result2 = await ctx.task("default-service", "test_logging", input={"message": "Step 2"})
    ctx.logger.info(f"Step 2 result: {result2}")

    # Step 3
    ctx.logger.info("✨ Step 3: Finalization")
    result3 = await ctx.task("default-service", "test_logging", input={"message": "Step 3"})
    ctx.logger.info(f"Step 3 result: {result3}")

    ctx.logger.info(f"🎉 Workflow completed for job: {job_id}")

    return {
        "status": "completed",
        "job_id": job_id,
        "steps": [result1, result2, result3]
    }


async def main():
    """Run the worker to test telemetry integration."""
    print("=" * 60)
    print("AGNT5 TELEMETRY TEST")
    print("=" * 60)
    print()
    print("This worker will test OpenTelemetry logging integration.")
    print()
    print("Expected behavior:")
    print("  ✅ Logs appear in console with invocation.id")
    print("  ✅ Logs sent to OpenTelemetry OTLP exporter")
    print("  ✅ Logs inherit span context for distributed tracing")
    print()
    print("To verify OpenTelemetry integration:")
    print("  1. Check console output has structured logs")
    print("  2. Check OpenTelemetry backend for log records")
    print("  3. Verify trace context propagation")
    print()
    print("=" * 60)
    print()

    worker = Worker(
        service_name="default-service",
        service_version="1.0.0",
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
