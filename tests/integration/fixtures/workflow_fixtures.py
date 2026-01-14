"""
Workflow fixtures for integration tests.

This module provides reusable workflow backends for testing various scenarios:
- Happy path: Basic multi-step workflows
- Error scenarios: Workflows that test failure modes
- Edge cases: Long-running, checkpoint-heavy workflows
- Recovery: Workflows for testing crash recovery

These fixtures are separate from pedagogical examples, allowing tests to cover
complex scenarios including failures, timeouts, and state recovery.
"""

import asyncio
from typing import Any, Optional

from agnt5 import workflow, WorkflowContext


# =============================================================================
# HAPPY PATH WORKFLOWS
# =============================================================================


@workflow
async def simple_two_step_workflow(ctx: WorkflowContext, value: int) -> dict:
    """
    Simple two-step workflow for basic tests.

    Args:
        value: Input value to process

    Returns:
        Dict with processing results
    """
    ctx.logger.info(f"Starting two-step workflow with value: {value}")

    # Step 1
    step1_result = await ctx.step(
        "step1",
        lambda: {"value": value, "doubled": value * 2}
    )
    ctx.logger.info(f"Step 1 complete: {step1_result}")

    # Step 2
    step2_result = await ctx.step(
        "step2",
        lambda: {"value": step1_result["doubled"], "tripled": step1_result["doubled"] * 3}
    )
    ctx.logger.info(f"Step 2 complete: {step2_result}")

    return {
        "original": value,
        "after_step1": step1_result["doubled"],
        "after_step2": step2_result["tripled"],
        "steps_completed": 2,
    }


@workflow
async def three_step_workflow(ctx: WorkflowContext, data: str) -> dict:
    """
    Three-step workflow for testing step sequencing.

    Args:
        data: Input data string

    Returns:
        Dict with results from all steps
    """
    ctx.logger.info(f"Starting three-step workflow with: {data}")

    # Step 1: Uppercase
    upper = await ctx.step("uppercase", lambda: data.upper())
    ctx.logger.info(f"After uppercase: {upper}")

    # Step 2: Reverse
    reversed_str = await ctx.step("reverse", lambda: upper[::-1])
    ctx.logger.info(f"After reverse: {reversed_str}")

    # Step 3: Add prefix/suffix
    final = await ctx.step("wrap", lambda: f"[{reversed_str}]")
    ctx.logger.info(f"Final result: {final}")

    return {
        "original": data,
        "step1": upper,
        "step2": reversed_str,
        "step3": final,
        "steps": 3,
    }


@workflow
async def long_running_workflow(ctx: WorkflowContext, steps: int = 10) -> dict:
    """
    Long-running workflow with many steps.

    Args:
        steps: Number of steps to execute

    Returns:
        Dict with step results
    """
    ctx.logger.info(f"Starting long workflow with {steps} steps")

    results = []
    for i in range(steps):
        result = await ctx.step(
            f"step_{i}",
            lambda i=i: {"step": i, "value": i * 10}
        )
        results.append(result)
        ctx.logger.info(f"Completed step {i}/{steps}")

    return {
        "total_steps": steps,
        "results": results,
        "sum": sum(r["value"] for r in results),
    }


@workflow
async def workflow_with_checkpoints(ctx: WorkflowContext, checkpoint_count: int = 5) -> dict:
    """
    Workflow with explicit checkpoints for recovery testing.

    Args:
        checkpoint_count: Number of checkpoints

    Returns:
        Dict with checkpoint info
    """
    ctx.logger.info(f"Starting workflow with {checkpoint_count} checkpoints")

    checkpoints = []
    for i in range(checkpoint_count):
        checkpoint = await ctx.step(
            f"checkpoint_{i}",
            lambda i=i: {
                "checkpoint_id": i,
                "timestamp": i * 100,
                "data": f"checkpoint_{i}_data",
            }
        )
        checkpoints.append(checkpoint)
        ctx.logger.info(f"Checkpoint {i} created")

    return {
        "checkpoint_count": checkpoint_count,
        "checkpoints": checkpoints,
        "final_checkpoint": checkpoints[-1] if checkpoints else None,
    }


@workflow
async def data_processing_workflow(ctx: WorkflowContext, items: list) -> dict:
    """
    Workflow that processes a list of items.

    Args:
        items: List of items to process

    Returns:
        Dict with processed results
    """
    ctx.logger.info(f"Processing {len(items)} items")

    # Step 1: Validate
    validation = await ctx.step(
        "validate",
        lambda: {"valid": len(items) > 0, "count": len(items)}
    )

    if not validation["valid"]:
        return {"status": "failed", "reason": "validation_failed"}

    # Step 2: Transform
    transformed = await ctx.step(
        "transform",
        lambda: [item * 2 if isinstance(item, (int, float)) else item for item in items]
    )

    # Step 3: Aggregate
    result = await ctx.step(
        "aggregate",
        lambda: {
            "count": len(transformed),
            "sum": sum(x for x in transformed if isinstance(x, (int, float))),
        }
    )

    return {
        "original_count": len(items),
        "transformed": transformed,
        "aggregate": result,
        "status": "completed",
    }


# =============================================================================
# ERROR SCENARIO WORKFLOWS
# =============================================================================


@workflow
async def failing_workflow(ctx: WorkflowContext, fail_at_step: int = 2) -> dict:
    """
    Workflow that fails at a specific step.

    Args:
        fail_at_step: Which step to fail at (0-indexed)

    Returns:
        Never completes (raises error)

    Raises:
        RuntimeError: At the specified step
    """
    ctx.logger.info(f"Starting workflow that will fail at step {fail_at_step}")

    # Step 0
    if fail_at_step == 0:
        raise RuntimeError("Failed at step 0")

    step0 = await ctx.step("step0", lambda: {"step": 0, "success": True})
    ctx.logger.info(f"Step 0 completed: {step0}")

    # Step 1
    if fail_at_step == 1:
        raise RuntimeError("Failed at step 1")

    step1 = await ctx.step("step1", lambda: {"step": 1, "success": True})
    ctx.logger.info(f"Step 1 completed: {step1}")

    # Step 2
    if fail_at_step == 2:
        raise RuntimeError("Failed at step 2")

    step2 = await ctx.step("step2", lambda: {"step": 2, "success": True})
    ctx.logger.info(f"Step 2 completed: {step2}")

    return {"status": "unexpected_completion"}


@workflow
async def sometimes_failing_workflow(ctx: WorkflowContext, fail_probability: float = 0.5) -> dict:
    """
    Workflow that randomly fails.

    Args:
        fail_probability: Probability of failure (0.0 to 1.0)

    Returns:
        Success dict or raises error

    Raises:
        RuntimeError: Random failure based on probability
    """
    import random

    ctx.logger.info(f"Starting workflow with {fail_probability*100}% failure rate")

    # Step 1
    step1 = await ctx.step("step1", lambda: {"step": 1})
    ctx.logger.info("Step 1 completed")

    # Random failure point
    if random.random() < fail_probability:
        raise RuntimeError(f"Random workflow failure (probability: {fail_probability})")

    # Step 2
    step2 = await ctx.step("step2", lambda: {"step": 2})
    ctx.logger.info("Step 2 completed")

    return {"success": True, "steps_completed": 2}


@workflow
async def timeout_workflow(ctx: WorkflowContext, step_duration: float = 5.0) -> dict:
    """
    Workflow with slow steps for timeout testing.

    Args:
        step_duration: Duration of each step in seconds

    Returns:
        Dict with timing info (if completes)
    """
    ctx.logger.info(f"Starting workflow with {step_duration}s steps")

    async def slow_step(step_num: int):
        await asyncio.sleep(step_duration)
        return {"step": step_num, "duration": step_duration}

    # Step 1 (slow)
    step1 = await ctx.step("slow_step1", lambda: asyncio.run(slow_step(1)))
    ctx.logger.info(f"Slow step 1 completed")

    # Step 2 (slow)
    step2 = await ctx.step("slow_step2", lambda: asyncio.run(slow_step(2)))
    ctx.logger.info(f"Slow step 2 completed")

    return {
        "steps_completed": 2,
        "total_duration": step_duration * 2,
    }


@workflow
async def step_failure_workflow(ctx: WorkflowContext, failing_step_name: str = "step2") -> dict:
    """
    Workflow where a specific step fails.

    Args:
        failing_step_name: Name of step that should fail

    Returns:
        Partial results before failure

    Raises:
        RuntimeError: When failing step is reached
    """
    ctx.logger.info(f"Starting workflow, step '{failing_step_name}' will fail")

    # Step 1
    if failing_step_name == "step1":
        await ctx.step("step1", lambda: None or (_ for _ in ()).throw(RuntimeError("Step 1 failed")))
    else:
        step1 = await ctx.step("step1", lambda: {"status": "success"})
        ctx.logger.info("Step 1 succeeded")

    # Step 2
    if failing_step_name == "step2":
        def failing_lambda():
            raise RuntimeError("Step 2 failed")
        await ctx.step("step2", failing_lambda)
    else:
        step2 = await ctx.step("step2", lambda: {"status": "success"})
        ctx.logger.info("Step 2 succeeded")

    # Step 3
    if failing_step_name == "step3":
        def failing_lambda():
            raise RuntimeError("Step 3 failed")
        await ctx.step("step3", failing_lambda)
    else:
        step3 = await ctx.step("step3", lambda: {"status": "success"})
        ctx.logger.info("Step 3 succeeded")

    return {"status": "completed"}


@workflow
async def corrupted_checkpoint_workflow(ctx: WorkflowContext) -> dict:
    """
    Workflow for testing checkpoint corruption recovery.

    Returns:
        Dict with checkpoint info
    """
    ctx.logger.info("Starting workflow with checkpoints")

    # Step 1
    step1 = await ctx.step("checkpoint1", lambda: {"data": "checkpoint_1", "valid": True})

    # Step 2 (where corruption might occur in tests)
    step2 = await ctx.step("checkpoint2", lambda: {"data": "checkpoint_2", "valid": True})

    # Step 3
    step3 = await ctx.step("checkpoint3", lambda: {"data": "checkpoint_3", "valid": True})

    return {
        "checkpoints": [step1, step2, step3],
        "status": "completed",
    }


# =============================================================================
# EDGE CASE WORKFLOWS
# =============================================================================


@workflow
async def empty_workflow(ctx: WorkflowContext) -> dict:
    """
    Workflow with no steps (edge case).

    Returns:
        Simple completion dict
    """
    ctx.logger.info("Empty workflow executed")
    return {"status": "completed", "steps": 0}


@workflow
async def single_step_workflow(ctx: WorkflowContext, value: Any) -> dict:
    """
    Workflow with only one step.

    Args:
        value: Input value

    Returns:
        Dict with single step result
    """
    result = await ctx.step("only_step", lambda: {"value": value, "processed": True})
    return result


@workflow
async def very_long_workflow(ctx: WorkflowContext, steps: int = 100) -> dict:
    """
    Very long workflow for stress testing.

    Args:
        steps: Number of steps (default: 100)

    Returns:
        Dict with aggregated results
    """
    ctx.logger.info(f"Starting very long workflow with {steps} steps")

    results = []
    for i in range(steps):
        result = await ctx.step(f"step_{i}", lambda i=i: {"step": i})
        results.append(result)

        # Log progress every 10 steps
        if (i + 1) % 10 == 0:
            ctx.logger.info(f"Progress: {i+1}/{steps} steps")

    return {
        "total_steps": steps,
        "completed": True,
    }


@workflow
async def conditional_workflow(ctx: WorkflowContext, condition: bool, branch: str = "a") -> dict:
    """
    Workflow with conditional logic.

    Args:
        condition: Boolean condition for branching
        branch: Which branch to take ("a" or "b")

    Returns:
        Dict with branch results
    """
    ctx.logger.info(f"Starting conditional workflow: condition={condition}, branch={branch}")

    # Common step
    init = await ctx.step("init", lambda: {"initialized": True})

    # Conditional branching
    if condition:
        branch_result = await ctx.step(
            "branch_true",
            lambda: {"condition": True, "branch": branch, "result": "true_branch"}
        )
    else:
        branch_result = await ctx.step(
            "branch_false",
            lambda: {"condition": False, "branch": branch, "result": "false_branch"}
        )

    # Final step
    final = await ctx.step("final", lambda: {"status": "completed"})

    return {
        "condition": condition,
        "branch": branch,
        "branch_result": branch_result,
        "final": final,
    }


@workflow
async def nested_data_workflow(ctx: WorkflowContext, data: dict) -> dict:
    """
    Workflow with nested data structures.

    Args:
        data: Nested dictionary input

    Returns:
        Dict with processed nested data
    """
    ctx.logger.info(f"Processing nested data with {len(data)} top-level keys")

    # Step 1: Validate structure
    validation = await ctx.step(
        "validate",
        lambda: {"valid": isinstance(data, dict), "keys": list(data.keys())}
    )

    # Step 2: Process
    processed = await ctx.step(
        "process",
        lambda: {
            "data": data,
            "flattened": str(data),
            "key_count": len(data),
        }
    )

    return {
        "validation": validation,
        "processed": processed,
        "status": "completed",
    }
