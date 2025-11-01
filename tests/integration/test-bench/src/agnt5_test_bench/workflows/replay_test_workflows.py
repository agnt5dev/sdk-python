"""
Workflow Replay Test Suite

This module contains workflows specifically designed to test and verify
the Phase 1 workflow replay functionality. These workflows help ensure that:
1. Completed steps are cached correctly
2. Retries use cached steps instead of re-executing
3. State is preserved across crash recovery
4. Replay provides significant time and cost savings

Usage:
    # Start test-bench dev server
    just platform start-dev-server python

    # Invoke crash test workflow
    curl -X POST http://localhost:34181/run \\
      -H "Content-Type: application/json" \\
      -d '{"serviceName": "agnt5-test-bench",
           "componentName": "workflow_that_crashes_at_step",
           "componentType": "workflow",
           "inputData": "{\\"crash_at_step\\": 3}"}'

    # Retry workflow (after it crashes)
    curl -X POST http://localhost:34181/retry/{run_id}
"""

from agnt5 import workflow, function, WorkflowContext, FunctionContext
import time
import asyncio
from typing import Optional, Dict, Any
import json


# ============================================================================
# Helper Functions for Testing Replay
# ============================================================================

@function
async def expensive_llm_call(ctx: FunctionContext, task: str, delay_seconds: int = 2) -> str:
    """
    Simulates an expensive LLM API call with configurable delay.

    Args:
        task: Description of the LLM task
        delay_seconds: Simulated API latency in seconds

    Returns:
        Result string with task completion info
    """
    start_time = time.time()
    ctx.logger.info(f"💰 Executing expensive LLM call: {task} (${delay_seconds * 0.25:.2f})")
    await asyncio.sleep(delay_seconds)
    elapsed = time.time() - start_time

    result = f"LLM Result: {task} (took {elapsed:.2f}s, cost $0.{delay_seconds * 25:02d})"
    ctx.logger.info(f"✅ Completed: {result}")
    return result


@function
async def simple_step(ctx: FunctionContext, step_num: int, data: Optional[str] = None) -> Dict[str, Any]:
    """
    A simple step that can be cached and replayed.

    Args:
        step_num: Step number in sequence
        data: Optional data to process

    Returns:
        Dictionary with step results
    """
    start_time = time.time()
    ctx.logger.info(f"📌 Executing step {step_num}")

    # Simulate some work
    await asyncio.sleep(0.5)

    elapsed = time.time() - start_time
    result = {
        "step": step_num,
        "status": "completed",
        "elapsed_ms": int(elapsed * 1000),
        "data": data or f"result_{step_num}",
        "timestamp": time.time()
    }

    ctx.logger.info(f"✅ Step {step_num} completed in {elapsed * 1000:.1f}ms")
    return result


@function
async def crash_step(ctx: FunctionContext, step_num: int, message: str) -> str:
    """
    A step that intentionally crashes to test replay recovery.

    Args:
        step_num: Step number where crash occurs
        message: Crash message

    Raises:
        RuntimeError: Always raises to simulate crash
    """
    ctx.logger.error(f"💥 CRASH at step {step_num}: {message}")
    raise RuntimeError(f"Intentional crash at step {step_num}: {message}")


@function
async def state_mutating_step(
    ctx: FunctionContext,
    step_num: int,
    key: str,
    value: Any
) -> Dict[str, Any]:
    """
    A step that mutates state and returns info about it.

    Args:
        step_num: Step number
        key: State key to set
        value: State value to set

    Returns:
        Dictionary with state mutation info
    """
    ctx.logger.info(f"🔧 Step {step_num}: Setting state[{key}] = {value}")

    # Note: We can't directly mutate workflow state from a function,
    # but we can return data that the workflow will use
    result = {
        "step": step_num,
        "state_mutation": {
            "key": key,
            "value": value
        },
        "timestamp": time.time()
    }

    ctx.logger.info(f"✅ Step {step_num}: State mutation prepared")
    return result


# ============================================================================
# Test Workflow 1: Expensive Steps (Time & Cost Savings)
# ============================================================================

@workflow
async def workflow_with_expensive_steps(
    ctx: WorkflowContext,
    step_count: int = 5,
    delay_per_step: int = 2
) -> Dict[str, Any]:
    """
    Workflow with expensive LLM-like operations to demonstrate replay cost savings.

    This workflow simulates a multi-step AI workflow with expensive LLM API calls.
    On retry, cached steps should return instantly, saving time and money.

    Args:
        step_count: Number of expensive steps to execute (default: 5)
        delay_per_step: Delay in seconds per step to simulate API latency (default: 2)

    Returns:
        Summary with execution time, cost, and replay statistics

    Expected Behavior:
        First Run: 5 steps × 2s = 10s total, cost = $2.50
        Retry: cached steps return in ~5ms total, cost = $0.00
        Savings: ~10s and $2.50

    Test Instructions:
        1. Run workflow normally - should take ~10 seconds
        2. Crash it manually or let it complete
        3. Retry workflow
        4. Should complete in < 1 second using cached steps
    """
    start_time = time.time()
    ctx.logger.info("=" * 60)
    ctx.logger.info(f"Starting Expensive Workflow: {step_count} steps @ {delay_per_step}s each")
    ctx.logger.info("=" * 60)

    # Track execution
    results = []
    total_cost = 0.0
    cached_steps = 0

    # Execute expensive steps
    for i in range(1, step_count + 1):
        step_start = time.time()

        task_description = f"Analyze data chunk {i}/{step_count}"
        result = await ctx.task(expensive_llm_call, task_description, delay_per_step)

        step_elapsed = time.time() - step_start

        # Detect if step was cached (< 100ms indicates replay)
        was_cached = step_elapsed < 0.1
        if was_cached:
            cached_steps += 1
            ctx.logger.info(f"⚡ Step {i} returned from cache ({step_elapsed * 1000:.1f}ms)")
        else:
            total_cost += delay_per_step * 0.25

        results.append({
            "step": i,
            "result": result,
            "elapsed_ms": int(step_elapsed * 1000),
            "cached": was_cached
        })

    # Calculate summary
    total_elapsed = time.time() - start_time
    estimated_savings = (step_count - cached_steps) * delay_per_step * 0.25

    summary = {
        "workflow": "workflow_with_expensive_steps",
        "total_steps": step_count,
        "cached_steps": cached_steps,
        "new_steps": step_count - cached_steps,
        "total_time_seconds": round(total_elapsed, 2),
        "total_cost_usd": round(total_cost, 2),
        "estimated_savings_usd": round(estimated_savings, 2),
        "results": results
    }

    ctx.logger.info("=" * 60)
    ctx.logger.info("WORKFLOW SUMMARY:")
    ctx.logger.info(f"  Total Time: {total_elapsed:.2f}s")
    ctx.logger.info(f"  Total Cost: ${total_cost:.2f}")
    ctx.logger.info(f"  Cached Steps: {cached_steps}/{step_count}")
    if cached_steps > 0:
        ctx.logger.info(f"  💰 Savings: ${estimated_savings:.2f} ({cached_steps * 100 // step_count}%)")
    ctx.logger.info("=" * 60)

    # Store summary in workflow state for later inspection
    ctx.state.set("execution_summary", summary)

    return summary


# ============================================================================
# Test Workflow 2: Crash and Recovery (Primary Replay Test)
# ============================================================================

@workflow
async def workflow_that_crashes_at_step(
    ctx: WorkflowContext,
    crash_at_step: int = 3,
    total_steps: int = 5
) -> Dict[str, Any]:
    """
    Workflow that intentionally crashes at a specific step to test replay recovery.

    This is the PRIMARY test for workflow replay functionality. It crashes at a
    configurable step, allowing you to verify that:
    1. Steps before crash are recorded in database
    2. Retry loads cached steps from database
    3. Cached steps return instantly without re-execution
    4. Workflow completes successfully after retry

    Args:
        crash_at_step: Which step number to crash at (1-based, default: 3)
        total_steps: Total number of steps in workflow (default: 5)

    Returns:
        Summary of execution including which steps were cached

    Expected Behavior:
        First Run:
          - Steps 1-2 execute successfully
          - Step 3 crashes with RuntimeError
          - Workflow status = "failed"
          - 2 steps recorded in database

        Retry Run:
          - Platform loads 2 completed steps
          - SDK logs "🔄 Replaying workflow with 2 cached steps"
          - Steps 1-2 return from cache (< 100ms)
          - Steps 3-5 execute normally
          - Workflow status = "completed"

    Test Instructions:
        1. Run: crash_at_step=3, total_steps=5
        2. Verify failure and 2 steps in database
        3. Retry workflow with same run_id
        4. Verify replay logs and success
    """
    start_time = time.time()
    ctx.logger.info("=" * 60)
    ctx.logger.info(f"Crash Test Workflow: Will crash at step {crash_at_step}/{total_steps}")
    ctx.logger.info("=" * 60)

    # Check if this is a retry by looking for cached steps
    workflow_entity = ctx._workflow_entity
    cached_count = len(workflow_entity._completed_steps)

    if cached_count > 0:
        ctx.logger.info(f"🔄 REPLAY DETECTED: {cached_count} cached steps available")
        ctx.logger.info(f"   Cached steps: {list(workflow_entity._completed_steps.keys())}")
    else:
        ctx.logger.info("▶️  FIRST RUN: No cached steps")

    # Track execution
    results = []

    # Execute steps
    for step_num in range(1, total_steps + 1):
        step_start = time.time()

        ctx.logger.info(f"\n--- Step {step_num}/{total_steps} ---")

        # Check if we should crash at this step
        if step_num == crash_at_step:
            # Only crash on first attempt (not on retry)
            if cached_count == 0:
                ctx.logger.warning(f"⚠️  About to crash at step {step_num} (as requested)")
                try:
                    await ctx.task(
                        crash_step,
                        step_num,
                        f"Testing crash recovery - step {step_num}"
                    )
                except Exception as e:
                    ctx.logger.error(f"💥 Workflow crashed as expected: {e}")
                    raise  # Re-raise to mark workflow as failed
            else:
                ctx.logger.info(f"🔄 RETRY: Skipping crash at step {step_num} (was cached: {cached_count > 0})")

        # Execute normal step
        result = await ctx.task(simple_step, step_num, f"data_for_step_{step_num}")

        step_elapsed = time.time() - step_start
        was_cached = step_elapsed < 0.1

        if was_cached:
            ctx.logger.info(f"⚡ Step {step_num} from cache ({step_elapsed * 1000:.1f}ms)")

        results.append({
            "step": step_num,
            "result": result,
            "elapsed_ms": int(step_elapsed * 1000),
            "cached": was_cached
        })

    # Success!
    total_elapsed = time.time() - start_time
    cached_steps = sum(1 for r in results if r["cached"])

    summary = {
        "workflow": "workflow_that_crashes_at_step",
        "total_steps": total_steps,
        "crash_at_step": crash_at_step,
        "cached_steps": cached_steps,
        "new_steps": total_steps - cached_steps,
        "total_time_seconds": round(total_elapsed, 2),
        "is_retry": cached_count > 0,
        "results": results
    }

    ctx.logger.info("\n" + "=" * 60)
    ctx.logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY")
    ctx.logger.info(f"   Total Time: {total_elapsed:.2f}s")
    ctx.logger.info(f"   Cached Steps: {cached_steps}/{total_steps}")
    ctx.logger.info(f"   Is Retry: {cached_count > 0}")
    ctx.logger.info("=" * 60)

    return summary


# ============================================================================
# Test Workflow 3: Mixed Step Types (Functions, State, etc.)
# ============================================================================

@workflow
async def workflow_with_mixed_steps(ctx: WorkflowContext) -> Dict[str, Any]:
    """
    Workflow with various step types to test replay compatibility.

    This workflow combines:
    - Regular function calls
    - State mutations
    - Different data types
    - Nested data structures

    All step types should be replayable on retry.

    Returns:
        Summary of execution with step types and replay info
    """
    start_time = time.time()
    ctx.logger.info("=" * 60)
    ctx.logger.info("Mixed Steps Workflow: Testing various step types")
    ctx.logger.info("=" * 60)

    results = []

    # Step 1: Simple function call
    ctx.logger.info("\n--- Step 1: Simple Function ---")
    result1 = await ctx.task(simple_step, 1, "simple_data")
    results.append(("simple_function", result1))
    ctx.state.set("step_1_result", result1)

    # Step 2: Expensive operation
    ctx.logger.info("\n--- Step 2: Expensive LLM Call ---")
    result2 = await ctx.task(expensive_llm_call, "Analyze user sentiment", 1)
    results.append(("expensive_llm", result2))
    ctx.state.set("step_2_result", result2)

    # Step 3: State mutation step
    ctx.logger.info("\n--- Step 3: State Mutation ---")
    result3 = await ctx.task(state_mutating_step, 3, "user_id", "user_123")
    results.append(("state_mutation", result3))
    # Apply the state mutation
    if "state_mutation" in result3:
        key = result3["state_mutation"]["key"]
        value = result3["state_mutation"]["value"]
        ctx.state.set(key, value)
        ctx.logger.info(f"   Applied: state[{key}] = {value}")

    # Step 4: Complex data structures
    ctx.logger.info("\n--- Step 4: Complex Data ---")
    complex_data = {
        "nested": {
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        },
        "array": ["a", "b", "c"]
    }
    result4 = await ctx.task(simple_step, 4, json.dumps(complex_data))
    results.append(("complex_data", result4))
    ctx.state.set("complex_result", complex_data)

    # Step 5: Final aggregation
    ctx.logger.info("\n--- Step 5: Aggregation ---")
    result5 = await ctx.task(simple_step, 5, "aggregation")
    results.append(("aggregation", result5))

    # Verify state consistency
    ctx.logger.info("\n--- Verifying State Consistency ---")
    assert ctx.state.get("user_id") == "user_123", "State mutation failed"
    assert ctx.state.get("step_1_result") is not None, "Step 1 result not in state"
    ctx.logger.info("✅ State consistency verified")

    total_elapsed = time.time() - start_time

    summary = {
        "workflow": "workflow_with_mixed_steps",
        "total_steps": len(results),
        "step_types": [step_type for step_type, _ in results],
        "total_time_seconds": round(total_elapsed, 2),
        "state_keys": list(ctx._workflow_entity.state._state.keys()),
        "results": [{"type": t, "data": r} for t, r in results]
    }

    ctx.logger.info("\n" + "=" * 60)
    ctx.logger.info("✅ MIXED STEPS WORKFLOW COMPLETED")
    ctx.logger.info(f"   Total Time: {total_elapsed:.2f}s")
    ctx.logger.info(f"   Step Types: {len(set(t for t, _ in results))}")
    ctx.logger.info("=" * 60)

    return summary


# ============================================================================
# Test Workflow 4: Heavy State Changes (State Consistency)
# ============================================================================

@workflow
async def workflow_with_state_changes(ctx: WorkflowContext, iterations: int = 5) -> Dict[str, Any]:
    """
    Workflow with heavy state manipulation to test state replay consistency.

    Each step depends on state from previous steps. On replay, state must be
    restored correctly for the workflow to complete successfully.

    Args:
        iterations: Number of state-mutating iterations (default: 5)

    Returns:
        Summary with state history and verification results
    """
    start_time = time.time()
    ctx.logger.info("=" * 60)
    ctx.logger.info(f"State Changes Workflow: {iterations} iterations")
    ctx.logger.info("=" * 60)

    # Initialize state
    ctx.state.set("counter", 0)
    ctx.state.set("accumulator", 0)
    ctx.state.set("history", [])

    results = []

    # Execute iterations
    for i in range(1, iterations + 1):
        ctx.logger.info(f"\n--- Iteration {i}/{iterations} ---")

        # Get current state
        counter = ctx.state.get("counter")
        accumulator = ctx.state.get("accumulator")
        history = ctx.state.get("history")

        ctx.logger.info(f"   Before: counter={counter}, accumulator={accumulator}")

        # Execute step that produces a value
        result = await ctx.task(simple_step, i, f"iteration_{i}")
        step_value = result["step"]

        # Update state based on result
        new_counter = counter + 1
        new_accumulator = accumulator + step_value
        new_history = history + [{"iteration": i, "value": step_value, "accumulated": new_accumulator}]

        ctx.state.set("counter", new_counter)
        ctx.state.set("accumulator", new_accumulator)
        ctx.state.set("history", new_history)

        ctx.logger.info(f"   After: counter={new_counter}, accumulator={new_accumulator}")

        results.append({
            "iteration": i,
            "counter": new_counter,
            "accumulator": new_accumulator
        })

    # Verify final state
    final_counter = ctx.state.get("counter")
    final_accumulator = ctx.state.get("accumulator")
    final_history = ctx.state.get("history")

    ctx.logger.info("\n--- State Verification ---")
    ctx.logger.info(f"   Final counter: {final_counter} (expected: {iterations})")
    ctx.logger.info(f"   Final accumulator: {final_accumulator}")
    ctx.logger.info(f"   History length: {len(final_history)}")

    # Assertions
    assert final_counter == iterations, f"Counter mismatch: {final_counter} != {iterations}"
    assert len(final_history) == iterations, f"History length mismatch: {len(final_history)} != {iterations}"

    # Calculate expected accumulator (sum of 1 to iterations)
    expected_accumulator = sum(range(1, iterations + 1))
    assert final_accumulator == expected_accumulator, \
        f"Accumulator mismatch: {final_accumulator} != {expected_accumulator}"

    ctx.logger.info("✅ State verification passed")

    total_elapsed = time.time() - start_time

    summary = {
        "workflow": "workflow_with_state_changes",
        "iterations": iterations,
        "final_counter": final_counter,
        "final_accumulator": final_accumulator,
        "history": final_history,
        "total_time_seconds": round(total_elapsed, 2),
        "results": results
    }

    ctx.logger.info("\n" + "=" * 60)
    ctx.logger.info("✅ STATE WORKFLOW COMPLETED")
    ctx.logger.info(f"   Total Time: {total_elapsed:.2f}s")
    ctx.logger.info(f"   State Consistency: VERIFIED")
    ctx.logger.info("=" * 60)

    return summary


# ============================================================================
# Test Workflow 5: Stress Test (Many Steps for Scale)
# ============================================================================

@workflow
async def workflow_replay_stress_test(
    ctx: WorkflowContext,
    step_count: int = 20,
    delay_ms: int = 100
) -> Dict[str, Any]:
    """
    Stress test workflow with many steps to test replay performance at scale.

    This workflow executes many lightweight steps to test:
    1. Replay performance with large step counts
    2. Memory usage with many cached steps
    3. Database query performance loading many entities

    Args:
        step_count: Number of steps to execute (default: 20)
        delay_ms: Delay per step in milliseconds (default: 100)

    Returns:
        Summary with performance metrics and replay statistics
    """
    start_time = time.time()
    ctx.logger.info("=" * 60)
    ctx.logger.info(f"Replay Stress Test: {step_count} steps @ {delay_ms}ms each")
    ctx.logger.info("=" * 60)

    # Check if replay
    cached_count = len(ctx._workflow_entity._completed_steps)
    if cached_count > 0:
        ctx.logger.info(f"🔄 REPLAY MODE: {cached_count} cached steps")

    results = []
    checkpoint_interval = max(5, step_count // 4)  # Log every ~25%

    # Execute many steps
    for i in range(1, step_count + 1):
        step_start = time.time()

        # Periodic logging
        if i % checkpoint_interval == 0 or i == 1:
            ctx.logger.info(f"📍 Progress: {i}/{step_count} ({i * 100 // step_count}%)")

        # Execute step with small delay
        await asyncio.sleep(delay_ms / 1000)
        result = await ctx.task(simple_step, i, f"data_{i}")

        step_elapsed = time.time() - step_start
        was_cached = step_elapsed < (delay_ms / 1000 / 2)  # Cache should be much faster

        results.append({
            "step": i,
            "elapsed_ms": int(step_elapsed * 1000),
            "cached": was_cached
        })

    total_elapsed = time.time() - start_time
    cached_steps = sum(1 for r in results if r["cached"])
    avg_step_time = total_elapsed / step_count

    # Performance metrics
    throughput = step_count / total_elapsed  # steps per second

    summary = {
        "workflow": "workflow_replay_stress_test",
        "total_steps": step_count,
        "cached_steps": cached_steps,
        "new_steps": step_count - cached_steps,
        "total_time_seconds": round(total_elapsed, 2),
        "avg_step_time_ms": round(avg_step_time * 1000, 2),
        "throughput_steps_per_sec": round(throughput, 2),
        "is_replay": cached_count > 0,
        "cache_hit_rate": round(cached_steps / step_count * 100, 1) if step_count > 0 else 0
    }

    ctx.logger.info("\n" + "=" * 60)
    ctx.logger.info("✅ STRESS TEST COMPLETED")
    ctx.logger.info(f"   Total Steps: {step_count}")
    ctx.logger.info(f"   Cached: {cached_steps}/{step_count} ({summary['cache_hit_rate']}%)")
    ctx.logger.info(f"   Total Time: {total_elapsed:.2f}s")
    ctx.logger.info(f"   Avg Step Time: {avg_step_time * 1000:.1f}ms")
    ctx.logger.info(f"   Throughput: {throughput:.1f} steps/sec")
    ctx.logger.info("=" * 60)

    return summary
