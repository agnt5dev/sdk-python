"""
Example: AGNT5 Parallel and Sequential Tool Calls

This example demonstrates tool execution patterns:
- Sequential tool calls: Tools run one after another, each depends on previous
- Parallel tool calls: Multiple independent tools run concurrently
- Mixed mode: Some tools parallel, some sequential
- Dependency resolution: Execute tools based on dependency graph

Tool execution patterns enable:
- Optimized performance through parallelization
- Correct ordering when dependencies exist
- Flexible orchestration strategies
- Efficient resource utilization

Each workflow/function can be:
- Run standalone: python ex_14_parallel_tool_calls.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("parallel_tool_workflow", {...}, component_type="workflow")

Prerequisites:
- Set OPENAI_API_KEY for agent-based examples
"""

import asyncio
import os
import time
from typing import Any, Dict, List

from agnt5 import Agent, Context, FunctionContext, WorkflowContext, function, tool, workflow


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@tool
async def fetch_data(ctx: Context, source: str) -> dict:
    """
    Fetch data from a source (simulated).

    Args:
        source: Data source identifier

    Returns:
        Fetched data dictionary
    """
    ctx.logger.info(f"[fetch_data] Fetching from {source}...")
    await asyncio.sleep(0.1)  # Simulate I/O
    return {
        "source": source,
        "data": [1, 2, 3, 4, 5],
        "timestamp": time.time(),
    }


@tool
async def transform_data(ctx: Context, data: List[int], operation: str = "double") -> dict:
    """
    Transform data with specified operation.

    Args:
        data: List of numbers to transform
        operation: Transformation operation (double, square, negate)

    Returns:
        Transformed data dictionary
    """
    ctx.logger.info(f"[transform_data] Transforming with {operation}...")
    await asyncio.sleep(0.05)  # Simulate processing

    if operation == "double":
        result = [x * 2 for x in data]
    elif operation == "square":
        result = [x ** 2 for x in data]
    elif operation == "negate":
        result = [-x for x in data]
    else:
        result = data

    return {
        "operation": operation,
        "input": data,
        "output": result,
    }


@tool
async def aggregate_data(ctx: Context, datasets: List[Dict]) -> dict:
    """
    Aggregate multiple datasets.

    Args:
        datasets: List of dataset dictionaries to aggregate

    Returns:
        Aggregated result dictionary
    """
    ctx.logger.info(f"[aggregate_data] Aggregating {len(datasets)} datasets...")
    await asyncio.sleep(0.05)

    all_values = []
    for ds in datasets:
        if "output" in ds:
            all_values.extend(ds["output"])
        elif "data" in ds:
            all_values.extend(ds["data"])

    return {
        "total_count": len(all_values),
        "sum": sum(all_values),
        "average": sum(all_values) / len(all_values) if all_values else 0,
        "datasets_merged": len(datasets),
    }


@tool
async def validate_result(ctx: Context, data: Dict) -> dict:
    """
    Validate processing result.

    Args:
        data: Data to validate

    Returns:
        Validation result dictionary
    """
    ctx.logger.info(f"[validate_result] Validating...")
    await asyncio.sleep(0.02)

    is_valid = "sum" in data or "output" in data
    return {
        "valid": is_valid,
        "checked_fields": list(data.keys()),
        "message": "Validation passed" if is_valid else "Missing required fields",
    }


@tool
async def format_output(ctx: Context, data: Dict, format_type: str = "json") -> str:
    """
    Format data for output.

    Args:
        data: Data to format
        format_type: Output format (json, text, csv)

    Returns:
        Formatted string output
    """
    ctx.logger.info(f"[format_output] Formatting as {format_type}...")
    await asyncio.sleep(0.02)

    if format_type == "json":
        import json
        return json.dumps(data, indent=2)
    elif format_type == "text":
        return "\n".join(f"{k}: {v}" for k, v in data.items())
    elif format_type == "csv":
        return ",".join(str(v) for v in data.values())
    return str(data)


@tool
async def slow_operation(ctx: Context, name: str, delay_ms: int = 100) -> dict:
    """
    Simulated slow operation for timing tests.

    Args:
        name: Operation name
        delay_ms: Delay in milliseconds

    Returns:
        Operation result with timing
    """
    start = time.time()
    ctx.logger.info(f"[slow_operation] Starting {name} (delay: {delay_ms}ms)...")
    await asyncio.sleep(delay_ms / 1000)
    elapsed = (time.time() - start) * 1000

    return {
        "name": name,
        "delay_ms": delay_ms,
        "actual_ms": round(elapsed, 2),
        "completed": True,
    }


# =============================================================================
# SEQUENTIAL TOOL PATTERNS
# =============================================================================


@workflow
async def sequential_tool_chain(ctx: WorkflowContext, source: str) -> dict:
    """
    Tools run one after another, each depending on the previous.

    Demonstrates strict sequential execution where each step
    needs the result of the previous step.

    Args:
        source: Initial data source

    Returns:
        Dict with chain results

    Example:
        result = client.run("sequential_tool_chain", {
            "source": "database"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting sequential tool chain from: {source}")
    start_time = time.time()

    # Step 1: Fetch data
    fetched = await fetch_data(ctx, source=source)
    ctx.logger.info(f"Fetched: {fetched}")

    # Step 2: Transform data (depends on step 1)
    transformed = await transform_data(ctx, data=fetched["data"], operation="double")
    ctx.logger.info(f"Transformed: {transformed}")

    # Step 3: Validate (depends on step 2)
    validation = await validate_result(ctx, data=transformed)
    ctx.logger.info(f"Validation: {validation}")

    # Step 4: Format output (depends on step 3)
    formatted = await format_output(ctx, data={
        "source": source,
        "transformed": transformed["output"],
        "valid": validation["valid"],
    }, format_type="text")
    ctx.logger.info(f"Formatted output ready")

    elapsed = (time.time() - start_time) * 1000

    return {
        "source": source,
        "steps": ["fetch", "transform", "validate", "format"],
        "final_output": formatted,
        "validation": validation,
        "execution_time_ms": round(elapsed, 2),
        "pattern": "sequential_chain",
    }


@workflow
async def conditional_tool_sequence(ctx: WorkflowContext, data: List[int], threshold: float = 10.0) -> dict:
    """
    Next tool depends on the result of the previous tool.

    Demonstrates conditional branching based on tool results.

    Args:
        data: Input data to process
        threshold: Threshold for conditional branching

    Returns:
        Dict with conditional processing results

    Example:
        result = client.run("conditional_tool_sequence", {
            "data": [1, 2, 3, 4, 5],
            "threshold": 10.0
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting conditional sequence with threshold: {threshold}")

    # Step 1: Transform data
    transformed = await transform_data(ctx, data=data, operation="double")

    # Step 2: Aggregate to check condition
    aggregated = await aggregate_data(ctx, datasets=[transformed])

    # Conditional: Choose next step based on result
    if aggregated["average"] > threshold:
        ctx.logger.info(f"Average {aggregated['average']} > threshold, applying square")
        # Apply additional transformation
        final_transform = await transform_data(ctx, data=transformed["output"], operation="square")
        branch = "high_value"
    else:
        ctx.logger.info(f"Average {aggregated['average']} <= threshold, applying negate")
        final_transform = await transform_data(ctx, data=transformed["output"], operation="negate")
        branch = "low_value"

    # Step 3: Validate final result
    validation = await validate_result(ctx, data=final_transform)

    return {
        "input_data": data,
        "threshold": threshold,
        "intermediate_average": aggregated["average"],
        "branch_taken": branch,
        "final_output": final_transform["output"],
        "validation": validation,
        "pattern": "conditional_sequence",
    }


@workflow
async def ordered_tool_pipeline(ctx: WorkflowContext, sources: List[str]) -> dict:
    """
    Explicit ordering with dependencies between tool calls.

    Each source is processed sequentially, maintaining order guarantees.

    Args:
        sources: List of data sources to process in order

    Returns:
        Dict with ordered processing results

    Example:
        result = client.run("ordered_tool_pipeline", {
            "sources": ["db1", "db2", "db3"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting ordered pipeline for {len(sources)} sources")

    all_results = []
    cumulative_sum = 0

    for i, source in enumerate(sources):
        ctx.logger.info(f"Processing source {i+1}/{len(sources)}: {source}")

        # Fetch data from this source
        fetched = await fetch_data(ctx, source=source)

        # Transform with awareness of previous cumulative sum
        transformed = await transform_data(
            ctx,
            data=fetched["data"],
            operation="double" if cumulative_sum < 50 else "square"
        )

        # Update cumulative sum (order matters!)
        cumulative_sum += sum(transformed["output"])

        all_results.append({
            "source": source,
            "order": i + 1,
            "transformed": transformed["output"],
            "cumulative_sum": cumulative_sum,
        })

    # Final aggregation respects order
    final = await aggregate_data(ctx, datasets=[{"output": r["transformed"]} for r in all_results])

    return {
        "sources": sources,
        "results": all_results,
        "final_aggregation": final,
        "processing_order_preserved": True,
        "pattern": "ordered_pipeline",
    }


# =============================================================================
# PARALLEL TOOL PATTERNS
# =============================================================================


@workflow
async def parallel_tool_execution(ctx: WorkflowContext, sources: List[str]) -> dict:
    """
    Multiple independent tools run concurrently.

    Demonstrates parallelization for improved performance.

    Args:
        sources: List of independent data sources

    Returns:
        Dict with parallel execution results

    Example:
        result = client.run("parallel_tool_execution", {
            "sources": ["api1", "api2", "api3", "api4"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting parallel execution for {len(sources)} sources")
    start_time = time.time()

    # Execute all fetches in parallel (no dependencies between them)
    fetch_tasks = [fetch_data(ctx, source=source) for source in sources]
    fetched_results = await asyncio.gather(*fetch_tasks)

    # Execute all transformations in parallel
    transform_tasks = [
        transform_data(ctx, data=result["data"], operation="double")
        for result in fetched_results
    ]
    transformed_results = await asyncio.gather(*transform_tasks)

    # Aggregate all results
    aggregated = await aggregate_data(ctx, datasets=transformed_results)

    elapsed = (time.time() - start_time) * 1000

    return {
        "sources": sources,
        "parallel_fetches": len(fetched_results),
        "parallel_transforms": len(transformed_results),
        "aggregated": aggregated,
        "execution_time_ms": round(elapsed, 2),
        "pattern": "parallel_execution",
    }


@workflow
async def mixed_parallel_sequential(ctx: WorkflowContext, items: List[str]) -> dict:
    """
    Some tools parallel, some sequential based on dependencies.

    Stage 1: Parallel fetch
    Stage 2: Sequential transform (each depends on aggregate of previous)
    Stage 3: Parallel validation

    Args:
        items: Items to process

    Returns:
        Dict with mixed execution results

    Example:
        result = client.run("mixed_parallel_sequential", {
            "items": ["a", "b", "c", "d"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting mixed execution for {len(items)} items")
    start_time = time.time()

    # STAGE 1: Parallel slow operations (independent)
    ctx.logger.info("Stage 1: Parallel slow operations")
    stage1_tasks = [
        slow_operation(ctx, name=f"fetch_{item}", delay_ms=50)
        for item in items
    ]
    stage1_results = await asyncio.gather(*stage1_tasks)
    stage1_time = (time.time() - start_time) * 1000

    # STAGE 2: Sequential dependent operations
    ctx.logger.info("Stage 2: Sequential dependent operations")
    stage2_start = time.time()
    cumulative = 0
    stage2_results = []
    for i, result in enumerate(stage1_results):
        # Each operation depends on cumulative state
        op_result = await slow_operation(ctx, name=f"process_{i}", delay_ms=30)
        cumulative += op_result["actual_ms"]
        stage2_results.append({**op_result, "cumulative_ms": cumulative})
    stage2_time = (time.time() - stage2_start) * 1000

    # STAGE 3: Parallel validation (independent again)
    ctx.logger.info("Stage 3: Parallel validation")
    stage3_start = time.time()
    stage3_tasks = [
        slow_operation(ctx, name=f"validate_{i}", delay_ms=20)
        for i in range(len(items))
    ]
    stage3_results = await asyncio.gather(*stage3_tasks)
    stage3_time = (time.time() - stage3_start) * 1000

    total_time = (time.time() - start_time) * 1000

    return {
        "items": items,
        "stage1": {
            "pattern": "parallel",
            "results": stage1_results,
            "time_ms": round(stage1_time, 2),
        },
        "stage2": {
            "pattern": "sequential",
            "results": stage2_results,
            "time_ms": round(stage2_time, 2),
        },
        "stage3": {
            "pattern": "parallel",
            "results": stage3_results,
            "time_ms": round(stage3_time, 2),
        },
        "total_time_ms": round(total_time, 2),
        "pattern": "mixed_parallel_sequential",
    }


@workflow
async def batch_tool_operations(ctx: WorkflowContext, numbers: List[int], batch_size: int = 3) -> dict:
    """
    Run same tool with different inputs in parallel batches.

    Demonstrates batched parallel execution for load control.

    Args:
        numbers: Numbers to process
        batch_size: Number of parallel operations per batch

    Returns:
        Dict with batched processing results

    Example:
        result = client.run("batch_tool_operations", {
            "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "batch_size": 3
        }, component_type="workflow")
    """
    ctx.logger.info(f"Processing {len(numbers)} numbers in batches of {batch_size}")
    start_time = time.time()

    # Split into batches
    batches = [numbers[i:i + batch_size] for i in range(0, len(numbers), batch_size)]
    ctx.logger.info(f"Created {len(batches)} batches")

    all_results = []

    for batch_idx, batch in enumerate(batches):
        ctx.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}: {batch}")

        # Process batch in parallel
        batch_tasks = [
            transform_data(ctx, data=[n], operation="square")
            for n in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)

    elapsed = (time.time() - start_time) * 1000

    # Flatten results
    all_outputs = [r["output"][0] for r in all_results]

    return {
        "input_numbers": numbers,
        "batch_size": batch_size,
        "num_batches": len(batches),
        "all_outputs": all_outputs,
        "sum_of_squares": sum(all_outputs),
        "execution_time_ms": round(elapsed, 2),
        "pattern": "batch_parallel",
    }


# =============================================================================
# DEPENDENCY RESOLUTION
# =============================================================================


@workflow
async def tool_dependency_graph(ctx: WorkflowContext, input_value: int) -> dict:
    """
    Execute tools based on dependency DAG.

    Dependency graph:
        A (no deps) -> C (depends on A, B)
        B (no deps) -> C
        C -> D (depends on C)

    A and B run in parallel, then C, then D.

    Args:
        input_value: Initial value for processing

    Returns:
        Dict with dependency-ordered results

    Example:
        result = client.run("tool_dependency_graph", {
            "input_value": 10
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting dependency graph execution with input: {input_value}")
    start_time = time.time()

    # Level 0: A and B have no dependencies (parallel)
    async def task_a():
        result = await slow_operation(ctx, name="task_A", delay_ms=100)
        return {**result, "output": input_value * 2}

    async def task_b():
        result = await slow_operation(ctx, name="task_B", delay_ms=80)
        return {**result, "output": input_value + 5}

    ctx.logger.info("Level 0: Running A and B in parallel")
    result_a, result_b = await asyncio.gather(task_a(), task_b())

    # Level 1: C depends on A and B
    async def task_c():
        combined = result_a["output"] + result_b["output"]
        result = await slow_operation(ctx, name="task_C", delay_ms=60)
        return {**result, "output": combined, "inputs": [result_a["output"], result_b["output"]]}

    ctx.logger.info("Level 1: Running C (depends on A, B)")
    result_c = await task_c()

    # Level 2: D depends on C
    async def task_d():
        final = result_c["output"] * 2
        result = await slow_operation(ctx, name="task_D", delay_ms=40)
        return {**result, "output": final, "input": result_c["output"]}

    ctx.logger.info("Level 2: Running D (depends on C)")
    result_d = await task_d()

    elapsed = (time.time() - start_time) * 1000

    return {
        "input_value": input_value,
        "execution_order": ["A||B", "C", "D"],
        "results": {
            "A": result_a,
            "B": result_b,
            "C": result_c,
            "D": result_d,
        },
        "final_output": result_d["output"],
        "execution_time_ms": round(elapsed, 2),
        "pattern": "dependency_graph",
    }


@workflow(chat=True)
async def agent_parallel_tools(ctx: WorkflowContext, query: str) -> dict:
    """
    Agent that can make parallel tool calls.

    The agent is instructed to gather information from multiple
    sources concurrently when possible.

    Args:
        query: Query requiring multiple data sources

    Returns:
        Dict with agent results and tool call info

    Example:
        result = client.run("agent_parallel_tools", {
            "query": "Get data from all three sources and combine them"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting agent with parallel tools for: {query}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "query": query,
        }

    # Create agent with multiple tools it can call in parallel
    data_agent = Agent(
        name="data_gatherer",
        model="openai/gpt-4o-mini",
        instructions="""You are a data gathering agent. When asked to get data from multiple sources,
        you should call the fetch_data tool multiple times - the system will execute them in parallel.
        After gathering data, use aggregate_data to combine the results.
        Be efficient and gather all needed data in a single round of tool calls when possible.""",
        tools=[fetch_data, aggregate_data, transform_data],
        temperature=0.3,
        max_iterations=10,
    )

    start_time = time.time()
    result = await data_agent.run_sync(query, context=ctx)
    elapsed = (time.time() - start_time) * 1000

    # Analyze tool calls
    tool_call_summary = {}
    for tc in result.tool_calls:
        name = tc.get("name", "unknown")
        tool_call_summary[name] = tool_call_summary.get(name, 0) + 1

    return {
        "query": query,
        "agent_output": result.output,
        "tool_calls_made": len(result.tool_calls),
        "tool_call_breakdown": tool_call_summary,
        "execution_time_ms": round(elapsed, 2),
        "pattern": "agent_parallel_tools",
    }


# =============================================================================
# TIMING COMPARISON
# =============================================================================


@workflow
async def timing_comparison(ctx: WorkflowContext, num_operations: int = 5) -> dict:
    """
    Compare sequential vs parallel execution timing.

    Runs the same set of operations both ways to show performance difference.

    Args:
        num_operations: Number of slow operations to run

    Returns:
        Dict with timing comparison

    Example:
        result = client.run("timing_comparison", {
            "num_operations": 5
        }, component_type="workflow")
    """
    ctx.logger.info(f"Timing comparison with {num_operations} operations")
    delay_per_op = 50  # ms

    # Sequential execution
    ctx.logger.info("Running SEQUENTIAL...")
    seq_start = time.time()
    for i in range(num_operations):
        await slow_operation(ctx, name=f"seq_{i}", delay_ms=delay_per_op)
    seq_time = (time.time() - seq_start) * 1000

    # Parallel execution
    ctx.logger.info("Running PARALLEL...")
    par_start = time.time()
    tasks = [
        slow_operation(ctx, name=f"par_{i}", delay_ms=delay_per_op)
        for i in range(num_operations)
    ]
    await asyncio.gather(*tasks)
    par_time = (time.time() - par_start) * 1000

    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0

    return {
        "num_operations": num_operations,
        "delay_per_op_ms": delay_per_op,
        "sequential_time_ms": round(seq_time, 2),
        "parallel_time_ms": round(par_time, 2),
        "speedup_factor": round(speedup, 2),
        "expected_sequential_ms": num_operations * delay_per_op,
        "expected_parallel_ms": delay_per_op,  # All run at once
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Parallel and Sequential Tool Calls")
    print("=" * 60)

    ctx = Context(
        run_id="tool-calls-demo",
        correlation_id="tool-calls-demo",
        parent_correlation_id="",
    )

    # Sequential chain
    print("\n--- Sequential Tool Chain ---")
    result = await sequential_tool_chain(ctx, source="database")
    print(f"Execution time: {result['execution_time_ms']}ms")

    # Conditional sequence
    print("\n--- Conditional Tool Sequence ---")
    result = await conditional_tool_sequence(ctx, data=[1, 2, 3, 4, 5], threshold=5.0)
    print(f"Branch taken: {result['branch_taken']}")

    # Parallel execution
    print("\n--- Parallel Tool Execution ---")
    result = await parallel_tool_execution(ctx, sources=["api1", "api2", "api3"])
    print(f"Execution time: {result['execution_time_ms']}ms")

    # Mixed pattern
    print("\n--- Mixed Parallel/Sequential ---")
    result = await mixed_parallel_sequential(ctx, items=["a", "b", "c", "d"])
    print(f"Total time: {result['total_time_ms']}ms")

    # Batch operations
    print("\n--- Batch Tool Operations ---")
    result = await batch_tool_operations(ctx, numbers=[1, 2, 3, 4, 5, 6], batch_size=2)
    print(f"Sum of squares: {result['sum_of_squares']}")

    # Dependency graph
    print("\n--- Dependency Graph ---")
    result = await tool_dependency_graph(ctx, input_value=10)
    print(f"Final output: {result['final_output']}")

    # Timing comparison
    print("\n--- Timing Comparison ---")
    result = await timing_comparison(ctx, num_operations=5)
    print(f"Sequential: {result['sequential_time_ms']}ms, Parallel: {result['parallel_time_ms']}ms")
    print(f"Speedup: {result['speedup_factor']}x")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
