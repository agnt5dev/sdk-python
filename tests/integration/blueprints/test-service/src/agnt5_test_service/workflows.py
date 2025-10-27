"""
Test Workflows for Integration Testing

Provides multi-step workflows for testing durability and state management.
"""

import asyncio
import os
from agnt5 import Agent, Context, WorkflowContext, workflow
from agnt5_test_service.tools import calculate_total, format_report, search_database, validate_data


@workflow
async def order_fulfillment(ctx: Context, order_id: str, items: list = None) -> dict:
    """
    Multi-step order processing workflow.

    Tests:
    - Multi-step workflow execution
    - State persistence across steps
    - Error handling in workflows
    """
    if items is None:
        items = []

    ctx.logger.info(f"Starting order fulfillment for order {order_id}")

    # Step 1: Validate order
    ctx.logger.info("Step 1: Validating order")
    await asyncio.sleep(0.1)  # Simulate validation

    # Step 2: Process payment
    ctx.logger.info("Step 2: Processing payment")
    await asyncio.sleep(0.1)  # Simulate payment processing
    payment_id = f"pay_{order_id}"

    # Step 3: Reserve inventory
    ctx.logger.info("Step 3: Reserving inventory")
    await asyncio.sleep(0.1)  # Simulate inventory check

    # Step 4: Ship order
    ctx.logger.info("Step 4: Shipping order")
    await asyncio.sleep(0.1)  # Simulate shipping
    tracking_number = f"TRACK_{order_id}"

    return {
        "order_id": order_id,
        "status": "completed",
        "payment_id": payment_id,
        "tracking_number": tracking_number,
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
        "steps_completed": steps,
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


@workflow
async def tool_orchestrated_workflow(ctx: Context, numbers: list[float], query: str) -> dict:
    """
    Workflow that directly invokes tools for processing.

    Tests:
    - Direct tool invocation in workflows
    - Tool result handling
    - Multiple tool coordination
    """
    ctx.logger.info(f"Starting tool orchestrated workflow")

    # Step 1: Validate input data
    ctx.logger.info("Step 1: Validating input data")
    validation_result = await validate_data(ctx, data={"numbers": numbers, "query": query}, required_fields=["numbers", "query"])

    if not validation_result["is_valid"]:
        return {
            "status": "failed",
            "error": validation_result["message"],
        }

    # Step 2: Calculate statistics
    ctx.logger.info("Step 2: Calculating statistics")
    sum_result = await calculate_total(ctx, numbers=numbers, operation="sum")
    avg_result = await calculate_total(ctx, numbers=numbers, operation="average")

    # Step 3: Search database
    ctx.logger.info("Step 3: Searching database")
    search_results = await search_database(ctx, query=query, limit=3)

    # Step 4: Format report
    ctx.logger.info("Step 4: Formatting report")
    report_data = {
        "total_sum": sum_result["result"],
        "average": avg_result["result"],
        "count": len(numbers),
        "search_results_count": len(search_results),
    }
    report = await format_report(ctx, data=report_data, format_style="summary")

    return {
        "status": "completed",
        "validation": validation_result,
        "statistics": {
            "sum": sum_result["result"],
            "average": avg_result["result"],
            "count": len(numbers),
        },
        "search_results": search_results,
        "report": report,
    }


@workflow
async def agent_research_workflow(ctx: Context, research_topic: str) -> dict:
    """
    Workflow using an agent with tools to perform research.

    Tests:
    - Agent integration in workflows
    - Agent tool orchestration
    - LLM-driven tool selection
    """
    ctx.logger.info(f"Starting agent research workflow for: {research_topic}")

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        ctx.logger.warning("OPENAI_API_KEY not set, skipping agent execution")
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "research_topic": research_topic,
        }

    # Step 1: Create research agent with tools
    ctx.logger.info("Step 1: Creating research agent")
    research_agent = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="""You are a research assistant. You MUST use the search_database tool to find information.

        Required steps:
        1. ALWAYS call search_database tool first to find relevant information
        2. Use calculate_total if you need statistics
        3. Use format_report to format your findings

        Do not answer from your own knowledge - you must use the tools provided.""",
        tools=[search_database, calculate_total, format_report],
        temperature=0.7,
        max_iterations=5,
    )

    # Step 2: Execute research task
    ctx.logger.info("Step 2: Agent executing research")
    result = await research_agent.run(
        f"Research the topic: {research_topic}. Search for information and provide a summary.",
        context=ctx,
    )

    # Step 3: Process results
    ctx.logger.info("Step 3: Processing research results")

    return {
        "status": "completed",
        "research_topic": research_topic,
        "agent_output": result.output,
        "tool_calls_made": len(result.tool_calls),
        "tools_used": [tc["name"] for tc in result.tool_calls],
    }


@workflow
async def agent_multi_step_workflow(ctx: Context, task: str, data_points: list[float]) -> dict:
    """
    Complex workflow with agent using multiple tools across steps.

    Tests:
    - Multi-step agent workflows
    - Agent state across workflow steps
    - Complex tool orchestration
    """
    ctx.logger.info(f"Starting multi-step agent workflow: {task}")

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        ctx.logger.warning("OPENAI_API_KEY not set, skipping agent execution")
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "task": task,
        }

    # Step 1: Data analysis phase
    ctx.logger.info("Step 1: Data analysis phase")
    analyst_agent = Agent(
        name="analyst",
        model="openai/gpt-4o-mini",
        instructions="""You are a data analyst. You MUST use the calculate_total tool to analyze data.

        Required: Call calculate_total tool multiple times with different operations (sum, average, min, max).
        Do not calculate manually - always use the tool.""",
        tools=[calculate_total],
        temperature=0.5,
    )

    analysis_result = await analyst_agent.run(
        f"Analyze these data points: {data_points}. Calculate sum, average, and provide insights.",
        context=ctx,
    )

    # Step 2: Research phase
    ctx.logger.info("Step 2: Research phase")
    researcher_agent = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="""You are a researcher. You MUST use the search_database tool to find information.
        Always call search_database before answering.""",
        tools=[search_database],
        temperature=0.7,
    )

    # Extract keywords from task for search
    search_query = task.split()[0] if task else "analytics"
    research_result = await researcher_agent.run(
        f"Search for information about: {search_query}",
        context=ctx,
    )

    # Step 3: Report generation phase
    ctx.logger.info("Step 3: Report generation phase")
    report_agent = Agent(
        name="report_writer",
        model="openai/gpt-4o-mini",
        instructions="""You are a report writer. Create a comprehensive report combining:
        1. Data analysis findings
        2. Research findings

        Format it clearly and professionally.""",
        tools=[format_report],
        temperature=0.6,
    )

    final_result = await report_agent.run(
        f"Create a report about '{task}' combining:\n"
        f"Analysis: {analysis_result.output}\n"
        f"Research: {research_result.output}",
        context=ctx,
    )

    return {
        "status": "completed",
        "task": task,
        "data_points_count": len(data_points),
        "analysis_output": analysis_result.output,
        "research_output": research_result.output,
        "final_report": final_result.output,
        "total_tool_calls": (
            len(analysis_result.tool_calls) + len(research_result.tool_calls) + len(final_result.tool_calls)
        ),
    }


@workflow(chat=True)
async def approval_workflow_hitl(ctx: WorkflowContext, message: str) -> dict:
    """
    Human-in-the-loop workflow with approval step.

    Tests:
    - Workflow pauses for user input
    - User approval/rejection flow
    - Workflow resumes after user response
    - State persistence during pause
    """
    ctx.logger.info(f"Starting approval workflow with message: {message}")

    # Step 1: Process the message
    ctx.logger.info("Step 1: Processing message")
    processed = message.upper()
    await asyncio.sleep(0.1)  # Simulate processing

    # Step 2: Ask for user approval (PAUSES HERE)
    ctx.logger.info("Step 2: Requesting user approval")
    decision = await ctx.wait_for_user(
        f"Approve processing result: '{processed}'?",
        input_type="approval",
        options=[
            {"id": "yes", "label": "Approve"},
            {"id": "no", "label": "Reject"}
        ]
    )

    # Step 3: Handle decision
    ctx.logger.info(f"Step 3: User decision received: {decision}")
    if decision == "yes":
        # Continue with approval path
        await asyncio.sleep(0.1)
        return {
            "status": "approved",
            "message": message,
            "processed": processed,
            "decision": decision
        }
    else:
        # Rejection path
        return {
            "status": "rejected",
            "message": message,
            "processed": processed,
            "decision": decision
        }


@workflow(chat=True)
async def multi_choice_workflow(ctx: WorkflowContext, task: str) -> dict:
    """
    Workflow with multiple choice input.

    Tests:
    - Choice input type
    - Multiple options handling
    - Workflow branching based on choice
    """
    ctx.logger.info(f"Starting multi-choice workflow: {task}")

    # Step 1: Process task
    await asyncio.sleep(0.1)

    # Step 2: Ask user to choose processing method
    method = await ctx.wait_for_user(
        "Which processing method?",
        input_type="choice",
        options=[
            {"id": "fast", "label": "Fast (low quality)"},
            {"id": "standard", "label": "Standard (balanced)"},
            {"id": "thorough", "label": "Thorough (high quality)"}
        ]
    )

    # Step 3: Process based on choice
    ctx.logger.info(f"Processing with {method} method")
    await asyncio.sleep(0.1)

    return {
        "status": "completed",
        "task": task,
        "method": method,
        "quality": {"fast": "low", "standard": "balanced", "thorough": "high"}[method]
    }


@workflow(chat=True)
async def text_input_workflow(ctx: WorkflowContext) -> dict:
    """
    Workflow with text input.

    Tests:
    - Text input type
    - Free-form user responses
    - Multiple text inputs in sequence
    """
    ctx.logger.info("Starting text input workflow")

    # Step 1: Ask for name
    name = await ctx.wait_for_user(
        "What is your name?",
        input_type="text"
    )

    # Step 2: Ask for city
    city = await ctx.wait_for_user(
        "Which city are you from?",
        input_type="text"
    )

    # Step 3: Generate personalized result
    await asyncio.sleep(0.1)

    return {
        "status": "completed",
        "greeting": f"Hello {name} from {city}!",
        "name": name,
        "city": city
    }


__all__ = [
    "order_fulfillment",
    "long_workflow",
    "data_pipeline",
    "tool_orchestrated_workflow",
    "agent_research_workflow",
    "agent_multi_step_workflow",
    "approval_workflow_hitl",
    "multi_choice_workflow",
    "text_input_workflow",
]
