"""
Example: AGNT5 Context Propagation

This example demonstrates context propagation patterns in workflows:
- Sequential context: Pass accumulated context through workflow steps
- Parallel context: Branch context to parallel steps, merge results
- State management: Using ctx.state for step-to-step data

Context propagation enables workflows to:
- Maintain state across multiple steps
- Share data between parallel branches
- Accumulate results as workflow progresses
- Handle partial failures gracefully

Each workflow can be:
- Run standalone: python ex_13_context_propagation.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("sequential_context_workflow", {...}, component_type="workflow")

Prerequisites:
- Set OPENAI_API_KEY for agent-based examples
"""

import asyncio
import os
import unicodedata
from typing import Any, Dict, List

from agnt5 import Agent, Context, FunctionContext, WorkflowContext, function, workflow


def is_punctuation(char: str) -> bool:
    """Check if a character is punctuation (Unicode-aware)."""
    return unicodedata.category(char).startswith('P')


def is_sentence_ender(char: str) -> bool:
    """Check if a character ends a sentence (Unicode-aware)."""
    # ASCII + common Unicode sentence enders
    return char in ".!?。！？⁈⁉‽"


# =============================================================================
# SEQUENTIAL CONTEXT PROPAGATION
# =============================================================================


@workflow
async def sequential_context_workflow(ctx: WorkflowContext, input_data: str) -> dict:
    """
    Sequential workflow demonstrating context propagation through steps.

    Each step receives context from previous steps and adds to it.
    Context accumulates as the workflow progresses.

    Args:
        input_data: Initial data to process

    Returns:
        Dict with accumulated context from all steps

    Example:
        result = client.run("sequential_context_workflow", {
            "input_data": "Hello World"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting sequential context workflow with: {input_data}")

    # Step 1: Parse input
    ctx.state.set("original_input", input_data)
    parsed = await ctx.step("parse", lambda: {
        "words": input_data.split(),
        "length": len(input_data),
        "uppercase": input_data.upper(),
    })
    ctx.state.set("parsed", parsed)
    ctx.logger.info(f"Step 1 - Parsed: {parsed}")

    # Step 2: Analyze (uses result from step 1)
    analysis = await ctx.step("analyze", lambda: {
        "word_count": len(parsed["words"]),
        "char_count": parsed["length"],
        "has_uppercase": any(c.isupper() for c in input_data),
        "previous_step": "parse",
    })
    ctx.state.set("analysis", analysis)
    ctx.logger.info(f"Step 2 - Analysis: {analysis}")

    # Step 3: Enrich (uses results from steps 1 and 2)
    enriched = await ctx.step("enrich", lambda: {
        "summary": f"{analysis['word_count']} words, {analysis['char_count']} chars",
        "transformed": parsed["uppercase"],
        "metadata": {
            "steps_completed": 3,
            "context_chain": ["parse", "analyze", "enrich"],
        },
    })
    ctx.state.set("enriched", enriched)
    ctx.logger.info(f"Step 3 - Enriched: {enriched}")

    return {
        "input": input_data,
        "parsed": parsed,
        "analysis": analysis,
        "enriched": enriched,
        "context_accumulated": True,
    }


@workflow
async def ctx_state_pipeline(ctx: WorkflowContext, numbers: List[float]) -> dict:
    """
    Pipeline using ctx.state for step-to-step data transfer.

    Demonstrates using WorkflowContext state to share data between steps.

    Args:
        numbers: List of numbers to process

    Returns:
        Dict with pipeline results

    Example:
        result = client.run("ctx_state_pipeline", {
            "numbers": [1.5, 2.5, 3.5, 4.5, 5.5]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting state pipeline with {len(numbers)} numbers")

    # Step 1: Store initial data in state
    ctx.state.set("raw_numbers", numbers)
    ctx.state.set("step_count", 0)

    # Step 2: Calculate statistics
    stats = await ctx.step("statistics", lambda: {
        "sum": sum(numbers),
        "count": len(numbers),
        "average": sum(numbers) / len(numbers) if numbers else 0,
        "min": min(numbers) if numbers else 0,
        "max": max(numbers) if numbers else 0,
    })
    ctx.state.set("statistics", stats)
    ctx.state.set("step_count", ctx.state.get("step_count", 0) + 1)
    ctx.logger.info(f"Statistics calculated: {stats}")

    # Step 3: Normalize (uses statistics from state)
    stored_stats = ctx.state.get("statistics")
    normalized = await ctx.step("normalize", lambda: {
        "normalized": [(n - stored_stats["min"]) / (stored_stats["max"] - stored_stats["min"])
                       if stored_stats["max"] != stored_stats["min"] else 0
                       for n in numbers],
        "range": stored_stats["max"] - stored_stats["min"],
    })
    ctx.state.set("normalized", normalized)
    ctx.state.set("step_count", ctx.state.get("step_count", 0) + 1)
    ctx.logger.info(f"Normalized: {normalized}")

    # Step 4: Categorize (uses normalized values)
    stored_normalized = ctx.state.get("normalized")
    categorized = await ctx.step("categorize", lambda: {
        "low": [n for n in stored_normalized["normalized"] if n < 0.33],
        "medium": [n for n in stored_normalized["normalized"] if 0.33 <= n < 0.67],
        "high": [n for n in stored_normalized["normalized"] if n >= 0.67],
    })
    ctx.state.set("categorized", categorized)
    ctx.state.set("step_count", ctx.state.get("step_count", 0) + 1)
    ctx.logger.info(f"Categorized: {categorized}")

    return {
        "input_numbers": numbers,
        "statistics": stats,
        "normalized": normalized,
        "categorized": categorized,
        "total_steps": ctx.state.get("step_count"),
        "state_keys": ["raw_numbers", "statistics", "normalized", "categorized", "step_count"],
    }


@workflow
async def explicit_context_chain(ctx: WorkflowContext, document: str) -> dict:
    """
    Explicit context dictionary passed through each step.

    Each step receives the accumulated context and adds its contribution.

    Args:
        document: Text document to process

    Returns:
        Dict with processing chain results

    Example:
        result = client.run("explicit_context_chain", {
            "document": "The quick brown fox jumps over the lazy dog."
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting explicit context chain")

    # Initialize context dictionary
    chain_context: Dict[str, Any] = {
        "original": document,
        "steps": [],
        "metadata": {},
    }

    # Step 1: Tokenize
    def tokenize():
        tokens = document.lower().split()
        chain_context["tokens"] = tokens
        chain_context["steps"].append("tokenize")
        chain_context["metadata"]["token_count"] = len(tokens)
        return {"tokens": tokens, "count": len(tokens)}

    token_result = await ctx.step("tokenize", tokenize)
    ctx.logger.info(f"Tokenized: {token_result['count']} tokens")

    # Step 2: Filter (uses tokens from chain_context)
    def filter_tokens():
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "over"}
        filtered = [t for t in chain_context["tokens"] if t not in stopwords]
        chain_context["filtered_tokens"] = filtered
        chain_context["steps"].append("filter")
        chain_context["metadata"]["filtered_count"] = len(filtered)
        return {"filtered": filtered, "removed": len(chain_context["tokens"]) - len(filtered)}

    filter_result = await ctx.step("filter", filter_tokens)
    ctx.logger.info(f"Filtered: removed {filter_result['removed']} stopwords")

    # Step 3: Analyze frequency (uses filtered_tokens from chain_context)
    def analyze_frequency():
        freq = {}
        for token in chain_context["filtered_tokens"]:
            freq[token] = freq.get(token, 0) + 1
        chain_context["frequency"] = freq
        chain_context["steps"].append("frequency")
        chain_context["metadata"]["unique_tokens"] = len(freq)
        return {"frequency": freq, "unique_count": len(freq)}

    freq_result = await ctx.step("frequency", analyze_frequency)
    ctx.logger.info(f"Frequency analysis: {freq_result['unique_count']} unique tokens")

    # Step 4: Generate summary (uses all accumulated context)
    def summarize():
        top_tokens = sorted(chain_context["frequency"].items(), key=lambda x: x[1], reverse=True)[:5]
        summary = {
            "document_length": len(chain_context["original"]),
            "total_tokens": chain_context["metadata"]["token_count"],
            "after_filtering": chain_context["metadata"]["filtered_count"],
            "unique_tokens": chain_context["metadata"]["unique_tokens"],
            "top_tokens": top_tokens,
            "processing_steps": chain_context["steps"],
        }
        chain_context["summary"] = summary
        chain_context["steps"].append("summarize")
        return summary

    summary = await ctx.step("summarize", summarize)
    ctx.logger.info(f"Summary: {summary}")

    return {
        "document": document,
        "chain_context": chain_context,
        "final_summary": summary,
    }


# =============================================================================
# PARALLEL CONTEXT PROPAGATION
# =============================================================================


@workflow
async def parallel_context_workflow(ctx: WorkflowContext, text: str) -> dict:
    """
    Parallel workflow that branches context and merges results.

    Multiple analysis branches run in parallel, results are merged.

    Args:
        text: Text to analyze in parallel

    Returns:
        Dict with merged parallel results

    Example:
        result = client.run("parallel_context_workflow", {
            "text": "The quick brown fox jumps over the lazy dog."
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting parallel context workflow")

    # Store shared input
    ctx.state.set("input_text", text)

    # Run multiple analysis tasks in parallel using asyncio.gather
    async def analyze_length():
        await asyncio.sleep(0.05)  # Simulate work
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines()) or 1,
        }

    async def analyze_content():
        await asyncio.sleep(0.05)  # Simulate work
        words = text.lower().split()
        return {
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "longest_word": max(words, key=len) if words else "",
        }

    async def analyze_structure():
        await asyncio.sleep(0.05)  # Simulate work
        return {
            "sentences": sum(1 for c in text if is_sentence_ender(c)),
            "has_punctuation": any(is_punctuation(c) for c in text),
            "starts_uppercase": text[0].isupper() if text else False,
        }

    # Execute all analyses in parallel
    ctx.logger.info("Running parallel analysis branches...")
    length_result, content_result, structure_result = await asyncio.gather(
        ctx.step("analyze_length", analyze_length()),
        ctx.step("analyze_content", analyze_content()),
        ctx.step("analyze_structure", analyze_structure()),
    )

    # Merge results from parallel branches
    merged = {
        "length_analysis": length_result,
        "content_analysis": content_result,
        "structure_analysis": structure_result,
        "parallel_branches": 3,
    }
    ctx.state.set("merged_results", merged)
    ctx.logger.info(f"Merged parallel results: {merged}")

    # Final aggregation step
    final = await ctx.step("aggregate", lambda: {
        "total_metrics": (
            len(length_result) + len(content_result) + len(structure_result)
        ),
        "summary": f"Analyzed {length_result['char_count']} chars, "
                   f"{content_result['unique_words']} unique words, "
                   f"{structure_result['sentences']} sentences",
    })

    return {
        "input": text,
        "parallel_results": merged,
        "aggregated": final,
    }


@workflow
async def parallel_fan_out_fan_in(ctx: WorkflowContext, items: List[str]) -> dict:
    """
    Fan-out/fan-in pattern: distribute work, then aggregate.

    Each item is processed independently in parallel, results collected.

    Args:
        items: List of items to process in parallel

    Returns:
        Dict with individual and aggregated results

    Example:
        result = client.run("parallel_fan_out_fan_in", {
            "items": ["apple", "banana", "cherry", "date"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting fan-out/fan-in with {len(items)} items")

    # Fan-out: Process each item in parallel
    async def process_item(item: str, index: int):
        await asyncio.sleep(0.02)  # Simulate varying work
        return {
            "item": item,
            "index": index,
            "length": len(item),
            "uppercase": item.upper(),
            "reversed": item[::-1],
        }

    # Create parallel tasks for all items
    tasks = [
        ctx.step(f"process_{i}", process_item(item, i))
        for i, item in enumerate(items)
    ]

    # Execute all in parallel
    ctx.logger.info(f"Fan-out: Processing {len(items)} items in parallel...")
    results = await asyncio.gather(*tasks)

    # Fan-in: Aggregate results
    ctx.logger.info("Fan-in: Aggregating results...")
    aggregated = await ctx.step("aggregate", lambda: {
        "total_items": len(results),
        "total_length": sum(r["length"] for r in results),
        "all_uppercase": [r["uppercase"] for r in results],
        "all_reversed": [r["reversed"] for r in results],
        "longest_item": max(results, key=lambda r: r["length"])["item"],
    })

    return {
        "input_items": items,
        "individual_results": results,
        "aggregated": aggregated,
        "pattern": "fan_out_fan_in",
    }


@workflow
async def context_merge_strategies(ctx: WorkflowContext, data: Dict[str, Any]) -> dict:
    """
    Demonstrates different strategies for merging parallel context.

    Shows override, combine, and conflict resolution approaches.

    Args:
        data: Data to process with different merge strategies

    Returns:
        Dict showing different merge strategy results

    Example:
        result = client.run("context_merge_strategies", {
            "data": {"value": 100, "items": ["a", "b", "c"]}
        }, component_type="workflow")
    """
    ctx.logger.info("Demonstrating context merge strategies")

    # Parallel branches that produce potentially conflicting results
    async def branch_a():
        await asyncio.sleep(0.03)
        return {
            "score": 85,
            "status": "good",
            "items": ["x", "y"],
            "metadata": {"source": "branch_a", "priority": 1},
        }

    async def branch_b():
        await asyncio.sleep(0.03)
        return {
            "score": 92,
            "status": "excellent",
            "items": ["y", "z"],
            "metadata": {"source": "branch_b", "priority": 2},
        }

    async def branch_c():
        await asyncio.sleep(0.03)
        return {
            "score": 78,
            "status": "acceptable",
            "items": ["a", "b"],
            "metadata": {"source": "branch_c", "priority": 1},
        }

    # Execute parallel branches
    result_a, result_b, result_c = await asyncio.gather(
        ctx.step("branch_a", branch_a()),
        ctx.step("branch_b", branch_b()),
        ctx.step("branch_c", branch_c()),
    )

    all_results = [result_a, result_b, result_c]

    # Strategy 1: Last Write Wins (simple override)
    def last_write_wins():
        merged = {}
        for r in all_results:
            merged.update(r)
        return merged

    # Strategy 2: Highest Priority Wins
    def priority_merge():
        sorted_results = sorted(all_results, key=lambda r: r["metadata"]["priority"], reverse=True)
        return sorted_results[0]

    # Strategy 3: Aggregate/Combine Values
    def aggregate_merge():
        return {
            "score": sum(r["score"] for r in all_results) / len(all_results),  # Average
            "status": max(all_results, key=lambda r: r["score"])["status"],  # Best status
            "items": list(set(item for r in all_results for item in r["items"])),  # Unique items
            "sources": [r["metadata"]["source"] for r in all_results],
        }

    # Strategy 4: Conflict Detection
    def conflict_detection():
        conflicts = []
        if len(set(r["status"] for r in all_results)) > 1:
            conflicts.append({
                "field": "status",
                "values": [r["status"] for r in all_results],
            })
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "resolution": "use_highest_score",
            "resolved_value": max(all_results, key=lambda r: r["score"]),
        }

    # Apply all strategies
    strategies = await ctx.step("apply_strategies", lambda: {
        "last_write_wins": last_write_wins(),
        "priority_merge": priority_merge(),
        "aggregate_merge": aggregate_merge(),
        "conflict_detection": conflict_detection(),
    })

    return {
        "input": data,
        "branch_results": {
            "a": result_a,
            "b": result_b,
            "c": result_c,
        },
        "merge_strategies": strategies,
        "recommended": "aggregate_merge",
    }


# =============================================================================
# ERROR HANDLING IN PARALLEL CONTEXT
# =============================================================================


@workflow
async def parallel_with_error_handling(ctx: WorkflowContext, items: List[str]) -> dict:
    """
    Parallel workflow with error handling for failed branches.

    Demonstrates handling partial failures in parallel execution.

    Args:
        items: Items to process (some may fail)

    Returns:
        Dict with successful and failed results

    Example:
        result = client.run("parallel_with_error_handling", {
            "items": ["valid1", "fail", "valid2", "error", "valid3"]
        }, component_type="workflow")
    """
    ctx.logger.info(f"Processing {len(items)} items with error handling")

    async def process_with_possible_failure(item: str, index: int):
        await asyncio.sleep(0.02)
        # Simulate failures for certain items
        if "fail" in item.lower() or "error" in item.lower():
            raise ValueError(f"Processing failed for item: {item}")
        return {
            "item": item,
            "index": index,
            "processed": item.upper(),
            "success": True,
        }

    # Process all items, capturing both successes and failures
    successful = []
    failed = []

    for i, item in enumerate(items):
        try:
            result = await ctx.step(
                f"process_{i}",
                process_with_possible_failure(item, i)
            )
            successful.append(result)
        except Exception as e:
            failed.append({
                "item": item,
                "index": i,
                "error": str(e),
                "success": False,
            })
            ctx.logger.warning(f"Item {item} failed: {e}")

    # Aggregate results despite partial failures
    aggregated = await ctx.step("aggregate", lambda: {
        "total_items": len(items),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "success_rate": len(successful) / len(items) if items else 0,
        "all_processed": [r["processed"] for r in successful],
    })

    return {
        "input_items": items,
        "successful": successful,
        "failed": failed,
        "aggregated": aggregated,
        "partial_success": len(failed) > 0 and len(successful) > 0,
    }


@workflow
async def parallel_with_fallback(ctx: WorkflowContext, query: str) -> dict:
    """
    Parallel workflow with fallback when primary path fails.

    Demonstrates graceful degradation in parallel execution.

    Args:
        query: Query to process with fallback options

    Returns:
        Dict with result from successful path

    Example:
        result = client.run("parallel_with_fallback", {
            "query": "search for something"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Processing query with fallback: {query}")

    # Primary path (may fail)
    async def primary_processing():
        await asyncio.sleep(0.05)
        # Simulate occasional failure
        if len(query) < 5:
            raise ValueError("Query too short for primary processing")
        return {
            "source": "primary",
            "result": f"Primary result for: {query}",
            "quality": "high",
        }

    # Fallback path (more reliable but lower quality)
    async def fallback_processing():
        await asyncio.sleep(0.03)
        return {
            "source": "fallback",
            "result": f"Fallback result for: {query}",
            "quality": "medium",
        }

    # Try primary first, use fallback if needed
    result = None
    used_fallback = False

    try:
        result = await ctx.step("primary", primary_processing())
        ctx.logger.info("Primary processing succeeded")
    except Exception as e:
        ctx.logger.warning(f"Primary failed: {e}, using fallback")
        used_fallback = True
        result = await ctx.step("fallback", fallback_processing())
        ctx.logger.info("Fallback processing succeeded")

    return {
        "query": query,
        "result": result,
        "used_fallback": used_fallback,
        "reliability": "degraded" if used_fallback else "full",
    }


# =============================================================================
# AGENT-BASED CONTEXT PROPAGATION
# =============================================================================


@workflow(chat=True)
async def agent_context_chain(ctx: WorkflowContext, topic: str) -> dict:
    """
    Context propagation through a chain of agents.

    Each agent receives context from previous agent and adds to it.

    Args:
        topic: Topic to research and analyze

    Returns:
        Dict with chained agent results

    Example:
        result = client.run("agent_context_chain", {
            "topic": "machine learning"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting agent context chain for: {topic}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "topic": topic,
        }

    # Chain context that accumulates through agents
    chain_context = {"topic": topic, "stages": []}

    # Agent 1: Research
    researcher = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="You are a researcher. Provide 3 key facts about the given topic. Be concise.",
        temperature=0.7,
        max_iterations=3,
    )

    research_result = await researcher.run_sync(f"Research: {topic}", context=ctx)
    chain_context["research"] = research_result.output
    chain_context["stages"].append("research")
    ctx.logger.info(f"Research complete: {research_result.output[:100]}...")

    # Agent 2: Analyzer (receives research context)
    analyzer = Agent(
        name="analyzer",
        model="openai/gpt-4o-mini",
        instructions="You are an analyst. Analyze the provided research and identify key insights. Be concise.",
        temperature=0.5,
        max_iterations=3,
    )

    analysis_result = await analyzer.run_sync(
        f"Analyze this research about {topic}:\n{chain_context['research']}",
        context=ctx,
    )
    chain_context["analysis"] = analysis_result.output
    chain_context["stages"].append("analysis")
    ctx.logger.info(f"Analysis complete: {analysis_result.output[:100]}...")

    # Agent 3: Summarizer (receives all previous context)
    summarizer = Agent(
        name="summarizer",
        model="openai/gpt-4o-mini",
        instructions="You are a summarizer. Create a one-paragraph summary combining research and analysis.",
        temperature=0.3,
        max_iterations=3,
    )

    summary_result = await summarizer.run_sync(
        f"Summarize:\nTopic: {topic}\nResearch: {chain_context['research']}\nAnalysis: {chain_context['analysis']}",
        context=ctx,
    )
    chain_context["summary"] = summary_result.output
    chain_context["stages"].append("summary")
    ctx.logger.info(f"Summary complete: {summary_result.output[:100]}...")

    return {
        "topic": topic,
        "chain_context": chain_context,
        "stages_completed": chain_context["stages"],
        "final_summary": summary_result.output,
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Context Propagation Examples")
    print("=" * 60)

    ctx = Context(
        run_id="context-propagation-demo",
        correlation_id="context-propagation-demo",
        parent_correlation_id="",
    )

    # Sequential context
    print("\n--- Sequential Context Workflow ---")
    result = await sequential_context_workflow(ctx, input_data="Hello World from AGNT5")
    print(f"Result: {result['enriched']['summary']}")

    # State pipeline
    print("\n--- State Pipeline ---")
    result = await ctx_state_pipeline(ctx, numbers=[10.0, 20.0, 30.0, 40.0, 50.0])
    print(f"Result: {result['statistics']}")

    # Explicit context chain
    print("\n--- Explicit Context Chain ---")
    result = await explicit_context_chain(ctx, document="The quick brown fox jumps over the lazy dog.")
    print(f"Result: {result['final_summary']}")

    # Parallel context
    print("\n--- Parallel Context Workflow ---")
    result = await parallel_context_workflow(ctx, text="The quick brown fox jumps over the lazy dog.")
    print(f"Result: {result['aggregated']}")

    # Fan-out/fan-in
    print("\n--- Fan-Out/Fan-In ---")
    result = await parallel_fan_out_fan_in(ctx, items=["apple", "banana", "cherry"])
    print(f"Result: {result['aggregated']}")

    # Error handling
    print("\n--- Parallel with Error Handling ---")
    result = await parallel_with_error_handling(ctx, items=["valid1", "fail", "valid2"])
    print(f"Result: Success rate: {result['aggregated']['success_rate']:.0%}")

    # Fallback
    print("\n--- Parallel with Fallback ---")
    result = await parallel_with_fallback(ctx, query="short")
    print(f"Result: Used fallback: {result['used_fallback']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
