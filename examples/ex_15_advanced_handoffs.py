"""
Example: AGNT5 Advanced Handoffs

This example demonstrates advanced agent handoff patterns:
- Chain handoffs: A -> B -> C linear handoff chain
- Conditional handoffs: Route based on analysis
- Handoffs with context: Pass full conversation history
- Handoff loops: Agent can hand back to original

Handoffs enable:
- Specialized agent routing
- Complex workflow orchestration
- Context preservation across agents
- Dynamic agent selection

Each workflow/function can be:
- Run standalone: python ex_15_advanced_handoffs.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("chain_handoff_workflow", {...}, component_type="workflow")

Prerequisites:
- Set OPENAI_API_KEY for agent-based examples
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from agnt5 import Agent, Context, FunctionContext, WorkflowContext, function, handoff, tool, workflow
from agnt5.lm import Message


# =============================================================================
# TOOL DEFINITIONS FOR HANDOFF EXAMPLES
# =============================================================================


@tool
async def analyze_request(ctx: Context, request: str) -> dict:
    """
    Analyze a request to determine its category.

    Args:
        request: The request to analyze

    Returns:
        Analysis result with category and confidence
    """
    ctx.logger.info(f"[analyze_request] Analyzing: {request[:50]}...")

    # Simple keyword-based categorization
    request_lower = request.lower()

    if any(word in request_lower for word in ["code", "programming", "bug", "error", "technical"]):
        category = "technical"
        confidence = 0.9
    elif any(word in request_lower for word in ["bill", "payment", "invoice", "charge", "refund"]):
        category = "billing"
        confidence = 0.85
    elif any(word in request_lower for word in ["how", "what", "when", "where", "help"]):
        category = "general"
        confidence = 0.7
    else:
        category = "unknown"
        confidence = 0.5

    return {
        "request": request,
        "category": category,
        "confidence": confidence,
        "keywords_found": [w for w in request_lower.split() if len(w) > 3],
    }


@tool
async def log_handoff(ctx: Context, from_agent: str, to_agent: str, reason: str) -> dict:
    """
    Log a handoff event for tracking.

    Args:
        from_agent: Source agent name
        to_agent: Target agent name
        reason: Reason for handoff

    Returns:
        Handoff log entry
    """
    ctx.logger.info(f"[log_handoff] {from_agent} -> {to_agent}: {reason}")

    return {
        "from": from_agent,
        "to": to_agent,
        "reason": reason,
        "logged": True,
    }


# =============================================================================
# CHAIN HANDOFFS
# =============================================================================


@function
async def chain_handoff_workflow(ctx: FunctionContext, query: str) -> dict:
    """
    Linear handoff chain: Intake -> Specialist -> Reviewer.

    Each agent handles their part and hands off to the next.

    Args:
        query: Initial query to process

    Returns:
        Dict with results from each agent in the chain

    Example:
        result = client.run("chain_handoff_workflow", {
            "query": "Help me fix a bug in my Python code"
        })
    """
    ctx.logger.info(f"Starting chain handoff for: {query}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "query": query,
        }

    # Final agent in chain (Reviewer)
    reviewer_agent = Agent(
        name="reviewer",
        model="openai/gpt-4o-mini",
        instructions="""You are a quality reviewer. Review the specialist's work and provide:
        1. A brief summary of what was done
        2. Any additional recommendations
        3. A quality rating (1-5 stars)
        Be concise.""",
        temperature=0.3,
        max_iterations=3,
    )

    # Middle agent (Specialist)
    specialist_agent = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a technical specialist. Provide a detailed solution to the problem.
        After providing your solution, hand off to the reviewer for quality check.""",
        handoffs=[handoff(reviewer_agent, "Hand off to reviewer for quality check")],
        temperature=0.5,
        max_iterations=5,
    )

    # First agent (Intake)
    intake_agent = Agent(
        name="intake",
        model="openai/gpt-4o-mini",
        instructions="""You are an intake agent. Your job is to:
        1. Understand the user's request
        2. Summarize the key points
        3. Hand off to the specialist for detailed handling
        Be brief and efficient.""",
        handoffs=[handoff(specialist_agent, "Hand off to specialist for detailed handling")],
        temperature=0.5,
        max_iterations=3,
    )

    # Start the chain
    result = await intake_agent.run(query)

    return {
        "query": query,
        "final_output": result.output,
        "handoff_chain": ["intake", "specialist", "reviewer"],
        "handoff_occurred": result.handoff_to is not None,
        "pattern": "chain_handoff",
    }


@function
async def three_stage_pipeline(ctx: FunctionContext, document: str) -> dict:
    """
    Three-stage document processing pipeline with handoffs.

    Stage 1: Parser extracts key information
    Stage 2: Analyzer finds insights
    Stage 3: Summarizer creates final output

    Args:
        document: Document text to process

    Returns:
        Dict with pipeline results

    Example:
        result = client.run("three_stage_pipeline", {
            "document": "AGNT5 is a platform for AI workflow orchestration..."
        })
    """
    ctx.logger.info(f"Starting three-stage pipeline")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "document": document[:100],
        }

    # Stage 3: Summarizer (final)
    summarizer = Agent(
        name="summarizer",
        model="openai/gpt-4o-mini",
        instructions="""You are a summarizer. Create a concise 2-3 sentence summary from the analysis provided.
        Focus on the most important points.""",
        temperature=0.3,
        max_iterations=3,
    )

    # Stage 2: Analyzer
    analyzer = Agent(
        name="analyzer",
        model="openai/gpt-4o-mini",
        instructions="""You are an analyst. Based on the parsed information:
        1. Identify 3 key insights
        2. Note any patterns or trends
        Then hand off to the summarizer.""",
        handoffs=[handoff(summarizer, "Hand off to summarizer for final summary")],
        temperature=0.5,
        max_iterations=4,
    )

    # Stage 1: Parser
    parser = Agent(
        name="parser",
        model="openai/gpt-4o-mini",
        instructions="""You are a document parser. Extract:
        1. Main topic
        2. Key entities mentioned
        3. Important facts
        Then hand off to the analyzer.""",
        handoffs=[handoff(analyzer, "Hand off to analyzer for insight extraction")],
        temperature=0.3,
        max_iterations=4,
    )

    result = await parser.run(f"Parse this document:\n{document}")

    return {
        "document": document[:100] + "..." if len(document) > 100 else document,
        "final_output": result.output,
        "stages": ["parser", "analyzer", "summarizer"],
        "pattern": "three_stage_pipeline",
    }


# =============================================================================
# CONDITIONAL HANDOFFS
# =============================================================================


@function
async def conditional_handoff_router(ctx: FunctionContext, request: str) -> dict:
    """
    Route to different specialists based on request analysis.

    The router analyzes the request and hands off to the appropriate specialist.

    Args:
        request: User request to route

    Returns:
        Dict with routing decision and specialist response

    Example:
        result = client.run("conditional_handoff_router", {
            "request": "I need help with my billing statement"
        })
    """
    ctx.logger.info(f"Starting conditional routing for: {request}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "request": request,
        }

    # Create specialist agents
    technical_agent = Agent(
        name="technical_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a technical support specialist. Help with:
        - Code issues and debugging
        - Technical configuration
        - System errors
        Provide clear, step-by-step solutions.""",
        temperature=0.5,
        max_iterations=5,
    )

    billing_agent = Agent(
        name="billing_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a billing specialist. Help with:
        - Invoice questions
        - Payment issues
        - Subscription management
        Be clear about policies and next steps.""",
        temperature=0.5,
        max_iterations=5,
    )

    general_agent = Agent(
        name="general_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a general support agent. Help with:
        - Product questions
        - How-to guidance
        - General inquiries
        Be friendly and helpful.""",
        temperature=0.7,
        max_iterations=5,
    )

    # Router agent with conditional handoffs
    router = Agent(
        name="router",
        model="openai/gpt-4o-mini",
        instructions="""You are a support router. Analyze the request and:
        1. Determine if it's technical, billing, or general
        2. Hand off to the appropriate specialist
        Do NOT try to answer yourself - always hand off.""",
        tools=[analyze_request],
        handoffs=[
            handoff(technical_agent, "Transfer to technical support for code/system issues"),
            handoff(billing_agent, "Transfer to billing for payment/invoice issues"),
            handoff(general_agent, "Transfer to general support for other questions"),
        ],
        temperature=0.3,
        max_iterations=5,
    )

    result = await router.run(request)

    return {
        "request": request,
        "final_output": result.output,
        "routed_to": result.handoff_to if result.handoff_to else "none",
        "tool_calls": len(result.tool_calls),
        "pattern": "conditional_handoff",
    }


@function
async def priority_based_routing(ctx: FunctionContext, issue: str, priority: str = "normal") -> dict:
    """
    Route based on both content and priority level.

    High priority issues go to senior agents, normal to regular agents.

    Args:
        issue: Issue description
        priority: Priority level (low, normal, high, critical)

    Returns:
        Dict with routing decision

    Example:
        result = client.run("priority_based_routing", {
            "issue": "System is completely down",
            "priority": "critical"
        })
    """
    ctx.logger.info(f"Routing issue with priority: {priority}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "issue": issue,
        }

    # Junior agent for low priority
    junior_agent = Agent(
        name="junior_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a junior support agent. Handle simple issues:
        - Basic questions
        - Documentation pointers
        - Simple troubleshooting
        Escalate if the issue is complex.""",
        temperature=0.7,
        max_iterations=3,
    )

    # Regular agent for normal priority
    regular_agent = Agent(
        name="regular_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a support agent. Handle standard issues:
        - Common problems
        - Standard procedures
        - Configuration help
        Be thorough but efficient.""",
        temperature=0.5,
        max_iterations=5,
    )

    # Senior agent for high/critical priority
    senior_agent = Agent(
        name="senior_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a senior support engineer. Handle complex issues:
        - Critical system problems
        - Escalated issues
        - Complex troubleshooting
        Be thorough and document everything.""",
        temperature=0.3,
        max_iterations=7,
    )

    # Priority router
    router = Agent(
        name="priority_router",
        model="openai/gpt-4o-mini",
        instructions=f"""You are a priority-aware router. The current priority is: {priority}

        Routing rules:
        - low/normal priority + simple issue -> junior_support
        - normal priority + standard issue -> regular_support
        - high/critical priority OR complex issue -> senior_support

        Analyze the issue and hand off accordingly.""",
        handoffs=[
            handoff(junior_agent, "Route to junior support for simple issues"),
            handoff(regular_agent, "Route to regular support for standard issues"),
            handoff(senior_agent, "Route to senior support for complex/critical issues"),
        ],
        temperature=0.3,
        max_iterations=4,
    )

    result = await router.run(f"Priority: {priority}\nIssue: {issue}")

    return {
        "issue": issue,
        "priority": priority,
        "final_output": result.output,
        "routed_to": result.handoff_to if result.handoff_to else "none",
        "pattern": "priority_routing",
    }


# =============================================================================
# HANDOFFS WITH CONTEXT
# =============================================================================


@function
async def handoff_with_full_history(ctx: FunctionContext, messages: List[Dict[str, str]]) -> dict:
    """
    Handoff that preserves full conversation history.

    The receiving agent gets the complete context of the conversation.

    Args:
        messages: List of previous messages [{role, content}]

    Returns:
        Dict with handoff result

    Example:
        result = client.run("handoff_with_full_history", {
            "messages": [
                {"role": "user", "content": "I need help with Python"},
                {"role": "assistant", "content": "What specifically about Python?"},
                {"role": "user", "content": "I'm getting a TypeError"}
            ]
        })
    """
    ctx.logger.info(f"Handoff with {len(messages)} messages of history")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "message_count": len(messages),
        }

    # Specialist agent that will receive the handoff
    specialist = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a specialist who has received a handoff.
        You have access to the full conversation history.
        Continue helping the user with their issue.
        Reference previous context when relevant.""",
        temperature=0.5,
        max_iterations=5,
    )

    # Initial agent that will hand off
    initial_agent = Agent(
        name="initial_agent",
        model="openai/gpt-4o-mini",
        instructions="""You are an initial contact agent.
        Review the conversation history and hand off to the specialist.
        When handing off, ensure the specialist has full context.""",
        handoffs=[handoff(specialist, "Hand off to specialist with full history")],
        temperature=0.5,
        max_iterations=3,
    )

    # Build context from messages
    context_summary = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    last_message = messages[-1]["content"] if messages else "No messages"

    # Convert dict messages to Message objects for history
    history_messages: List[Message] = []
    if len(messages) > 1:
        for m in messages[:-1]:
            if m["role"] == "user":
                history_messages.append(Message.user(m["content"]))
            elif m["role"] == "assistant":
                history_messages.append(Message.assistant(m["content"]))

    result = await initial_agent.run(
        f"Previous conversation:\n{context_summary}\n\nContinue from the last message.",
        history=history_messages if history_messages else None,
    )

    return {
        "messages_received": len(messages),
        "final_output": result.output,
        "handoff_occurred": result.handoff_to is not None,
        "handoff_to": result.handoff_to,
        "pattern": "handoff_with_history",
    }


@function
async def handoff_state_transfer(ctx: FunctionContext, task: str, initial_state: Optional[Dict[str, Any]] = None) -> dict:
    """
    Explicit state transfer during handoff.

    State is explicitly passed between agents during handoff.

    Args:
        task: Task to process
        initial_state: Initial state dictionary

    Returns:
        Dict with state transfer result

    Example:
        result = client.run("handoff_state_transfer", {
            "task": "Analyze and summarize data",
            "initial_state": {"data_source": "api", "format": "json"}
        })
    """
    ctx.logger.info(f"Handoff with state transfer")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "task": task,
        }

    state = initial_state or {}
    state["task"] = task
    state["stages_completed"] = []

    # Second stage agent
    stage2_agent = Agent(
        name="stage2",
        model="openai/gpt-4o-mini",
        instructions="""You are the second stage processor.
        You receive state from stage 1.
        Complete the processing based on the state provided.
        Summarize what was done and the final state.""",
        temperature=0.5,
        max_iterations=4,
    )

    # First stage agent
    stage1_agent = Agent(
        name="stage1",
        model="openai/gpt-4o-mini",
        instructions="""You are the first stage processor.
        Process the initial task and update the state.
        Then hand off to stage 2 with the updated state.
        Include all state information in your handoff message.""",
        handoffs=[handoff(stage2_agent, "Hand off to stage 2 with updated state")],
        temperature=0.5,
        max_iterations=4,
    )

    # Include state in the message
    state_str = "\n".join([f"- {k}: {v}" for k, v in state.items()])
    result = await stage1_agent.run(
        f"Task: {task}\n\nCurrent State:\n{state_str}\n\nProcess stage 1 and hand off."
    )

    return {
        "task": task,
        "initial_state": initial_state,
        "final_output": result.output,
        "handoff_occurred": result.handoff_to is not None,
        "pattern": "state_transfer",
    }


# =============================================================================
# HANDOFF LOOPS
# =============================================================================


@function
async def handoff_with_escalation(ctx: FunctionContext, issue: str, max_escalations: int = 2) -> dict:
    """
    Agent can escalate back or forward based on complexity.

    Demonstrates bidirectional handoff capability.

    Args:
        issue: Issue to handle
        max_escalations: Maximum number of escalations allowed

    Returns:
        Dict with escalation path and result

    Example:
        result = client.run("handoff_with_escalation", {
            "issue": "Complex system integration failure",
            "max_escalations": 2
        })
    """
    ctx.logger.info(f"Handoff with escalation support")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "issue": issue,
        }

    # Level 3: Senior (final escalation)
    level3_agent = Agent(
        name="senior_engineer",
        model="openai/gpt-4o-mini",
        instructions="""You are a senior engineer (Level 3 support).
        You handle the most complex issues.
        Provide comprehensive solutions.
        You cannot escalate further.""",
        temperature=0.3,
        max_iterations=5,
    )

    # Level 2: Intermediate
    level2_agent = Agent(
        name="engineer",
        model="openai/gpt-4o-mini",
        instructions="""You are a support engineer (Level 2).
        Handle moderately complex issues.
        If too complex, escalate to Level 3.
        If simple, you can handle it yourself.""",
        handoffs=[handoff(level3_agent, "Escalate to Level 3 senior engineer")],
        temperature=0.5,
        max_iterations=5,
    )

    # Level 1: Initial support
    level1_agent = Agent(
        name="support_agent",
        model="openai/gpt-4o-mini",
        instructions="""You are a Level 1 support agent.
        Handle simple issues yourself.
        Escalate to Level 2 if the issue is technical or complex.
        Always try to help first before escalating.""",
        handoffs=[handoff(level2_agent, "Escalate to Level 2 engineer")],
        temperature=0.7,
        max_iterations=4,
    )

    result = await level1_agent.run(issue)

    # Track escalation path
    escalation_path = ["level1"]
    if result.handoff_to:
        escalation_path.append(result.handoff_to)

    return {
        "issue": issue,
        "final_output": result.output,
        "escalation_path": escalation_path,
        "final_handler": result.handoff_to or "level1",
        "pattern": "escalation_handoff",
    }


@workflow(chat=True)
async def collaborative_handoff(ctx: WorkflowContext, problem: str) -> dict:
    """
    Agents collaborate by handing back and forth.

    Demonstrates iterative collaboration between agents.

    Args:
        problem: Problem requiring collaboration

    Returns:
        Dict with collaboration result

    Example:
        result = client.run("collaborative_handoff", {
            "problem": "Design a scalable API architecture"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting collaborative handoff for: {problem}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "problem": problem,
        }

    # Track collaboration rounds
    ctx.state.set("rounds", 0)
    ctx.state.set("contributions", [])

    # Architect agent
    architect = Agent(
        name="architect",
        model="openai/gpt-4o-mini",
        instructions="""You are a software architect.
        Propose high-level design solutions.
        Focus on structure and patterns.
        Be concise - one key point per response.""",
        temperature=0.5,
        max_iterations=3,
    )

    # Developer agent
    developer = Agent(
        name="developer",
        model="openai/gpt-4o-mini",
        instructions="""You are a senior developer.
        Review architectural proposals.
        Add implementation considerations.
        Be concise - one key point per response.""",
        temperature=0.5,
        max_iterations=3,
    )

    # Simulate collaboration rounds
    contributions = []

    # Round 1: Architect proposes
    arch_result = await architect.run(f"Propose a solution for: {problem}")
    contributions.append({"agent": "architect", "content": arch_result.output})

    # Round 2: Developer responds
    dev_result = await developer.run(
        f"Review and add to this proposal:\n{arch_result.output}"
    )
    contributions.append({"agent": "developer", "content": dev_result.output})

    # Round 3: Architect refines
    final_result = await architect.run(
        f"Refine based on developer feedback:\n{dev_result.output}"
    )
    contributions.append({"agent": "architect", "content": final_result.output})

    return {
        "problem": problem,
        "contributions": contributions,
        "collaboration_rounds": len(contributions),
        "final_solution": final_result.output,
        "pattern": "collaborative_handoff",
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Advanced Handoffs Examples")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set. Set it to run agent examples.")
        print("   export OPENAI_API_KEY=sk-...")
        return

    ctx = Context(
        run_id="handoffs-demo",
        correlation_id="handoffs-demo",
        parent_correlation_id="",
    )

    # Chain handoff
    print("\n--- Chain Handoff ---")
    result = await chain_handoff_workflow(ctx, query="Help me fix a Python import error")
    print(f"Chain: {result.get('handoff_chain', [])}")
    print(f"Output: {result.get('final_output', '')[:200]}...")

    # Conditional routing
    print("\n--- Conditional Handoff Router ---")
    result = await conditional_handoff_router(ctx, request="I have a billing question")
    print(f"Routed to: {result.get('routed_to', 'none')}")
    print(f"Output: {result.get('final_output', '')[:200]}...")

    # Priority routing
    print("\n--- Priority-Based Routing ---")
    result = await priority_based_routing(ctx, issue="System is slow", priority="high")
    print(f"Routed to: {result.get('routed_to', 'none')}")

    # Escalation
    print("\n--- Escalation Handoff ---")
    result = await handoff_with_escalation(ctx, issue="Complex database migration issue")
    print(f"Escalation path: {result.get('escalation_path', [])}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
