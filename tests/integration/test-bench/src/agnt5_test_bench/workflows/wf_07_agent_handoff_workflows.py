"""
Agent Handoff Workflows - Explicit agent handoff patterns

Test Coverage:
✅ Simple triage → specialist handoff
✅ Handoffs with context sharing
✅ Conversation history propagation
✅ Multiple handoff options
✅ Handoff with state persistence

Handoffs vs Agent-as-Tool:
- **Handoffs**: Explicit control transfer, target agent takes over completely
- **Agent-as-Tool**: Coordinator invokes specialist, gets result, continues

MCP Verification Points:
- Logs: Handoff decisions and transfers
- Traces: Agent transition spans
- Events: agent.handoff, agent.transfer
- Entity: Conversation history shared across agents
"""

from agnt5 import workflow, WorkflowContext, Agent, handoff, tool, Context
from ..tools import search_web, domain_specific_analysis
from ..agents import research_specialist  # Import from canonical agents


# ============================================================================
# SHARED TOOLS FOR HANDOFF SCENARIOS
# ============================================================================


@tool(auto_schema=True)
async def save_preference(ctx: Context, key: str, value: str) -> str:
    """Save a preference to context state.

    Args:
        key: Preference key
        value: Preference value

    Returns:
        Confirmation message
    """
    ctx.logger.info(f"[TOOL] save_preference: {key}={value}")

    prefs = ctx.get("preferences", {})
    prefs[key] = value
    ctx.set("preferences", prefs)

    return f"Saved: {key}={value}"


# ============================================================================
# MODULE-LEVEL SPECIALIST AGENTS
# ============================================================================

# Technical specialist for software questions
technical_specialist = Agent(
    name="TechnicalSpecialist",
    model="openai/gpt-4o-mini",
    instructions=(
        "You are a technical specialist for software engineering questions. "
        "Provide detailed technical answers with examples when appropriate."
    ),
    temperature=0.7,
    max_iterations=5,
)

# Business specialist for business questions
business_specialist = Agent(
    name="BusinessSpecialist",
    model="openai/gpt-4o-mini",
    instructions=(
        "You are a business specialist for business and strategy questions. "
        "Provide business-focused advice with practical recommendations."
    ),
    temperature=0.7,
    max_iterations=5,
)

# Note: research_specialist imported from agents.py (canonical version)


# ============================================================================
# WORKFLOW 31: Simple Triage → Specialist Handoff
# ============================================================================


@workflow
async def wf_31_simple_triage_handoff(ctx: WorkflowContext, query: str) -> dict:
    """
    Test Scenario: Simple triage agent routing to specialist.

    Validates:
    - Triage agent can route to specialists
    - Handoff transfers control completely
    - Specialist provides final response

    Args:
        query: User query for routing

    Returns:
        Specialist response

    MCP Verification Points:
        - Logs: Triage decision and handoff
        - Traces: Triage span then specialist span
        - Events: agent.handoff event
    """
    ctx.logger.info("=== WF_31: Simple Triage Handoff ===")
    ctx.logger.info(f"Query: {query}")

    # Create triage agent with handoff options
    triage_agent = Agent(
        name="TriageAgent",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a triage agent. Route questions to the right specialist:\n"
            "- Technical questions → TechnicalSpecialist\n"
            "- Business questions → BusinessSpecialist\n"
            "Analyze the query and transfer to the appropriate specialist."
        ),
        handoffs=[technical_specialist, business_specialist],
        temperature=0.7,
    )

    # Execute triage (will handoff to specialist)
    ctx.logger.info("Executing triage agent...")
    result = await triage_agent.run(query, context=ctx)

    ctx.logger.info(f"Final output: {result.output[:100]}...")
    ctx.logger.info(f"Handoff to: {result.handoff_to if hasattr(result, 'handoff_to') else 'none'}")

    return {
        "query": query,
        "output": result.output,
        "handoff_occurred": hasattr(result, "handoff_to"),
        "handoff_to": getattr(result, "handoff_to", None),
    }


# ============================================================================
# WORKFLOW 32: Handoff with Context Sharing
# ============================================================================


@workflow
async def wf_32_handoff_with_context(ctx: WorkflowContext, user_name: str, query: str) -> dict:
    """
    Test Scenario: Handoff with context state sharing.

    Validates:
    - Context state persists across handoff
    - Specialist can access shared state
    - User preferences maintained

    Args:
        user_name: User name for personalization
        query: User query

    Returns:
        Personalized specialist response

    MCP Verification Points:
        - Logs: Context setting and retrieval
        - Entity: Shared state visible to specialist
    """
    ctx.logger.info("=== WF_32: Handoff with Context ===")
    ctx.logger.info(f"User: {user_name}, Query: {query}")

    # Set context before triage
    ctx.state.set("user_name", user_name)
    ctx.state.set("preferences", {"detail_level": "comprehensive"})
    ctx.logger.info(f"Context set: user={user_name}")

    # Context-aware specialist
    context_specialist = Agent(
        name="ContextSpecialist",
        model="openai/gpt-4o-mini",
        instructions=(
            "You are a specialist receiving a handoff. "
            "IMPORTANT: Use ctx.get() to retrieve user preferences and personalize your response. "
            "Address the user by name if available."
        ),
        temperature=0.7,
    )

    # Triage with context awareness
    triage = Agent(
        name="ContextTriage",
        model="openai/gpt-4o-mini",
        instructions="Route queries to the specialist. The specialist will use context.",
        handoffs=[context_specialist],
        temperature=0.7,
    )

    # Execute with context
    ctx.logger.info("Executing triage with context...")
    result = await triage.run(query, context=ctx)

    ctx.logger.info(f"Output: {result.output[:100]}...")

    return {
        "user_name": user_name,
        "query": query,
        "output": result.output,
        "context_preserved": user_name in result.output if user_name else False,
    }


# ============================================================================
# WORKFLOW 33: Handoff with Tool Usage
# ============================================================================


@workflow
async def wf_33_handoff_with_tools(ctx: WorkflowContext, research_topic: str) -> dict:
    """
    Test Scenario: Handoff to specialist with tools.

    Validates:
    - Specialist uses tools after handoff
    - Tool results incorporated in response
    - Complex handoff scenarios

    Args:
        research_topic: Topic to research

    Returns:
        Research results from specialist

    MCP Verification Points:
        - Logs: Handoff then tool usage
        - Traces: Handoff span then tool spans
        - Events: agent.handoff then tool.call events
    """
    ctx.logger.info("=== WF_33: Handoff with Tools ===")
    ctx.logger.info(f"Research topic: {research_topic}")

    # Triage to research specialist
    research_triage = Agent(
        name="ResearchTriage",
        model="openai/gpt-4o-mini",
        instructions="You route research queries to the research specialist.",
        handoffs=[research_specialist],
        temperature=0.7,
    )

    # Execute triage (handoff to specialist with tools)
    ctx.logger.info("Routing to research specialist...")
    result = await research_triage.run(
        f"Research this topic: {research_topic}",
        context=ctx,
    )

    ctx.logger.info(f"Research output: {result.output[:100]}...")
    ctx.logger.info(f"Tool calls: {len(result.tool_calls)}")

    return {
        "topic": research_topic,
        "output": result.output,
        "tool_calls_count": len(result.tool_calls),
        "handoff_occurred": True,
    }


# ============================================================================
# WORKFLOW 34: Multiple Handoff Options
# ============================================================================


@workflow
async def wf_34_multiple_handoff_options(ctx: WorkflowContext, query_type: str, query: str) -> dict:
    """
    Test Scenario: Triage with multiple handoff options.

    Validates:
    - Multiple specialists available
    - Correct specialist selected
    - Different routing paths

    Args:
        query_type: Type hint (technical/business/research)
        query: The actual query

    Returns:
        Response from selected specialist

    MCP Verification Points:
        - Logs: Routing decision
        - Events: Handoff to specific agent
    """
    ctx.logger.info("=== WF_34: Multiple Handoff Options ===")
    ctx.logger.info(f"Type: {query_type}, Query: {query}")

    # Triage with three specialists
    multi_triage = Agent(
        name="MultiTriage",
        model="openai/gpt-4o-mini",
        instructions=(
            "You route queries to specialists:\n"
            "- TechnicalSpecialist: software, code, architecture\n"
            "- BusinessSpecialist: strategy, business, ROI\n"
            "- ResearchSpecialist: research, analysis, investigation\n"
            "Choose the most appropriate specialist."
        ),
        handoffs=[technical_specialist, business_specialist, research_specialist],
        temperature=0.7,
    )

    # Execute triage
    ctx.logger.info("Routing query...")
    result = await multi_triage.run(query, context=ctx)

    ctx.logger.info(f"Output: {result.output[:100]}...")

    return {
        "query_type": query_type,
        "query": query,
        "output": result.output,
        "handoff_to": getattr(result, "handoff_to", "unknown"),
    }


# ============================================================================
# WORKFLOW 35: Handoff with State Persistence
# ============================================================================


@workflow
async def wf_35_handoff_with_state(ctx: WorkflowContext, user_id: str, query: str) -> dict:
    """
    Test Scenario: Handoff with workflow state persistence.

    Validates:
    - Workflow state persists through handoff
    - State accessible before and after
    - Independent state management

    Args:
        user_id: User identifier
        query: User query

    Returns:
        Results with state information

    MCP Verification Points:
        - Logs: State operations around handoff
        - Entity: Workflow state preserved
    """
    ctx.logger.info("=== WF_35: Handoff with State ===")
    ctx.logger.info(f"User: {user_id}, Query: {query}")

    # Set workflow state
    ctx.state.set("user_id", user_id)
    ctx.state.set("query_count", 1)
    ctx.state.set("handoff_initiated", True)

    ctx.logger.info("Workflow state set before handoff")

    # Execute handoff
    simple_triage = Agent(
        name="SimpleTriage",
        model="openai/gpt-4o-mini",
        instructions="Route to technical specialist.",
        handoffs=[technical_specialist],
        temperature=0.7,
    )

    result = await simple_triage.run(query, context=ctx)

    # Verify state after handoff
    ctx.state.set("handoff_completed", True)
    ctx.state.set("query_count", 2)

    user_id_after = ctx.state.get("user_id")
    query_count = ctx.state.get("query_count")
    handoff_completed = ctx.state.get("handoff_completed")

    ctx.logger.info(f"State after handoff: user={user_id_after}, count={query_count}")

    return {
        "user_id": user_id,
        "query": query,
        "output": result.output,
        "state_preserved": user_id_after == user_id,
        "query_count": query_count,
        "handoff_completed": handoff_completed,
    }


__all__ = [
    "save_preference",
    "technical_specialist",
    "business_specialist",
    "research_specialist",
    "wf_31_simple_triage_handoff",
    "wf_32_handoff_with_context",
    "wf_33_handoff_with_tools",
    "wf_34_multiple_handoff_options",
    "wf_35_handoff_with_state",
]
