"""
Agent Handoff Test Workflows

This module contains test cases for explicit agent handoffs - where control
is transferred from one agent to another using the handoff() pattern.

Handoffs vs Agent-as-Tool:
- **Handoffs**: Explicit transfer of control, target agent takes over completely
- **Agent-as-Tool**: Coordinator invokes specialist, gets result, continues

Test Coverage:
- Simple triage → specialist handoff
- Handoffs with context sharing and state persistence
- Complex workflows combining handoffs + agent-as-tool patterns
- Conversation history propagation across handoffs

Shared Components:
- create_technical_specialist(): Technical domain agent
- create_business_specialist(): Business domain agent
- create_research_specialist(): Research specialist agent
- save_user_preference: Tool for saving state

See also:
- agent_collaboration_workflows.py: Agent-as-tool pattern (different from handoffs)
- agent_tools_workflows.py: Regular tool usage
"""

from typing import Dict
from agnt5 import workflow, WorkflowContext, Agent, tool, Context, handoff
from .agent_tools_workflows import search_web, simple_calculator
from .agent_collaboration_workflows import domain_specific_analysis


# ============================================================================
# SHARED TOOLS FOR HANDOFF SCENARIOS
# ============================================================================


@tool(auto_schema=True)
async def save_user_preference(ctx: Context, key: str, value: str) -> str:
    """
    Save a user preference to context state.

    Args:
        key: Preference key
        value: Preference value

    Returns:
        Confirmation message
    """
    ctx.logger.info(f"[TOOL] save_user_preference: {key}={value}")

    prefs = ctx.get("user_preferences", {})
    prefs[key] = value
    ctx.set("user_preferences", prefs)

    return f"Saved preference: {key}={value}"


# ============================================================================
# MODULE-LEVEL AGENTS (for auto-registration)
# ============================================================================

# Technical specialist agent for software engineering questions
technical_specialist = Agent(
    name="TechnicalSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a technical specialist for software engineering questions.
    Provide detailed technical answers with code examples when appropriate.
    Be thorough and precise.""",
    temperature=0.7,
    max_iterations=5,
)

# Business specialist agent for business and strategy questions
business_specialist = Agent(
    name="BusinessSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a business specialist for business and strategy questions.
    Provide business-focused advice with practical recommendations.
    Consider ROI and business impact.""",
    temperature=0.7,
    max_iterations=5,
)

# Research specialist with search and analysis capabilities
research_specialist = Agent(
    name="ResearchSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a research specialist. You can:
    - Search the web for information
    - Perform data analysis
    Be thorough and provide detailed insights.""",
    tools=[search_web, domain_specific_analysis],
    temperature=0.7,
    max_iterations=5,
)

# Context-aware specialist for handoff scenarios
context_aware_specialist = Agent(
    name="ContextAwareSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a specialist agent receiving a handoff.

    IMPORTANT: Check the context for user preferences:
    - Use ctx.get('user_preferences', {}) to retrieve preferences
    - Personalize your response based on the user's name and preferences
    - Acknowledge the user by name in your response""",
    tools=[],
    temperature=0.7,
    max_iterations=5,
)

# Researcher agent for gathering information
researcher_agent = Agent(
    name="Researcher",
    model="openai/gpt-4o-mini",
    instructions="Research topics and gather information using web search.",
    tools=[search_web],
    temperature=0.7,
    max_iterations=3,
)

# Analyst agent for data analysis
analyst_agent = Agent(
    name="Analyst",
    model="openai/gpt-4o-mini",
    instructions="Analyze data and provide insights using analysis tools.",
    tools=[domain_specific_analysis],
    temperature=0.7,
    max_iterations=3,
)

# Triage agent with handoffs to specialists
triage_agent = Agent(
    name="TriageAgent",
    model="openai/gpt-4o-mini",
    instructions="""You are a triage agent that routes requests to specialists.

    Available specialists:
    - TechnicalSpecialist: For technical/engineering questions
    - BusinessSpecialist: For business/strategy questions

    Analyze the user's request:
    - If it's technical (code, architecture, engineering), transfer to TechnicalSpecialist
    - If it's business (strategy, pricing, marketing), transfer to BusinessSpecialist
    - Only handle simple general questions yourself

    Use the transfer_to_{specialist} tools for delegation.""",
    handoffs=[technical_specialist, business_specialist],
    temperature=0.7,
    max_iterations=3,
)

# Report writer that uses other agents as tools
report_writer_agent = Agent(
    name="ReportWriter",
    model="openai/gpt-4o-mini",
    instructions="""You are a report writer. You can:
    - Use the Researcher agent to gather information
    - Use the Analyst agent to analyze data
    - Use simple_calculator for calculations

    Synthesize findings into a well-structured report with:
    1. Executive Summary
    2. Research Findings
    3. Data Analysis
    4. Recommendations""",
    tools=[researcher_agent, analyst_agent, simple_calculator],  # Agents as tools!
    temperature=0.7,
    max_iterations=10,
)

# Executive that can handoff to report writer
executive_agent = Agent(
    name="Executive",
    model="openai/gpt-4o-mini",
    instructions="""You are an executive assistant.

    For simple questions, answer directly.
    For complex requests requiring comprehensive reports, transfer to ReportWriter.

    Transfer if the request asks for:
    - Research + analysis
    - Comprehensive reports
    - Multi-step investigations""",
    handoffs=[handoff(report_writer_agent, "Transfer to report writer for detailed reports")],
    temperature=0.7,
    max_iterations=3,
)

# Intake agent that collects info and hands off
intake_agent = Agent(
    name="IntakeAgent",
    model="openai/gpt-4o-mini",
    instructions="""You are an intake agent. Your job:
    1. Save the user's information using save_user_preference tool
    2. Save their communication preference (concise or detailed) by asking or inferring
    3. Transfer to ContextAwareSpecialist to handle their actual request

    Always transfer after collecting info.""",
    tools=[save_user_preference],
    handoffs=[
        handoff(
            context_aware_specialist,
            "Transfer to specialist after collecting user preferences",
            pass_full_history=True,  # Pass conversation history
        )
    ],
    temperature=0.7,
    max_iterations=5,
)


# ============================================================================
# TEST WORKFLOW 7: Simple Handoff (Triage → Specialist)
# ============================================================================


@workflow
async def test_handoff_simple(ctx: WorkflowContext, user_request: str) -> dict:
    """
    Test Scenario 7: Simple Handoff Pattern

    A triage agent analyzes the request and transfers control to the
    appropriate specialist using explicit handoffs. The specialist
    takes over completely and returns the final result.

    Args:
        user_request: User's request (technical, business, or general)

    Returns:
        Result with handoff metadata and final specialist output

    MCP Verification Points:
        - Logs: Triage decision and handoff transfer
        - Traces: Handoff span with original trace_id continuation
        - Events: Agent transition events (triage → specialist)
        - Entity: Both agents in workflow entity with handoff metadata

    Example requests:
        - "How do I implement OAuth2 in Python?" → TechnicalSpecialist
        - "What's a good pricing strategy for SaaS?" → BusinessSpecialist
        - "What is machine learning?" → Handled by triage (general)
    """
    ctx.logger.info("=== TEST 7: Simple Handoff ===")
    ctx.logger.info(f"User request: {user_request}")

    # Run triage agent (using module-level agent instance)
    ctx.logger.info("Executing triage agent...")
    result = await triage_agent.run(user_request, context=ctx)

    # Extract handoff info
    handoff_occurred = result.handoff_to is not None
    final_agent = result.handoff_to if handoff_occurred else "TriageAgent"

    ctx.logger.info(f"Final output: {result.output[:100]}...")
    ctx.logger.info(f"Handoff occurred: {handoff_occurred}")
    ctx.logger.info(f"Final agent: {final_agent}")

    # Check workflow entity
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    return {
        "scenario": "test_handoff_simple",
        "output": result.output,
        "handoff_occurred": handoff_occurred,
        "handoff_to": result.handoff_to,
        "final_agent": final_agent,
        "agents_in_workflow": all_agents,
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }


# ============================================================================
# TEST WORKFLOW 8: Handoff with Context Sharing
# ============================================================================


@workflow
async def test_handoff_with_context(ctx: WorkflowContext, user_name: str, user_request: str) -> dict:
    """
    Test Scenario 8: Handoff with Context Sharing

    An intake agent collects user information, saves it to context,
    then transfers to a specialist with full conversation history.
    The specialist uses the saved context to personalize the response.

    Args:
        user_name: User's name
        user_request: User's actual request

    Returns:
        Result with context sharing verification

    MCP Verification Points:
        - Logs: State mutations in both agents
        - Traces: Context propagation across handoff
        - Events: State update events from both agents
        - Entity: Shared state accessible to both agents

    Tests:
        - Context persistence via ctx.set()
        - Context retrieval via ctx.get()
        - Conversation history passing with pass_full_history=True
        - Personalized responses using saved context
    """
    ctx.logger.info("=== TEST 8: Handoff with Context Sharing ===")
    ctx.logger.info(f"User: {user_name}, Request: {user_request}")

    # Construct the full task with user name embedded
    full_task = f"User name is {user_name}. User request: {user_request}. Please save the user's name first, then handle their request."

    # Run intake agent (using module-level agent instance)
    ctx.logger.info("Executing intake agent...")
    result = await intake_agent.run(full_task, context=ctx)

    # Extract handoff and context info
    handoff_occurred = result.handoff_to is not None
    user_prefs = ctx.get("user_preferences", {})

    ctx.logger.info(f"Final output: {result.output[:100]}...")
    ctx.logger.info(f"Handoff occurred: {handoff_occurred}")
    ctx.logger.info(f"User preferences saved: {user_prefs}")

    # Check workflow entity
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    return {
        "scenario": "test_handoff_with_context",
        "output": result.output,
        "handoff_occurred": handoff_occurred,
        "handoff_to": result.handoff_to,
        "user_preferences_saved": user_prefs,
        "agents_in_workflow": all_agents,
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }


# ============================================================================
# TEST WORKFLOW 9: Complex Multi-Agent Workflow (Handoff + Agent-as-Tool)
# ============================================================================


@workflow
async def test_handoff_complex(ctx: WorkflowContext, task: str) -> dict:
    """
    Test Scenario 9: Complex Multi-Agent Workflow

    Combines BOTH patterns in a single workflow:
    - Executive agent with simple tools
    - Can handoff to report_writer
    - Report_writer uses researcher + analyst as TOOLS (not handoffs)

    This tests:
    - Nested agent execution with different collaboration patterns
    - Mixing handoffs and agent-as-tool in one workflow
    - Complex trace propagation across both patterns

    Args:
        task: Complex task requiring multiple agents and patterns

    Returns:
        Result with comprehensive multi-pattern metadata

    MCP Verification Points:
        - Logs: Mixed pattern execution (handoff + delegation)
        - Traces: Multiple nested spans (handoff + tool invocations)
        - Events: Both handoff and agent-as-tool events
        - Entity: 4 agents (executive, report_writer, researcher, analyst)

    Flow:
        1. Executive receives complex request
        2. Executive hands off to report_writer
        3. Report_writer uses researcher agent AS A TOOL
        4. Report_writer uses analyst agent AS A TOOL
        5. Report_writer synthesizes and returns final report
    """
    ctx.logger.info("=== TEST 9: Complex Multi-Agent Workflow ===")
    ctx.logger.info(f"Task: {task}")

    # Run executive agent (using module-level agent instance)
    ctx.logger.info("Executing executive agent...")
    result = await executive_agent.run(task, context=ctx)

    # Extract comprehensive metadata
    tool_calls = result.tool_calls
    handoff_occurred = result.handoff_to is not None

    # Classify tool calls
    agent_tool_calls = [tc for tc in tool_calls if tc["name"] in ["Researcher", "Analyst"]]
    regular_tool_calls = [tc for tc in tool_calls if tc["name"] not in ["Researcher", "Analyst"]]

    ctx.logger.info(f"Final output: {result.output[:100]}...")
    ctx.logger.info(f"Handoff occurred: {handoff_occurred}")
    ctx.logger.info(f"Total tool calls: {len(tool_calls)}")
    ctx.logger.info(f"Agent-as-tool calls: {len(agent_tool_calls)}")

    # Check workflow entity
    workflow_entity = ctx._workflow_entity
    all_agents = workflow_entity.list_agents()

    return {
        "scenario": "test_handoff_complex",
        "output": result.output,
        "handoff_occurred": handoff_occurred,
        "handoff_to": result.handoff_to,
        "total_tool_calls": len(tool_calls),
        "agent_as_tool_calls": len(agent_tool_calls),
        "regular_tool_calls": len(regular_tool_calls),
        "agents_in_workflow": all_agents,
        "tool_call_breakdown": {
            "agents_as_tools": [tc["name"] for tc in agent_tool_calls],
            "regular_tools": [tc["name"] for tc in regular_tool_calls],
        },
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
    }
