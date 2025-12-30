"""
Example: AGNT5 Multi-Agent Orchestration

This example demonstrates advanced multi-agent coordination patterns:
- Supervisor/Worker: Supervisor delegates subtasks to workers
- Debate pattern: Two agents argue, third decides
- Consensus pattern: Multiple agents must agree
- Pipeline pattern: Agents form processing pipeline
- Shared state: Agents communicate via shared state

Multi-agent orchestration enables:
- Complex task decomposition
- Diverse perspectives on problems
- Quality through consensus
- Efficient parallel processing

Each workflow/function can be:
- Run standalone: python ex_16_multi_agent_orchestration.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("supervisor_worker_workflow", {...}, component_type="workflow")

Prerequisites:
- Set OPENAI_API_KEY for agent-based examples
"""

import asyncio
import os
from typing import Any, Dict, List

from agnt5 import Agent, Context, FunctionContext, WorkflowContext, function, tool, workflow


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@tool(auto_schema=True)
async def register_contribution(ctx: Context, agent_name: str, contribution: str) -> dict:
    """
    Register an agent's contribution to shared state.

    Args:
        agent_name: Name of the contributing agent
        contribution: The contribution content

    Returns:
        Registration confirmation
    """
    ctx.logger.info(f"[register_contribution] {agent_name}: {contribution[:50]}...")
    return {
        "agent": agent_name,
        "contribution": contribution,
        "registered": True,
    }


@tool(auto_schema=True)
async def read_contributions(ctx: Context) -> list:
    """
    Read all registered contributions.

    Returns:
        List of all contributions
    """
    ctx.logger.info("[read_contributions] Reading all contributions")
    # In a real implementation, this would read from shared state
    return [
        {"agent": "agent1", "contribution": "Sample contribution 1"},
        {"agent": "agent2", "contribution": "Sample contribution 2"},
    ]


@tool(auto_schema=True)
async def vote(ctx: Context, agent_name: str, choice: str, reasoning: str) -> dict:
    """
    Cast a vote with reasoning.

    Args:
        agent_name: Name of the voting agent
        choice: The choice/vote
        reasoning: Reasoning for the vote

    Returns:
        Vote confirmation
    """
    ctx.logger.info(f"[vote] {agent_name} votes: {choice}")
    return {
        "agent": agent_name,
        "choice": choice,
        "reasoning": reasoning,
        "recorded": True,
    }


# =============================================================================
# SUPERVISOR/WORKER PATTERN
# =============================================================================


@workflow(chat=True)
async def supervisor_worker_workflow(ctx: WorkflowContext, task: str) -> dict:
    """
    Supervisor delegates subtasks to specialized workers.

    The supervisor breaks down the task and assigns to workers.
    Workers complete their subtasks and report back.
    Supervisor synthesizes the final result.

    Args:
        task: Complex task to be decomposed

    Returns:
        Dict with supervisor synthesis and worker results

    Example:
        result = client.run("supervisor_worker_workflow", {
            "task": "Research and summarize AI trends"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting supervisor/worker pattern for: {task}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "task": task,
        }

    # Create specialized worker agents
    research_worker = Agent(
        name="research_worker",
        model="openai/gpt-4o-mini",
        instructions="""You are a research worker.
        When given a research task, gather key information.
        Provide 3-5 bullet points of findings.
        Be factual and concise.""",
        temperature=0.5,
        max_iterations=3,
    )

    analysis_worker = Agent(
        name="analysis_worker",
        model="openai/gpt-4o-mini",
        instructions="""You are an analysis worker.
        When given data or findings, analyze patterns and implications.
        Provide 2-3 key insights.
        Be analytical and precise.""",
        temperature=0.5,
        max_iterations=3,
    )

    writing_worker = Agent(
        name="writing_worker",
        model="openai/gpt-4o-mini",
        instructions="""You are a writing worker.
        When given content, create a well-structured summary.
        Focus on clarity and readability.
        Keep it to one paragraph.""",
        temperature=0.6,
        max_iterations=3,
    )

    # Supervisor coordinates workers
    supervisor = Agent(
        name="supervisor",
        model="openai/gpt-4o-mini",
        instructions="""You are a supervisor coordinating a team.
        Break down complex tasks into subtasks.
        Assign work to your team members.
        Synthesize their results into a cohesive output.

        Your team:
        - research_worker: Gathers information
        - analysis_worker: Analyzes findings
        - writing_worker: Creates summaries

        Use each worker as a tool to complete the task.""",
        tools=[research_worker, analysis_worker, writing_worker],
        temperature=0.5,
        max_iterations=10,
    )

    result = await supervisor.run_sync(f"Complete this task: {task}")

    return {
        "task": task,
        "final_output": result.output,
        "tool_calls": len(result.tool_calls),
        "workers_used": [tc.get("name", "unknown") for tc in result.tool_calls],
        "pattern": "supervisor_worker",
    }


@workflow(chat=True)
async def hierarchical_delegation(ctx: WorkflowContext, project: str) -> dict:
    """
    Multi-level delegation hierarchy.

    Director -> Managers -> Workers

    Args:
        project: Project to complete

    Returns:
        Dict with hierarchical results

    Example:
        result = client.run("hierarchical_delegation", {
            "project": "Build a REST API"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting hierarchical delegation for: {project}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "project": project,
        }

    # Level 3: Workers
    backend_worker = Agent(
        name="backend_worker",
        model="openai/gpt-4o-mini",
        instructions="You implement backend tasks. Report what you would build.",
        temperature=0.5,
        max_iterations=3,
    )

    frontend_worker = Agent(
        name="frontend_worker",
        model="openai/gpt-4o-mini",
        instructions="You implement frontend tasks. Report what you would build.",
        temperature=0.5,
        max_iterations=3,
    )

    # Level 2: Managers (use workers)
    tech_manager = Agent(
        name="tech_manager",
        model="openai/gpt-4o-mini",
        instructions="""You are a technical manager.
        Coordinate backend and frontend workers.
        Delegate specific technical tasks.
        Report the overall technical plan.""",
        tools=[backend_worker, frontend_worker],
        temperature=0.5,
        max_iterations=6,
    )

    # Level 1: Director (uses managers)
    director = Agent(
        name="director",
        model="openai/gpt-4o-mini",
        instructions="""You are a project director.
        Coordinate the technical manager.
        Provide high-level project plan.
        Synthesize all reports into executive summary.""",
        tools=[tech_manager],
        temperature=0.5,
        max_iterations=8,
    )

    result = await director.run_sync(f"Execute project: {project}")

    return {
        "project": project,
        "final_output": result.output,
        "tool_calls": len(result.tool_calls),
        "hierarchy": ["director", "tech_manager", "backend_worker", "frontend_worker"],
        "pattern": "hierarchical_delegation",
    }


# =============================================================================
# DEBATE PATTERN
# =============================================================================


@workflow(chat=True)
async def debate_agents(ctx: WorkflowContext, topic: str) -> dict:
    """
    Two agents argue different perspectives, third decides.

    Agent A: Argues for
    Agent B: Argues against
    Judge: Makes final decision

    Args:
        topic: Topic to debate

    Returns:
        Dict with debate results and decision

    Example:
        result = client.run("debate_agents", {
            "topic": "Should AI systems be regulated?"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting debate on: {topic}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "topic": topic,
        }

    # Proponent agent
    proponent = Agent(
        name="proponent",
        model="openai/gpt-4o-mini",
        instructions="""You argue IN FAVOR of the topic.
        Present 3 strong arguments supporting the position.
        Be persuasive but fair.
        Keep response to 2-3 sentences per argument.""",
        temperature=0.7,
        max_iterations=3,
    )

    # Opponent agent
    opponent = Agent(
        name="opponent",
        model="openai/gpt-4o-mini",
        instructions="""You argue AGAINST the topic.
        Present 3 strong counterarguments.
        Be persuasive but fair.
        Keep response to 2-3 sentences per argument.""",
        temperature=0.7,
        max_iterations=3,
    )

    # Judge agent
    judge = Agent(
        name="judge",
        model="openai/gpt-4o-mini",
        instructions="""You are an impartial judge.
        Evaluate both sides of the debate objectively.
        Consider the strength of arguments.
        Make a reasoned decision with explanation.
        Format: Decision: [FOR/AGAINST], Reasoning: [explanation]""",
        temperature=0.3,
        max_iterations=3,
    )

    # Round 1: Opening statements
    pro_opening = await proponent.run_sync(f"Argue FOR: {topic}")
    con_opening = await opponent.run_sync(f"Argue AGAINST: {topic}")

    ctx.logger.info(f"Pro arguments: {pro_opening.output[:100]}...")
    ctx.logger.info(f"Con arguments: {con_opening.output[:100]}...")

    # Round 2: Rebuttals
    pro_rebuttal = await proponent.run_sync(
        f"Respond to this opposition:\n{con_opening.output}"
    )
    con_rebuttal = await opponent.run_sync(
        f"Respond to this argument:\n{pro_opening.output}"
    )

    # Judge's decision
    judgment = await judge.run_sync(
        f"""Topic: {topic}

FOR arguments:
{pro_opening.output}

AGAINST arguments:
{con_opening.output}

FOR rebuttal:
{pro_rebuttal.output}

AGAINST rebuttal:
{con_rebuttal.output}

Please evaluate and decide."""
    )

    return {
        "topic": topic,
        "pro_arguments": pro_opening.output,
        "con_arguments": con_opening.output,
        "pro_rebuttal": pro_rebuttal.output,
        "con_rebuttal": con_rebuttal.output,
        "judgment": judgment.output,
        "pattern": "debate",
    }


# =============================================================================
# CONSENSUS PATTERN
# =============================================================================


@workflow(chat=True)
async def consensus_agents(ctx: WorkflowContext, question: str, required_agreement: float = 0.6) -> dict:
    """
    Multiple agents must reach consensus on a decision.

    Each agent votes, consensus is determined by agreement threshold.

    Args:
        question: Question requiring consensus
        required_agreement: Required agreement ratio (0.0-1.0)

    Returns:
        Dict with votes and consensus result

    Example:
        result = client.run("consensus_agents", {
            "question": "Should we use microservices architecture?",
            "required_agreement": 0.6
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting consensus process for: {question}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "question": question,
        }

    # Create a panel of agents with different perspectives
    agents = [
        Agent(
            name="pragmatist",
            model="openai/gpt-4o-mini",
            instructions="""You are a pragmatic decision-maker.
            Consider practical implications and feasibility.
            Vote YES or NO with brief reasoning (1-2 sentences).""",
            temperature=0.5,
            max_iterations=3,
        ),
        Agent(
            name="innovator",
            model="openai/gpt-4o-mini",
            instructions="""You are an innovation-focused advisor.
            Consider long-term benefits and cutting-edge approaches.
            Vote YES or NO with brief reasoning (1-2 sentences).""",
            temperature=0.7,
            max_iterations=3,
        ),
        Agent(
            name="conservative",
            model="openai/gpt-4o-mini",
            instructions="""You are a risk-averse advisor.
            Consider stability, risks, and proven solutions.
            Vote YES or NO with brief reasoning (1-2 sentences).""",
            temperature=0.3,
            max_iterations=3,
        ),
        Agent(
            name="user_advocate",
            model="openai/gpt-4o-mini",
            instructions="""You advocate for end-user experience.
            Consider usability, accessibility, and user impact.
            Vote YES or NO with brief reasoning (1-2 sentences).""",
            temperature=0.5,
            max_iterations=3,
        ),
        Agent(
            name="cost_analyst",
            model="openai/gpt-4o-mini",
            instructions="""You focus on cost and resources.
            Consider budget, time, and resource requirements.
            Vote YES or NO with brief reasoning (1-2 sentences).""",
            temperature=0.4,
            max_iterations=3,
        ),
    ]

    # Collect votes from all agents
    votes = []
    for agent in agents:
        result = await agent.run_sync(f"Question: {question}\nPlease vote YES or NO with reasoning.")
        vote_text = result.output.upper()
        vote = "YES" if "YES" in vote_text else "NO" if "NO" in vote_text else "ABSTAIN"
        votes.append({
            "agent": agent.name,
            "vote": vote,
            "reasoning": result.output,
        })
        ctx.logger.info(f"{agent.name} votes: {vote}")

    # Calculate consensus
    yes_votes = sum(1 for v in votes if v["vote"] == "YES")
    no_votes = sum(1 for v in votes if v["vote"] == "NO")
    total_valid = yes_votes + no_votes

    if total_valid > 0:
        yes_ratio = yes_votes / total_valid
        no_ratio = no_votes / total_valid
    else:
        yes_ratio = no_ratio = 0

    if yes_ratio >= required_agreement:
        consensus = "YES"
        consensus_reached = True
    elif no_ratio >= required_agreement:
        consensus = "NO"
        consensus_reached = True
    else:
        consensus = "NO_CONSENSUS"
        consensus_reached = False

    return {
        "question": question,
        "required_agreement": required_agreement,
        "votes": votes,
        "vote_summary": {
            "yes": yes_votes,
            "no": no_votes,
            "abstain": len(votes) - total_valid,
        },
        "agreement_ratio": max(yes_ratio, no_ratio),
        "consensus_reached": consensus_reached,
        "consensus_decision": consensus,
        "pattern": "consensus",
    }


# =============================================================================
# PIPELINE PATTERN
# =============================================================================


@workflow(chat=True)
async def pipeline_agents(ctx: WorkflowContext, content: str) -> dict:
    """
    Agents form a processing pipeline.

    Each agent transforms the content and passes to the next.

    Args:
        content: Initial content to process

    Returns:
        Dict with pipeline stages and final output

    Example:
        result = client.run("pipeline_agents", {
            "content": "A new AI model was released that can understand images."
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting pipeline for content: {content[:50]}...")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "content": content,
        }

    # Stage 1: Extractor
    extractor = Agent(
        name="extractor",
        model="openai/gpt-4o-mini",
        instructions="""You extract key facts from content.
        List 3-5 key facts as bullet points.
        Be concise and factual.""",
        temperature=0.3,
        max_iterations=3,
    )

    # Stage 2: Enricher
    enricher = Agent(
        name="enricher",
        model="openai/gpt-4o-mini",
        instructions="""You enrich extracted facts with context.
        Add relevant background or implications for each fact.
        Keep additions brief.""",
        temperature=0.5,
        max_iterations=3,
    )

    # Stage 3: Synthesizer
    synthesizer = Agent(
        name="synthesizer",
        model="openai/gpt-4o-mini",
        instructions="""You synthesize enriched information into a coherent summary.
        Create a well-structured paragraph.
        Maintain all key information.""",
        temperature=0.5,
        max_iterations=3,
    )

    # Stage 4: Formatter
    formatter = Agent(
        name="formatter",
        model="openai/gpt-4o-mini",
        instructions="""You format content for presentation.
        Add a title, clean up formatting.
        Ensure professional appearance.""",
        temperature=0.3,
        max_iterations=3,
    )

    # Execute pipeline
    stages = []

    # Stage 1
    stage1_result = await extractor.run_sync(f"Extract key facts from:\n{content}")
    stages.append({"stage": "extract", "output": stage1_result.output})

    # Stage 2
    stage2_result = await enricher.run_sync(f"Enrich these facts:\n{stage1_result.output}")
    stages.append({"stage": "enrich", "output": stage2_result.output})

    # Stage 3
    stage3_result = await synthesizer.run_sync(f"Synthesize:\n{stage2_result.output}")
    stages.append({"stage": "synthesize", "output": stage3_result.output})

    # Stage 4
    stage4_result = await formatter.run_sync(f"Format:\n{stage3_result.output}")
    stages.append({"stage": "format", "output": stage4_result.output})

    return {
        "input_content": content,
        "stages": stages,
        "final_output": stage4_result.output,
        "pipeline_length": len(stages),
        "pattern": "pipeline",
    }


# =============================================================================
# SHARED STATE PATTERN
# =============================================================================


@workflow(chat=True)
async def shared_blackboard(ctx: WorkflowContext, problem: str) -> dict:
    """
    Agents communicate via shared state (blackboard pattern).

    Multiple agents read from and write to shared state.
    Each agent contributes their expertise to the solution.

    Args:
        problem: Problem requiring multiple perspectives

    Returns:
        Dict with blackboard state and solution

    Example:
        result = client.run("shared_blackboard", {
            "problem": "Design a user authentication system"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting shared blackboard for: {problem}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "problem": problem,
        }

    # Initialize shared blackboard state
    blackboard = {
        "problem": problem,
        "contributions": [],
        "constraints": [],
        "solutions": [],
    }

    # Define specialist agents
    security_expert = Agent(
        name="security_expert",
        model="openai/gpt-4o-mini",
        instructions="""You are a security expert.
        Review the problem and add security considerations.
        Contribute 2-3 security requirements or constraints.
        Format: CONSTRAINT: [description]""",
        temperature=0.5,
        max_iterations=3,
    )

    architect = Agent(
        name="architect",
        model="openai/gpt-4o-mini",
        instructions="""You are a system architect.
        Review the problem and existing constraints.
        Propose an architectural solution.
        Format: SOLUTION: [description]""",
        temperature=0.5,
        max_iterations=3,
    )

    reviewer = Agent(
        name="reviewer",
        model="openai/gpt-4o-mini",
        instructions="""You are a code reviewer.
        Review the proposed solutions.
        Identify potential issues or improvements.
        Format: REVIEW: [observations]""",
        temperature=0.5,
        max_iterations=3,
    )

    # Phase 1: Security expert adds constraints
    security_result = await security_expert.run_sync(
        f"Problem: {problem}\n\nAdd security constraints."
    )
    blackboard["contributions"].append({
        "agent": "security_expert",
        "content": security_result.output,
    })
    blackboard["constraints"].append(security_result.output)

    # Phase 2: Architect proposes solution considering constraints
    architect_result = await architect.run_sync(
        f"""Problem: {problem}

Constraints:
{blackboard['constraints'][-1]}

Propose a solution."""
    )
    blackboard["contributions"].append({
        "agent": "architect",
        "content": architect_result.output,
    })
    blackboard["solutions"].append(architect_result.output)

    # Phase 3: Reviewer evaluates
    reviewer_result = await reviewer.run_sync(
        f"""Problem: {problem}

Constraints:
{blackboard['constraints'][-1]}

Proposed Solution:
{blackboard['solutions'][-1]}

Review and provide feedback."""
    )
    blackboard["contributions"].append({
        "agent": "reviewer",
        "content": reviewer_result.output,
    })

    # Final synthesis
    final_solution = f"""
Problem: {problem}

Security Constraints:
{blackboard['constraints'][-1]}

Proposed Solution:
{blackboard['solutions'][-1]}

Review Feedback:
{reviewer_result.output}
"""

    return {
        "problem": problem,
        "blackboard": blackboard,
        "final_solution": final_solution.strip(),
        "contributors": ["security_expert", "architect", "reviewer"],
        "pattern": "shared_blackboard",
    }


@workflow(chat=True)
async def agent_messaging(ctx: WorkflowContext, task: str) -> dict:
    """
    Agents pass explicit messages to each other.

    Demonstrates structured message passing between agents.

    Args:
        task: Task requiring agent collaboration

    Returns:
        Dict with message history and result

    Example:
        result = client.run("agent_messaging", {
            "task": "Create a marketing plan"
        }, component_type="workflow")
    """
    ctx.logger.info(f"Starting agent messaging for: {task}")

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "task": task,
        }

    # Message queue
    messages = []

    # Define agents
    planner = Agent(
        name="planner",
        model="openai/gpt-4o-mini",
        instructions="""You are a strategic planner.
        Create high-level plans.
        Send clear task assignments to other team members.
        Format messages as: TO: [agent] MESSAGE: [content]""",
        temperature=0.5,
        max_iterations=3,
    )

    executor = Agent(
        name="executor",
        model="openai/gpt-4o-mini",
        instructions="""You are an executor who implements plans.
        When you receive a message, acknowledge and execute.
        Report completion or issues.
        Format: TO: [agent] STATUS: [complete/issue] DETAILS: [content]""",
        temperature=0.5,
        max_iterations=3,
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="""You coordinate between team members.
        Track progress and resolve issues.
        Provide final summary when complete.""",
        temperature=0.5,
        max_iterations=3,
    )

    # Message 1: Coordinator initiates
    msg1 = await coordinator.run_sync(
        f"New task: {task}\nAssign to planner."
    )
    messages.append({"from": "coordinator", "to": "planner", "content": msg1.output})

    # Message 2: Planner creates plan
    msg2 = await planner.run_sync(
        f"Received: {msg1.output}\nCreate plan and assign to executor."
    )
    messages.append({"from": "planner", "to": "executor", "content": msg2.output})

    # Message 3: Executor responds
    msg3 = await executor.run_sync(
        f"Received plan: {msg2.output}\nExecute and report."
    )
    messages.append({"from": "executor", "to": "coordinator", "content": msg3.output})

    # Message 4: Coordinator summarizes
    msg4 = await coordinator.run_sync(
        f"Received status: {msg3.output}\nProvide final summary."
    )
    messages.append({"from": "coordinator", "to": "all", "content": msg4.output})

    return {
        "task": task,
        "messages": messages,
        "message_count": len(messages),
        "final_summary": msg4.output,
        "pattern": "agent_messaging",
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Multi-Agent Orchestration Examples")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set. Set it to run agent examples.")
        print("   export OPENAI_API_KEY=sk-...")
        return

    ctx = Context(run_id="multi-agent-demo")

    # Supervisor/Worker
    print("\n--- Supervisor/Worker Pattern ---")
    result = await supervisor_worker_workflow(ctx, task="Summarize recent AI developments")
    print(f"Workers used: {result.get('workers_used', [])}")
    print(f"Output: {result.get('final_output', '')[:200]}...")

    # Debate
    print("\n--- Debate Pattern ---")
    result = await debate_agents(ctx, topic="Remote work is better than office work")
    print(f"Judgment: {result.get('judgment', '')[:200]}...")

    # Consensus
    print("\n--- Consensus Pattern ---")
    result = await consensus_agents(ctx, question="Should we use Python for this project?")
    print(f"Consensus: {result.get('consensus_decision', 'N/A')}")
    print(f"Agreement: {result.get('agreement_ratio', 0):.0%}")

    # Pipeline
    print("\n--- Pipeline Pattern ---")
    result = await pipeline_agents(ctx, content="A new AI model can generate images from text.")
    print(f"Stages: {[s['stage'] for s in result.get('stages', [])]}")
    print(f"Output: {result.get('final_output', '')[:200]}...")

    # Shared Blackboard
    print("\n--- Shared Blackboard Pattern ---")
    result = await shared_blackboard(ctx, problem="Design a login system")
    print(f"Contributors: {result.get('contributors', [])}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
