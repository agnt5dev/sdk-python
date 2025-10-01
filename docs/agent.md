# Agent Component

## What is an Agent?

An **Agent** in AGNT5 is an autonomous LLM-driven system that can reason, plan, and execute tasks using tools. Agents orchestrate complex multi-step workflows by breaking down problems, selecting appropriate tools, and iterating until tasks are complete. Agents combine language models, tools, memory, and sessions to deliver intelligent, context-aware interactions.

**Key Characteristics:**
- **LLM-Powered**: Driven by language models for reasoning and decision-making
- **Tool Orchestration**: Automatically selects and executes appropriate tools
- **Memory Integration**: Maintains long-term knowledge across conversations
- **Session Aware**: Uses sessions for conversation context and multi-agent coordination
- **Streaming Support**: Real-time event streaming for responsive UX
- **Durable by Default**: Built on AGNT5 primitives for automatic fault tolerance

## Why are Agents Needed?

### 1. Autonomous Task Execution

Agents break down complex tasks and execute them autonomously:

```python
from agnt5 import Agent, LanguageModel, tool

@tool(auto_schema=True)
def search_papers(query: str) -> List[Dict]:
    """Search academic papers."""
    pass

@tool(auto_schema=True)
def analyze_paper(paper_url: str) -> Dict:
    """Analyze paper content."""
    pass

lm = LanguageModel()
agent = Agent(
    name="researcher",
    model=lm,
    instructions="You are a research assistant. Break down complex research tasks.",
    tools=[search_papers, analyze_paper]
)

# Agent autonomously:
# 1. Searches for relevant papers
# 2. Analyzes each paper
# 3. Synthesizes findings
result = await agent.run("Summarize recent work on transformer architectures")
```

### 2. Multi-Step Reasoning with Tools

Agents chain tool calls based on reasoning:

```python
# Agent decides tool execution order based on context
agent = Agent(
    name="analyst",
    model=lm,
    tools=[
        search_web,
        fetch_stock_data,
        calculate_metrics,
        generate_chart
    ],
    instructions="Analyze companies thoroughly before making recommendations."
)

result = await agent.run("Should I invest in Tesla?")

# Agent's reasoning chain:
# 1. search_web("Tesla recent news")
# 2. fetch_stock_data("TSLA")
# 3. calculate_metrics(stock_data)
# 4. generate_chart(metrics)
# 5. Synthesize analysis and recommendation
```

### 3. Multi-Agent Collaboration

Multiple specialized agents work together on complex tasks:

```python
from agnt5 import Session

# Shared session for agent coordination
session = Session(id="code-review-123", user_id="developer-456")

# Specialized agents
code_analyzer = Agent(
    name="analyzer",
    model=lm,
    tools=[lint_tool, complexity_tool],
    session=session
)

security_checker = Agent(
    name="security",
    model=lm,
    tools=[vuln_scan_tool, dependency_check_tool],
    session=session
)

# Agents share context through session
analysis = await code_analyzer.run("Analyze code quality")
security = await security_checker.run("Check for security issues")
# Both agents see shared context and each other's findings
```

## How to Use Agents

### Basic Agent Creation

```python
from agnt5 import Agent, LanguageModel

lm = LanguageModel()

agent = Agent(
    name="assistant",
    model=lm,
    instructions="You are a helpful coding assistant."
)

# Simple agent without tools
result = await agent.run("Explain recursion")
print(result.output)
```

### Agent with Tools

```python
from agnt5 import Agent, tool

@tool(auto_schema=True)
def search_docs(query: str, language: str = "python") -> List[Dict]:
    """Search programming language documentation."""
    pass

@tool(auto_schema=True)
def run_code(code: str, language: str = "python") -> Dict[str, str]:
    """Execute code and return output."""
    pass

agent = Agent(
    name="coding_assistant",
    model=lm,
    instructions="""You are a coding assistant. Help users write and test code.
    Use search_docs to find API references.
    Use run_code to test code examples.""",
    tools=[search_docs, run_code]
)

result = await agent.run("How do I read a file in Python? Show me an example.")
# Agent searches docs, generates example, tests it with run_code
```

### Agent with Session and Memory

```python
from agnt5 import Agent, Session, Memory, LanguageModel

# Create session for conversation
session = Session(
    id="tutoring-session-789",
    user_id="student-123",
    metadata={"subject": "mathematics"}
)

# Create memory for long-term knowledge
memory = Memory(service=VectorMemoryService())
await memory.store("student_level", "Advanced calculus, struggles with proofs")

# Create agent with context
lm = LanguageModel()
agent = Agent(
    name="math_tutor",
    model=lm,
    instructions="You are a patient math tutor. Adapt to student's level.",
    tools=[solve_equation_tool, plot_function_tool],
    session=session,
    memory=memory
)

# Agent uses memory and session for personalized tutoring
result = await agent.run("Help me understand the epsilon-delta definition")
# Agent recalls student level from memory
# Agent maintains conversation in session
```

### Streaming Agent Responses

```python
async for event in agent.stream("Analyze this large dataset", session=session):
    match event.type:
        case "thinking":
            print(f"🤔 {event.content}")
        case "tool_call":
            print(f"🔧 Calling {event.tool_name}({event.arguments})")
        case "tool_result":
            print(f"✓ Result: {event.result}")
        case "response":
            print(f"💬 {event.content}")
        case "error":
            print(f"❌ Error: {event.error}")
```

### Agent Planning

Preview what an agent will do before execution:

```python
# Get execution plan without running
plan = agent.plan("Analyze competitor pricing strategies")

print(f"Estimated steps: {len(plan.steps)}")
for step in plan.steps:
    print(f"- {step.type}: {step.description}")
    if step.tool:
        print(f"  Tool: {step.tool.name}")

# Review plan, then execute if approved
if user_approves(plan):
    result = await agent.run("Analyze competitor pricing strategies")
```

## Common Patterns

### Research Agent Pattern

```python
from agnt5 import Agent, Session, Memory, tool

@tool(auto_schema=True)
def search_academic(query: str, year_from: int = 2020) -> List[Dict]:
    """Search academic papers."""
    pass

@tool(auto_schema=True)
def extract_insights(paper_text: str) -> Dict[str, List[str]]:
    """Extract key insights from paper."""
    pass

# Research session
session = Session(id="research-ai-safety-001", user_id="researcher-123")
memory = Memory(service=VectorMemoryService())

research_agent = Agent(
    name="research_agent",
    model=lm,
    instructions="""You are a research assistant specializing in AI safety.

    Research process:
    1. Search for relevant recent papers
    2. Extract key insights from each paper
    3. Identify common themes and gaps
    4. Synthesize findings into comprehensive summary

    Focus on papers from 2020 onwards.""",
    tools=[search_academic, extract_insights],
    session=session,
    memory=memory
)

result = await research_agent.run(
    "Survey the current state of AI alignment research"
)

# Store findings in long-term memory
await memory.ingest_from_session(session, strategy="smart")
```

### Multi-Agent Workflow

```python
# Coordinator pattern for complex workflows
session = Session(id="product-launch-001", user_id="pm-456")

# Specialized agents
market_researcher = Agent(
    name="market_analyst",
    model=lm,
    tools=[market_data_tool, competitor_analysis_tool],
    session=session,
    instructions="Analyze market opportunities and competitive landscape."
)

product_designer = Agent(
    name="designer",
    model=lm,
    tools=[design_tool, user_research_tool],
    session=session,
    instructions="Design products based on market research and user needs."
)

technical_lead = Agent(
    name="tech_lead",
    model=lm,
    tools=[architecture_tool, feasibility_tool],
    session=session,
    instructions="Assess technical feasibility and propose architecture."
)

# Sequential execution with shared context
market_analysis = await market_researcher.run(
    "Analyze market for AI-powered code review tools"
)

product_specs = await product_designer.run(
    "Design product based on market analysis"
)

tech_assessment = await technical_lead.run(
    "Evaluate technical feasibility of proposed product"
)

# All agents see shared context and previous outputs
```

### Agent Handoff Pattern

```python
from agnt5.tools import AgentTool

# Create specialized agents
billing_agent = Agent(
    name="billing_specialist",
    model=lm,
    tools=[payment_tool, invoice_tool, refund_tool],
    instructions="Handle billing, payments, and refunds."
)

technical_agent = Agent(
    name="tech_support",
    model=lm,
    tools=[diagnostic_tool, fix_tool, escalation_tool],
    instructions="Diagnose and fix technical issues."
)

# Coordinator with agent handoff capability
coordinator = Agent(
    name="coordinator",
    model=lm,
    tools=[
        classify_request_tool,
        AgentTool(target_agent=billing_agent),
        AgentTool(target_agent=technical_agent)
    ],
    instructions="""You are a support coordinator.
    Classify requests and hand off to appropriate specialist.

    Hand off to:
    - billing_specialist: payment, invoice, refund questions
    - tech_support: technical issues, bugs, troubleshooting"""
)

session = Session(id="support-ticket-789", user_id="customer-123")

result = await coordinator.run(
    "I was charged twice for my subscription",
    session=session
)
# Coordinator automatically hands off to billing_agent
```

### Agent with Human-in-the-Loop

```python
@tool(auto_schema=True, confirmation=True)
def deploy_to_production(version: str) -> Dict[str, str]:
    """Deploy application to production.

    Warning: Requires human approval.
    """
    pass

deployment_agent = Agent(
    name="deployer",
    model=lm,
    tools=[run_tests_tool, deploy_to_production],
    instructions="""Run all tests before deploying.
    Always request human approval for production deployments."""
)

result = await deployment_agent.run("Deploy version 2.0 to production")
# Agent runs tests automatically
# Requests human approval before deploy_to_production
# Waits for approval signal before proceeding
```

### Iterative Problem Solving

```python
debugging_agent = Agent(
    name="debugger",
    model=lm,
    tools=[
        analyze_logs_tool,
        run_diagnostic_tool,
        apply_fix_tool,
        verify_fix_tool
    ],
    instructions="""You are a debugging assistant.

    Process:
    1. Analyze error logs to identify root cause
    2. Run diagnostics to confirm hypothesis
    3. Apply potential fix
    4. Verify fix works
    5. If not fixed, iterate (max 3 attempts)

    Always verify fixes before considering issue resolved."""
)

result = await debugging_agent.run(
    "Users are experiencing 500 errors on the checkout page"
)
# Agent iteratively debugs until issue is resolved or max attempts reached
```

## Best Practices

### 1. Write Clear Instructions

Good instructions help agents make better decisions:

```python
# Good - Specific, actionable instructions
agent = Agent(
    name="code_reviewer",
    model=lm,
    tools=[analyze_code_tool, suggest_improvements_tool],
    instructions="""You are an expert code reviewer specializing in Python.

    Review process:
    1. Analyze code for common issues (complexity, duplication, style)
    2. Check for security vulnerabilities and edge cases
    3. Suggest specific improvements with code examples
    4. Prioritize: security > correctness > performance > style

    Be constructive and explain your reasoning."""
)

# Avoid - Vague instructions
agent = Agent(
    name="helper",
    model=lm,
    tools=[tool1, tool2],
    instructions="Help the user with stuff."  # Too vague
)
```

### 2. Use Sessions for Multi-Agent Coordination

Share context across agents:

```python
# Create shared session
session = Session(id="project-workflow-123", user_id="user-456")

# Set shared context
session.set_state("project_name", "ai-safety-research")
session.set_state("deadline", "2024-12-31")

# All agents access shared context
agent1 = Agent(name="agent1", session=session, ...)
agent2 = Agent(name="agent2", session=session, ...)

# Context automatically shared
```

### 3. Leverage Memory for Long-Term Context

Use Memory for knowledge that persists across sessions:

```python
# Store user expertise in long-term memory
await memory.store("user_expertise", "Expert in React and TypeScript")
await memory.store("coding_style", "Prefers functional programming")

# Agent recalls context automatically
agent = Agent(
    name="assistant",
    model=lm,
    tools=[code_gen_tool],
    memory=memory
)

# Memory influences agent's responses
result = await agent.run("Help me build a component")
# Agent generates React/TypeScript with functional style
```

## Architecture

Agents orchestrate AGNT5 primitives:

1. **LLM Core**: Language model for reasoning and planning
2. **Tool Execution**: Tools built on Function primitive with durability
3. **State Management**: Sessions use Entity for conversation state
4. **Long-Term Storage**: Memory uses Entity for knowledge persistence
5. **Orchestration**: Agent decision loop uses Workflow patterns internally
6. **Streaming**: Real-time event emission for responsive UX

```
Agent
├── LanguageModel (reasoning)
├── Tools (actions via Function)
├── Session (context via Entity)
├── Memory (knowledge via Entity)
└── Planner (orchestration via Workflow patterns)
```

## Comparison with Other Components

| Aspect          | Function         | Workflow           | Agent              |
| --------------- | ---------------- | ------------------ | ------------------ |
| Autonomy        | None             | Scripted           | Autonomous         |
| Decision Making | Pre-programmed   | Control flow       | LLM-driven         |
| Tool Use        | N/A              | Explicit calls     | Dynamic selection  |
| Adaptability    | Fixed            | Fixed steps        | Adaptive reasoning |
| Use Case        | Single operation | Multi-step process | Complex tasks      |

**When to use Function:**
- Single, deterministic operation
- No decision-making needed

**When to use Workflow:**
- Pre-defined multi-step process
- Explicit control flow

**When to use Agent:**
- Complex, open-ended tasks
- Requires reasoning and adaptation
- Dynamic tool selection needed

## See Also

- [Session Component](session.md) - Agent conversation context
- [Tool Component](tool.md) - Agent capabilities
- [Memory Component](memory.md) - Agent long-term knowledge
- [Workflow Component](workflow.md) - Orchestration patterns
- [Context API](context.md) - Agent execution context