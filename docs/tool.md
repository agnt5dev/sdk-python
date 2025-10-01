# Tool Component

## What is a Tool?

A **Tool** in AGNT5 is a callable capability that extends what agents can do. Tools provide structured interfaces to functions, APIs, services, and other agents, with automatic schema extraction from Python code. Tools are the hands and eyes of agents - enabling them to search, analyze, compute, and interact with external systems.

**Key Characteristics:**
- **Automatic Schema**: Extract input/output schemas from Python docstrings and type hints
- **Multiple Types**: Function, Hosted, MCP, OpenAPI, and Agent tools
- **Built on Function**: Inherits durability and retry logic from Function primitive
- **Confirmation Policies**: Optional user approval for dangerous operations
- **Rich Metadata**: Descriptions, examples, and parameter constraints

## Why are Tools Needed?

### 1. Extend Agent Capabilities

Agents alone can only generate text - tools give them real-world abilities:

```python
from agnt5 import Agent, tool

@tool(auto_schema=True)
def search_web(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search the web for information.

    Args:
        query: The search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, and snippet
    """
    # Implementation
    return search_results

agent = Agent(
    name="researcher",
    model=lm,
    tools=[search_web]  # Agent can now search the web
)

result = await agent.run("What are the latest developments in quantum computing?")
# Agent automatically calls search_web tool with appropriate query
```

### 2. Reusable Function Libraries

Define tools once, use across multiple agents:

```python
# Define domain-specific tools
@tool(auto_schema=True)
def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code for quality issues."""
    pass

@tool(auto_schema=True)
def run_tests(test_file: str) -> Dict[str, Any]:
    """Execute test suite and return results."""
    pass

@tool(auto_schema=True)
def format_code(code: str, style: str = "pep8") -> str:
    """Format code according to style guide."""
    pass

# Multiple agents share the same toolset
code_reviewer = Agent(name="reviewer", tools=[analyze_code, run_tests])
code_fixer = Agent(name="fixer", tools=[analyze_code, format_code])
```

### 3. Safe Execution with Confirmation

Require approval for dangerous operations:

```python
@tool(auto_schema=True, confirmation=True)
def delete_database(database_name: str) -> Dict[str, str]:
    """Delete a database permanently.

    Args:
        database_name: Name of the database to delete

    Returns:
        Status of deletion operation

    Warning:
        This operation is irreversible and will delete all data.
    """
    # Requires human approval before execution
    pass

# Agent proposes deletion but waits for approval
agent = Agent(name="admin", tools=[delete_database])
result = await agent.run("Clean up the test database")
# User receives confirmation prompt before tool executes
```

## How to Use Tools

### Function Tools with Auto-Schema

The simplest way to create tools is with the `@tool()` decorator:

```python
from agnt5 import tool

@tool(auto_schema=True)
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.

    Args:
        length: Length of the rectangle in meters
        width: Width of the rectangle in meters

    Returns:
        Area in square meters

    Examples:
        >>> calculate_area(5.0, 3.0)
        15.0
    """
    return length * width

# Schema automatically extracted:
# {
#   "name": "calculate_area",
#   "description": "Calculate the area of a rectangle.",
#   "input_schema": {
#     "type": "object",
#     "properties": {
#       "length": {"type": "number", "description": "Length of the rectangle in meters"},
#       "width": {"type": "number", "description": "Width of the rectangle in meters"}
#     },
#     "required": ["length", "width"]
#   }
# }
```

### Manual Schema Definition

For more control, define schemas explicitly:

```python
from agnt5 import Tool

search_tool = Tool(
    name="search",
    description="Search for information",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "filters": {"type": "object"}
        },
        "required": ["query"]
    },
    handler=search_function
)
```

### Hosted Tools (AGNT5 Workers)

Tools can be deployed as durable AGNT5 workers:

```python
from agnt5 import worker
from agnt5.tools import HostedTool

# Define worker function
@worker.handler
def analyze_data(data: Dict) -> Dict:
    """Worker function for complex data analysis."""
    # Heavy computation here
    return analysis_results

# Create hosted tool pointing to worker
analysis_tool = HostedTool(
    name="analyze_data",
    description="Perform complex data analysis",
    endpoint="agnt5://data-analysis-service/analyze_data"
)

# Agent uses hosted tool with automatic retries and durability
agent = Agent(name="analyst", tools=[analysis_tool])
```

### MCP Tools (Model Context Protocol)

Integrate with MCP servers:

```python
from agnt5.tools import MCPTool

# Connect to MCP server
filesystem_tool = MCPTool(
    name="filesystem",
    mcp_server_url="http://localhost:3000/mcp",
    capabilities=["read_file", "write_file", "list_directory"]
)

agent = Agent(name="file_assistant", tools=[filesystem_tool])
```

### OpenAPI Tools

Automatically generate tools from OpenAPI specs:

```python
from agnt5.tools import OpenAPITool

# Create tools from OpenAPI specification
github_tools = OpenAPITool.from_spec(
    spec_url="https://api.github.com/openapi.json",
    operations=["get_repo", "list_issues", "create_issue"]
)

agent = Agent(name="github_bot", tools=github_tools)
```

## Common Patterns

### Tool Composition

Combine multiple tools for complex capabilities:

```python
@tool(auto_schema=True)
def search_papers(query: str, year_from: int = 2020) -> List[Dict]:
    """Search academic papers."""
    pass

@tool(auto_schema=True)
def download_pdf(url: str) -> bytes:
    """Download PDF document."""
    pass

@tool(auto_schema=True)
def extract_text(pdf_data: bytes) -> str:
    """Extract text from PDF."""
    pass

# Agent orchestrates multiple tools
research_agent = Agent(
    name="researcher",
    tools=[search_papers, download_pdf, extract_text],
    instructions="Search papers, download them, and extract key findings."
)

result = await research_agent.run("Survey recent work on transformer architectures")
# Agent chains: search_papers → download_pdf → extract_text
```

### Conditional Tool Execution

Tools with prerequisite checks:

```python
@tool(auto_schema=True)
def check_balance(account_id: str) -> Dict[str, float]:
    """Check account balance."""
    return {"account_id": account_id, "balance": 1000.0}

@tool(auto_schema=True, confirmation=True)
def transfer_funds(from_account: str, to_account: str, amount: float) -> Dict:
    """Transfer funds between accounts.

    Args:
        from_account: Source account ID
        to_account: Destination account ID
        amount: Amount to transfer

    Returns:
        Transfer confirmation with transaction ID
    """
    # Check balance first (agent learns to do this)
    balance = check_balance(from_account)
    if balance["balance"] < amount:
        raise ValueError("Insufficient funds")

    # Perform transfer
    return {"transaction_id": "txn_123", "status": "completed"}

agent = Agent(
    name="banking_assistant",
    tools=[check_balance, transfer_funds],
    instructions="Always check balance before transfers."
)
```

### Tool Error Handling

Tools with robust error handling:

```python
@tool(auto_schema=True)
def fetch_stock_price(symbol: str) -> Dict[str, Any]:
    """Fetch current stock price.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Stock price data with current price, change, and volume

    Raises:
        ValueError: If symbol is invalid
        ConnectionError: If market data service is unavailable
    """
    try:
        # Fetch from market data API
        price_data = market_api.get_price(symbol)
        return {
            "symbol": symbol,
            "price": price_data.current,
            "change": price_data.change,
            "volume": price_data.volume
        }
    except InvalidSymbolError:
        raise ValueError(f"Invalid stock symbol: {symbol}")
    except MarketAPIError as e:
        raise ConnectionError(f"Market data unavailable: {e}")

# Agent handles tool errors gracefully
agent = Agent(name="stock_advisor", tools=[fetch_stock_price])
result = await agent.run("What's the price of AAPL?")
# If tool fails, agent can retry or inform user
```

### Dynamic Tool Registration

Register tools at runtime based on context:

```python
# Base toolset
base_tools = [search_tool, calculate_tool]

# Add specialized tools based on user role
if user.role == "admin":
    admin_tools = [delete_user_tool, modify_permissions_tool]
    all_tools = base_tools + admin_tools
else:
    all_tools = base_tools

agent = Agent(
    name="assistant",
    tools=all_tools,
    instructions=f"You are assisting a {user.role}."
)
```

### Tool with Context Access

Tools can access execution context for advanced operations:

```python
from agnt5 import tool, Context

@tool(auto_schema=True)
async def store_memory(ctx: Context, key: str, value: str) -> Dict[str, str]:
    """Store information in long-term memory.

    Args:
        ctx: Execution context (automatically provided)
        key: Memory key
        value: Content to store

    Returns:
        Confirmation of storage
    """
    # Access context for durable storage
    await ctx.memory.set(key, value)

    return {
        "status": "stored",
        "key": key,
        "timestamp": ctx.now()
    }

# Context is automatically injected when tool is called
agent = Agent(name="memory_agent", tools=[store_memory])
```

## Best Practices

### 1. Write Clear Tool Descriptions

Good descriptions help agents use tools correctly:

```python
# Good - Clear, specific description
@tool(auto_schema=True)
def search_documentation(query: str, language: str = "python") -> List[Dict]:
    """Search official language documentation for code examples and API references.

    Use this tool when you need to find specific functions, classes, or usage
    examples from official documentation. Returns relevant documentation sections
    with code examples.

    Args:
        query: Specific function name, class, or concept to search for
        language: Programming language (python, javascript, go, rust)

    Returns:
        List of documentation sections with title, url, and code examples

    Examples:
        >>> search_documentation("asyncio.gather", "python")
        [{"title": "asyncio.gather", "url": "...", "example": "await asyncio.gather(...)"}]
    """
    pass

# Avoid - Vague description
@tool(auto_schema=True)
def search(q: str) -> List:
    """Search for stuff."""  # Too vague - agent won't know when to use this
    pass
```

### 2. Use Type Hints and Docstrings

Enable automatic schema extraction:

```python
from typing import List, Dict, Optional

@tool(auto_schema=True)
def analyze_sentiment(
    text: str,
    language: str = "en",
    return_scores: bool = False
) -> Dict[str, Any]:
    """Analyze sentiment of text.

    Args:
        text: Text to analyze (minimum 10 characters)
        language: ISO language code (en, es, fr, de)
        return_scores: Include detailed confidence scores

    Returns:
        Sentiment analysis with label (positive/negative/neutral)
        and optional confidence scores
    """
    # Type hints + docstring = complete schema
    pass
```

### 3. Implement Confirmation for Dangerous Operations

Protect users from destructive actions:

```python
# Dangerous operations should require confirmation
@tool(auto_schema=True, confirmation=True)
def execute_code(code: str, language: str = "python") -> Dict[str, str]:
    """Execute arbitrary code in a sandboxed environment.

    Warning:
        Code execution can be dangerous. This tool requires explicit user approval.
    """
    pass

@tool(auto_schema=True, confirmation=True)
def send_email_blast(recipients: List[str], subject: str, body: str) -> Dict:
    """Send email to multiple recipients.

    Warning:
        Bulk email requires confirmation to prevent spam.
    """
    pass
```

## Architecture

Tools are built on AGNT5's Function primitive:

1. **Function Foundation**: Each tool wraps a durable function with retry policies
2. **Schema Layer**: Automatic extraction from Python type hints and docstrings
3. **Agent Integration**: Tools registered with agents for LLM-driven invocation
4. **Execution Modes**:
   - **FunctionTool**: Direct Python function execution
   - **HostedTool**: Remote execution via AGNT5 workers
   - **MCPTool**: Proxy to MCP servers
   - **OpenAPITool**: Generated from OpenAPI specs
5. **Durability**: All tool executions benefit from function-level durability and checkpointing

## Comparison with Function

| Aspect | Function | Tool |
|--------|----------|------|
| Purpose | General computation | Agent capability |
| Schema | Optional | Required (auto-generated) |
| Discovery | Manual invocation | Agent-driven selection |
| Metadata | Basic | Rich (description, examples, confirmation) |
| Use Case | Backend logic | Agent actions |

**When to use Function:**
- Backend processing
- Internal system operations
- Not exposed to agents

**When to use Tool:**
- Agent capabilities
- External system integration
- User-facing operations

## See Also

- [Function Component](function.md) - Underlying primitive for tools
- [Agent Component](agent.md) - Agents use tools for actions
- [Context API](context.md) - Tool context operations
- [Worker](../sdk/python/workers.md) - Hosted tool deployment