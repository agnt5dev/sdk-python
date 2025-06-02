# AGNT5 Python SDK

Python SDK for building durable, resilient agent-first applications with the AGNT5 platform.

## Features

- High-level AI components (Agents, Tools, Workflows)
- Durable execution with automatic failure recovery
- Memory systems (working, episodic, semantic, procedural)
- Multi-provider LLM support (Anthropic, OpenAI, Google, etc.)
- Rust-powered core for optimal performance

## Installation

```bash
pip install agnt5
```

## Quick Start

```python
from agnt5 import Agent, tool

@tool
def search_web(query: str) -> str:
    # Your search implementation
    return f"Results for: {query}"

agent = Agent(
    name="assistant",
    instructions="You are a helpful AI assistant",
    tools=[search_web],
    memory=True
)

response = await agent.run("Search for Python tutorials")
```

## Documentation

See the full documentation at [docs.agnt5.dev](https://docs.agnt5.dev)
