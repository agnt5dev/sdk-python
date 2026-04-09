"""
Example: AGNT5 MCP Server

Expose AGNT5 tools, agents, workflows, prompts, and resources as an MCP server.

Run with:
    python ex_18_mcp_server.py

Then connect from an MCP-compatible client over stdio.
"""

import asyncio

from agnt5 import Agent, Context, MCPServer, Prompt, Resource, tool, workflow
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, TokenUsage


class DemoLanguageModel(LanguageModel):
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            text="This is a demo agent response.",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, request: GenerateRequest):
        if False:
            yield None


@tool
async def echo(ctx: Context, message: str) -> str:
    """Echo a message."""
    return f"echo:{message}"


@workflow()
async def summarize_topic(ctx, input: dict) -> dict:
    return {"topic": input["topic"], "summary": "demo-summary"}


async def build_prompt(topic: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": {"type": "text", "text": f"Research {topic}"},
            }
        ]
    }


async def read_handbook() -> str:
    return "# Demo Handbook"


async def main() -> None:
    server = MCPServer(
        id="demo-mcp",
        name="Demo MCP Server",
        version="1.0.0",
        tools={"echo": echo},
        agents={
            "research_agent": Agent(
                name="research_agent",
                model=DemoLanguageModel(),
                instructions="You are a helpful demo agent.",
            )
        },
        workflows={"summarize_topic": summarize_topic},
        prompts={
            "research_brief": Prompt(
                name="research_brief",
                description="Build a research brief",
                arguments_schema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
                handler=build_prompt,
            )
        },
        resources={
            "docs://handbook": Resource.text(
                uri="docs://handbook",
                name="Demo Handbook",
                mime_type="text/markdown",
                read=read_handbook,
            )
        },
    )
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
