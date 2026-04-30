"""Example: Agent with provider-hosted built-in tools.

Demonstrates wiring OpenAI's server-side `web_search_preview` tool into an
Agent. Requires a real OpenAI API key in `OPENAI_API_KEY`.

Usage:
    OPENAI_API_KEY=sk-... python examples/ex_19_built_in_tools.py
"""

import asyncio
import os

from agnt5 import Agent
from agnt5.lm import BuiltInTool


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    agent = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions=(
            "You research questions using web search when public information "
            "would help. Cite sources you used in your answer."
        ),
        built_in_tools=[BuiltInTool.WEB_SEARCH],
        max_iterations=4,
    )

    result = await agent.run_sync(
        user_message="What is the current version of the AGNT5 Python SDK?",
    )

    print("Output:\n", result.output)
    print("\nTool calls:")
    for call in result.tool_calls:
        marker = "[built-in]" if call.get("built_in") else "[local]   "
        print(f"  {marker} {call['name']}({call['arguments']})")


if __name__ == "__main__":
    asyncio.run(main())
