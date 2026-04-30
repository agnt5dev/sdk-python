"""Built-in AGNT5 tools usable from `Agent(tools=[...])`.

Each entry here is a *factory*: calling it returns a configured `Tool` instance.

    from agnt5.tools import web_fetch
    agent = Agent(..., tools=[web_fetch(max_bytes=200_000)])
"""

from .web_fetch import web_fetch
from .web_search import web_search

__all__ = ["web_fetch", "web_search"]
