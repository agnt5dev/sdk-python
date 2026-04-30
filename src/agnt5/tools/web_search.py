"""`web_search` — AGNT5-owned web search tool with provider abstraction.

Lazy provider resolution: the chosen provider is determined the first time
the tool is actually invoked, not when the factory is called. That way an
agent built at import time doesn't need search keys present in the
environment until the agent actually runs.
"""

# Note: do NOT add `from __future__ import annotations` here — the @tool
# decorator compares the first parameter's annotation against the Context
# class identity at decoration time, which requires eager annotations.

from typing import Any, Dict, List, Optional

from ..context import Context
from ..tool import Tool, tool
from .search.base import SearchProvider, resolve_provider

_DEFAULT_MAX_RESULTS = 5


def web_search(
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    provider: Optional[str] = None,
    name: Optional[str] = None,
) -> Tool:
    """Build a Tool that runs a web search via Brave, Tavily, or SearXNG.

    Provider selection (resolved at first call, not at factory time):
      1. The `provider` argument, if given.
      2. `AGNT5_WEB_SEARCH_PROVIDER` environment variable.
      3. First match by env-key presence:
         AGNT5_BRAVE_SEARCH_API_KEY → brave
         AGNT5_TAVILY_API_KEY       → tavily
         AGNT5_SEARXNG_URL          → searxng

    Args:
        max_results: Maximum results to return. Defaults to 5.
        provider: One of "brave", "tavily", "searxng".
        name: Override the tool name (default: "web_search").

    Returns:
        A `Tool` exposing `web_search(query: str)` returning a list of dicts
        with keys `title`, `url`, `snippet`.

    Raises:
        ConfigurationError (at first call): no provider could be resolved or
            the resolved provider's required env var is missing.
    """
    cached: Dict[str, Optional[SearchProvider]] = {"provider": None}

    @tool(name=name or "web_search")
    async def _web_search(ctx: Context, query: str) -> List[Dict[str, Any]]:
        """Run a web search and return up to N results.

        Each result is a dict with `title`, `url`, and `snippet` keys.
        """
        if cached["provider"] is None:
            cached["provider"] = resolve_provider(provider)
        results = await cached["provider"].search(query, max_results=max_results)
        return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]

    return _web_search
