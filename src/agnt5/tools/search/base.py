"""SearchProvider protocol + provider resolution for `agnt5.tools.web_search`."""

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class SearchResult:
    """A single search result, normalized across providers."""

    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    """A web search provider. Each provider hits its own API and normalizes
    results into a flat list of `SearchResult`."""

    name: str

    async def search(self, query: str, *, max_results: int) -> List[SearchResult]:
        ...


_PROVIDER_KEY_ENV = {
    "brave": "AGNT5_BRAVE_SEARCH_API_KEY",
    "tavily": "AGNT5_TAVILY_API_KEY",
    "searxng": "AGNT5_SEARXNG_URL",
}


def resolve_provider(explicit: Optional[str] = None) -> SearchProvider:
    """Pick a SearchProvider based on the explicit arg, the
    `AGNT5_WEB_SEARCH_PROVIDER` env var, or the first env-key match.

    Resolution order:
      1. `explicit` ("brave" | "tavily" | "searxng")
      2. `AGNT5_WEB_SEARCH_PROVIDER`
      3. First match in (Brave, Tavily, SearXNG) by env var presence

    Raises:
        ConfigurationError: if no provider can be resolved or the chosen
            provider is missing its required env var.
    """
    # Imported here to avoid a circular import at module load.
    from ...exceptions import ConfigurationError
    from .brave import BraveSearchProvider
    from .searxng import SearXNGSearchProvider
    from .tavily import TavilySearchProvider

    chosen = (explicit or os.getenv("AGNT5_WEB_SEARCH_PROVIDER") or "").strip().lower()

    if not chosen:
        for name, env in _PROVIDER_KEY_ENV.items():
            if os.getenv(env):
                chosen = name
                break

    if not chosen:
        raise ConfigurationError(
            "agnt5.tools.web_search has no provider configured. Set "
            "AGNT5_BRAVE_SEARCH_API_KEY, AGNT5_TAVILY_API_KEY, or "
            "AGNT5_SEARXNG_URL — or pass provider=... explicitly. "
            "For zero-config search, enable provider-hosted search via "
            "Agent(built_in_tools=[BuiltInTool.WEB_SEARCH]) instead."
        )

    if chosen == "brave":
        key = os.getenv("AGNT5_BRAVE_SEARCH_API_KEY")
        if not key:
            raise ConfigurationError("Brave search selected but AGNT5_BRAVE_SEARCH_API_KEY is unset")
        return BraveSearchProvider(api_key=key)
    if chosen == "tavily":
        key = os.getenv("AGNT5_TAVILY_API_KEY")
        if not key:
            raise ConfigurationError("Tavily search selected but AGNT5_TAVILY_API_KEY is unset")
        return TavilySearchProvider(api_key=key)
    if chosen == "searxng":
        url = os.getenv("AGNT5_SEARXNG_URL")
        if not url:
            raise ConfigurationError("SearXNG search selected but AGNT5_SEARXNG_URL is unset")
        return SearXNGSearchProvider(base_url=url)

    raise ConfigurationError(
        f"Unknown search provider: {chosen!r}. Valid: brave, tavily, searxng."
    )
