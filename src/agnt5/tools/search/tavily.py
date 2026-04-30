"""Tavily Search provider."""

from typing import List

import httpx

from .base import SearchResult

_ENDPOINT = "https://api.tavily.com/search"


class TavilySearchProvider:
    """Tavily Search.

    Docs: https://docs.tavily.com/docs/rest-api/api-reference
    """

    name = "tavily"

    def __init__(self, api_key: str, *, timeout: float = 15.0):
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> List[SearchResult]:
        body = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(max_results, 20)),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_ENDPOINT, json=body)
            response.raise_for_status()
            data = response.json()

        results = data.get("results") or []
        out: List[SearchResult] = []
        for r in results[:max_results]:
            out.append(
                SearchResult(
                    title=r.get("title", "") or "",
                    url=r.get("url", "") or "",
                    snippet=r.get("content", "") or "",
                )
            )
        return out
