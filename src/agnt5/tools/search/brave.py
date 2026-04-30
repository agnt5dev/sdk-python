"""Brave Search provider."""

from typing import List

import httpx

from .base import SearchResult

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider:
    """Brave Web Search.

    Docs: https://api-dashboard.search.brave.com/app/documentation/web-search
    """

    name = "brave"

    def __init__(self, api_key: str, *, timeout: float = 15.0):
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> List[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": max(1, min(max_results, 20))}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(_ENDPOINT, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        web_results = (data.get("web") or {}).get("results") or []
        out: List[SearchResult] = []
        for r in web_results[:max_results]:
            out.append(
                SearchResult(
                    title=r.get("title", "") or "",
                    url=r.get("url", "") or "",
                    snippet=r.get("description", "") or "",
                )
            )
        return out
