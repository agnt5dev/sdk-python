"""SearXNG provider — works against any SearXNG instance with JSON output enabled."""

from typing import List

import httpx

from .base import SearchResult


class SearXNGSearchProvider:
    """Self-hosted or public SearXNG instance.

    Set `AGNT5_SEARXNG_URL` to the instance base URL (no trailing /search). A
    public instance must have JSON output enabled and `bot_detection.ip_lists`
    permissive enough to allow programmatic queries; many public instances
    block these by default.
    """

    name = "searxng"

    def __init__(self, base_url: str, *, timeout: float = 15.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int) -> List[SearchResult]:
        params = {"q": query, "format": "json"}
        url = f"{self._base_url}/search"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
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
