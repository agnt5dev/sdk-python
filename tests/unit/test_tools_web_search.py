"""Tests for `agnt5.tools.web_search` and its provider resolution."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agnt5 import Context
from agnt5.exceptions import ConfigurationError
from agnt5.tool import Tool
from agnt5.tools import web_search
from agnt5.tools.search import (
    BraveSearchProvider,
    SearchResult,
    SearXNGSearchProvider,
    TavilySearchProvider,
    resolve_provider,
)


def _ctx() -> Context:
    return Context(run_id="test", correlation_id="corr", parent_correlation_id="parent")


# ---------- resolve_provider ----------

def test_resolve_provider_no_config_raises(monkeypatch):
    for var in ("AGNT5_WEB_SEARCH_PROVIDER", "AGNT5_BRAVE_SEARCH_API_KEY",
                "AGNT5_TAVILY_API_KEY", "AGNT5_SEARXNG_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigurationError):
        resolve_provider()


def test_resolve_provider_explicit_brave_requires_key(monkeypatch):
    monkeypatch.delenv("AGNT5_BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        resolve_provider("brave")


def test_resolve_provider_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("AGNT5_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("AGNT5_TAVILY_API_KEY", "tk")
    monkeypatch.setenv("AGNT5_BRAVE_SEARCH_API_KEY", "bk")
    p = resolve_provider("brave")
    assert isinstance(p, BraveSearchProvider)


def test_resolve_provider_uses_env_var_when_no_explicit(monkeypatch):
    monkeypatch.setenv("AGNT5_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("AGNT5_TAVILY_API_KEY", "tk")
    p = resolve_provider()
    assert isinstance(p, TavilySearchProvider)


def test_resolve_provider_falls_back_to_first_key_match(monkeypatch):
    monkeypatch.delenv("AGNT5_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("AGNT5_BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("AGNT5_TAVILY_API_KEY", "tk")
    p = resolve_provider()
    assert isinstance(p, TavilySearchProvider)


def test_resolve_provider_searxng_requires_url(monkeypatch):
    monkeypatch.delenv("AGNT5_SEARXNG_URL", raising=False)
    with pytest.raises(ConfigurationError):
        resolve_provider("searxng")


def test_resolve_provider_unknown_name_raises(monkeypatch):
    with pytest.raises(ConfigurationError):
        resolve_provider("yahoo")


# ---------- factory ----------

def test_factory_returns_tool_instance():
    t = web_search()
    assert isinstance(t, Tool)
    assert t.name == "web_search"


def test_factory_respects_name_override():
    assert web_search(name="search").name == "search"


# ---------- providers (mocked HTTP) ----------

def _patch_get(client_module: str, response: httpx.Response):
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return patch(f"{client_module}.httpx.AsyncClient", return_value=cm)


def _patch_post(client_module: str, response: httpx.Response):
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return patch(f"{client_module}.httpx.AsyncClient", return_value=cm)


@pytest.mark.asyncio
async def test_brave_provider_normalizes_results():
    payload = {
        "web": {
            "results": [
                {"title": "T1", "url": "https://a", "description": "S1"},
                {"title": "T2", "url": "https://b", "description": "S2"},
            ]
        }
    }
    response = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://api.search.brave.com/"),
    )
    with _patch_get("agnt5.tools.search.brave", response):
        provider = BraveSearchProvider(api_key="k")
        results = await provider.search("query", max_results=5)
    assert len(results) == 2
    assert results[0] == SearchResult(title="T1", url="https://a", snippet="S1")


@pytest.mark.asyncio
async def test_tavily_provider_normalizes_results():
    payload = {
        "results": [
            {"title": "T1", "url": "https://a", "content": "C1"},
        ]
    }
    response = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )
    with _patch_post("agnt5.tools.search.tavily", response):
        provider = TavilySearchProvider(api_key="k")
        results = await provider.search("query", max_results=3)
    assert results == [SearchResult(title="T1", url="https://a", snippet="C1")]


@pytest.mark.asyncio
async def test_searxng_provider_normalizes_results():
    payload = {
        "results": [
            {"title": "T1", "url": "https://a", "content": "C1"},
            {"title": "T2", "url": "https://b", "content": "C2"},
        ]
    }
    response = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://searx.example.com/search"),
    )
    with _patch_get("agnt5.tools.search.searxng", response):
        provider = SearXNGSearchProvider(base_url="https://searx.example.com")
        results = await provider.search("query", max_results=2)
    assert len(results) == 2
    assert results[1].url == "https://b"


# ---------- factory + provider end-to-end ----------

@pytest.mark.asyncio
async def test_factory_lazy_resolution_uses_env(monkeypatch):
    monkeypatch.setenv("AGNT5_BRAVE_SEARCH_API_KEY", "k")
    monkeypatch.delenv("AGNT5_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("AGNT5_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("AGNT5_SEARXNG_URL", raising=False)

    payload = {"web": {"results": [{"title": "T", "url": "https://a", "description": "S"}]}}
    response = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://api.search.brave.com/"),
    )
    with _patch_get("agnt5.tools.search.brave", response):
        searcher = web_search(max_results=1)
        out = await searcher.handler(_ctx(), "anything")

    assert out == [{"title": "T", "url": "https://a", "snippet": "S"}]


@pytest.mark.asyncio
async def test_factory_raises_at_first_call_when_unconfigured(monkeypatch):
    for var in ("AGNT5_WEB_SEARCH_PROVIDER", "AGNT5_BRAVE_SEARCH_API_KEY",
                "AGNT5_TAVILY_API_KEY", "AGNT5_SEARXNG_URL"):
        monkeypatch.delenv(var, raising=False)
    searcher = web_search()  # build-time: no error
    with pytest.raises(ConfigurationError):
        await searcher.handler(_ctx(), "q")
