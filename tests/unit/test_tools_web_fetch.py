"""Tests for `agnt5.tools.web_fetch`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agnt5 import Context
from agnt5.tool import Tool
from agnt5.tools import web_fetch
from agnt5.tools.web_fetch import WebFetchError, _check_url


def test_factory_returns_tool_instance():
    fetcher = web_fetch()
    assert isinstance(fetcher, Tool)
    assert fetcher.name == "web_fetch"


def test_factory_respects_name_override():
    assert web_fetch(name="grab").name == "grab"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/foo",
        "gopher://example.com",
        "javascript:alert(1)",
        "",
    ],
)
def test_check_url_rejects_unsupported_schemes(url):
    with pytest.raises(WebFetchError):
        _check_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.5/",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "http://[::1]/",
    ],
)
def test_check_url_rejects_private_or_loopback(url):
    with pytest.raises(WebFetchError):
        _check_url(url)


def test_check_url_allows_public_hostnames_and_ips():
    assert _check_url("https://example.com/path") == "example.com"
    assert _check_url("http://1.1.1.1/") == "1.1.1.1"


def test_check_url_requires_a_host():
    with pytest.raises(WebFetchError):
        _check_url("https:///no-host")


def _patch_async_client(response: httpx.Response):
    """Patch httpx.AsyncClient so .get returns a fixed response."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return patch("agnt5.tools.web_fetch.httpx.AsyncClient", return_value=cm)


@pytest.mark.asyncio
async def test_happy_path_returns_url_and_body():
    body = b"<html><body>hello world</body></html>"
    response = httpx.Response(
        status_code=200,
        content=body,
        request=httpx.Request("GET", "https://example.com/"),
    )

    with _patch_async_client(response):
        fetcher = web_fetch()
        out = await fetcher.handler(Context(run_id="test", correlation_id="corr", parent_correlation_id="parent"), "https://example.com/")

    assert out.startswith("https://example.com/")
    assert "hello world" in out
    assert "[truncated]" not in out


@pytest.mark.asyncio
async def test_oversize_body_is_truncated():
    body = b"x" * 1000
    response = httpx.Response(
        status_code=200,
        content=body,
        request=httpx.Request("GET", "https://example.com/"),
    )

    with _patch_async_client(response):
        fetcher = web_fetch(max_bytes=100)
        out = await fetcher.handler(Context(run_id="test", correlation_id="corr", parent_correlation_id="parent"), "https://example.com/")

    assert "[truncated]" in out
    body_part = out.split("\n\n", 1)[1]
    assert len(body_part) == 100 + len("\n\n[truncated]")


@pytest.mark.asyncio
async def test_handler_rejects_private_url_before_request():
    fetcher = web_fetch()
    with pytest.raises(WebFetchError):
        await fetcher.handler(Context(run_id="test", correlation_id="corr", parent_correlation_id="parent"), "http://127.0.0.1/")


@pytest.mark.asyncio
async def test_redirect_final_url_is_reported():
    final = httpx.Response(
        status_code=200,
        content=b"final body",
        request=httpx.Request("GET", "https://final.example.com/"),
    )
    # httpx.Response.url comes from the request when constructed this way; the
    # follow-redirects machinery is the AsyncClient's responsibility, which we
    # mock. The contract under test: web_fetch surfaces response.url verbatim.
    with _patch_async_client(final):
        fetcher = web_fetch()
        out = await fetcher.handler(Context(run_id="test", correlation_id="corr", parent_correlation_id="parent"), "https://start.example.com/")
    assert out.startswith("https://final.example.com/")
