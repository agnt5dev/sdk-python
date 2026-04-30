"""`web_fetch` — fetch a URL and return its contents as text.

v1 returns the response body decoded as UTF-8 (replacing invalid bytes), capped
at `max_bytes`. HTML is returned as-is; LLMs handle markup fine and a v1
quickstart doesn't need a heavyweight HTML→text dependency. A future revision
can add structured text extraction (e.g. via selectolax) behind an opt-in flag.
"""

# Note: do NOT add `from __future__ import annotations` here — the @tool
# decorator compares the first parameter's annotation against the Context
# class identity at decoration time, which only works with eager annotations.

import ipaddress
from typing import Optional
from urllib.parse import urlparse

import httpx

from ..context import Context
from ..tool import Tool, tool

_DEFAULT_MAX_BYTES = 200_000
_DEFAULT_TIMEOUT = 15.0
_ALLOWED_SCHEMES = {"http", "https"}


class WebFetchError(Exception):
    """Raised when `web_fetch` rejects a request before issuing it."""


def _check_url(url: str) -> str:
    """Validate the URL scheme and host. Returns the parsed hostname.

    SSRF guard: this rejects non-http(s) schemes, the literal hostname
    "localhost", and IP literals that fall in private/loopback/link-local
    ranges. It does NOT do DNS resolution; a hostile DNS record can still
    point at an internal IP. For quickstart usage this is acceptable; harden
    callers facing untrusted input.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise WebFetchError(f"web_fetch only supports http(s); got scheme={scheme!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise WebFetchError("web_fetch requires a hostname")
    if host in {"localhost", "ip6-localhost", "ip6-loopback"}:
        raise WebFetchError("web_fetch refuses localhost")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host  # not an IP literal; allow

    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        raise WebFetchError(f"web_fetch refuses non-public IP literal: {host}")
    return host


async def _fetch(url: str, *, max_bytes: int, timeout: float) -> str:
    _check_url(url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
    final_url = str(response.url)
    body = response.content[: max_bytes + 1]
    truncated = len(body) > max_bytes
    if truncated:
        body = body[:max_bytes]
    text = body.decode("utf-8", errors="replace")
    suffix = "\n\n[truncated]" if truncated else ""
    return f"{final_url}\n\n{text}{suffix}"


def web_fetch(
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = _DEFAULT_TIMEOUT,
    name: Optional[str] = None,
) -> Tool:
    """Build a Tool that fetches a URL and returns its body as text.

    The returned Tool exposes a single argument, `url: str`. Output is the final
    URL (after redirects) followed by a blank line and up to `max_bytes` of body
    text. Bodies longer than `max_bytes` are truncated with a trailing
    `[truncated]` marker.

    Args:
        max_bytes: Maximum body bytes to return. Defaults to 200_000.
        timeout: Per-request timeout in seconds. Defaults to 15.
        name: Override the tool name (default: "web_fetch").

    Raises (at call time, not build time):
        WebFetchError: scheme is not http(s), or host is local/private.
        httpx.HTTPError: transport failure (timeout, connect error, etc.).
    """

    @tool(name=name or "web_fetch")
    async def _web_fetch(ctx: Context, url: str) -> str:
        """Fetch a URL over HTTP(S) and return the response body as text.

        Refuses non-http(s) schemes and private/loopback hosts. Truncates the
        body at the configured byte cap.
        """
        return await _fetch(url, max_bytes=max_bytes, timeout=timeout)

    return _web_fetch
