"""Chat bot tokens for hosted workers, fetched when a reply needs one.

A chat run's input names the integration the message came through, but not
its bot token: run input is stored with the run's history, so a token there
could be read by anyone who can read the run. A hosted worker fetches the
token from the platform with its workload identity instead, and keeps it only
in memory until it expires.

Outside a hosted worker (``agnt5 dev``, tests) there is no workload identity,
and the token configured on the ``ChatBot`` is used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BROKER_URL_ENV = "AGNT5_WORKLOAD_BROKER_URL"
TOKEN_FILE_ENV = "AGNT5_WORKLOAD_TOKEN_FILE"
CREDENTIAL_AUDIENCE = "agnt5-credentials"
SCOPE = "chat_bot_token"
# Fetch again this long before the platform's expiry, so a reply never starts
# with a token about to lapse.
_EXPIRY_MARGIN_S = 60.0
_DEFAULT_TTL_S = 300.0
_TIMEOUT_S = 10.0


@dataclass
class _Cached:
    token: str
    expires_at: float


class BotTokenSource:
    """Fetches and caches bot tokens by integration id."""

    def __init__(
        self,
        broker_url: Optional[str] = None,
        token_file: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._broker_url = (broker_url or os.environ.get(BROKER_URL_ENV, "")).rstrip("/")
        self._token_file = token_file or os.environ.get(TOKEN_FILE_ENV, "")
        self._transport = transport
        self._cache: dict[str, _Cached] = {}
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """Whether this worker has a workload identity to fetch with."""
        return bool(self._broker_url and self._token_file)

    async def get(self, integration_id: str) -> Optional[str]:
        """The integration's bot token, or ``None`` when it can't be fetched.

        A failed fetch is logged, without the token, and returns ``None`` so
        the caller can fall back to a configured token.
        """
        if not integration_id or not self.available:
            return None
        cached = self._cache.get(integration_id)
        if cached and cached.expires_at - _EXPIRY_MARGIN_S > time.time():
            return cached.token
        async with self._lock:
            cached = self._cache.get(integration_id)
            if cached and cached.expires_at - _EXPIRY_MARGIN_S > time.time():
                return cached.token
            try:
                fetched = await self._fetch(integration_id)
            except Exception as error:  # noqa: BLE001 - reported, then fall back
                logger.warning(
                    "Could not fetch the chat bot token for integration %s: %s",
                    integration_id,
                    error,
                )
                return None
            self._cache[integration_id] = fetched
            return fetched.token

    async def _fetch(self, integration_id: str) -> _Cached:
        with open(self._token_file, encoding="utf-8") as handle:
            projected = handle.read().strip()
        if not projected:
            raise RuntimeError("the workload token file is empty")
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, transport=self._transport) as client:
            exchanged = await client.post(
                f"{self._broker_url}/internal/workload/token",
                json={"token": projected, "audience": [CREDENTIAL_AUDIENCE]},
            )
            exchanged.raise_for_status()
            workload_token = exchanged.json().get("token", "")
            if not workload_token:
                raise RuntimeError("the workload token exchange returned no token")
            credential = await client.post(
                f"{self._broker_url}/internal/workload/credential",
                json={"scope": SCOPE, "ref": integration_id, "purpose": "chat_reply"},
                headers={"Authorization": f"Bearer {workload_token}"},
            )
            if credential.status_code >= 400:
                # The broker's error says why (an integration that's gone, no
                # bot token on it); the response never carries a credential.
                try:
                    reason = credential.json().get("error", "")
                except ValueError:
                    reason = ""
                raise RuntimeError(
                    f"credential request failed ({credential.status_code}) {reason}".strip()
                )
            body = credential.json()
        token = body.get("value", "")
        if not token:
            raise RuntimeError("the credential response held no token")
        return _Cached(token=token, expires_at=_expiry(body.get("expires_at")))


def _expiry(value: object) -> float:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time() + _DEFAULT_TTL_S


_default_source: Optional[BotTokenSource] = None


def default_source() -> BotTokenSource:
    """The process-wide source, built from the worker's environment."""
    global _default_source
    if _default_source is None:
        _default_source = BotTokenSource()
    return _default_source
