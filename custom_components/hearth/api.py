"""Small async client for the Hearth hub HTTP API.

The contract is the hub's OpenAPI document (docs/openapi.json in the Hearth
repository). Only the calls this integration needs are implemented.
"""

from datetime import UTC, datetime
import json
from typing import Any

import aiohttp

from .const import REQUEST_TIMEOUT, SUMMARY_TIMEOUT

_MAX_DETAIL = 300


class HearthError(Exception):
    """Base class for hub errors."""


class HearthConnectionError(HearthError):
    """The hub could not be reached, or did not answer in time."""


class HearthAuthError(HearthError):
    """The hub rejected the bearer token."""


class HearthApiError(HearthError):
    """The hub answered with an HTTP error or an unusable body."""

    def __init__(self, status: int, detail: str) -> None:
        """Keep the HTTP status and the hub's own explanation."""
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _detail(body: str) -> str:
    """Pull the human-readable part out of a FastAPI error body."""
    try:
        detail = json.loads(body)["detail"]
    except (ValueError, KeyError, TypeError):
        detail = body.strip()
    if isinstance(detail, list):  # request validation: [{"loc": [...], "msg": "..."}]
        parts = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                loc = ".".join(str(p) for p in item.get("loc", ())[1:])
                parts.append(f"{loc}: {item['msg']}" if loc else str(item["msg"]))
            else:
                parts.append(str(item))
        detail = "; ".join(parts)
    detail = str(detail)
    if len(detail) > _MAX_DETAIL:
        detail = detail[: _MAX_DETAIL - 1] + "…"
    return detail or "no details given"


def _iso_utc(when: datetime) -> str:
    """Format an aware datetime the way the hub stores them (UTC, 'Z')."""
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class HearthClient:
    """Talks to one Hearth hub."""

    def __init__(self, session: aiohttp.ClientSession, url: str, token: str) -> None:
        """Store the session, the hub base URL and the bearer token."""
        self._session = session
        self._url = url.rstrip("/")
        self._token = token

    @property
    def url(self) -> str:
        """The hub base URL."""
        return self._url

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        try:
            response = await self._session.request(
                method,
                f"{self._url}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                params=params,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            text = await response.text()
        except TimeoutError as err:
            raise HearthConnectionError(f"no answer within {timeout:g} s") from err
        except aiohttp.ClientError as err:
            raise HearthConnectionError(str(err) or type(err).__name__) from err

        if response.status in (401, 403):
            raise HearthAuthError(_detail(text))
        if response.status >= 400:
            raise HearthApiError(response.status, _detail(text))
        try:
            data = json.loads(text)
        except ValueError as err:
            raise HearthApiError(response.status, "the answer is not JSON") from err
        if not isinstance(data, dict):
            raise HearthApiError(response.status, "the answer is not a JSON object")
        return data

    async def status(self) -> dict[str, Any]:
        """GET /api/v1/status: version, counters and switch states."""
        return await self._request("GET", "/api/v1/status")

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """GET /api/v1/search: full-text search, newest first."""
        data = await self._request(
            "GET", "/api/v1/search", params={"q": query, "limit": limit}
        )
        return data.get("segments") or []

    async def segments(
        self, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """GET /api/v1/segments: transcripts that start in [start, end)."""
        data = await self._request(
            "GET",
            "/api/v1/segments",
            params={"start": _iso_utc(start), "end": _iso_utc(end), "limit": limit},
        )
        return data.get("segments") or []

    async def run_summary(self, hours: float, kind: str) -> dict[str, Any] | None:
        """POST /api/v1/summaries/run: summarise the last hours now.

        Returns None when there was nothing to summarise.
        """
        data = await self._request(
            "POST",
            "/api/v1/summaries/run",
            body={"kind": kind, "hours": hours},
            timeout=SUMMARY_TIMEOUT,
        )
        return data.get("summary")

    async def latest_summary(self, kind: str | None = None) -> dict[str, Any] | None:
        """GET /api/v1/summaries/latest: the newest stored summary, if any."""
        data = await self._request(
            "GET", "/api/v1/summaries/latest", params={"kind": kind} if kind else None
        )
        return data.get("summary")
