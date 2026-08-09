"""Shared OAuth refresh-token plumbing for the vendor gateways.

Calendar sync and email sending both authenticate to Google and Microsoft with
the same refresh-token grant, differing only in scope. The exchange lives here
so a fix to the expiry handling or the failure semantics lands once.

Like the gateways that call it, this is **never fatal**: a refusal from the
token endpoint returns ``None`` and the caller falls back to whatever access
token it already has, letting the real request produce the error the user sees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Refresh a little before expiry so a token does not lapse mid-request.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)

DEFAULT_TIMEOUT = 20.0


def microsoft_token_url(tenant: str) -> str:
    return f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"


@dataclass
class TokenRefresh:
    """A newly minted access token the caller should persist."""

    access_token: str
    expires_at: datetime | None = None


def is_expired(expires_at: datetime | None) -> bool:
    """Whether a token should be refreshed before use.

    An unknown expiry is treated as valid: guessing "expired" would burn a
    refresh on every call, so a 401 from the real request drives it instead.
    """
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at - TOKEN_REFRESH_MARGIN <= datetime.now(UTC)


def _expires_at(expires_in: object) -> datetime | None:
    """Turn a token endpoint's ``expires_in`` seconds into an absolute time."""
    if isinstance(expires_in, bool) or not isinstance(expires_in, int | float | str):
        return None
    try:
        seconds = int(float(expires_in))
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.now(UTC) + timedelta(seconds=seconds)


async def refresh_access_token(
    token_url: str,
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    scope: str | None = None,
    provider: str = "oauth",
    timeout: float = DEFAULT_TIMEOUT,
) -> TokenRefresh | None:
    """Exchange a refresh token for a new access token.

    Returns ``None`` for every failure mode — missing configuration, a rejected
    grant, a network fault, or a response without a token — because no caller
    can do anything with the distinction beyond logging it, which happens here.
    """
    if not (token_url and client_id and client_secret and refresh_token):
        return None

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    if scope:
        data["scope"] = scope

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(token_url, data=data)
        if response.status_code >= 400:
            logger.warning(
                "%s token refresh failed: %s %s",
                provider,
                response.status_code,
                response.text[:200],
            )
            return None
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("%s token refresh error: %s", provider, exc)
        return None

    token = body.get("access_token") if isinstance(body, dict) else None
    if not token:
        return None
    return TokenRefresh(
        access_token=token, expires_at=_expires_at(body.get("expires_in"))
    )
