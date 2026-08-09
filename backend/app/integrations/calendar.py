"""Google and Outlook calendar gateway (design §4.3).

Two operations matter for scheduling: reading when an interviewer is busy, and
writing the event once a slot is booked. Both follow the same contract as the
OpenRouter gateway — **never fatal**. A calendar that cannot be reached returns
``synced=False`` rather than raising, and the caller falls back to the
interviewer's configured working hours. Losing calendar sync should cost slot
accuracy, not the ability to schedule at all.

Access tokens are refreshed here but persisted by the caller: this module does
no database work, so a refreshed token is handed back on the result object for
the service layer to store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GOOGLE = "google"
OUTLOOK = "outlook"

GOOGLE_API = "https://www.googleapis.com/calendar/v3"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
MICROSOFT_API = "https://graph.microsoft.com/v1.0"

# Refresh a little before expiry so a token does not lapse mid-request.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)

_TIMEOUT = 20.0


@dataclass
class CalendarCredentials:
    """The parts of a ``CalendarAccount`` a provider needs, without the ORM row."""

    provider: str
    email: str
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: datetime | None = None
    calendar_id: str | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            # Unknown expiry: assume valid and let a 401 drive the refresh.
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires - TOKEN_REFRESH_MARGIN <= datetime.now(UTC)


@dataclass
class TokenRefresh:
    """A newly minted access token the caller should persist."""

    access_token: str
    expires_at: datetime | None = None


@dataclass
class BusyResult:
    """Busy blocks for one account.

    ``synced=False`` means the blocks are not trustworthy — usually empty
    because the fetch failed — and the caller should say so rather than
    presenting working hours as verified availability.
    """

    blocks: list[tuple[datetime, datetime]] = field(default_factory=list)
    synced: bool = True
    error: str | None = None
    refreshed: TokenRefresh | None = None


@dataclass
class CalendarEvent:
    """The event to write into an interviewer's calendar."""

    summary: str
    start: datetime
    end: datetime
    description: str | None = None
    location: str | None = None
    attendees: list[str] = field(default_factory=list)
    timezone: str = "UTC"
    # Ask the provider to mint a Meet/Teams link for the event.
    create_conference: bool = True


@dataclass
class EventResult:
    ok: bool
    external_event_id: str | None = None
    meeting_url: str | None = None
    error: str | None = None
    refreshed: TokenRefresh | None = None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Unparseable calendar timestamp %r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class CalendarProvider:
    """Base provider. Subclasses talk to one vendor's API."""

    name = "unknown"

    @property
    def is_configured(self) -> bool:
        return False

    async def fetch_busy(
        self, credentials: CalendarCredentials, start: datetime, end: datetime
    ) -> BusyResult:
        raise NotImplementedError

    async def create_event(
        self, credentials: CalendarCredentials, event: CalendarEvent
    ) -> EventResult:
        raise NotImplementedError

    async def delete_event(
        self, credentials: CalendarCredentials, external_event_id: str
    ) -> EventResult:
        raise NotImplementedError


class UnavailableProvider(CalendarProvider):
    """Stand-in for a provider that is not configured or not recognised.

    Reports "not synced" instead of raising, so an organization with no
    calendar integration still schedules off working hours.
    """

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    async def fetch_busy(
        self, credentials: CalendarCredentials, start: datetime, end: datetime
    ) -> BusyResult:
        return BusyResult(blocks=[], synced=False, error=self.reason)

    async def create_event(
        self, credentials: CalendarCredentials, event: CalendarEvent
    ) -> EventResult:
        return EventResult(ok=False, error=self.reason)

    async def delete_event(
        self, credentials: CalendarCredentials, external_event_id: str
    ) -> EventResult:
        return EventResult(ok=False, error=self.reason)


class _OAuthProvider(CalendarProvider):
    """Shared OAuth refresh plumbing for the real providers."""

    token_url = ""

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self.client_id = client_id
        self.client_secret = client_secret

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _refresh_payload(self, refresh_token: str) -> dict[str, str]:
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

    async def _refresh(self, credentials: CalendarCredentials) -> TokenRefresh | None:
        """Exchange the refresh token for a new access token."""
        if not credentials.refresh_token or not self.is_configured:
            return None
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    self.token_url,
                    data=self._refresh_payload(credentials.refresh_token),
                )
            if response.status_code >= 400:
                logger.warning(
                    "%s token refresh failed: %s %s",
                    self.name,
                    response.status_code,
                    response.text[:200],
                )
                return None
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("%s token refresh error: %s", self.name, exc)
            return None

        token = body.get("access_token")
        if not token:
            return None
        expires_in = body.get("expires_in")
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=int(expires_in))
            if isinstance(expires_in, int | float | str) and str(expires_in).isdigit()
            else None
        )
        return TokenRefresh(access_token=token, expires_at=expires_at)

    async def _authorize(
        self, credentials: CalendarCredentials
    ) -> tuple[str | None, TokenRefresh | None]:
        """Return a usable access token, refreshing first if it has expired."""
        if credentials.access_token and not credentials.is_expired:
            return credentials.access_token, None
        refreshed = await self._refresh(credentials)
        if refreshed is not None:
            return refreshed.access_token, refreshed
        # No refresh available: try the existing token anyway. It may still
        # work if our expiry bookkeeping is wrong.
        return credentials.access_token, None


class GoogleCalendarProvider(_OAuthProvider):
    """Google Calendar via the freeBusy and events APIs."""

    name = GOOGLE
    token_url = GOOGLE_TOKEN_URL

    async def fetch_busy(
        self, credentials: CalendarCredentials, start: datetime, end: datetime
    ) -> BusyResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return BusyResult(synced=False, error="No usable Google access token")

        calendar_id = credentials.calendar_id or credentials.email or "primary"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{GOOGLE_API}/freeBusy",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "timeMin": _iso(start),
                        "timeMax": _iso(end),
                        "items": [{"id": calendar_id}],
                    },
                )
            if response.status_code >= 400:
                return BusyResult(
                    synced=False,
                    error=f"Google freeBusy HTTP {response.status_code}",
                    refreshed=refreshed,
                )
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return BusyResult(synced=False, error=str(exc), refreshed=refreshed)

        calendars = body.get("calendars") or {}
        entry = calendars.get(calendar_id) or next(iter(calendars.values()), {})
        if entry.get("errors"):
            return BusyResult(
                synced=False,
                error=str(entry["errors"][:1]),
                refreshed=refreshed,
            )

        return BusyResult(
            blocks=_coerce_blocks(entry.get("busy") or [], "start", "end"),
            synced=True,
            refreshed=refreshed,
        )

    async def create_event(
        self, credentials: CalendarCredentials, event: CalendarEvent
    ) -> EventResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return EventResult(ok=False, error="No usable Google access token")

        calendar_id = credentials.calendar_id or "primary"
        body: dict = {
            "summary": event.summary,
            "description": event.description,
            "location": event.location,
            "start": {"dateTime": _iso(event.start), "timeZone": "UTC"},
            "end": {"dateTime": _iso(event.end), "timeZone": "UTC"},
            "attendees": [{"email": e} for e in event.attendees],
        }
        params = {"sendUpdates": "all"}
        if event.create_conference:
            body["conferenceData"] = {
                "createRequest": {
                    # Google requires a caller-supplied idempotency key here.
                    "requestId": f"hireagent-{int(event.start.timestamp())}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
            params["conferenceDataVersion"] = "1"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{GOOGLE_API}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=body,
                )
            if response.status_code >= 400:
                return EventResult(
                    ok=False,
                    error=f"Google event create HTTP {response.status_code}",
                    refreshed=refreshed,
                )
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return EventResult(ok=False, error=str(exc), refreshed=refreshed)

        return EventResult(
            ok=True,
            external_event_id=data.get("id"),
            meeting_url=data.get("hangoutLink") or data.get("htmlLink"),
            refreshed=refreshed,
        )

    async def delete_event(
        self, credentials: CalendarCredentials, external_event_id: str
    ) -> EventResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return EventResult(ok=False, error="No usable Google access token")
        calendar_id = credentials.calendar_id or "primary"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.delete(
                    f"{GOOGLE_API}/calendars/{calendar_id}/events/{external_event_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"sendUpdates": "all"},
                )
        except httpx.HTTPError as exc:
            return EventResult(ok=False, error=str(exc), refreshed=refreshed)

        # 410 Gone means somebody already deleted it, which is the state we want.
        ok = response.status_code < 400 or response.status_code == 410
        return EventResult(
            ok=ok,
            error=None if ok else f"HTTP {response.status_code}",
            refreshed=refreshed,
        )


class OutlookCalendarProvider(_OAuthProvider):
    """Outlook / Microsoft 365 via Microsoft Graph."""

    name = OUTLOOK

    def __init__(
        self, client_id: str = "", client_secret: str = "", tenant: str = "common"
    ) -> None:
        super().__init__(client_id, client_secret)
        self.tenant = tenant or "common"

    @property
    def token_url(self) -> str:  # type: ignore[override]
        return f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0/token"

    def _refresh_payload(self, refresh_token: str) -> dict[str, str]:
        payload = super()._refresh_payload(refresh_token)
        # Graph requires an explicit scope on refresh.
        payload["scope"] = "https://graph.microsoft.com/.default offline_access"
        return payload

    async def fetch_busy(
        self, credentials: CalendarCredentials, start: datetime, end: datetime
    ) -> BusyResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return BusyResult(synced=False, error="No usable Outlook access token")

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{MICROSOFT_API}/me/calendar/getSchedule",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "schedules": [credentials.email],
                        "startTime": {"dateTime": _iso(start), "timeZone": "UTC"},
                        "endTime": {"dateTime": _iso(end), "timeZone": "UTC"},
                        "availabilityViewInterval": 15,
                    },
                )
            if response.status_code >= 400:
                return BusyResult(
                    synced=False,
                    error=f"Graph getSchedule HTTP {response.status_code}",
                    refreshed=refreshed,
                )
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return BusyResult(synced=False, error=str(exc), refreshed=refreshed)

        schedules = body.get("value") or []
        if not schedules:
            return BusyResult(blocks=[], synced=True, refreshed=refreshed)

        blocks: list[tuple[datetime, datetime]] = []
        for item in schedules[0].get("scheduleItems") or []:
            # "free" items are placeholders the user is not actually blocked by.
            if str(item.get("status", "")).lower() == "free":
                continue
            begin = _parse_dt((item.get("start") or {}).get("dateTime"))
            finish = _parse_dt((item.get("end") or {}).get("dateTime"))
            if begin and finish and finish > begin:
                blocks.append((begin, finish))
        return BusyResult(blocks=blocks, synced=True, refreshed=refreshed)

    async def create_event(
        self, credentials: CalendarCredentials, event: CalendarEvent
    ) -> EventResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return EventResult(ok=False, error="No usable Outlook access token")

        body: dict = {
            "subject": event.summary,
            "body": {"contentType": "HTML", "content": event.description or ""},
            "start": {"dateTime": _iso(event.start), "timeZone": "UTC"},
            "end": {"dateTime": _iso(event.end), "timeZone": "UTC"},
            "attendees": [
                {"emailAddress": {"address": e}, "type": "required"}
                for e in event.attendees
            ],
        }
        if event.location:
            body["location"] = {"displayName": event.location}
        if event.create_conference:
            body["isOnlineMeeting"] = True
            body["onlineMeetingProvider"] = "teamsForBusiness"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{MICROSOFT_API}/me/events",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
            if response.status_code >= 400:
                return EventResult(
                    ok=False,
                    error=f"Graph event create HTTP {response.status_code}",
                    refreshed=refreshed,
                )
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return EventResult(ok=False, error=str(exc), refreshed=refreshed)

        meeting = data.get("onlineMeeting") or {}
        return EventResult(
            ok=True,
            external_event_id=data.get("id"),
            meeting_url=meeting.get("joinUrl") or data.get("webLink"),
            refreshed=refreshed,
        )

    async def delete_event(
        self, credentials: CalendarCredentials, external_event_id: str
    ) -> EventResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return EventResult(ok=False, error="No usable Outlook access token")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.delete(
                    f"{MICROSOFT_API}/me/events/{external_event_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as exc:
            return EventResult(ok=False, error=str(exc), refreshed=refreshed)

        ok = response.status_code < 400 or response.status_code == 404
        return EventResult(
            ok=ok,
            error=None if ok else f"HTTP {response.status_code}",
            refreshed=refreshed,
        )


def _coerce_blocks(
    raw: list, start_key: str, end_key: str
) -> list[tuple[datetime, datetime]]:
    """Turn a provider's busy array into clean, ordered datetime pairs."""
    blocks: list[tuple[datetime, datetime]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        begin = _parse_dt(entry.get(start_key))
        finish = _parse_dt(entry.get(end_key))
        # Drop zero-length and inverted ranges rather than letting them
        # corrupt the interval algebra downstream.
        if begin and finish and finish > begin:
            blocks.append((begin, finish))
    return blocks


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_registry: dict[str, CalendarProvider] | None = None


def _build_registry() -> dict[str, CalendarProvider]:
    google = GoogleCalendarProvider(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
    )
    outlook = OutlookCalendarProvider(
        client_id=settings.microsoft_client_id,
        client_secret=settings.microsoft_client_secret,
        tenant=settings.microsoft_tenant_id,
    )
    registry: dict[str, CalendarProvider] = {}
    for provider in (google, outlook):
        registry[provider.name] = (
            provider
            if settings.calendar_sync_enabled and provider.is_configured
            else UnavailableProvider(
                provider.name,
                f"{provider.name} calendar sync is not configured",
            )
        )
    return registry


def get_provider(name: str) -> CalendarProvider:
    """The provider for ``name``; never raises for an unknown vendor."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    provider = _registry.get((name or "").lower())
    if provider is None:
        return UnavailableProvider(name or "unknown", f"Unsupported calendar provider {name!r}")
    return provider


def set_providers(registry: dict[str, CalendarProvider] | None) -> None:
    """Swap the process-wide provider registry (used by tests)."""
    global _registry
    _registry = registry
