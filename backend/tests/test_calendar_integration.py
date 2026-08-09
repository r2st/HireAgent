"""The Google/Outlook calendar gateway (design §4.3).

The contract under test is "never fatal": every failure mode a vendor can
present — an HTTP error, a dead socket, junk JSON, a per-calendar error array —
must come back as ``synced=False`` rather than an exception, because the
scheduler above falls back to working hours and carries on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from app.integrations import calendar as calendar_api
from app.integrations.calendar import (
    BusyResult,
    CalendarCredentials,
    CalendarEvent,
    GoogleCalendarProvider,
    OutlookCalendarProvider,
    UnavailableProvider,
    _coerce_blocks,
    _parse_dt,
)

WINDOW_START = datetime(2027, 1, 4, 0, tzinfo=UTC)
WINDOW_END = datetime(2027, 1, 11, 0, tzinfo=UTC)


class FakeResponse:
    def __init__(self, status_code: int, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeHTTP:
    """Stands in for ``httpx`` inside the calendar module.

    Records every call so a test can assert on the request the provider built,
    not merely on what it did with the reply.
    """

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        outer = self

        class _Client:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> bool:
                return False

            async def _record(self, method: str, url: str, **kwargs):
                outer.calls.append({"method": method, "url": url, **kwargs})
                if not outer.responses:
                    raise AssertionError(f"unexpected {method} {url}")
                nxt = outer.responses.pop(0)
                if isinstance(nxt, Exception):
                    raise nxt
                return nxt

            async def post(self, url, **kwargs):
                return await self._record("POST", url, **kwargs)

            async def delete(self, url, **kwargs):
                return await self._record("DELETE", url, **kwargs)

        self.AsyncClient = _Client
        self.HTTPError = httpx.HTTPError


@pytest.fixture
def google() -> GoogleCalendarProvider:
    return GoogleCalendarProvider(client_id="cid", client_secret="secret")


@pytest.fixture
def outlook() -> OutlookCalendarProvider:
    return OutlookCalendarProvider(client_id="cid", client_secret="secret")


def creds(**overrides) -> CalendarCredentials:
    base = {
        "provider": "google",
        "email": "interviewer@acme.test",
        "access_token": "at-live",
        "refresh_token": "rt",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
        "calendar_id": None,
    }
    base.update(overrides)
    return CalendarCredentials(**base)  # type: ignore[arg-type]


def install(monkeypatch: pytest.MonkeyPatch, http: FakeHTTP) -> FakeHTTP:
    monkeypatch.setattr(calendar_api, "httpx", http)
    return http


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
class TestCredentials:
    def test_a_future_expiry_is_not_expired(self) -> None:
        assert not creds(expires_at=datetime.now(UTC) + timedelta(hours=2)).is_expired

    def test_a_past_expiry_is_expired(self) -> None:
        assert creds(expires_at=datetime.now(UTC) - timedelta(minutes=1)).is_expired

    def test_expiry_inside_the_refresh_margin_counts_as_expired(self) -> None:
        # Refreshing early avoids a token lapsing mid-request.
        assert creds(expires_at=datetime.now(UTC) + timedelta(minutes=2)).is_expired

    def test_an_unknown_expiry_is_assumed_valid(self) -> None:
        # Let a 401 drive the refresh rather than guessing.
        assert not creds(expires_at=None).is_expired

    def test_a_naive_expiry_is_read_as_utc(self) -> None:
        naive = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)
        assert creds(expires_at=naive).is_expired


class TestParsing:
    def test_parses_z_suffixed_timestamps(self) -> None:
        assert _parse_dt("2027-01-04T09:00:00Z") == datetime(2027, 1, 4, 9, tzinfo=UTC)

    def test_assumes_utc_for_a_naive_timestamp(self) -> None:
        # Graph returns naive strings alongside a separate timeZone field.
        assert _parse_dt("2027-01-04T09:00:00") == datetime(2027, 1, 4, 9, tzinfo=UTC)

    def test_returns_none_for_junk(self) -> None:
        assert _parse_dt("not a date") is None
        assert _parse_dt(None) is None
        assert _parse_dt("") is None

    def test_coerce_blocks_drops_inverted_and_empty_ranges(self) -> None:
        blocks = _coerce_blocks(
            [
                {"start": "2027-01-04T09:00:00Z", "end": "2027-01-04T10:00:00Z"},
                {"start": "2027-01-04T11:00:00Z", "end": "2027-01-04T11:00:00Z"},
                {"start": "2027-01-04T13:00:00Z", "end": "2027-01-04T12:00:00Z"},
                {"start": "junk", "end": "2027-01-04T12:00:00Z"},
                "not a dict",
            ],
            "start",
            "end",
        )
        assert blocks == [
            (datetime(2027, 1, 4, 9, tzinfo=UTC), datetime(2027, 1, 4, 10, tzinfo=UTC))
        ]


# --------------------------------------------------------------------------- #
# Unavailable provider and registry
# --------------------------------------------------------------------------- #
class TestUnavailableProvider:
    async def test_reports_not_synced_rather_than_raising(self) -> None:
        provider = UnavailableProvider("google", "not configured")
        result = await provider.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is False
        assert result.blocks == []
        assert result.error == "not configured"

    async def test_event_writes_fail_softly(self) -> None:
        provider = UnavailableProvider("google", "not configured")
        event = CalendarEvent(summary="x", start=WINDOW_START, end=WINDOW_END)
        assert (await provider.create_event(creds(), event)).ok is False
        assert (await provider.delete_event(creds(), "evt")).ok is False


class TestRegistry:
    def test_an_unknown_vendor_yields_an_unavailable_provider(self) -> None:
        calendar_api.set_providers(None)
        provider = calendar_api.get_provider("carrier-pigeon")
        assert isinstance(provider, UnavailableProvider)

    def test_providers_are_unavailable_when_oauth_is_unconfigured(self) -> None:
        # The test environment sets no Google/Microsoft client credentials.
        calendar_api.set_providers(None)
        assert isinstance(calendar_api.get_provider("google"), UnavailableProvider)

    def test_set_providers_overrides_the_registry(self) -> None:
        sentinel = UnavailableProvider("google", "stub")
        calendar_api.set_providers({"google": sentinel})
        assert calendar_api.get_provider("google") is sentinel
        assert calendar_api.get_provider("GOOGLE") is sentinel
        calendar_api.set_providers(None)


# --------------------------------------------------------------------------- #
# Google
# --------------------------------------------------------------------------- #
class TestGoogleFreeBusy:
    async def test_parses_busy_blocks(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(
                    200,
                    {
                        "calendars": {
                            "interviewer@acme.test": {
                                "busy": [
                                    {
                                        "start": "2027-01-04T09:00:00Z",
                                        "end": "2027-01-04T10:00:00Z",
                                    }
                                ]
                            }
                        }
                    },
                )
            ),
        )
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is True
        assert result.blocks == [
            (
                datetime(2027, 1, 4, 9, tzinfo=UTC),
                datetime(2027, 1, 4, 10, tzinfo=UTC),
            )
        ]
        assert http.calls[0]["json"]["items"] == [{"id": "interviewer@acme.test"}]

    async def test_an_http_error_is_not_an_empty_calendar(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(500, text="boom")))
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is False
        assert "500" in (result.error or "")

    async def test_a_transport_failure_is_swallowed(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(httpx.ConnectError("no route")))
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is False

    async def test_unparseable_json_is_swallowed(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(200, ValueError("bad json"))))
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is False

    async def test_a_per_calendar_error_marks_the_read_unsynced(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # HTTP 200 with an errors array is Google's way of saying "no access".
        install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(
                    200,
                    {
                        "calendars": {
                            "interviewer@acme.test": {
                                "errors": [{"reason": "notFound"}],
                                "busy": [],
                            }
                        }
                    },
                )
            ),
        )
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is False
        assert "notFound" in (result.error or "")

    async def test_an_empty_calendar_is_synced_with_no_blocks(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(200, {"calendars": {"interviewer@acme.test": {"busy": []}}})
            ),
        )
        result = await google.fetch_busy(creds(), WINDOW_START, WINDOW_END)
        assert result.synced is True and result.blocks == []


class TestGoogleTokenRefresh:
    async def test_an_expired_token_is_refreshed_before_the_call(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(200, {"access_token": "at-new", "expires_in": 3600}),
                FakeResponse(200, {"calendars": {}}),
            ),
        )
        result = await google.fetch_busy(
            creds(expires_at=datetime.now(UTC) - timedelta(hours=1)),
            WINDOW_START,
            WINDOW_END,
        )
        assert result.refreshed is not None
        assert result.refreshed.access_token == "at-new"
        # The refreshed token, not the stale one, is used for the real call.
        assert http.calls[1]["headers"]["Authorization"] == "Bearer at-new"

    async def test_a_failed_refresh_falls_back_to_the_existing_token(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Our expiry bookkeeping may simply be wrong; the old token may work.
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(400, {"error": "invalid_grant"}),
                FakeResponse(200, {"calendars": {}}),
            ),
        )
        result = await google.fetch_busy(
            creds(expires_at=datetime.now(UTC) - timedelta(hours=1)),
            WINDOW_START,
            WINDOW_END,
        )
        assert result.refreshed is None
        assert http.calls[1]["headers"]["Authorization"] == "Bearer at-live"

    async def test_no_token_at_all_reports_unsynced(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP())
        result = await google.fetch_busy(
            creds(access_token=None, refresh_token=None), WINDOW_START, WINDOW_END
        )
        assert result.synced is False
        assert "No usable Google access token" in (result.error or "")


class TestGoogleEvents:
    async def test_creates_an_event_with_a_meet_link(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(
                    200,
                    {
                        "id": "evt-1",
                        "hangoutLink": "https://meet.google.com/abc-defg-hij",
                        "htmlLink": "https://calendar.google.com/evt-1",
                    },
                )
            ),
        )
        event = CalendarEvent(
            summary="Technical interview",
            start=datetime(2027, 1, 4, 9, tzinfo=UTC),
            end=datetime(2027, 1, 4, 10, tzinfo=UTC),
            attendees=["a@acme.test", "candidate@example.com"],
        )
        result = await google.create_event(creds(), event)
        assert result.ok is True
        assert result.external_event_id == "evt-1"
        assert result.meeting_url == "https://meet.google.com/abc-defg-hij"

        body = http.calls[0]["json"]
        assert body["conferenceData"]["createRequest"]["conferenceSolutionKey"] == {
            "type": "hangoutsMeet"
        }
        assert [a["email"] for a in body["attendees"]] == [
            "a@acme.test",
            "candidate@example.com",
        ]

    async def test_falls_back_to_the_html_link_without_a_meet_link(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            FakeHTTP(FakeResponse(200, {"id": "e", "htmlLink": "https://cal/e"})),
        )
        event = CalendarEvent(
            summary="Phone screen",
            start=datetime(2027, 1, 4, 9, tzinfo=UTC),
            end=datetime(2027, 1, 4, 10, tzinfo=UTC),
            create_conference=False,
        )
        result = await google.create_event(creds(), event)
        assert result.meeting_url == "https://cal/e"

    async def test_a_rejected_create_returns_not_ok(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(403, {"error": "forbidden"})))
        event = CalendarEvent(
            summary="x",
            start=datetime(2027, 1, 4, 9, tzinfo=UTC),
            end=datetime(2027, 1, 4, 10, tzinfo=UTC),
        )
        result = await google.create_event(creds(), event)
        assert result.ok is False and "403" in (result.error or "")

    async def test_delete_succeeds(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(204)))
        assert (await google.delete_event(creds(), "evt-1")).ok is True

    async def test_an_already_deleted_event_counts_as_deleted(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 410 Gone is the state we wanted anyway.
        install(monkeypatch, FakeHTTP(FakeResponse(410)))
        assert (await google.delete_event(creds(), "evt-1")).ok is True

    async def test_a_failed_delete_reports_the_status(
        self, google: GoogleCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(500)))
        result = await google.delete_event(creds(), "evt-1")
        assert result.ok is False and "500" in (result.error or "")


# --------------------------------------------------------------------------- #
# Outlook
# --------------------------------------------------------------------------- #
class TestOutlook:
    async def test_parses_schedule_items_and_skips_free_ones(
        self, outlook: OutlookCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(
                    200,
                    {
                        "value": [
                            {
                                "scheduleItems": [
                                    {
                                        "status": "busy",
                                        "start": {"dateTime": "2027-01-04T09:00:00"},
                                        "end": {"dateTime": "2027-01-04T10:00:00"},
                                    },
                                    {
                                        # "free" items are placeholders, not conflicts.
                                        "status": "free",
                                        "start": {"dateTime": "2027-01-04T11:00:00"},
                                        "end": {"dateTime": "2027-01-04T12:00:00"},
                                    },
                                    {
                                        "status": "oof",
                                        "start": {"dateTime": "2027-01-04T14:00:00"},
                                        "end": {"dateTime": "2027-01-04T15:00:00"},
                                    },
                                ]
                            }
                        ]
                    },
                )
            ),
        )
        result = await outlook.fetch_busy(
            creds(provider="outlook"), WINDOW_START, WINDOW_END
        )
        assert result.synced is True
        assert result.blocks == [
            (
                datetime(2027, 1, 4, 9, tzinfo=UTC),
                datetime(2027, 1, 4, 10, tzinfo=UTC),
            ),
            (
                datetime(2027, 1, 4, 14, tzinfo=UTC),
                datetime(2027, 1, 4, 15, tzinfo=UTC),
            ),
        ]

    async def test_an_empty_schedule_response_is_synced(
        self, outlook: OutlookCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(200, {"value": []})))
        result = await outlook.fetch_busy(
            creds(provider="outlook"), WINDOW_START, WINDOW_END
        )
        assert result.synced is True and result.blocks == []

    async def test_an_http_error_reports_unsynced(
        self, outlook: OutlookCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(503)))
        result = await outlook.fetch_busy(
            creds(provider="outlook"), WINDOW_START, WINDOW_END
        )
        assert result.synced is False

    async def test_creates_an_event_with_a_teams_link(
        self, outlook: OutlookCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(
                    200,
                    {
                        "id": "AAMk-1",
                        "onlineMeeting": {"joinUrl": "https://teams.microsoft.com/l/x"},
                    },
                )
            ),
        )
        event = CalendarEvent(
            summary="Panel",
            start=datetime(2027, 1, 4, 9, tzinfo=UTC),
            end=datetime(2027, 1, 4, 10, tzinfo=UTC),
            attendees=["a@acme.test"],
        )
        result = await outlook.create_event(creds(provider="outlook"), event)
        assert result.ok is True
        assert result.external_event_id == "AAMk-1"
        assert result.meeting_url == "https://teams.microsoft.com/l/x"
        assert http.calls[0]["json"]["isOnlineMeeting"] is True

    async def test_refresh_requests_the_graph_scope(
        self, outlook: OutlookCalendarProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Graph rejects a refresh that omits the scope.
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(200, {"access_token": "at-new", "expires_in": 3600}),
                FakeResponse(200, {"value": []}),
            ),
        )
        await outlook.fetch_busy(
            creds(
                provider="outlook", expires_at=datetime.now(UTC) - timedelta(hours=1)
            ),
            WINDOW_START,
            WINDOW_END,
        )
        assert "scope" in http.calls[0]["data"]
        assert "login.microsoftonline.com" in http.calls[0]["url"]

    async def test_a_missing_tenant_defaults_to_common(self) -> None:
        assert "/common/" in OutlookCalendarProvider(tenant="").token_url


class TestProviderConfiguration:
    def test_a_provider_without_credentials_is_unconfigured(self) -> None:
        assert not GoogleCalendarProvider().is_configured
        assert GoogleCalendarProvider(client_id="a", client_secret="b").is_configured

    async def test_refresh_is_skipped_without_client_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No client secret means no refresh is even attempted.
        install(monkeypatch, FakeHTTP(FakeResponse(200, {"calendars": {}})))
        provider = GoogleCalendarProvider()
        result = await provider.fetch_busy(
            creds(expires_at=datetime.now(UTC) - timedelta(hours=1)),
            WINDOW_START,
            WINDOW_END,
        )
        assert result.refreshed is None


class TestBusyResultDefaults:
    def test_defaults_to_synced_with_no_blocks(self) -> None:
        result = BusyResult()
        assert result.synced is True and result.blocks == [] and result.error is None

    def test_namespace_shim_exposes_the_httpx_error_type(self) -> None:
        # Guards the monkeypatch itself: the providers' except clauses
        # reference httpx.HTTPError through the module attribute.
        http = FakeHTTP()
        assert issubclass(http.HTTPError, Exception)
        assert isinstance(SimpleNamespace(AsyncClient=http.AsyncClient), SimpleNamespace)
