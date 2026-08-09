"""Interview scheduling end to end (design §4.3).

Covers the whole arc the design describes: read availability, offer slots, let
the candidate book from a link with no account, remind, reschedule, cancel,
and collect scorecards — plus the tenant and consent boundaries around it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta

import pytest
from httpx import AsyncClient, Response

from app.integrations import calendar as calendar_api
from app.models.enums import UserRole
from tests.conftest import make_user
from tests.factories import FakeCalendarProvider
from tests.test_applications import apply_candidate, create_candidate, create_job

WORKING_HOURS = {
    "mon": [["09:00", "17:00"]],
    "tue": [["09:00", "17:00"]],
    "wed": [["09:00", "17:00"]],
    "thu": [["09:00", "17:00"]],
    "fri": [["09:00", "17:00"]],
}


def future_monday(weeks: int = 4) -> datetime:
    """Midnight UTC on a Monday at least ``weeks`` out.

    Anchoring on a weekday keeps working-hours expansion deterministic, and
    anchoring on "now" keeps the suite from expiring on a hard-coded date.
    """
    today = datetime.now(UTC).date()
    days_ahead = (7 - today.weekday()) % 7 or 7
    monday = today + timedelta(days=days_ahead, weeks=weeks)
    return datetime.combine(monday, time(0), tzinfo=UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


WINDOW_START = future_monday()
WINDOW_END = WINDOW_START + timedelta(days=5)


async def connect_calendar(
    client: AsyncClient, headers: dict, *, email: str, **overrides
) -> dict:
    payload = {
        "provider": "google",
        "email": email,
        "access_token": "at",
        "refresh_token": "rt",
        "working_hours": WORKING_HOURS,
        "timezone": "UTC",
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/calendar/accounts", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
async def scene(client: AsyncClient, auth_headers: dict, registered: dict) -> dict:
    """A job, a consenting candidate, an application, and one interviewer.

    The candidate grants email consent up front because §8.2 blocks sending
    slots without it — the tests that care about the gate revoke or omit it.
    """
    job = await create_job(client, auth_headers)
    candidate = await create_candidate(
        client,
        auth_headers,
        consents=[{"consent_type": "email_communication", "granted": True}],
    )
    application = await apply_candidate(
        client, auth_headers, job["id"], candidate["id"]
    )
    interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
    await connect_calendar(
        client,
        interviewer["headers"],
        email=interviewer["user"]["email"],
    )
    return {
        "job": job,
        "candidate": candidate,
        "application": application,
        "interviewer": interviewer,
        "admin_id": registered["user"]["id"],
    }


async def propose(
    client: AsyncClient, headers: dict, scene: dict, **overrides
) -> dict:
    payload = {
        "application_id": scene["application"]["id"],
        "interviewer_ids": [scene["interviewer"]["user"]["id"]],
        "duration_minutes": 60,
        "window_start": iso(WINDOW_START),
        "window_end": iso(WINDOW_END),
        "min_notice_hours": 0,
        "slot_count": 4,
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/interviews/propose", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def schedule(
    client: AsyncClient, headers: dict, scene: dict, **overrides
) -> Response:
    """Returns the raw response — callers assert on failure statuses too."""
    payload = {
        "application_id": scene["application"]["id"],
        "interviewer_ids": [scene["interviewer"]["user"]["id"]],
        "scheduled_at": iso(WINDOW_START + timedelta(hours=10)),
        "duration_minutes": 60,
    }
    payload.update(overrides)
    return await client.post(
        "/api/v1/interviews/schedule", headers=headers, json=payload
    )


async def booking_token(client: AsyncClient, headers: dict, interview_id: str) -> str:
    resp = await client.get(f"/api/v1/interviews/{interview_id}", headers=headers)
    assert resp.status_code == 200, resp.text
    url = resp.json()["booking_url"]
    assert url, "expected a booking url"
    return url.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- #
# Calendar accounts
# --------------------------------------------------------------------------- #
class TestCalendarAccounts:
    async def test_connecting_never_echoes_the_oauth_tokens(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        account = await connect_calendar(
            client, auth_headers, email="admin@acme.test"
        )
        assert "access_token" not in account
        assert "refresh_token" not in account
        assert account["provider"] == "google"
        assert account["is_active"] is True

    async def test_reconnecting_updates_rather_than_duplicating(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        first = await connect_calendar(client, auth_headers, email="admin@acme.test")
        second = await connect_calendar(
            client, auth_headers, email="admin@acme.test", timezone="Asia/Kolkata"
        )
        assert first["id"] == second["id"]
        assert second["timezone"] == "Asia/Kolkata"

        listed = await client.get("/api/v1/calendar/accounts", headers=auth_headers)
        assert len(listed.json()) == 1

    async def test_an_unsupported_provider_is_rejected(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/calendar/accounts",
            headers=auth_headers,
            json={"provider": "carrier-pigeon", "email": "a@b.test"},
        )
        assert resp.status_code == 422

    async def test_a_non_admin_cannot_connect_for_someone_else(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
        resp = await client.post(
            "/api/v1/calendar/accounts",
            headers=recruiter["headers"],
            json={
                "provider": "google",
                "email": "someone@acme.test",
                "user_id": scene["interviewer"]["user"]["id"],
            },
        )
        assert resp.status_code == 403

    async def test_an_admin_can_connect_for_someone_else(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/calendar/accounts",
            headers=auth_headers,
            json={
                "provider": "outlook",
                "email": "second@acme.test",
                "user_id": scene["interviewer"]["user"]["id"],
            },
        )
        assert resp.status_code == 201

    async def test_a_non_admin_only_sees_their_own_calendars(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await connect_calendar(client, auth_headers, email="admin@acme.test")
        resp = await client.get(
            "/api/v1/calendar/accounts", headers=scene["interviewer"]["headers"]
        )
        emails = {a["email"] for a in resp.json()}
        assert emails == {scene["interviewer"]["user"]["email"]}

    async def test_working_hours_can_be_updated(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        account = await connect_calendar(client, auth_headers, email="a@acme.test")
        resp = await client.patch(
            f"/api/v1/calendar/accounts/{account['id']}",
            headers=auth_headers,
            json={"working_hours": {"sat": [["10:00", "14:00"]]}, "timezone": "UTC"},
        )
        assert resp.status_code == 200
        assert resp.json()["working_hours"] == {"sat": [["10:00", "14:00"]]}

    @pytest.mark.parametrize(
        "hours",
        [
            {"funday": [["09:00", "17:00"]]},
            {"mon": [["17:00", "09:00"]]},
            {"mon": [["nonsense", "17:00"]]},
            {"mon": [["09:00"]]},
            {"mon": "09:00-17:00"},
        ],
    )
    async def test_malformed_working_hours_are_rejected_at_the_edge(
        self, client: AsyncClient, auth_headers: dict, hours: dict
    ) -> None:
        # Silently skipping these would surface later as "no availability".
        resp = await client.post(
            "/api/v1/calendar/accounts",
            headers=auth_headers,
            json={
                "provider": "google",
                "email": "a@acme.test",
                "working_hours": hours,
            },
        )
        assert resp.status_code == 422

    async def test_disconnecting_discards_the_stored_tokens(
        self, client: AsyncClient, auth_headers: dict, session
    ) -> None:
        from sqlalchemy import select

        from app.models.interview import CalendarAccount

        account = await connect_calendar(client, auth_headers, email="a@acme.test")
        resp = await client.delete(
            f"/api/v1/calendar/accounts/{account['id']}", headers=auth_headers
        )
        assert resp.status_code == 200

        row = await session.scalar(
            select(CalendarAccount).where(CalendarAccount.id == uuid.UUID(account["id"]))
        )
        assert row.deleted_at is not None
        assert row.access_token is None
        assert row.refresh_token is None

    async def test_another_tenant_cannot_read_the_account(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        account = await connect_calendar(client, auth_headers, email="a@acme.test")
        resp = await client.patch(
            f"/api/v1/calendar/accounts/{account['id']}",
            headers=second_org["headers"],
            json={"timezone": "UTC"},
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
class TestAvailability:
    async def test_slots_come_from_the_configured_working_hours(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=auth_headers,
            json={
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "duration_minutes": 60,
                "window_start": iso(WINDOW_START),
                "window_end": iso(WINDOW_END),
                "min_notice_hours": 0,
                "limit": 5,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["slots"]) == 5
        for slot in body["slots"]:
            start = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
            assert 9 <= start.hour < 17
            assert start.weekday() < 5

    async def test_an_unsynced_calendar_is_reported_not_hidden(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # No OAuth client is configured in tests, so the provider is
        # unavailable and slots rest on working hours alone.
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=auth_headers,
            json={
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "window_start": iso(WINDOW_START),
                "window_end": iso(WINDOW_END),
                "min_notice_hours": 0,
            },
        )
        body = resp.json()
        assert body["warnings"]
        assert body["interviewers"][0]["calendar_synced"] is False

    async def test_busy_time_is_subtracted_from_the_offer(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Block the whole first working day; its slots must disappear.
        fake = FakeCalendarProvider(
            busy=[(WINDOW_START, WINDOW_START + timedelta(days=1))]
        )
        calendar_api.set_providers({"google": fake})

        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=auth_headers,
            json={
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "duration_minutes": 60,
                "window_start": iso(WINDOW_START),
                "window_end": iso(WINDOW_END),
                "min_notice_hours": 0,
                "limit": 20,
            },
        )
        body = resp.json()
        assert fake.busy_calls == 1
        assert body["interviewers"][0]["calendar_synced"] is True
        booked_day = WINDOW_START.date()
        assert all(
            datetime.fromisoformat(s["start"].replace("Z", "+00:00")).date()
            != booked_day
            for s in body["slots"]
        )

    async def test_a_provider_that_raises_does_not_break_the_lookup(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        calendar_api.set_providers({"google": FakeCalendarProvider(raises=True)})
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=auth_headers,
            json={
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "window_start": iso(WINDOW_START),
                "window_end": iso(WINDOW_END),
                "min_notice_hours": 0,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["interviewers"][0]["calendar_synced"] is False

    async def test_an_interviewer_role_cannot_probe_availability(
        self, client: AsyncClient, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=scene["interviewer"]["headers"],
            json={"interviewer_ids": [scene["interviewer"]["user"]["id"]]},
        )
        assert resp.status_code == 403

    async def test_an_inverted_window_is_rejected(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=auth_headers,
            json={
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "window_start": iso(WINDOW_END),
                "window_end": iso(WINDOW_START),
            },
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Direct scheduling
# --------------------------------------------------------------------------- #
class TestScheduleInterview:
    async def test_scheduling_books_the_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await schedule(client, auth_headers, scene)
        assert resp.status_code == 201, resp.text
        interview = resp.json()["interview"]
        assert interview["status"] == "scheduled"
        assert interview["round_number"] == 1
        assert interview["scheduled_at"] is not None

    async def test_rounds_increment_per_application(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        first = await schedule(client, auth_headers, scene)
        second = await schedule(
            client,
            auth_headers,
            scene,
            scheduled_at=iso(WINDOW_START + timedelta(days=1, hours=10)),
        )
        assert first.json()["interview"]["round_number"] == 1
        assert second.json()["interview"]["round_number"] == 2

    async def test_the_past_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await schedule(
            client,
            auth_headers,
            scene,
            scheduled_at=iso(datetime.now(UTC) - timedelta(hours=1)),
        )
        assert resp.status_code == 422
        assert "past" in resp.json()["detail"]["message"]

    async def test_a_double_booked_interviewer_is_a_conflict(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        # A second candidate, same interviewer, overlapping time.
        other = await create_candidate(client, auth_headers)
        other_application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], other["id"]
        )
        resp = await schedule(
            client,
            auth_headers,
            scene,
            application_id=other_application["id"],
            scheduled_at=iso(WINDOW_START + timedelta(hours=10, minutes=30)),
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["details"]["conflicts"]

    async def test_a_back_to_back_booking_is_not_a_conflict(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Half-open intervals: 10:00-11:00 and 11:00-12:00 are fine.
        await schedule(client, auth_headers, scene)
        other = await create_candidate(client, auth_headers)
        other_application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], other["id"]
        )
        resp = await schedule(
            client,
            auth_headers,
            scene,
            application_id=other_application["id"],
            scheduled_at=iso(WINDOW_START + timedelta(hours=11)),
        )
        assert resp.status_code == 201

    async def test_conflicts_can_be_overridden_deliberately(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        other = await create_candidate(client, auth_headers)
        other_application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], other["id"]
        )
        resp = await schedule(
            client,
            auth_headers,
            scene,
            application_id=other_application["id"],
            allow_conflicts=True,
        )
        assert resp.status_code == 201

    async def test_a_live_interview_needs_an_interviewer(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await schedule(client, auth_headers, scene, interviewer_ids=[])
        assert resp.status_code == 422
        assert "at least one interviewer" in resp.json()["detail"]["message"]

    async def test_an_async_screening_needs_no_interviewer(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await schedule(
            client, auth_headers, scene, interviewer_ids=[], type="async_video"
        )
        assert resp.status_code == 201

    async def test_an_interviewer_from_another_tenant_is_not_found(
        self, client: AsyncClient, auth_headers: dict, scene: dict, second_org: dict
    ) -> None:
        resp = await schedule(
            client, auth_headers, scene, interviewer_ids=[second_org["user"]["id"]]
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["details"]["missing"]

    async def test_a_rejected_application_cannot_gain_an_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await client.post(
            f"/api/v1/applications/{scene['application']['id']}/reject",
            headers=auth_headers,
            json={"reason": "not a fit"},
        )
        resp = await schedule(client, auth_headers, scene)
        assert resp.status_code == 409

    async def test_an_interviewer_role_cannot_schedule(
        self, client: AsyncClient, scene: dict
    ) -> None:
        resp = await schedule(client, scene["interviewer"]["headers"], scene)
        assert resp.status_code == 403

    async def test_a_calendar_event_is_written_and_linked(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        fake = FakeCalendarProvider()
        calendar_api.set_providers({"google": fake})

        resp = await schedule(client, auth_headers, scene)
        interview = resp.json()["interview"]
        assert len(fake.created) == 1
        assert interview["external_event_id"] == "ext-event-1"
        assert interview["meeting_url"] == "https://meet.example.test/abc"
        # The candidate is invited alongside the interviewer.
        assert len(fake.created[0].attendees) == 2

    async def test_a_calendar_failure_warns_but_still_books(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        calendar_api.set_providers({"google": FakeCalendarProvider(create_ok=False)})
        resp = await schedule(client, auth_headers, scene)
        assert resp.status_code == 201
        body = resp.json()
        assert body["interview"]["status"] == "scheduled"
        assert any("could not be created" in w for w in body["warnings"])

    async def test_no_connected_calendar_warns_but_still_books(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        from sqlalchemy import delete

        from app.models.interview import CalendarAccount

        await session.execute(delete(CalendarAccount))
        await session.commit()

        resp = await schedule(client, auth_headers, scene)
        assert resp.status_code == 201
        assert any("No connected calendar" in w for w in resp.json()["warnings"])


# --------------------------------------------------------------------------- #
# Proposing slots and candidate booking
# --------------------------------------------------------------------------- #
class TestProposeSlots:
    async def test_proposing_creates_a_pending_interview_with_slots(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        assert body["interview"]["status"] == "pending"
        assert body["interview"]["scheduled_at"] is None
        assert len(body["slots"]) == 4
        assert len(body["interview"]["proposed_slots"]) == 4

    async def test_the_booking_link_is_returned_to_staff_only(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        detail = await client.get(
            f"/api/v1/interviews/{body['interview']['id']}", headers=auth_headers
        )
        assert "/book/" in detail.json()["booking_url"]
        # The token itself is never part of the interview payload.
        assert "booking_token" not in body["interview"]

    async def test_slots_are_spread_across_days(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene, slot_count=6)
        days = {s["start"][:10] for s in body["slots"]}
        assert len(days) >= 2

    async def test_sending_slots_requires_communication_consent(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Design §8.2: no consent, no contact.
        candidate = await create_candidate(client, auth_headers)
        application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], candidate["id"]
        )
        resp = await client.post(
            "/api/v1/interviews/propose",
            headers=auth_headers,
            json={
                "application_id": application["id"],
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
            },
        )
        assert resp.status_code == 422
        assert "consent" in resp.json()["detail"]["message"]

    async def test_consent_can_be_waived_explicitly(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Slots relayed by phone, say — the recruiter takes responsibility.
        candidate = await create_candidate(client, auth_headers)
        application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], candidate["id"]
        )
        resp = await client.post(
            "/api/v1/interviews/propose",
            headers=auth_headers,
            json={
                "application_id": application["id"],
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
                "require_consent": False,
            },
        )
        assert resp.status_code == 201

    async def test_withdrawn_consent_blocks_sending_slots(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await client.post(
            f"/api/v1/candidates/{scene['candidate']['id']}/consents",
            headers=auth_headers,
            json={"consent_type": "email_communication", "granted": False},
        )
        resp = await client.post(
            "/api/v1/interviews/propose",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "interviewer_ids": [scene["interviewer"]["user"]["id"]],
            },
        )
        assert resp.status_code == 422

    async def test_no_availability_still_creates_a_pending_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # The recruiter keeps the setup work and can widen the window.
        calendar_api.set_providers(
            {
                "google": FakeCalendarProvider(
                    busy=[(WINDOW_START, WINDOW_START + timedelta(days=5))]
                )
            }
        )
        body = await propose(client, auth_headers, scene)
        assert body["interview"]["status"] == "pending"
        assert body["slots"] == []
        assert any("No common availability" in w for w in body["warnings"])


class TestCandidateBooking:
    async def test_the_public_view_shows_the_meeting_not_the_pipeline(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])

        resp = await client.get(f"/api/v1/booking/{token}")
        assert resp.status_code == 200
        view = resp.json()
        assert view["job_title"] == scene["job"]["title"]
        assert view["organization_name"] == "Acme Talent"
        assert view["can_book"] is True
        assert len(view["proposed_slots"]) == 4
        # Nothing about the candidate, the interviewers, or the score leaks.
        for leaked in ("candidate", "application_id", "participants", "notes"):
            assert leaked not in view

    async def test_the_public_view_needs_no_authentication(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        resp = await client.get(f"/api/v1/booking/{token}")
        assert resp.status_code == 200

    async def test_booking_a_slot_confirms_the_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        chosen = body["slots"][1]["start"]

        resp = await client.post(
            f"/api/v1/booking/{token}/book", json={"slot_start": chosen}
        )
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["status"] == "confirmed"
        assert view["scheduled_at"].replace("+00:00", "Z") == chosen
        assert view["can_book"] is False

    async def test_booking_writes_the_calendar_event(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        fake = FakeCalendarProvider()
        calendar_api.set_providers({"google": fake})
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])

        await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": body["slots"][0]["start"]},
        )
        assert len(fake.created) == 1

    async def test_a_time_that_was_not_offered_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Otherwise a candidate could book straight over another meeting.
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        resp = await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": iso(WINDOW_START + timedelta(hours=3))},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["details"]["offered"]

    async def test_booking_twice_is_a_conflict(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": body["slots"][0]["start"]},
        )
        resp = await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": body["slots"][1]["start"]},
        )
        assert resp.status_code == 409
        assert "reschedule" in resp.json()["detail"]["message"]

    @pytest.mark.parametrize("token", ["nope", "a" * 43])
    async def test_a_bad_token_is_indistinguishable_from_a_guess(
        self, client: AsyncClient, token: str
    ) -> None:
        """A wrong token and a well-formed one that does not exist look alike.

        Both must answer with the same 404 body, or the endpoint becomes an
        oracle for which booking tokens are live.
        """
        resp = await client.get(f"/api/v1/booking/{token}")
        assert resp.status_code == 404
        assert resp.json()["detail"]["message"] == "This booking link is not valid"

    async def test_an_empty_token_does_not_reach_the_endpoint(
        self, client: AsyncClient
    ) -> None:
        """``/booking/`` matches no route; it still answers in our envelope."""
        resp = await client.get("/api/v1/booking/")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "not_found"

    async def test_an_expired_link_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        from sqlalchemy import select

        from app.models.interview import Interview

        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])

        interview = await session.scalar(
            select(Interview).where(Interview.booking_token == token)
        )
        interview.booking_expires_at = datetime.now(UTC) - timedelta(hours=1)
        await session.commit()

        resp = await client.get(f"/api/v1/booking/{token}")
        assert resp.status_code == 404
        assert "expired" in resp.json()["detail"]["message"]

    async def test_a_slot_taken_since_it_was_offered_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """The re-check at booking time, not proposal time, is what catches this."""
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        chosen = body["slots"][0]["start"]

        # The same interviewer gets booked into that exact slot meanwhile.
        other = await create_candidate(client, auth_headers)
        other_application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], other["id"]
        )
        blocking = await schedule(
            client,
            auth_headers,
            scene,
            application_id=other_application["id"],
            scheduled_at=chosen,
        )
        assert blocking.status_code == 201

        resp = await client.post(
            f"/api/v1/booking/{token}/book", json={"slot_start": chosen}
        )
        assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# Rescheduling and cancellation
# --------------------------------------------------------------------------- #
class TestReschedule:
    async def test_rescheduling_supersedes_rather_than_mutates(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        original = (await schedule(client, auth_headers, scene)).json()["interview"]
        new_time = iso(WINDOW_START + timedelta(days=1, hours=14))

        resp = await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={"scheduled_at": new_time, "reason": "clash"},
        )
        assert resp.status_code == 200, resp.text
        replacement = resp.json()["interview"]
        assert replacement["id"] != original["id"]
        assert replacement["rescheduled_from_id"] == original["id"]
        assert replacement["status"] == "scheduled"

        stale = await client.get(
            f"/api/v1/interviews/{original['id']}", headers=auth_headers
        )
        assert stale.json()["status"] == "rescheduled"

    async def test_participants_carry_over_and_must_re_accept(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        original = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{original['id']}/respond",
            headers=scene["interviewer"]["headers"],
            json={"response_status": "accepted"},
        )
        resp = await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={"scheduled_at": iso(WINDOW_START + timedelta(days=1, hours=14))},
        )
        replacement = resp.json()["interview"]

        participants = await client.get(
            f"/api/v1/interviews/{replacement['id']}/participants",
            headers=auth_headers,
        )
        rows = participants.json()
        assert len(rows) == 1
        # The old yes was for the old slot.
        assert rows[0]["response_status"] == "pending"

    async def test_the_old_calendar_hold_is_released_before_the_new_one(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        fake = FakeCalendarProvider()
        calendar_api.set_providers({"google": fake})
        original = (await schedule(client, auth_headers, scene)).json()["interview"]

        await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={"scheduled_at": iso(WINDOW_START + timedelta(days=1, hours=14))},
        )
        assert fake.deleted == ["ext-event-1"]
        assert len(fake.created) == 2

    async def test_the_booking_link_follows_the_new_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])

        resp = await client.post(
            f"/api/v1/interviews/{body['interview']['id']}/reschedule",
            headers=auth_headers,
            json={"scheduled_at": iso(WINDOW_START + timedelta(days=1, hours=14))},
        )
        replacement = resp.json()["interview"]
        view = await client.get(f"/api/v1/booking/{token}")
        assert view.status_code == 200
        assert view.json()["interview_id"] == replacement["id"]

    async def test_a_candidate_can_move_to_another_offered_slot(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": body["slots"][0]["start"]},
        )

        resp = await client.post(
            f"/api/v1/booking/{token}/reschedule",
            json={"slot_start": body["slots"][2]["start"]},
        )
        assert resp.status_code == 200, resp.text
        view = resp.json()
        assert view["status"] == "confirmed"
        assert view["scheduled_at"].replace("+00:00", "Z") == body["slots"][2]["start"]

    async def test_a_candidate_cannot_move_to_a_time_never_offered(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        resp = await client.post(
            f"/api/v1/booking/{token}/reschedule",
            json={"slot_start": iso(WINDOW_START + timedelta(hours=3))},
        )
        assert resp.status_code == 422

    async def test_rescheduling_needs_a_time_or_new_slots(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        original = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={"reason": "just because"},
        )
        assert resp.status_code == 422

    async def test_new_slots_can_be_proposed_instead_of_a_time(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        original = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={
                "propose_new_slots": True,
                "slot_count": 3,
                "window_start": iso(WINDOW_START),
                "window_end": iso(WINDOW_END),
            },
        )
        assert resp.status_code == 200
        replacement = resp.json()["interview"]
        assert replacement["status"] == "pending"
        assert len(replacement["proposed_slots"]) == 3

    async def test_a_completed_interview_cannot_be_rescheduled(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        original = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{original['id']}/complete", headers=auth_headers
        )
        resp = await client.post(
            f"/api/v1/interviews/{original['id']}/reschedule",
            headers=auth_headers,
            json={"scheduled_at": iso(WINDOW_START + timedelta(days=2, hours=10))},
        )
        assert resp.status_code == 409


class TestCancel:
    async def test_cancelling_kills_the_booking_link(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])

        resp = await client.post(
            f"/api/v1/interviews/{body['interview']['id']}/cancel",
            headers=auth_headers,
            json={"reason": "role closed"},
        )
        assert resp.status_code == 200
        assert resp.json()["interview"]["status"] == "cancelled"
        assert (await client.get(f"/api/v1/booking/{token}")).status_code == 404

    async def test_cancelling_releases_the_calendar_hold(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        fake = FakeCalendarProvider()
        calendar_api.set_providers({"google": fake})
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]

        await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel",
            headers=auth_headers,
            json={},
        )
        assert fake.deleted == ["ext-event-1"]

    async def test_a_candidate_can_cancel_from_the_link(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        body = await propose(client, auth_headers, scene)
        token = await booking_token(client, auth_headers, body["interview"]["id"])
        await client.post(
            f"/api/v1/booking/{token}/book",
            json={"slot_start": body["slots"][0]["start"]},
        )

        resp = await client.post(
            f"/api/v1/booking/{token}/cancel", json={"reason": "took another offer"}
        )
        assert resp.status_code == 200

        detail = await client.get(
            f"/api/v1/interviews/{body['interview']['id']}", headers=auth_headers
        )
        assert detail.json()["status"] == "cancelled"
        assert detail.json()["cancellation_reason"] == "took another offer"

    async def test_cancelling_twice_is_idempotent(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        first = await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel", headers=auth_headers, json={}
        )
        second = await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel", headers=auth_headers, json={}
        )
        assert first.status_code == 200 and second.status_code == 200

    async def test_the_freed_slot_becomes_bookable_again(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel", headers=auth_headers, json={}
        )

        other = await create_candidate(client, auth_headers)
        other_application = await apply_candidate(
            client, auth_headers, scene["job"]["id"], other["id"]
        )
        resp = await schedule(
            client, auth_headers, scene, application_id=other_application["id"]
        )
        assert resp.status_code == 201


# --------------------------------------------------------------------------- #
# Outcomes and feedback
# --------------------------------------------------------------------------- #
class TestOutcomes:
    async def test_completing_advances_the_pipeline_card(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/complete",
            headers=auth_headers,
            json={"notes": "went well"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "interviewed"

    async def test_completing_never_drags_a_later_stage_backwards(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # A late-recorded first round must not undo an offer.
        await client.put(
            f"/api/v1/applications/{scene['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "offered", "force": True},
        )
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{interview['id']}/complete", headers=auth_headers
        )
        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "offered"

    async def test_advancing_can_be_declined(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{interview['id']}/complete",
            headers=auth_headers,
            json={"advance_application": False},
        )
        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "applied"

    async def test_a_cancelled_interview_cannot_be_completed(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel", headers=auth_headers, json={}
        )
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/complete", headers=auth_headers
        )
        assert resp.status_code == 409

    async def test_a_no_show_is_recorded(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/no-show",
            headers=auth_headers,
            json={"reason": "candidate did not join"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_show"

    async def test_an_interviewer_can_accept_their_invitation(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/respond",
            headers=scene["interviewer"]["headers"],
            json={"response_status": "declined"},
        )
        assert resp.status_code == 200
        assert resp.json()["response_status"] == "declined"

    async def test_a_non_participant_cannot_respond(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/respond",
            headers=auth_headers,
            json={"response_status": "accepted"},
        )
        assert resp.status_code == 404


class TestFeedback:
    async def test_a_participant_can_file_a_scorecard(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=scene["interviewer"]["headers"],
            json={
                "rating": 8.5,
                "recommendation": "yes",
                "feedback": "Strong on systems design.",
                "scorecard": {"systems_design": 9, "communication": 8},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert float(body["rating"]) == 8.5
        assert body["recommendation"] == "yes"
        assert body["feedback_submitted_at"] is not None

    async def test_feedback_is_always_attributed_to_the_caller(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # An uninvolved colleague has no participant row to write to.
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=auth_headers,
            json={"rating": 10, "recommendation": "strong_yes"},
        )
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "payload",
        [
            {"rating": 11},
            {"rating": -1},
            {"recommendation": "maybe"},
        ],
    )
    async def test_out_of_range_scorecards_are_rejected(
        self, client: AsyncClient, auth_headers: dict, scene: dict, payload: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=scene["interviewer"]["headers"],
            json=payload,
        )
        assert resp.status_code == 422

    async def test_the_summary_aggregates_the_panel(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        second = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await schedule(
            client,
            auth_headers,
            scene,
            interviewer_ids=[
                scene["interviewer"]["user"]["id"],
                second["user"]["id"],
            ],
        )
        interview = resp.json()["interview"]

        await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=scene["interviewer"]["headers"],
            json={"rating": 8, "recommendation": "yes"},
        )
        await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=second["headers"],
            json={"rating": 6, "recommendation": "no"},
        )

        summary = await client.get(
            f"/api/v1/interviews/{interview['id']}/feedback", headers=auth_headers
        )
        body = summary.json()
        assert body["participants"] == 2
        assert body["submitted"] == 2
        assert body["average_rating"] == 7.0
        assert body["recommendations"] == {"yes": 1, "no": 1}

    async def test_feedback_cannot_be_left_on_a_cancelled_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel", headers=auth_headers, json={}
        )
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/feedback",
            headers=scene["interviewer"]["headers"],
            json={"rating": 7},
        )
        assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# Listing and isolation
# --------------------------------------------------------------------------- #
class TestListing:
    async def test_filters_by_application_and_status(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        await propose(client, auth_headers, scene)

        resp = await client.get(
            "/api/v1/interviews",
            headers=auth_headers,
            params={"application_id": scene["application"]["id"]},
        )
        assert resp.json()["total"] == 2

        pending = await client.get(
            "/api/v1/interviews", headers=auth_headers, params={"status": "pending"}
        )
        assert pending.json()["total"] == 1

    async def test_mine_scopes_to_the_callers_own_interviews(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        resp = await client.get(
            "/api/v1/interviews",
            headers=scene["interviewer"]["headers"],
            params={"mine": True},
        )
        assert resp.json()["total"] == 1

        # An interviewer with nothing assigned sees nothing.
        other = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        empty = await client.get(
            "/api/v1/interviews", headers=other["headers"], params={"mine": True}
        )
        assert empty.json()["total"] == 0

    async def test_filters_by_candidate_and_job(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        by_candidate = await client.get(
            "/api/v1/interviews",
            headers=auth_headers,
            params={"candidate_id": scene["candidate"]["id"]},
        )
        by_job = await client.get(
            "/api/v1/interviews",
            headers=auth_headers,
            params={"job_id": scene["job"]["id"]},
        )
        assert by_candidate.json()["total"] == 1
        assert by_job.json()["total"] == 1

    async def test_unscheduled_interviews_sort_first(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        # Pending work belongs at the top, not sorted as if it were in 1970.
        await schedule(client, auth_headers, scene)
        await propose(client, auth_headers, scene)
        resp = await client.get("/api/v1/interviews", headers=auth_headers)
        assert resp.json()["items"][0]["scheduled_at"] is None

    async def test_a_deleted_interview_disappears(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        await client.delete(
            f"/api/v1/interviews/{interview['id']}", headers=auth_headers
        )
        listed = await client.get("/api/v1/interviews", headers=auth_headers)
        assert listed.json()["total"] == 0
        gone = await client.get(
            f"/api/v1/interviews/{interview['id']}", headers=auth_headers
        )
        assert gone.status_code == 404


class TestTenantIsolation:
    async def test_another_tenant_cannot_read_the_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict, second_org: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.get(
            f"/api/v1/interviews/{interview['id']}", headers=second_org["headers"]
        )
        assert resp.status_code == 404

    async def test_another_tenant_cannot_cancel_the_interview(
        self, client: AsyncClient, auth_headers: dict, scene: dict, second_org: dict
    ) -> None:
        interview = (await schedule(client, auth_headers, scene)).json()["interview"]
        resp = await client.post(
            f"/api/v1/interviews/{interview['id']}/cancel",
            headers=second_org["headers"],
            json={},
        )
        assert resp.status_code == 404

    async def test_another_tenants_interviews_are_absent_from_the_list(
        self, client: AsyncClient, auth_headers: dict, scene: dict, second_org: dict
    ) -> None:
        await schedule(client, auth_headers, scene)
        resp = await client.get("/api/v1/interviews", headers=second_org["headers"])
        assert resp.json()["total"] == 0

    async def test_availability_cannot_be_probed_across_tenants(
        self, client: AsyncClient, scene: dict, second_org: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/interviews/availability",
            headers=second_org["headers"],
            json={"interviewer_ids": [scene["interviewer"]["user"]["id"]]},
        )
        # The foreign user simply has no calendar in this tenant.
        assert resp.status_code == 200
        assert resp.json()["interviewers"][0]["calendar_synced"] is False

    async def test_authentication_is_required(
        self, client: AsyncClient, scene: dict
    ) -> None:
        # ``scene`` is requested but unused on purpose: it seeds interviews, so
        # the 401 proves auth rejected the call rather than the list being empty.
        assert (await client.get("/api/v1/interviews")).status_code == 401
