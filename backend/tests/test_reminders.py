"""The interview reminder sweep (design §4.3).

The sweep's contract is "starts within the lead time and has not been reminded
yet", which is what makes it safe to miss a run. These tests pin that phrasing:
a worker that was down still catches up, and a worker that runs twice does not
send twice.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import ValidationError
from app.models.application import Application
from app.models.candidate import Candidate
from app.models.enums import InterviewStatus
from app.models.interview import Interview
from app.models.job import Job
from app.models.organization import Organization
from app.services import interview_service
from app.workers import reminders

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


class RecordingNotifier:
    """Captures payloads; can be told to decline or to blow up."""

    def __init__(self, *, result: bool = True, raises: Exception | None = None):
        self.sent: list[reminders.ReminderPayload] = []
        self._result = result
        self._raises = raises

    async def send(self, payload: reminders.ReminderPayload) -> bool:
        self.sent.append(payload)
        if self._raises is not None:
            raise self._raises
        return self._result


async def make_org(session, name: str = "Acme") -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


async def make_interview(
    session,
    org: Organization,
    *,
    starts_in: timedelta,
    status: InterviewStatus = InterviewStatus.CONFIRMED,
    candidate_name: str = "Casey Candidate",
    job_title: str = "Staff Engineer",
    **fields,
) -> Interview:
    suffix = uuid.uuid4().hex[:8]
    job = Job(organization_id=org.id, title=job_title, slug=f"job-{suffix}")
    candidate = Candidate(
        organization_id=org.id,
        full_name=candidate_name,
        email=f"casey-{suffix}@example.com",
        email_index=suffix,
    )
    session.add_all([job, candidate])
    await session.flush()

    application = Application(
        organization_id=org.id,
        job_id=job.id,
        candidate_id=candidate.id,
        applied_at=NOW - timedelta(days=7),
    )
    session.add(application)
    await session.flush()

    interview = Interview(
        organization_id=org.id,
        application_id=application.id,
        scheduled_at=NOW + starts_in,
        duration_minutes=60,
        status=status,
        **fields,
    )
    session.add(interview)
    await session.commit()
    return interview


class TestDueReminders:
    """Which interviews the query picks up."""

    @pytest.mark.parametrize(
        "kind,starts_in,expected",
        [
            # Inside the lead window — due.
            ("24h", timedelta(hours=23), True),
            ("24h", timedelta(hours=1), True),
            ("1h", timedelta(minutes=30), True),
            # Exactly at the boundary is still due (<=).
            ("24h", timedelta(hours=24), True),
            ("1h", timedelta(hours=1), True),
            # Beyond the lead — not yet.
            ("24h", timedelta(hours=25), False),
            ("1h", timedelta(hours=2), False),
            # Already started or past — never.
            ("24h", timedelta(0), False),
            ("24h", timedelta(hours=-1), False),
            ("1h", timedelta(minutes=-5), False),
        ],
    )
    async def test_lead_time_window(
        self, session, kind: str, starts_in: timedelta, expected: bool
    ) -> None:
        org = await make_org(session)
        await make_interview(session, org, starts_in=starts_in)

        due = await interview_service.due_reminders(session, kind, now=NOW)
        assert bool(due) is expected

    @pytest.mark.parametrize(
        "status,expected",
        [
            (InterviewStatus.CONFIRMED, True),
            (InterviewStatus.SCHEDULED, True),
            (InterviewStatus.PENDING, False),
            (InterviewStatus.CANCELLED, False),
            (InterviewStatus.COMPLETED, False),
        ],
    )
    async def test_only_live_interviews_are_reminded(
        self, session, status: InterviewStatus, expected: bool
    ) -> None:
        org = await make_org(session)
        await make_interview(
            session, org, starts_in=timedelta(hours=2), status=status
        )

        due = await interview_service.due_reminders(session, "24h", now=NOW)
        assert bool(due) is expected

    async def test_an_already_sent_reminder_is_not_due_again(self, session) -> None:
        org = await make_org(session)
        # Inside 1h, so both windows apply — but only the 24h one has been sent.
        await make_interview(
            session,
            org,
            starts_in=timedelta(minutes=30),
            reminder_24h_sent_at=NOW - timedelta(hours=23),
        )

        assert await interview_service.due_reminders(session, "24h", now=NOW) == []
        # The 1h reminder is tracked in its own column and is still pending.
        assert len(await interview_service.due_reminders(session, "1h", now=NOW)) == 1

    async def test_a_soft_deleted_interview_is_skipped(self, session) -> None:
        org = await make_org(session)
        interview = await make_interview(session, org, starts_in=timedelta(hours=2))
        interview.soft_delete()
        await session.commit()

        assert await interview_service.due_reminders(session, "24h", now=NOW) == []

    async def test_results_are_scoped_to_one_organization(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        await make_interview(session, mine, starts_in=timedelta(hours=2))
        await make_interview(session, theirs, starts_in=timedelta(hours=2))

        assert len(await interview_service.due_reminders(session, "24h", now=NOW)) == 2
        scoped = await interview_service.due_reminders(
            session, "24h", now=NOW, organization_id=mine.id
        )
        assert len(scoped) == 1
        assert scoped[0].organization_id == mine.id

    async def test_results_are_ordered_by_start_and_capped(self, session) -> None:
        org = await make_org(session)
        for hours in (5, 2, 8):
            await make_interview(session, org, starts_in=timedelta(hours=hours))

        due = await interview_service.due_reminders(session, "24h", now=NOW)
        starts = [i.scheduled_at for i in due]
        assert all(s is not None for s in starts)
        assert starts == sorted(starts)  # type: ignore[type-var]

        capped = await interview_service.due_reminders(
            session, "24h", now=NOW, limit=2
        )
        assert len(capped) == 2
        # The cap keeps the soonest, which are the ones that matter most.
        assert capped[0].scheduled_at == NOW + timedelta(hours=2)

    async def test_an_unknown_kind_is_rejected(self, session) -> None:
        with pytest.raises(ValidationError) as exc:
            await interview_service.due_reminders(session, "7d", now=NOW)
        assert "7d" in str(exc.value)
        assert exc.value.details["allowed"] == ["1h", "24h"]


class TestSendDueReminders:
    async def test_a_due_reminder_is_sent_and_marked(self, session) -> None:
        org = await make_org(session)
        interview = await make_interview(session, org, starts_in=timedelta(hours=2))
        notifier = RecordingNotifier()

        result = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )

        assert result == {"kind": "24h", "due": 1, "sent": 1, "failed": []}
        assert len(notifier.sent) == 1
        await session.refresh(interview)
        assert interview.reminder_24h_sent_at == NOW

    async def test_a_second_pass_sends_nothing(self, session) -> None:
        """Idempotency is the whole point of the sent-at columns."""
        org = await make_org(session)
        await make_interview(session, org, starts_in=timedelta(hours=2))
        notifier = RecordingNotifier()

        first = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )
        second = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )

        assert first["sent"] == 1
        assert second == {"kind": "24h", "due": 0, "sent": 0, "failed": []}
        assert len(notifier.sent) == 1

    async def test_a_declined_send_is_not_marked(self, session) -> None:
        """A notifier returning False must leave the reminder for the next pass."""
        org = await make_org(session)
        interview = await make_interview(session, org, starts_in=timedelta(hours=2))
        notifier = RecordingNotifier(result=False)

        result = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )

        assert result["sent"] == 0
        assert result["failed"] == [
            {"interview_id": str(interview.id), "error": "notifier declined"}
        ]
        await session.refresh(interview)
        assert interview.reminder_24h_sent_at is None

        retry = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=RecordingNotifier()
        )
        assert retry["sent"] == 1

    async def test_one_failure_does_not_stop_the_sweep(self, session) -> None:
        org = await make_org(session)
        await make_interview(session, org, starts_in=timedelta(hours=2))
        await make_interview(session, org, starts_in=timedelta(hours=3))

        class FailsOnce(RecordingNotifier):
            async def send(self, payload) -> bool:
                self.sent.append(payload)
                if len(self.sent) == 1:
                    raise RuntimeError("smtp exploded")
                return True

        notifier = FailsOnce()
        result = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )

        assert len(notifier.sent) == 2, "the sweep must continue past a failure"
        assert result["due"] == 2
        assert result["sent"] == 1
        assert len(result["failed"]) == 1
        assert "smtp exploded" in result["failed"][0]["error"]

    async def test_the_payload_carries_the_candidate_and_job(self, session) -> None:
        org = await make_org(session, "Acme Talent")
        interview = await make_interview(
            session,
            org,
            starts_in=timedelta(hours=2),
            candidate_name="Dana Dev",
            job_title="Principal Engineer",
            meeting_url="https://meet.example.com/abc",
            timezone="Asia/Kolkata",
        )
        notifier = RecordingNotifier()

        await reminders.send_due_reminders(session, "24h", now=NOW, notifier=notifier)

        payload = notifier.sent[0]
        assert payload.interview_id == interview.id
        assert payload.organization_id == org.id
        assert payload.kind == "24h"
        assert payload.candidate_name == "Dana Dev"
        assert payload.candidate_email is not None
        assert payload.job_title == "Principal Engineer"
        assert payload.meeting_url == "https://meet.example.com/abc"
        assert payload.timezone == "Asia/Kolkata"
        assert payload.duration_minutes == 60
        assert payload.scheduled_at == NOW + timedelta(hours=2)
        assert payload.scheduled_at.tzinfo is not None

    async def test_the_payload_survives_a_missing_application(self, session) -> None:
        """A soft-deleted application must not take the reminder down with it."""
        org = await make_org(session)
        interview = await make_interview(session, org, starts_in=timedelta(hours=2))
        application = await session.get(Application, interview.application_id)
        assert application is not None
        application.soft_delete()
        await session.commit()

        notifier = RecordingNotifier()
        result = await reminders.send_due_reminders(
            session, "24h", now=NOW, notifier=notifier
        )

        assert result["sent"] == 1
        payload = notifier.sent[0]
        assert payload.candidate_name is None
        assert payload.job_title is None

    async def test_the_booking_url_is_included_when_a_token_exists(
        self, session
    ) -> None:
        org = await make_org(session)
        await make_interview(
            session,
            org,
            starts_in=timedelta(hours=2),
            booking_token="tok_" + uuid.uuid4().hex,
        )
        notifier = RecordingNotifier()

        await reminders.send_due_reminders(session, "24h", now=NOW, notifier=notifier)
        assert notifier.sent[0].booking_url is not None

        org2 = await make_org(session, "No Token")
        await make_interview(session, org2, starts_in=timedelta(hours=2))
        notifier2 = RecordingNotifier()
        await reminders.send_due_reminders(
            session, "24h", now=NOW, organization_id=org2.id, notifier=notifier2
        )
        assert notifier2.sent[0].booking_url is None

    async def test_sending_is_scoped_to_one_organization(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        await make_interview(session, mine, starts_in=timedelta(hours=2))
        other = await make_interview(session, theirs, starts_in=timedelta(hours=2))
        notifier = RecordingNotifier()

        result = await reminders.send_due_reminders(
            session, "24h", now=NOW, organization_id=mine.id, notifier=notifier
        )

        assert result["sent"] == 1
        await session.refresh(other)
        assert other.reminder_24h_sent_at is None


class TestSweep:
    async def test_sweep_runs_every_kind(self, session) -> None:
        org = await make_org(session)
        # Inside 1h, so both kinds are due for this one.
        interview = await make_interview(session, org, starts_in=timedelta(minutes=30))
        notifier = RecordingNotifier()
        reminders.set_notifier(notifier)

        results = await reminders.sweep(session, now=NOW)

        assert [r["kind"] for r in results] == ["24h", "1h"]
        assert [r["sent"] for r in results] == [1, 1]
        await session.refresh(interview)
        assert interview.reminder_24h_sent_at == NOW
        assert interview.reminder_1h_sent_at == NOW

    async def test_sweep_uses_the_configured_notifier(self, session) -> None:
        org = await make_org(session)
        await make_interview(session, org, starts_in=timedelta(hours=2))
        notifier = RecordingNotifier()
        reminders.set_notifier(notifier)

        await reminders.sweep(session, now=NOW)

        assert len(notifier.sent) == 1

    async def test_sweep_with_nothing_due_is_quiet(self, session) -> None:
        org = await make_org(session)
        await make_interview(session, org, starts_in=timedelta(days=30))
        notifier = RecordingNotifier()
        reminders.set_notifier(notifier)

        results = await reminders.sweep(session, now=NOW)

        assert [r["due"] for r in results] == [0, 0]
        assert notifier.sent == []


class TestDefaultNotifier:
    async def test_the_default_logs_and_reports_success(self, session) -> None:
        reminders.set_notifier(None)
        notifier = reminders.get_notifier()
        assert isinstance(notifier, reminders.LoggingNotifier)
        # Cached, so a sweep and a later inspection see the same instance.
        assert reminders.get_notifier() is notifier

        org = await make_org(session)
        interview = await make_interview(session, org, starts_in=timedelta(hours=2))
        payload = await reminders.build_payload(session, interview, "24h")

        assert await notifier.send(payload) is True
        assert notifier.sent == [payload]
