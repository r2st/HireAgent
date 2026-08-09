"""Column type behaviour that the rest of the code assumes.

``UTCDateTime`` exists because SQLite has no time zone type: a naive datetime
coming back out of the database serialised as ``2026-09-07T09:30:00`` where the
same value before a refresh serialised as ``2026-09-07T09:30:00Z``. Scheduling
compares and emits those values, so the aware/naive split has to be closed at
the column rather than at each call site.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import select

from app.models.application import Application
from app.models.candidate import Candidate
from app.models.interview import Interview
from app.models.job import Job
from app.models.organization import Organization


async def _org(session) -> Organization:
    org = Organization(name="Tz Test", slug=f"tz-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


async def _application(session, org: Organization) -> Application:
    """The minimum chain an interview needs: job + candidate + application."""
    suffix = uuid.uuid4().hex[:8]
    job = Job(organization_id=org.id, title="Engineer", slug=f"eng-{suffix}")
    candidate = Candidate(
        organization_id=org.id,
        full_name="Casey Candidate",
        email=f"casey-{suffix}@example.com",
        email_index=suffix,
    )
    session.add_all([job, candidate])
    await session.flush()

    application = Application(
        organization_id=org.id,
        job_id=job.id,
        candidate_id=candidate.id,
        applied_at=datetime.now(UTC),
    )
    session.add(application)
    await session.commit()
    return application


class TestUTCDateTime:
    async def test_value_is_aware_after_refresh(self, session) -> None:
        """The exact path that produced the bug: write, commit, refresh."""
        org = await _org(session)
        scheduled = datetime(2026, 9, 7, 9, 30, tzinfo=UTC)
        application = await _application(session, org)
        interview = Interview(
            organization_id=org.id,
            application_id=application.id,
            scheduled_at=scheduled,
            duration_minutes=60,
        )
        session.add(interview)
        await session.commit()
        await session.refresh(interview)

        assert interview.scheduled_at is not None
        assert interview.scheduled_at.tzinfo is not None
        assert interview.scheduled_at == scheduled
        assert interview.scheduled_at.isoformat() == "2026-09-07T09:30:00+00:00"

    async def test_value_is_aware_when_loaded_in_a_fresh_query(
        self, session
    ) -> None:
        org = await _org(session)
        scheduled = datetime(2026, 9, 7, 9, 30, tzinfo=UTC)
        session.add(
            Interview(
                organization_id=org.id,
                application_id=(await _application(session, org)).id,
                scheduled_at=scheduled,
                duration_minutes=60,
            )
        )
        await session.commit()
        session.expunge_all()

        loaded = await session.scalar(select(Interview))
        assert loaded is not None
        assert loaded.scheduled_at == scheduled
        assert loaded.scheduled_at.tzinfo is not None

    async def test_a_naive_write_is_read_back_as_utc(self, session) -> None:
        """Naive input is taken to already be UTC, matching the app convention."""
        org = await _org(session)
        session.add(
            Interview(
                organization_id=org.id,
                application_id=(await _application(session, org)).id,
                scheduled_at=datetime(2026, 9, 7, 9, 30),
                duration_minutes=60,
            )
        )
        await session.commit()
        session.expunge_all()

        loaded = await session.scalar(select(Interview))
        assert loaded is not None
        assert loaded.scheduled_at == datetime(2026, 9, 7, 9, 30, tzinfo=UTC)

    async def test_a_non_utc_offset_is_normalised_not_truncated(
        self, session
    ) -> None:
        """An IST-tagged time must survive as the same instant, not the same clock."""
        org = await _org(session)
        ist = timezone(timedelta(hours=5, minutes=30))
        session.add(
            Interview(
                organization_id=org.id,
                application_id=(await _application(session, org)).id,
                scheduled_at=datetime(2026, 9, 7, 15, 0, tzinfo=ist),
                duration_minutes=60,
            )
        )
        await session.commit()
        session.expunge_all()

        loaded = await session.scalar(select(Interview))
        assert loaded is not None
        assert loaded.scheduled_at == datetime(2026, 9, 7, 9, 30, tzinfo=UTC)

    async def test_null_stays_null(self, session) -> None:
        org = await _org(session)
        session.add(
            Interview(
                organization_id=org.id,
                application_id=(await _application(session, org)).id,
                scheduled_at=None,
                duration_minutes=60,
            )
        )
        await session.commit()
        session.expunge_all()

        loaded = await session.scalar(select(Interview))
        assert loaded is not None
        assert loaded.scheduled_at is None

    async def test_mixin_timestamps_are_aware(self, session) -> None:
        """``created_at``/``updated_at`` come from a server default, not Python."""
        org = await _org(session)
        session.expunge_all()

        loaded = await session.get(Organization, org.id)
        assert loaded is not None
        assert loaded.created_at.tzinfo is not None
        assert loaded.updated_at.tzinfo is not None
        # A server-generated timestamp must be comparable with an aware "now"
        # without raising TypeError — the thing naive values break.
        assert loaded.created_at <= datetime.now(UTC) + timedelta(minutes=1)

    async def test_soft_delete_timestamp_is_aware(self, session) -> None:
        org = await _org(session)
        org.soft_delete()
        await session.commit()
        session.expunge_all()

        loaded = await session.get(Organization, org.id)
        assert loaded is not None
        assert loaded.deleted_at is not None
        assert loaded.deleted_at.tzinfo is not None
