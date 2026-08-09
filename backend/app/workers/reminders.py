"""Interview reminder sweep (design §4.3: 24 hours and 1 hour before).

Run periodically — every few minutes is plenty, since the query is "starts
within the lead time and has not been reminded yet" rather than "starts in
exactly N hours". That phrasing is what makes the sweep safe to miss: a worker
that was down for an hour catches up on its next pass instead of silently
skipping everyone in the gap.

Delivery goes through a pluggable notifier. The default one logs, because the
outreach service that will actually send the email and WhatsApp message is not
built yet — the scheduling side is complete and records what it sent, and
swapping in the real transport is a one-line ``set_notifier`` call.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.tenancy import get_scoped
from app.models.application import Application
from app.models.candidate import Candidate
from app.models.interview import Interview
from app.models.job import Job
from app.services import interview_service
from app.services.availability import to_iso, to_utc

logger = logging.getLogger(__name__)


@dataclass
class ReminderPayload:
    """Everything a transport needs to render the reminder."""

    interview_id: uuid.UUID
    organization_id: uuid.UUID
    kind: str
    scheduled_at: datetime
    timezone: str
    duration_minutes: int
    candidate_name: str | None
    candidate_email: str | None
    job_title: str | None
    meeting_url: str | None
    location: str | None
    booking_url: str | None


class ReminderNotifier(Protocol):
    async def send(self, payload: ReminderPayload) -> bool:
        """Deliver one reminder. Return ``False`` to leave it unmarked."""
        ...


class LoggingNotifier:
    """Default transport: records the intent without sending anything."""

    def __init__(self) -> None:
        self.sent: list[ReminderPayload] = []

    async def send(self, payload: ReminderPayload) -> bool:
        self.sent.append(payload)
        logger.info(
            "Interview reminder (%s) for %s at %s -> %s",
            payload.kind,
            payload.interview_id,
            to_iso(payload.scheduled_at),
            payload.candidate_email or "no candidate email",
        )
        return True


_notifier: ReminderNotifier | None = None


def get_notifier() -> ReminderNotifier:
    global _notifier
    if _notifier is None:
        _notifier = LoggingNotifier()
    return _notifier


def set_notifier(notifier: ReminderNotifier | None) -> None:
    """Swap the process-wide reminder transport (used by tests and wiring)."""
    global _notifier
    _notifier = notifier


async def build_payload(
    session: AsyncSession, interview: Interview, kind: str
) -> ReminderPayload:
    """Gather the candidate and job context for one reminder."""
    organization_id = interview.organization_id
    application = await get_scoped(
        session, Application, interview.application_id, organization_id
    )
    candidate = None
    job = None
    if application is not None:
        candidate = await get_scoped(
            session, Candidate, application.candidate_id, organization_id
        )
        job = await get_scoped(session, Job, application.job_id, organization_id)

    return ReminderPayload(
        interview_id=interview.id,
        organization_id=organization_id,
        kind=kind,
        scheduled_at=to_utc(interview.scheduled_at) if interview.scheduled_at else datetime.now(UTC),
        timezone=interview.timezone,
        duration_minutes=interview.duration_minutes,
        candidate_name=candidate.full_name if candidate else None,
        candidate_email=candidate.email if candidate else None,
        job_title=job.title if job else None,
        meeting_url=interview.meeting_url,
        location=interview.location,
        booking_url=interview_service.booking_url(interview),
    )


async def send_due_reminders(
    session: AsyncSession,
    kind: str,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 500,
    notifier: ReminderNotifier | None = None,
) -> dict:
    """Send every due ``kind`` reminder, marking each one as it succeeds.

    Marking happens per interview rather than in a batch at the end: if the
    transport dies halfway through, the ones already sent stay sent and the
    rest are picked up next pass. A duplicate reminder is a worse outcome than
    a slightly late one.
    """
    transport = notifier or get_notifier()
    due = await interview_service.due_reminders(
        session, kind, now=now, organization_id=organization_id, limit=limit
    )

    sent = 0
    failed: list[dict] = []
    for interview in due:
        payload = await build_payload(session, interview, kind)
        try:
            delivered = await transport.send(payload)
        except Exception as exc:  # noqa: BLE001 - one bad send must not stop the sweep
            logger.exception("Reminder delivery failed for %s", interview.id)
            failed.append({"interview_id": str(interview.id), "error": str(exc)})
            continue

        if delivered:
            await interview_service.mark_reminder_sent(session, interview, kind, at=now)
            sent += 1
        else:
            failed.append(
                {"interview_id": str(interview.id), "error": "notifier declined"}
            )

    if due:
        logger.info(
            "Reminder sweep (%s): %d due, %d sent, %d failed",
            kind,
            len(due),
            sent,
            len(failed),
        )
    return {"kind": kind, "due": len(due), "sent": sent, "failed": failed}


async def sweep(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
) -> list[dict]:
    """Run every reminder kind once."""
    return [
        await send_due_reminders(
            session, kind, now=now, organization_id=organization_id
        )
        for kind in interview_service.REMINDER_KINDS
    ]
