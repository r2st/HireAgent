"""Interview scheduling (design §4.3).

The flow the design asks for: read interviewer availability, offer the
candidate a set of slots, book the one they pick, remind them at 24h and 1h,
and let them reschedule or cancel themselves.

Two decisions shape this module.

**The booking token is the candidate's credential.** Candidates have no
account, so a random per-interview token authorises the self-service booking,
reschedule, and cancel routes. Everything reached by token is looked up by
token alone and derives its tenant from the row it finds — never from
caller-supplied input.

**Rescheduling appends rather than mutates.** A moved interview becomes a new
row pointing back at the old one through ``rescheduled_from_id``, so "this was
moved twice before it happened" survives in the record. The booking token
migrates to the new row, which is why the old one must be cleared first: the
column is unique.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.integrations import calendar as calendar_api
from app.models.application import Application
from app.models.candidate import Candidate
from app.models.enums import (
    STAGE_INDEX,
    ApplicationStatus,
    ConsentType,
    InterviewStatus,
    InterviewType,
    PipelineStage,
)
from app.models.interview import CalendarAccount, Interview, InterviewParticipant
from app.schemas.common import PaginationParams
from app.services import application_service, auth_service, calendar_service
from app.services.availability import (
    DEFAULT_GRANULARITY_MINUTES,
    Interval,
    ParticipantAvailability,
    find_slots,
    to_iso,
    to_utc,
)

logger = logging.getLogger(__name__)

# Interview types that are a live conversation and therefore need a time, a
# calendar hold, and at least one interviewer.
LIVE_TYPES = frozenset(
    {
        InterviewType.PHONE,
        InterviewType.VIDEO,
        InterviewType.ONSITE,
        InterviewType.TECHNICAL,
        InterviewType.PANEL,
    }
)

# Statuses from which an interview can still be moved or acted on.
OPEN_STATUSES = frozenset(
    {InterviewStatus.PENDING, InterviewStatus.SCHEDULED, InterviewStatus.CONFIRMED}
)

# Statuses that are the end of the line for a row.
TERMINAL_STATUSES = frozenset(
    {
        InterviewStatus.COMPLETED,
        InterviewStatus.CANCELLED,
        InterviewStatus.NO_SHOW,
        InterviewStatus.RESCHEDULED,
    }
)

VALID_RESPONSES = frozenset({"pending", "accepted", "declined", "tentative"})

VALID_RECOMMENDATIONS = frozenset(
    {"strong_yes", "yes", "neutral", "no", "strong_no"}
)

# Applications in these states should not be gaining new interviews.
CLOSED_APPLICATION_STATUSES = frozenset(
    {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}
)

MAX_PROPOSED_SLOTS = 20


@dataclass
class SchedulingResult:
    """An interview plus what the caller needs to know about how it was made.

    ``warnings`` carries the soft failures — an interviewer whose calendar
    could not be read, a calendar event that did not get written. None of these
    should block scheduling, but a recruiter must be able to see them.
    """

    interview: Interview
    slots: list[Interval] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
async def get_interview(
    session: AsyncSession, organization_id: uuid.UUID, interview_id: uuid.UUID
) -> Interview:
    interview = await get_scoped(session, Interview, interview_id, organization_id)
    if interview is None:
        raise NotFoundError("Interview not found")
    return interview


async def list_interviews(
    session: AsyncSession,
    organization_id: uuid.UUID,
    params: PaginationParams,
    *,
    application_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
    interviewer_id: uuid.UUID | None = None,
    status: InterviewStatus | None = None,
    type: InterviewType | None = None,
    scheduled_from: datetime | None = None,
    scheduled_to: datetime | None = None,
    upcoming_only: bool = False,
) -> tuple[list[Interview], int]:
    stmt = scoped_select(Interview, organization_id)
    count_stmt = (
        select(func.count(func.distinct(Interview.id)))
        .select_from(Interview)
        .where(
            Interview.organization_id == organization_id,
            Interview.deleted_at.is_(None),
        )
    )

    # Candidate and job are attributes of the application, so both filters
    # reach them through a join rather than a denormalised column.
    if candidate_id is not None or job_id is not None:
        stmt = stmt.join(Application, Application.id == Interview.application_id)
        count_stmt = count_stmt.join(
            Application, Application.id == Interview.application_id
        )

    if interviewer_id is not None:
        stmt = stmt.join(
            InterviewParticipant,
            InterviewParticipant.interview_id == Interview.id,
        )
        count_stmt = count_stmt.join(
            InterviewParticipant,
            InterviewParticipant.interview_id == Interview.id,
        )

    filters = []
    if application_id is not None:
        filters.append(Interview.application_id == application_id)
    if candidate_id is not None:
        filters.append(Application.candidate_id == candidate_id)
    if job_id is not None:
        filters.append(Application.job_id == job_id)
    if interviewer_id is not None:
        filters.append(InterviewParticipant.user_id == interviewer_id)
        filters.append(InterviewParticipant.deleted_at.is_(None))
    if status is not None:
        filters.append(Interview.status == status)
    if type is not None:
        filters.append(Interview.type == type)
    if scheduled_from is not None:
        filters.append(Interview.scheduled_at >= scheduled_from)
    if scheduled_to is not None:
        filters.append(Interview.scheduled_at <= scheduled_to)
    if upcoming_only:
        filters.append(Interview.scheduled_at >= datetime.now(UTC))
        filters.append(Interview.status.in_(list(OPEN_STATUSES)))

    for f in filters:
        stmt = stmt.where(f)
        count_stmt = count_stmt.where(f)

    total = int(await session.scalar(count_stmt) or 0)
    rows = list(
        (
            await session.execute(
                stmt.order_by(
                    # Unscheduled interviews are pending work; they belong at
                    # the top rather than sorted as if they were in 1970.
                    Interview.scheduled_at.is_(None).desc(),
                    Interview.scheduled_at.asc(),
                )
                .distinct()
                .offset(params.offset)
                .limit(params.page_size)
            )
        )
        .scalars()
        .all()
    )
    return rows, total


async def list_participants(
    session: AsyncSession, organization_id: uuid.UUID, interview_id: uuid.UUID
) -> list[InterviewParticipant]:
    result = await session.execute(
        scoped_select(InterviewParticipant, organization_id)
        .where(InterviewParticipant.interview_id == interview_id)
        .order_by(InterviewParticipant.is_organizer.desc(), InterviewParticipant.created_at)
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
async def find_available_slots(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interviewer_ids: list[uuid.UUID],
    *,
    duration_minutes: int = 45,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    granularity_minutes: int = DEFAULT_GRANULARITY_MINUTES,
    buffer_minutes: int = 0,
    min_notice_hours: int | None = None,
    limit: int = 10,
    per_day: int = 3,
    default_timezone: str = "UTC",
    exclude_interview_id: uuid.UUID | None = None,
) -> tuple[list[Interval], list[ParticipantAvailability]]:
    """Slots every listed interviewer can make."""
    window = _search_window(window_start, window_end)
    availability = await calendar_service.participant_availability(
        session,
        organization_id,
        interviewer_ids,
        window,
        default_timezone=default_timezone,
        exclude_interview_id=exclude_interview_id,
    )
    slots = find_slots(
        availability,
        window,
        duration_minutes=duration_minutes,
        granularity_minutes=granularity_minutes,
        buffer_minutes=buffer_minutes,
        min_notice_hours=(
            settings.interview_min_notice_hours
            if min_notice_hours is None
            else min_notice_hours
        ),
        limit=limit,
        per_day=per_day,
    )
    return slots, availability


def _search_window(start: datetime | None, end: datetime | None) -> Interval:
    now = datetime.now(UTC)
    window_start = to_utc(start) if start else now
    window_end = (
        to_utc(end)
        if end
        else window_start + timedelta(days=settings.interview_horizon_days)
    )
    if window_end <= window_start:
        raise ValidationError("The availability window must end after it starts")
    return Interval(window_start, window_end)


# --------------------------------------------------------------------------- #
# Creating interviews
# --------------------------------------------------------------------------- #
async def propose_slots(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    interviewer_ids: list[uuid.UUID],
    type: InterviewType = InterviewType.VIDEO,
    duration_minutes: int = 45,
    round_number: int | None = None,
    title: str | None = None,
    location: str | None = None,
    timezone: str = "UTC",
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    granularity_minutes: int = DEFAULT_GRANULARITY_MINUTES,
    buffer_minutes: int = 0,
    min_notice_hours: int | None = None,
    slot_count: int = 5,
    organizer_id: uuid.UUID | None = None,
    require_consent: bool = True,
) -> SchedulingResult:
    """Create a pending interview and offer the candidate a set of times.

    The candidate is not booked yet — they receive a link and pick. If no slot
    can be found the interview is still created in ``pending`` so the recruiter
    has something to widen the window on, rather than losing the setup work.
    """
    application = await _open_application(session, organization_id, application_id)
    interviewers = await _resolve_interviewers(
        session, organization_id, interviewer_ids, type
    )
    if require_consent:
        await _require_communication_consent(
            session, organization_id, application.candidate_id
        )

    slot_count = max(1, min(slot_count, MAX_PROPOSED_SLOTS))
    slots, availability = await find_available_slots(
        session,
        organization_id,
        [u.id for u in interviewers],
        duration_minutes=duration_minutes,
        window_start=window_start,
        window_end=window_end,
        granularity_minutes=granularity_minutes,
        buffer_minutes=buffer_minutes,
        min_notice_hours=min_notice_hours,
        limit=slot_count,
        default_timezone=timezone,
    )

    interview = Interview(
        organization_id=organization_id,
        application_id=application_id,
        type=type,
        status=InterviewStatus.PENDING,
        round_number=(
            round_number
            if round_number is not None
            else await _next_round_number(session, organization_id, application_id)
        ),
        title=title,
        duration_minutes=duration_minutes,
        timezone=timezone,
        location=location,
        proposed_slots=[s.to_dict() for s in slots],
        booking_token=_new_booking_token(),
        booking_expires_at=datetime.now(UTC)
        + timedelta(hours=settings.booking_token_ttl_hours),
    )
    session.add(interview)
    await session.flush()

    _attach_participants(
        session, organization_id, interview, interviewers, organizer_id
    )
    await session.commit()
    await session.refresh(interview)

    warnings = _availability_warnings(availability, interviewers)
    if not slots:
        warnings.append(
            "No common availability was found in the requested window; "
            "widen the window or reduce the number of interviewers."
        )
    return SchedulingResult(interview=interview, slots=slots, warnings=warnings)


async def schedule_interview(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    scheduled_at: datetime,
    interviewer_ids: list[uuid.UUID],
    type: InterviewType = InterviewType.VIDEO,
    duration_minutes: int = 45,
    round_number: int | None = None,
    title: str | None = None,
    location: str | None = None,
    meeting_url: str | None = None,
    timezone: str = "UTC",
    notes: str | None = None,
    organizer_id: uuid.UUID | None = None,
    allow_conflicts: bool = False,
    create_calendar_event: bool = True,
) -> SchedulingResult:
    """Book an interview at a known time (design §6.1 ``POST /interviews/schedule``).

    Conflicts are refused rather than warned about: a recruiter who typed a
    time that collides with an existing interview wants to know now, not after
    the invitation goes out. ``allow_conflicts`` is the deliberate override.
    """
    await _open_application(session, organization_id, application_id)
    interviewers = await _resolve_interviewers(
        session, organization_id, interviewer_ids, type
    )

    start = to_utc(scheduled_at)
    if start <= datetime.now(UTC):
        raise ValidationError("Interviews cannot be scheduled in the past")
    slot = Interval(start, start + timedelta(minutes=duration_minutes))

    if not allow_conflicts:
        await _assert_no_conflict(
            session, organization_id, [u.id for u in interviewers], slot
        )

    interview = Interview(
        organization_id=organization_id,
        application_id=application_id,
        type=type,
        status=InterviewStatus.SCHEDULED,
        round_number=(
            round_number
            if round_number is not None
            else await _next_round_number(session, organization_id, application_id)
        ),
        title=title,
        scheduled_at=start,
        duration_minutes=duration_minutes,
        timezone=timezone,
        location=location,
        meeting_url=meeting_url,
        notes=notes,
        proposed_slots=[],
        booking_token=_new_booking_token(),
        booking_expires_at=start + timedelta(days=1),
    )
    session.add(interview)
    await session.flush()

    _attach_participants(
        session, organization_id, interview, interviewers, organizer_id
    )
    await session.flush()

    warnings: list[str] = []
    if create_calendar_event:
        warnings += await _write_calendar_event(session, organization_id, interview)

    await session.commit()
    await session.refresh(interview)
    return SchedulingResult(interview=interview, slots=[slot], warnings=warnings)


# --------------------------------------------------------------------------- #
# Candidate self-service (token-authenticated)
# --------------------------------------------------------------------------- #
async def get_by_booking_token(session: AsyncSession, token: str) -> Interview:
    """Resolve a booking token to its interview.

    The token is the only credential here, so a bad one and an expired one both
    surface as a plain "not found or expired" — an attacker should not be able
    to tell a real token from a guess.
    """
    if not token:
        raise NotFoundError("This booking link is not valid")

    interview = await session.scalar(
        select(Interview).where(
            Interview.booking_token == token,
            Interview.deleted_at.is_(None),
        )
    )
    if interview is None:
        raise NotFoundError("This booking link is not valid")
    if interview.booking_expires_at is not None:
        expires = to_utc(interview.booking_expires_at)
        if expires < datetime.now(UTC):
            raise NotFoundError("This booking link has expired")
    return interview


async def book_slot(
    session: AsyncSession,
    token: str,
    slot_start: datetime,
    *,
    booked_by: str = "candidate",
) -> SchedulingResult:
    """Book one of the proposed slots. Used by the candidate-facing page."""
    interview = await get_by_booking_token(session, token)
    if interview.status not in OPEN_STATUSES:
        raise ConflictError(
            f"This interview is {interview.status} and can no longer be booked"
        )
    if interview.scheduled_at is not None:
        raise ConflictError(
            "This interview is already booked; use the reschedule link to move it"
        )

    start = _match_proposed_slot(interview, slot_start)
    slot = Interval(start, start + timedelta(minutes=interview.duration_minutes))

    interviewer_ids = [
        p.user_id
        for p in await list_participants(
            session, interview.organization_id, interview.id
        )
    ]
    # Re-check at booking time: the slots were computed when they were offered,
    # and an interviewer may have been booked into that time since.
    await _assert_no_conflict(
        session,
        interview.organization_id,
        interviewer_ids,
        slot,
        exclude_interview_id=interview.id,
    )

    interview.scheduled_at = start
    interview.status = (
        InterviewStatus.CONFIRMED
        if booked_by == "candidate"
        else InterviewStatus.SCHEDULED
    )
    # Keep the link alive past the interview so self-service reschedule and
    # cancel still work right up to the day.
    interview.booking_expires_at = start + timedelta(days=1)

    warnings = await _write_calendar_event(
        session, interview.organization_id, interview
    )
    await session.commit()
    await session.refresh(interview)
    return SchedulingResult(interview=interview, slots=[slot], warnings=warnings)


async def reschedule(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    *,
    scheduled_at: datetime | None = None,
    proposed_slots: list[Interval] | None = None,
    reason: str | None = None,
    allow_conflicts: bool = False,
    rescheduled_by: str = "recruiter",
) -> SchedulingResult:
    """Move an interview by superseding it with a new row.

    The original is marked ``rescheduled`` and keeps its history; the new row
    inherits the participants and the booking token.
    """
    original = await get_interview(session, organization_id, interview_id)
    if original.status in TERMINAL_STATUSES:
        raise ConflictError(
            f"A {original.status} interview cannot be rescheduled",
            details={"status": original.status},
        )
    if scheduled_at is None and not proposed_slots:
        raise ValidationError(
            "Rescheduling needs either a new time or a new set of proposed slots"
        )

    participants = await list_participants(session, organization_id, interview_id)
    interviewer_ids = [p.user_id for p in participants]

    start: datetime | None = None
    slot: Interval | None = None
    if scheduled_at is not None:
        start = to_utc(scheduled_at)
        if start <= datetime.now(UTC):
            raise ValidationError("Interviews cannot be rescheduled into the past")
        slot = Interval(start, start + timedelta(minutes=original.duration_minutes))
        if not allow_conflicts:
            await _assert_no_conflict(
                session,
                organization_id,
                interviewer_ids,
                slot,
                exclude_interview_id=original.id,
            )

    warnings: list[str] = []
    # Release the old calendar hold before writing the new one, so the
    # interviewer's diary does not end up with both.
    warnings += await _remove_calendar_event(session, organization_id, original)

    token = original.booking_token or _new_booking_token()
    # booking_token is unique: it has to leave the old row before it can land
    # on the new one.
    original.booking_token = None
    original.status = InterviewStatus.RESCHEDULED
    original.cancellation_reason = reason
    await session.flush()

    replacement = Interview(
        organization_id=organization_id,
        application_id=original.application_id,
        type=original.type,
        status=(
            InterviewStatus.PENDING
            if start is None
            else (
                InterviewStatus.CONFIRMED
                if rescheduled_by == "candidate"
                else InterviewStatus.SCHEDULED
            )
        ),
        round_number=original.round_number,
        title=original.title,
        scheduled_at=start,
        duration_minutes=original.duration_minutes,
        timezone=original.timezone,
        location=original.location,
        meeting_url=original.meeting_url if start is not None else None,
        proposed_slots=[s.to_dict() for s in (proposed_slots or [])],
        booking_token=token,
        booking_expires_at=(
            start + timedelta(days=1)
            if start is not None
            else datetime.now(UTC) + timedelta(hours=settings.booking_token_ttl_hours)
        ),
        calendar_account_id=original.calendar_account_id,
        notes=original.notes,
        rescheduled_from_id=original.id,
    )
    session.add(replacement)
    await session.flush()

    for participant in participants:
        session.add(
            InterviewParticipant(
                organization_id=organization_id,
                interview_id=replacement.id,
                user_id=participant.user_id,
                is_organizer=participant.is_organizer,
                # A new time needs a fresh accept: the old yes was for the old slot.
                response_status="pending",
            )
        )
    await session.flush()

    if start is not None:
        warnings += await _write_calendar_event(
            session, organization_id, replacement
        )

    await session.commit()
    await session.refresh(replacement)
    return SchedulingResult(
        interview=replacement,
        slots=[slot] if slot else (proposed_slots or []),
        warnings=warnings,
    )


async def reschedule_by_token(
    session: AsyncSession, token: str, slot_start: datetime
) -> SchedulingResult:
    """Candidate-driven reschedule onto another proposed slot."""
    interview = await get_by_booking_token(session, token)
    if interview.status in TERMINAL_STATUSES:
        raise ConflictError(
            f"This interview is {interview.status} and can no longer be moved"
        )
    start = _match_proposed_slot(interview, slot_start)
    # Carry the remaining offers forward so the candidate can move again.
    remaining = [
        _slot_from_dict(entry)
        for entry in interview.proposed_slots or []
        if _slot_from_dict(entry) is not None
    ]
    return await reschedule(
        session,
        interview.organization_id,
        interview.id,
        scheduled_at=start,
        proposed_slots=[s for s in remaining if s is not None],
        reason="Rescheduled by the candidate",
        rescheduled_by="candidate",
    )


async def cancel(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    *,
    reason: str | None = None,
    cancelled_by: str = "recruiter",
) -> SchedulingResult:
    """Cancel an interview and release its calendar hold."""
    interview = await get_interview(session, organization_id, interview_id)
    if interview.status == InterviewStatus.CANCELLED:
        return SchedulingResult(interview=interview)
    if interview.status in TERMINAL_STATUSES:
        raise ConflictError(
            f"A {interview.status} interview cannot be cancelled",
            details={"status": interview.status},
        )

    warnings = await _remove_calendar_event(session, organization_id, interview)
    interview.status = InterviewStatus.CANCELLED
    interview.cancelled_at = datetime.now(UTC)
    interview.cancellation_reason = reason or f"Cancelled by {cancelled_by}"
    # The link dies with the interview.
    interview.booking_token = None
    interview.booking_expires_at = None
    await session.commit()
    await session.refresh(interview)
    return SchedulingResult(interview=interview, warnings=warnings)


async def cancel_by_token(
    session: AsyncSession, token: str, *, reason: str | None = None
) -> SchedulingResult:
    interview = await get_by_booking_token(session, token)
    return await cancel(
        session,
        interview.organization_id,
        interview.id,
        reason=reason or "Cancelled by the candidate",
        cancelled_by="candidate",
    )


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
async def complete(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    *,
    notes: str | None = None,
    advance_application: bool = True,
    changed_by_id: uuid.UUID | None = None,
) -> Interview:
    """Mark an interview as held, and move the pipeline card along.

    The stage only ever moves forward: an application already at ``offered``
    is not dragged back to ``interviewed`` by a late-recorded first round.
    """
    interview = await get_interview(session, organization_id, interview_id)
    if interview.status == InterviewStatus.COMPLETED:
        return interview
    if interview.status in TERMINAL_STATUSES:
        raise ConflictError(
            f"A {interview.status} interview cannot be completed",
            details={"status": interview.status},
        )

    interview.status = InterviewStatus.COMPLETED
    interview.completed_at = datetime.now(UTC)
    if notes:
        interview.notes = notes
    await session.commit()
    await session.refresh(interview)

    if advance_application:
        application = await session.get(Application, interview.application_id)
        if application is not None and application.status not in CLOSED_APPLICATION_STATUSES:
            current = STAGE_INDEX[PipelineStage(application.stage)]
            if current < STAGE_INDEX[PipelineStage.INTERVIEWED]:
                await application_service.move_stage(
                    session,
                    organization_id,
                    application.id,
                    PipelineStage.INTERVIEWED,
                    changed_by_id=changed_by_id,
                    trigger="interview_completed",
                    note=f"Round {interview.round_number} interview completed",
                )
        await session.refresh(interview)
    return interview


async def mark_no_show(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    *,
    notes: str | None = None,
) -> Interview:
    interview = await get_interview(session, organization_id, interview_id)
    if interview.status in TERMINAL_STATUSES:
        raise ConflictError(
            f"A {interview.status} interview cannot be marked as a no-show",
            details={"status": interview.status},
        )
    interview.status = InterviewStatus.NO_SHOW
    interview.completed_at = datetime.now(UTC)
    if notes:
        interview.notes = notes
    await session.commit()
    await session.refresh(interview)
    return interview


async def respond(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    user_id: uuid.UUID,
    response_status: str,
) -> InterviewParticipant:
    """Record an interviewer accepting or declining their invitation."""
    if response_status not in VALID_RESPONSES:
        raise ValidationError(
            f"Invalid response '{response_status}'",
            details={"allowed": sorted(VALID_RESPONSES)},
        )
    participant = await _get_participant(
        session, organization_id, interview_id, user_id
    )
    participant.response_status = response_status
    await session.commit()
    await session.refresh(participant)
    return participant


async def submit_feedback(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    rating: float | None = None,
    recommendation: str | None = None,
    feedback: str | None = None,
    scorecard: dict | None = None,
) -> InterviewParticipant:
    """Record one interviewer's scorecard.

    Only an assigned participant can leave feedback — that is what makes a
    scorecard attributable, and it keeps an uninvolved colleague from voting.
    """
    if recommendation is not None and recommendation not in VALID_RECOMMENDATIONS:
        raise ValidationError(
            f"Invalid recommendation '{recommendation}'",
            details={"allowed": sorted(VALID_RECOMMENDATIONS)},
        )
    if rating is not None and not (0 <= rating <= 10):
        raise ValidationError("Rating must be between 0 and 10")

    interview = await get_interview(session, organization_id, interview_id)
    if interview.status in {InterviewStatus.CANCELLED, InterviewStatus.RESCHEDULED}:
        raise ConflictError(
            f"Feedback cannot be left on a {interview.status} interview"
        )

    participant = await _get_participant(
        session, organization_id, interview_id, user_id
    )
    if rating is not None:
        participant.rating = rating
    if recommendation is not None:
        participant.recommendation = recommendation
    if feedback is not None:
        participant.feedback = feedback
    if scorecard is not None:
        participant.scorecard_json = scorecard
    participant.feedback_submitted_at = datetime.now(UTC)

    await session.commit()
    await session.refresh(participant)
    return participant


async def feedback_summary(
    session: AsyncSession, organization_id: uuid.UUID, interview_id: uuid.UUID
) -> dict:
    """Aggregate the scorecards for one interview."""
    participants = await list_participants(session, organization_id, interview_id)
    rated = [p for p in participants if p.rating is not None]
    ratings = [float(p.rating) for p in rated if p.rating is not None]
    recommendations: dict[str, int] = {}
    for p in participants:
        if p.recommendation:
            recommendations[p.recommendation] = recommendations.get(p.recommendation, 0) + 1
    return {
        "participants": len(participants),
        "submitted": sum(1 for p in participants if p.feedback_submitted_at),
        "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "recommendations": recommendations,
    }


async def delete_interview(
    session: AsyncSession, organization_id: uuid.UUID, interview_id: uuid.UUID
) -> None:
    interview = await get_interview(session, organization_id, interview_id)
    await _remove_calendar_event(session, organization_id, interview)
    interview.booking_token = None
    interview.soft_delete()
    await session.commit()


# --------------------------------------------------------------------------- #
# Reminders (design §4.3: 24h and 1h before)
# --------------------------------------------------------------------------- #
# ``kind`` -> (lead time, the column that records the send)
REMINDER_KINDS: dict[str, tuple[timedelta, str]] = {
    "24h": (timedelta(hours=24), "reminder_24h_sent_at"),
    "1h": (timedelta(hours=1), "reminder_1h_sent_at"),
}


async def due_reminders(
    session: AsyncSession,
    kind: str,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 500,
) -> list[Interview]:
    """Interviews whose ``kind`` reminder is due and unsent.

    The test is "starts within the lead time" rather than "starts in exactly
    24 hours", so a worker that was down for an hour still sends on its next
    pass instead of skipping the window entirely.
    """
    if kind not in REMINDER_KINDS:
        raise ValidationError(
            f"Unknown reminder kind '{kind}'",
            details={"allowed": sorted(REMINDER_KINDS)},
        )
    lead, column = REMINDER_KINDS[kind]
    now = to_utc(now or datetime.now(UTC))

    stmt = (
        select(Interview)
        .where(
            Interview.deleted_at.is_(None),
            Interview.status.in_([InterviewStatus.SCHEDULED, InterviewStatus.CONFIRMED]),
            Interview.scheduled_at.is_not(None),
            Interview.scheduled_at > now,
            Interview.scheduled_at <= now + lead,
            getattr(Interview, column).is_(None),
        )
        .order_by(Interview.scheduled_at)
        .limit(limit)
    )
    if organization_id is not None:
        stmt = stmt.where(Interview.organization_id == organization_id)
    return list((await session.execute(stmt)).scalars().all())


async def mark_reminder_sent(
    session: AsyncSession,
    interview: Interview,
    kind: str,
    *,
    at: datetime | None = None,
) -> Interview:
    if kind not in REMINDER_KINDS:
        raise ValidationError(f"Unknown reminder kind '{kind}'")
    _, column = REMINDER_KINDS[kind]
    setattr(interview, column, to_utc(at or datetime.now(UTC)))
    await session.commit()
    await session.refresh(interview)
    return interview


def proposed_intervals(interview: Interview) -> list[Interval]:
    """The interview's offered slots, with any malformed entries dropped."""
    parsed = (_slot_from_dict(entry) for entry in interview.proposed_slots or [])
    return [slot for slot in parsed if slot is not None]


def booking_url(interview: Interview) -> str | None:
    """The candidate-facing link for this interview, if it still has a token."""
    if not interview.booking_token:
        return None
    return f"{settings.public_base_url.rstrip('/')}/book/{interview.booking_token}"


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _new_booking_token() -> str:
    # 32 bytes of urlsafe base64 is 43 characters, inside the column's 64.
    return secrets.token_urlsafe(32)


async def _open_application(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> Application:
    application = await application_service.get_application(
        session, organization_id, application_id
    )
    if application.status in CLOSED_APPLICATION_STATUSES:
        raise ConflictError(
            f"Application is {application.status}; reopen it before scheduling",
            details={"status": application.status},
        )
    return application


async def _resolve_interviewers(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interviewer_ids: list[uuid.UUID],
    type: InterviewType,
) -> list:
    """Load the interviewers, refusing ids from another tenant."""
    # De-duplicate while preserving order: the first id listed is the default
    # organizer, so ordering is meaningful.
    unique_ids = list(dict.fromkeys(interviewer_ids or []))
    if not unique_ids:
        if type in LIVE_TYPES:
            raise ValidationError(
                f"A {type} interview needs at least one interviewer"
            )
        return []

    users = await auth_service.get_users(session, organization_id, unique_ids)
    if len(users) != len(unique_ids):
        found = {u.id for u in users}
        missing = [str(i) for i in unique_ids if i not in found]
        raise NotFoundError(
            "One or more interviewers were not found",
            details={"missing": missing},
        )
    return users


async def _require_communication_consent(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> None:
    """Design §8.2: do not contact a candidate who has not agreed to it."""
    from app.services import candidate_service

    granted = await candidate_service.has_consent(
        session, organization_id, candidate_id, ConsentType.EMAIL_COMMUNICATION
    )
    if not granted:
        raise ValidationError(
            "The candidate has not consented to email communication, so "
            "interview slots cannot be sent to them",
            details={"consent_type": ConsentType.EMAIL_COMMUNICATION.value},
        )


async def _next_round_number(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> int:
    """One past the highest round already on this application."""
    highest = await session.scalar(
        select(func.max(Interview.round_number)).where(
            Interview.organization_id == organization_id,
            Interview.application_id == application_id,
            Interview.deleted_at.is_(None),
        )
    )
    return int(highest or 0) + 1


def _attach_participants(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview: Interview,
    interviewers: list,
    organizer_id: uuid.UUID | None,
) -> None:
    """Add participant rows, making exactly one of them the organizer.

    The organizer's calendar is the one the event is written to, so if the
    requested organizer is not actually an interviewer the first interviewer
    takes the role rather than the event having nowhere to go.
    """
    if not interviewers:
        return
    ids = {u.id for u in interviewers}
    chosen = organizer_id if organizer_id in ids else interviewers[0].id
    for user in interviewers:
        session.add(
            InterviewParticipant(
                organization_id=organization_id,
                interview_id=interview.id,
                user_id=user.id,
                is_organizer=user.id == chosen,
                response_status="pending",
            )
        )


def _availability_warnings(
    availability: list[ParticipantAvailability], interviewers: list
) -> list[str]:
    """Turn unsynced calendars into messages a recruiter can act on."""
    names = {u.id: u.full_name for u in interviewers}
    warnings = []
    for entry in availability:
        if entry.calendar_synced:
            continue
        who = names.get(entry.user_id, entry.user_id)
        warnings.append(
            f"Calendar for {who} could not be read "
            f"({entry.error or 'unknown reason'}); slots are based on working "
            "hours only and may conflict with existing meetings."
        )
    return warnings


def _slot_from_dict(entry: object) -> Interval | None:
    if not isinstance(entry, dict):
        return None
    try:
        start = datetime.fromisoformat(str(entry["start"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(entry["end"]).replace("Z", "+00:00"))
        return Interval(to_utc(start), to_utc(end))
    except (KeyError, ValueError):
        logger.warning("Discarding malformed proposed slot %r", entry)
        return None


def _match_proposed_slot(interview: Interview, slot_start: datetime) -> datetime:
    """Check the requested start is one that was actually offered.

    Booking is restricted to the offered slots because those are the times
    availability was verified for; accepting an arbitrary start would let a
    candidate book straight over an interviewer's other meeting.
    """
    wanted = to_utc(slot_start)
    for entry in interview.proposed_slots or []:
        slot = _slot_from_dict(entry)
        if slot is not None and slot.start == wanted:
            return slot.start
    raise ValidationError(
        "That time is not one of the offered slots",
        details={
            "requested": to_iso(wanted),
            "offered": [
                to_iso(s.start)
                for s in (
                    _slot_from_dict(e) for e in interview.proposed_slots or []
                )
                if s is not None
            ],
        },
    )


async def _assert_no_conflict(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interviewer_ids: list[uuid.UUID],
    slot: Interval,
    *,
    exclude_interview_id: uuid.UUID | None = None,
) -> None:
    """Refuse a time that collides with an interviewer's existing interview."""
    if not interviewer_ids:
        return

    stmt = (
        scoped_select(Interview, organization_id)
        .join(
            InterviewParticipant,
            InterviewParticipant.interview_id == Interview.id,
        )
        .where(
            InterviewParticipant.user_id.in_(interviewer_ids),
            InterviewParticipant.deleted_at.is_(None),
            Interview.status.in_(list(calendar_service.BLOCKING_STATUSES)),
            Interview.scheduled_at.is_not(None),
            # Cheap pre-filter in SQL; the exact overlap test needs the
            # duration, which lives on the row.
            Interview.scheduled_at < slot.end,
        )
        .add_columns(InterviewParticipant.user_id)
    )
    if exclude_interview_id is not None:
        stmt = stmt.where(Interview.id != exclude_interview_id)

    clashes: list[dict] = []
    for interview, user_id in (await session.execute(stmt)).all():
        booked = calendar_service.interview_interval(interview)
        if booked is not None and booked.overlaps(slot):
            clashes.append(
                {
                    "interview_id": str(interview.id),
                    "user_id": str(user_id),
                    "start": to_iso(booked.start),
                    "end": to_iso(booked.end),
                }
            )
    if clashes:
        raise ConflictError(
            "An interviewer is already booked during that time",
            details={"conflicts": clashes},
        )


async def _organizer_account(
    session: AsyncSession, organization_id: uuid.UUID, interview: Interview
) -> CalendarAccount | None:
    """The calendar the event should be written to."""
    participants = await list_participants(session, organization_id, interview.id)
    ordered = [p for p in participants if p.is_organizer] + [
        p for p in participants if not p.is_organizer
    ]
    for participant in ordered:
        account = await session.scalar(
            scoped_select(CalendarAccount, organization_id)
            .where(
                CalendarAccount.user_id == participant.user_id,
                CalendarAccount.is_active.is_(True),
            )
            .order_by(CalendarAccount.created_at)
            .limit(1)
        )
        if account is not None:
            return account
    return None


async def _write_calendar_event(
    session: AsyncSession, organization_id: uuid.UUID, interview: Interview
) -> list[str]:
    """Create the external calendar event, returning warnings rather than raising."""
    if interview.scheduled_at is None:
        return []

    account = await _organizer_account(session, organization_id, interview)
    if account is None:
        return [
            "No connected calendar for the interviewers; no calendar invite was sent."
        ]

    candidate_email = await _candidate_email(session, organization_id, interview)
    attendees = [account.email] + ([candidate_email] if candidate_email else [])
    start = to_utc(interview.scheduled_at)

    event = calendar_api.CalendarEvent(
        summary=interview.title or f"Interview — round {interview.round_number}",
        start=start,
        end=start + timedelta(minutes=interview.duration_minutes),
        description=interview.notes,
        location=interview.location,
        attendees=attendees,
        timezone=interview.timezone or "UTC",
        create_conference=interview.type
        in {InterviewType.VIDEO, InterviewType.PANEL, InterviewType.TECHNICAL},
    )

    provider = calendar_api.get_provider(account.provider)
    credentials = _credentials(account)
    try:
        result = await provider.create_event(credentials, event)
    except Exception as exc:  # noqa: BLE001 - a provider fault must not lose the booking
        logger.exception("Calendar event creation failed for interview %s", interview.id)
        return [f"Calendar invite could not be created: {exc}"]

    if result.refreshed is not None:
        account.access_token = result.refreshed.access_token
        account.token_expires_at = result.refreshed.expires_at
    if not result.ok:
        return [f"Calendar invite could not be created: {result.error}"]

    interview.calendar_account_id = account.id
    interview.external_event_id = result.external_event_id
    if result.meeting_url and not interview.meeting_url:
        interview.meeting_url = result.meeting_url
    await session.flush()
    return []


async def _remove_calendar_event(
    session: AsyncSession, organization_id: uuid.UUID, interview: Interview
) -> list[str]:
    """Delete the external event, if there is one."""
    if not interview.external_event_id or interview.calendar_account_id is None:
        return []

    account = await get_scoped(
        session,
        CalendarAccount,
        interview.calendar_account_id,
        organization_id,
        include_deleted=True,
    )
    if account is None:
        return []

    provider = calendar_api.get_provider(account.provider)
    try:
        result = await provider.delete_event(
            _credentials(account), interview.external_event_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calendar event deletion failed for interview %s", interview.id)
        return [f"The calendar invite could not be withdrawn: {exc}"]

    if result.refreshed is not None:
        account.access_token = result.refreshed.access_token
        account.token_expires_at = result.refreshed.expires_at
    if not result.ok:
        return [f"The calendar invite could not be withdrawn: {result.error}"]

    interview.external_event_id = None
    await session.flush()
    return []


def _credentials(account: CalendarAccount) -> calendar_api.CalendarCredentials:
    return calendar_api.CalendarCredentials(
        provider=account.provider,
        email=account.email,
        access_token=account.access_token,
        refresh_token=account.refresh_token,
        expires_at=account.token_expires_at,
        calendar_id=account.calendar_id,
    )


async def _candidate_email(
    session: AsyncSession, organization_id: uuid.UUID, interview: Interview
) -> str | None:
    application = await get_scoped(
        session, Application, interview.application_id, organization_id
    )
    if application is None:
        return None
    candidate = await get_scoped(
        session, Candidate, application.candidate_id, organization_id
    )
    return candidate.email if candidate else None


async def _get_participant(
    session: AsyncSession,
    organization_id: uuid.UUID,
    interview_id: uuid.UUID,
    user_id: uuid.UUID,
) -> InterviewParticipant:
    await get_interview(session, organization_id, interview_id)
    participant = await session.scalar(
        scoped_select(InterviewParticipant, organization_id).where(
            InterviewParticipant.interview_id == interview_id,
            InterviewParticipant.user_id == user_id,
        )
    )
    if participant is None:
        raise NotFoundError("That user is not a participant in this interview")
    return participant
