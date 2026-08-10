"""Outreach sequences: steps, enrollment, and send scheduling (design §4.2).

A sequence is a list of steps — "wait two days, then send this" — that each
enrolled candidate walks independently. Three decisions shape the module.

**The enrollment carries the clock, not the sequence.** ``next_send_at`` lives
on the enrollment, so two candidates enrolled a week apart are each at their
own point in the ramp, and a sequence can be edited without disturbing anyone
already partway through it.

**Enrolment is filtered, not rejected.** Handing a recruiter fifty candidates
and refusing all of them because one has no email address is useless. Every
candidate is judged separately and the skipped ones come back with a reason, so
the caller can show "44 enrolled, 6 skipped: 4 no consent, 2 already in this
sequence" instead of a 422.

**A send time is snapped into the window, never cancelled by it.** A step that
comes due at 3am does not get dropped; it moves to the next moment the window
allows. Outreach that silently skips a step is far worse than outreach that
arrives at nine the next morning — the candidate simply never hears the middle
of the story.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.models.application import Application
from app.models.candidate import Candidate, CandidateConsent
from app.models.enums import (
    ApplicationStatus,
    ConsentStatus,
    ConsentType,
    EnrollmentStatus,
    MessageStatus,
    OutreachChannel,
    SequenceStatus,
)
from app.models.organization import Organization
from app.models.outreach import (
    MessageTemplate,
    OutreachMessage,
    OutreachSequence,
    SequenceEnrollment,
    SequenceStep,
)
from app.schemas.common import PaginationParams
from app.services.availability import load_zone, to_utc

logger = logging.getLogger(__name__)

# Sequence states that still do work. Enrollments under anything else sit
# still, which is what makes pausing a campaign immediate.
RUNNING_STATUSES = frozenset({SequenceStatus.ACTIVE})

# Enrollment states that are the end of the line.
TERMINAL_ENROLLMENT_STATUSES = frozenset(
    {
        EnrollmentStatus.COMPLETED,
        EnrollmentStatus.REPLIED,
        EnrollmentStatus.BOUNCED,
        EnrollmentStatus.UNSUBSCRIBED,
        EnrollmentStatus.FAILED,
    }
)

# Which consent a channel requires before a candidate may be contacted on it
# (design §8.2). LinkedIn is absent deliberately: an InMail is sent inside
# LinkedIn's own consent regime, not ours.
CHANNEL_CONSENT: dict[OutreachChannel, ConsentType] = {
    OutreachChannel.EMAIL: ConsentType.EMAIL_COMMUNICATION,
    OutreachChannel.WHATSAPP: ConsentType.WHATSAPP_COMMUNICATION,
    OutreachChannel.SMS: ConsentType.SMS_COMMUNICATION,
}

# How far ahead the window search will look before giving up. A sequence whose
# window is unsatisfiable (no allowed weekday) would otherwise loop forever.
MAX_WINDOW_SEARCH_DAYS = 14

MAX_STEPS = 25

# Where steps are parked while their order is being rewritten. See _renumber.
RENUMBER_OFFSET = 1000

# Where a freshly inserted step waits until it has been given a real position.
# It must sit clear of the parking range as well as of the final 0..MAX_STEPS
# one: the parking pass is about to walk every sibling through RENUMBER_OFFSET+n
# and would collide with a new row already sitting in that range.
STAGING_ORDER = 2 * RENUMBER_OFFSET

# Applications in these states should not be gaining new outreach.
CLOSED_APPLICATION_STATUSES = frozenset(
    {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}
)

STAT_KEYS = (
    "enrolled",
    "sent",
    "delivered",
    "opened",
    "clicked",
    "replied",
    "bounced",
    "failed",
    "unsubscribed",
)


def _now(value: datetime | None = None) -> datetime:
    return to_utc(value) if value is not None else datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Send windows
# --------------------------------------------------------------------------- #
def _local_hour(day: date, hour: int, zone) -> datetime:
    """Wall-clock ``hour`` on ``day`` in ``zone``.

    Built naive and then localised rather than by adding a timedelta to an
    aware midnight: absolute arithmetic across a DST boundary lands an hour
    off, and "we send at nine" means nine on the clock.
    """
    naive = datetime.combine(day, time(0)) + timedelta(hours=hour)
    return naive.replace(tzinfo=zone)


def day_is_allowed(day: date, sequence: OutreachSequence) -> bool:
    return bool(sequence.send_on_weekends) or day.weekday() < 5


def is_within_window(moment: datetime, sequence: OutreachSequence) -> bool:
    """Whether ``moment`` falls inside the sequence's sending window."""
    zone = load_zone(sequence.timezone)
    local = to_utc(moment).astimezone(zone)
    if not day_is_allowed(local.date(), sequence):
        return False
    start = _local_hour(local.date(), sequence.send_window_start_hour, zone)
    end = _local_hour(local.date(), sequence.send_window_end_hour, zone)
    return start <= local < end


def next_send_time(after: datetime, sequence: OutreachSequence) -> datetime:
    """The first moment at or after ``after`` that the window allows.

    Always returns a time. A step that comes due at 3am is moved to nine the
    next morning rather than dropped: a candidate who never hears the middle of
    a sequence is a worse outcome than one who hears it a few hours late.
    """
    zone = load_zone(sequence.timezone)
    local = to_utc(after).astimezone(zone)
    day = local.date()

    for offset in range(MAX_WINDOW_SEARCH_DAYS + 1):
        current = day + timedelta(days=offset)
        if not day_is_allowed(current, sequence):
            continue
        start = _local_hour(current, sequence.send_window_start_hour, zone)
        end = _local_hour(current, sequence.send_window_end_hour, zone)
        if local < start:
            return start.astimezone(UTC)
        if local < end:
            return local.astimezone(UTC)

    # An unsatisfiable window (misconfigured hours, say) must not strand the
    # enrollment forever; send at the horizon and let the log explain.
    logger.warning(
        "Sequence %s has no send window within %d days; sending at the horizon",
        sequence.id,
        MAX_WINDOW_SEARCH_DAYS,
    )
    return (local + timedelta(days=MAX_WINDOW_SEARCH_DAYS)).astimezone(UTC)


def step_due_at(
    step: SequenceStep, *, after: datetime, sequence: OutreachSequence
) -> datetime:
    """When ``step`` should fire, given the previous step finished at ``after``.

    The delay is applied first and the window second, so "wait two days" means
    two days and not "two days, rounded down to whenever the window next
    opened".
    """
    delay = timedelta(days=max(0, step.delay_days), hours=max(0, step.delay_hours))
    return next_send_time(to_utc(after) + delay, sequence)


# --------------------------------------------------------------------------- #
# Sequence CRUD
# --------------------------------------------------------------------------- #
def _validate_window(start_hour: int, end_hour: int) -> None:
    if not 0 <= start_hour <= 23:
        raise ValidationError("Send window must start between hour 0 and 23")
    if not 1 <= end_hour <= 24:
        raise ValidationError("Send window must end between hour 1 and 24")
    if start_hour >= end_hour:
        raise ValidationError("Send window must start before it ends")


def _validate_timezone(name: str) -> None:
    """Reject an unknown zone at authoring time.

    ``load_zone`` deliberately falls back to UTC rather than raising, which is
    right at send time but wrong here: a typo would leave the sequence quietly
    sending at UTC hours instead of the recruiter's, and nobody would notice
    until a candidate got a 2am email.
    """
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValidationError(f"Unknown timezone {name!r}") from exc


async def create_sequence(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    name: str,
    job_id: uuid.UUID | None = None,
    sender_account_ids: list[uuid.UUID] | None = None,
    timezone: str = "UTC",
    send_window_start_hour: int = 9,
    send_window_end_hour: int = 18,
    send_on_weekends: bool = False,
    stop_on_reply: bool = True,
    daily_cap: int | None = None,
) -> OutreachSequence:
    label = (name or "").strip()
    if not label:
        raise ValidationError("A sequence needs a name")
    _validate_window(send_window_start_hour, send_window_end_hour)
    _validate_timezone(timezone)
    if daily_cap is not None and daily_cap < 1:
        raise ValidationError("A daily cap must be at least 1")

    sequence = OutreachSequence(
        organization_id=organization_id,
        name=label,
        job_id=job_id,
        status=SequenceStatus.DRAFT,
        sender_account_ids=[str(i) for i in (sender_account_ids or [])],
        timezone=timezone,
        send_window_start_hour=send_window_start_hour,
        send_window_end_hour=send_window_end_hour,
        send_on_weekends=send_on_weekends,
        stop_on_reply=stop_on_reply,
        daily_cap=daily_cap,
        stats_json=dict.fromkeys(STAT_KEYS, 0),
    )
    session.add(sequence)
    await session.commit()
    await session.refresh(sequence)
    return sequence


async def get_sequence(
    session: AsyncSession, organization_id: uuid.UUID, sequence_id: uuid.UUID
) -> OutreachSequence:
    sequence = await get_scoped(session, OutreachSequence, sequence_id, organization_id)
    if sequence is None:
        raise NotFoundError("Sequence not found")
    return sequence


async def list_sequences(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    status: SequenceStatus | None = None,
    job_id: uuid.UUID | None = None,
) -> list[OutreachSequence]:
    stmt = scoped_select(OutreachSequence, organization_id)
    if status is not None:
        stmt = stmt.where(OutreachSequence.status == status)
    if job_id is not None:
        stmt = stmt.where(OutreachSequence.job_id == job_id)
    stmt = stmt.order_by(OutreachSequence.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def update_sequence(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    *,
    changes: dict,
) -> OutreachSequence:
    sequence = await get_sequence(session, organization_id, sequence_id)

    for field_name in (
        "name",
        "timezone",
        "send_window_start_hour",
        "send_window_end_hour",
        "send_on_weekends",
        "stop_on_reply",
        "daily_cap",
        "job_id",
    ):
        if field_name in changes and changes[field_name] is not None:
            setattr(sequence, field_name, changes[field_name])
    if "sender_account_ids" in changes and changes["sender_account_ids"] is not None:
        sequence.sender_account_ids = [str(i) for i in changes["sender_account_ids"]]

    _validate_window(sequence.send_window_start_hour, sequence.send_window_end_hour)
    _validate_timezone(sequence.timezone)
    await session.commit()
    await session.refresh(sequence)
    return sequence


async def activate(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> OutreachSequence:
    """Start (or resume) a sequence so its enrollments begin sending."""
    sequence = await get_sequence(session, organization_id, sequence_id)
    if sequence.status == SequenceStatus.ACTIVE:
        return sequence
    if sequence.status == SequenceStatus.ARCHIVED:
        raise ConflictError("An archived sequence cannot be activated")
    if not sequence.steps:
        raise ValidationError("A sequence needs at least one step before it runs")

    sequence.status = SequenceStatus.ACTIVE
    if sequence.started_at is None:
        sequence.started_at = _now(now)
    sequence.completed_at = None
    await session.commit()
    await session.refresh(sequence)
    return sequence


async def pause(
    session: AsyncSession, organization_id: uuid.UUID, sequence_id: uuid.UUID
) -> OutreachSequence:
    """Halt sending without losing anyone's position in the sequence."""
    sequence = await get_sequence(session, organization_id, sequence_id)
    if sequence.status in (SequenceStatus.COMPLETED, SequenceStatus.ARCHIVED):
        raise ConflictError(f"A {sequence.status} sequence cannot be paused")
    sequence.status = SequenceStatus.PAUSED
    await session.commit()
    await session.refresh(sequence)
    return sequence


async def complete_sequence(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> OutreachSequence:
    sequence = await get_sequence(session, organization_id, sequence_id)
    sequence.status = SequenceStatus.COMPLETED
    sequence.completed_at = _now(now)
    await session.commit()
    await session.refresh(sequence)
    return sequence


async def delete_sequence(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> None:
    sequence = await get_sequence(session, organization_id, sequence_id)
    sequence.deleted_at = _now(now)
    sequence.status = SequenceStatus.ARCHIVED
    await session.commit()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
async def _renumber(session: AsyncSession, ordered: list[SequenceStep]) -> None:
    """Rewrite ``step_order`` over ``ordered`` as 0, 1, 2, … in two passes.

    ``(sequence_id, step_order)`` is unique, and a flush emits its UPDATEs in
    no particular order, so assigning the final numbers directly can collide
    with a sibling that has not moved yet. Parking every row above the range
    first makes the second pass collision-free whatever order it runs in.

    The parking range must therefore be empty when this starts, which is why a
    newly inserted step waits at ``STAGING_ORDER`` rather than inside it.
    """
    for offset, step in enumerate(ordered):
        step.step_order = RENUMBER_OFFSET + offset
    await session.flush()
    for index, step in enumerate(ordered):
        step.step_order = index
    await session.flush()


async def add_step(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    *,
    template_id: uuid.UUID | None = None,
    channel: OutreachChannel = OutreachChannel.EMAIL,
    delay_days: int = 0,
    delay_hours: int = 0,
    subject_override: str | None = None,
    body_override: str | None = None,
    variant_group: str | None = None,
    step_order: int | None = None,
) -> SequenceStep:
    """Append (or insert) a step.

    A step must end up with something to say: either a template or an inline
    body override. Allowing neither would produce an enrollment that advances
    through a step and sends nothing, which looks identical to a bug.
    """
    sequence = await get_sequence(session, organization_id, sequence_id)
    if len(sequence.steps) >= MAX_STEPS:
        raise ValidationError(f"A sequence may have at most {MAX_STEPS} steps")
    if delay_days < 0 or delay_hours < 0:
        raise ValidationError("A step delay cannot be negative")
    if template_id is None and not (body_override or "").strip():
        raise ValidationError("A step needs either a template or a body override")

    if template_id is not None:
        template = await get_scoped(
            session, MessageTemplate, template_id, organization_id
        )
        if template is None:
            raise NotFoundError("Template not found")
        if OutreachChannel(template.channel) != channel:
            raise ValidationError(
                f"Template is for {template.channel}, not {channel}",
                details={"template_channel": str(template.channel)},
            )

    existing = sorted(sequence.steps, key=lambda s: s.step_order)
    position = len(existing) if step_order is None else max(0, step_order)
    position = min(position, len(existing))

    step = SequenceStep(
        organization_id=organization_id,
        sequence_id=sequence.id,
        # Parked clear of both numbering ranges so neither the insert nor the
        # renumbering that follows it can collide with a sibling.
        step_order=STAGING_ORDER,
        channel=channel,
        template_id=template_id,
        delay_days=delay_days,
        delay_hours=delay_hours,
        subject_override=subject_override,
        body_override=body_override,
        variant_group=variant_group,
    )
    session.add(step)
    await session.flush()

    existing.insert(position, step)
    await _renumber(session, existing)
    await session.commit()
    await session.refresh(sequence)
    return step


async def get_step(
    session: AsyncSession, organization_id: uuid.UUID, step_id: uuid.UUID
) -> SequenceStep:
    step = await get_scoped(session, SequenceStep, step_id, organization_id)
    if step is None:
        raise NotFoundError("Step not found")
    return step


async def update_step(
    session: AsyncSession,
    organization_id: uuid.UUID,
    step_id: uuid.UUID,
    *,
    changes: dict,
) -> SequenceStep:
    step = await get_step(session, organization_id, step_id)
    for field_name in (
        "delay_days",
        "delay_hours",
        "subject_override",
        "body_override",
        "variant_group",
        "template_id",
    ):
        if field_name in changes and changes[field_name] is not None:
            setattr(step, field_name, changes[field_name])

    if step.delay_days < 0 or step.delay_hours < 0:
        raise ValidationError("A step delay cannot be negative")
    if step.template_id is None and not (step.body_override or "").strip():
        raise ValidationError("A step needs either a template or a body override")
    await session.commit()
    await session.refresh(step)
    return step


async def delete_step(
    session: AsyncSession, organization_id: uuid.UUID, step_id: uuid.UUID
) -> None:
    """Remove a step and close the gap in the ordering.

    Enrollments index into the sequence by ``current_step``, so leaving a hole
    would silently skip whichever step happened to sit after it.
    """
    step = await get_step(session, organization_id, step_id)
    sequence = await get_sequence(session, organization_id, step.sequence_id)
    survivors = sorted(
        (s for s in sequence.steps if s.id != step.id), key=lambda s: s.step_order
    )

    await session.delete(step)
    await session.flush()
    await _renumber(session, survivors)
    await session.commit()
    await session.refresh(sequence)


async def reorder_steps(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    ordered_step_ids: list[uuid.UUID],
) -> list[SequenceStep]:
    """Rewrite a sequence's step order from an explicit list.

    The list must name every step exactly once: a partial reorder would leave
    the rest at numbers that no longer mean anything, and enrollments index
    into the sequence by position.
    """
    sequence = await get_sequence(session, organization_id, sequence_id)
    by_id = {step.id: step for step in sequence.steps}
    if len(ordered_step_ids) != len(by_id) or set(ordered_step_ids) != set(by_id):
        raise ValidationError(
            "A reorder must list every step of the sequence exactly once",
            details={"expected": len(by_id), "received": len(ordered_step_ids)},
        )

    ordered = [by_id[step_id] for step_id in ordered_step_ids]
    await _renumber(session, ordered)
    await session.commit()
    await session.refresh(sequence)
    return sorted(sequence.steps, key=lambda s: s.step_order)


def step_at(sequence: OutreachSequence, index: int) -> SequenceStep | None:
    """The step an enrollment at ``index`` should send next, if any."""
    for step in sorted(sequence.steps, key=lambda s: s.step_order):
        if step.step_order == index:
            return step
    return None


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
@dataclass
class EnrollmentReport:
    """What one enrolment call did, candidate by candidate.

    Skips are data, not errors: a recruiter enrolling fifty candidates needs
    "44 enrolled, 6 skipped and here is why", not a 422 that enrolls nobody
    because one of them opted out.
    """

    enrolled: list[SequenceEnrollment] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    @property
    def enrolled_count(self) -> int:
        return len(self.enrolled)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def required_consents(sequence: OutreachSequence) -> set[ConsentType]:
    """Every consent the sequence's channels need before it may contact anyone."""
    return {
        CHANNEL_CONSENT[channel]
        for step in sequence.steps
        if (channel := OutreachChannel(step.channel)) in CHANNEL_CONSENT
    }


async def _application_for(
    session: AsyncSession,
    organization_id: uuid.UUID,
    candidate_id: uuid.UUID,
    job_id: uuid.UUID | None,
) -> Application | None:
    """The candidate's open application to the sequence's job, if there is one."""
    if job_id is None:
        return None
    return await session.scalar(
        scoped_select(Application, organization_id)
        .where(
            Application.candidate_id == candidate_id,
            Application.job_id == job_id,
            Application.status.not_in([s.value for s in CLOSED_APPLICATION_STATUSES]),
        )
        .order_by(Application.created_at.desc())
        .limit(1)
    )


async def enroll(
    session: AsyncSession,
    organization_id: uuid.UUID,
    sequence_id: uuid.UUID,
    candidate_ids: list[uuid.UUID],
    *,
    now: datetime | None = None,
) -> EnrollmentReport:
    """Enrol candidates, skipping the ones who must not be contacted.

    The first step's delay is honoured from the moment of enrolment, so a
    sequence whose step 0 has no delay starts as soon as the window allows.
    """
    from app.services import candidate_service

    sequence = await get_sequence(session, organization_id, sequence_id)
    if sequence.status == SequenceStatus.ARCHIVED:
        raise ConflictError("An archived sequence cannot take enrollments")
    if not sequence.steps:
        raise ValidationError("A sequence needs at least one step before enrolling")

    stamp = _now(now)
    first_step = step_at(sequence, 0)
    consents = required_consents(sequence)
    report = EnrollmentReport()

    already = set(
        (
            await session.execute(
                scoped_select(SequenceEnrollment, organization_id).where(
                    SequenceEnrollment.sequence_id == sequence.id,
                    SequenceEnrollment.candidate_id.in_(candidate_ids),
                )
            )
        )
        .scalars()
        .all()
    )
    already_ids = {e.candidate_id for e in already}

    def skip(candidate_id: uuid.UUID, reason: str, detail: str) -> None:
        report.skipped.append(
            {"candidate_id": str(candidate_id), "reason": reason, "detail": detail}
        )

    for candidate_id in dict.fromkeys(candidate_ids):
        if candidate_id in already_ids:
            skip(candidate_id, "already_enrolled", "Already in this sequence")
            continue

        candidate = await get_scoped(session, Candidate, candidate_id, organization_id)
        if candidate is None:
            skip(candidate_id, "not_found", "Candidate not found")
            continue
        if candidate.is_blacklisted:
            skip(candidate_id, "blacklisted", "Candidate is blacklisted")
            continue
        if not (candidate.email or "").strip():
            skip(candidate_id, "no_email", "Candidate has no email address")
            continue

        missing_consent = [
            consent
            for consent in sorted(consents)
            if not await candidate_service.has_consent(
                session, organization_id, candidate_id, consent
            )
        ]
        if missing_consent:
            skip(
                candidate_id,
                "no_consent",
                f"No consent for {', '.join(str(c) for c in missing_consent)}",
            )
            continue

        application = await _application_for(
            session, organization_id, candidate_id, sequence.job_id
        )
        enrollment = SequenceEnrollment(
            organization_id=organization_id,
            sequence_id=sequence.id,
            candidate_id=candidate_id,
            application_id=application.id if application else None,
            status=EnrollmentStatus.ACTIVE,
            current_step=0,
            enrolled_at=stamp,
            next_send_at=(
                step_due_at(first_step, after=stamp, sequence=sequence)
                if first_step
                else None
            ),
        )
        session.add(enrollment)
        report.enrolled.append(enrollment)

    if report.enrolled:
        bump_stats(sequence, "enrolled", len(report.enrolled))
    await session.commit()
    for enrollment in report.enrolled:
        await session.refresh(enrollment)
    return report


async def get_enrollment(
    session: AsyncSession, organization_id: uuid.UUID, enrollment_id: uuid.UUID
) -> SequenceEnrollment:
    enrollment = await get_scoped(
        session, SequenceEnrollment, enrollment_id, organization_id
    )
    if enrollment is None:
        raise NotFoundError("Enrollment not found")
    return enrollment


async def list_enrollments(
    session: AsyncSession,
    organization_id: uuid.UUID,
    params: PaginationParams | None = None,
    *,
    sequence_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    status: EnrollmentStatus | None = None,
) -> tuple[list[SequenceEnrollment], int]:
    """List enrollments, newest first.

    ``params`` is optional so internal callers (e.g. tests exercising a
    single campaign) don't have to construct one; it defaults to page 1 at
    the standard page size rather than returning every row, since a live
    campaign can enroll thousands of candidates.
    """
    params = params or PaginationParams()
    stmt = scoped_select(SequenceEnrollment, organization_id)
    count_stmt = (
        select(func.count())
        .select_from(SequenceEnrollment)
        .where(
            SequenceEnrollment.organization_id == organization_id,
            SequenceEnrollment.deleted_at.is_(None),
        )
    )
    if sequence_id is not None:
        stmt = stmt.where(SequenceEnrollment.sequence_id == sequence_id)
        count_stmt = count_stmt.where(SequenceEnrollment.sequence_id == sequence_id)
    if candidate_id is not None:
        stmt = stmt.where(SequenceEnrollment.candidate_id == candidate_id)
        count_stmt = count_stmt.where(SequenceEnrollment.candidate_id == candidate_id)
    if status is not None:
        stmt = stmt.where(SequenceEnrollment.status == status)
        count_stmt = count_stmt.where(SequenceEnrollment.status == status)
    stmt = (
        stmt.order_by(SequenceEnrollment.enrolled_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    rows = list((await session.execute(stmt)).scalars().all())
    total = int(await session.scalar(count_stmt) or 0)
    return rows, total


async def due_enrollments(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 200,
) -> list[SequenceEnrollment]:
    """Active enrollments whose next step is due, oldest first.

    "Due at or before now" rather than "due in this tick" is what makes the
    dispatcher safe to miss: a worker that was down for an hour catches up
    instead of skipping everyone in the gap.
    """
    stamp = _now(now)
    stmt = (
        select(SequenceEnrollment)
        .join(
            OutreachSequence,
            OutreachSequence.id == SequenceEnrollment.sequence_id,
        )
        .where(
            SequenceEnrollment.deleted_at.is_(None),
            SequenceEnrollment.status == EnrollmentStatus.ACTIVE.value,
            SequenceEnrollment.next_send_at.is_not(None),
            SequenceEnrollment.next_send_at <= stamp,
            # A paused or draft sequence stops its enrollments dead, which is
            # what makes pausing a campaign take effect on the next tick.
            OutreachSequence.deleted_at.is_(None),
            OutreachSequence.status == SequenceStatus.ACTIVE.value,
        )
    )
    if organization_id is not None:
        stmt = stmt.where(SequenceEnrollment.organization_id == organization_id)
    stmt = stmt.order_by(SequenceEnrollment.next_send_at.asc()).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


def advance(
    enrollment: SequenceEnrollment,
    sequence: OutreachSequence,
    *,
    now: datetime | None = None,
) -> SequenceEnrollment:
    """Move an enrollment past the step it just sent.

    Reaching the end completes the enrollment rather than leaving it ACTIVE
    with nothing to do, so "still being worked" and "finished the sequence"
    stay distinguishable in the pipeline view.
    """
    stamp = _now(now)
    enrollment.current_step += 1
    following = step_at(sequence, enrollment.current_step)
    if following is None:
        enrollment.status = EnrollmentStatus.COMPLETED
        enrollment.completed_at = stamp
        enrollment.next_send_at = None
    else:
        enrollment.next_send_at = step_due_at(following, after=stamp, sequence=sequence)
    return enrollment


def halt(
    enrollment: SequenceEnrollment,
    status: EnrollmentStatus,
    *,
    reason: str | None = None,
    now: datetime | None = None,
) -> SequenceEnrollment:
    """Stop an enrollment for good, clearing its clock so nothing fires again."""
    enrollment.status = status
    enrollment.next_send_at = None
    enrollment.paused_reason = reason
    if status in TERMINAL_ENROLLMENT_STATUSES:
        enrollment.completed_at = _now(now)
    return enrollment


async def pause_enrollment(
    session: AsyncSession,
    organization_id: uuid.UUID,
    enrollment_id: uuid.UUID,
    *,
    reason: str | None = None,
) -> SequenceEnrollment:
    enrollment = await get_enrollment(session, organization_id, enrollment_id)
    if enrollment.status in TERMINAL_ENROLLMENT_STATUSES:
        raise ConflictError(f"A {enrollment.status} enrollment cannot be paused")
    enrollment.status = EnrollmentStatus.PAUSED
    enrollment.paused_reason = reason
    # next_send_at is deliberately left in place: resuming should pick the
    # thread back up rather than restart the candidate's clock.
    await session.commit()
    await session.refresh(enrollment)
    return enrollment


async def resume_enrollment(
    session: AsyncSession,
    organization_id: uuid.UUID,
    enrollment_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> SequenceEnrollment:
    """Put a paused enrollment back in the queue.

    A step whose time passed while it was paused becomes due immediately —
    snapped into the window — rather than being skipped.
    """
    enrollment = await get_enrollment(session, organization_id, enrollment_id)
    if enrollment.status != EnrollmentStatus.PAUSED:
        raise ConflictError("Only a paused enrollment can be resumed")

    sequence = await get_sequence(session, organization_id, enrollment.sequence_id)
    stamp = _now(now)
    enrollment.status = EnrollmentStatus.ACTIVE
    enrollment.paused_reason = None
    pending = step_at(sequence, enrollment.current_step)
    if pending is None:
        enrollment.status = EnrollmentStatus.COMPLETED
        enrollment.completed_at = stamp
        enrollment.next_send_at = None
    else:
        scheduled = (
            to_utc(enrollment.next_send_at) if enrollment.next_send_at else stamp
        )
        enrollment.next_send_at = next_send_time(max(scheduled, stamp), sequence)
    await session.commit()
    await session.refresh(enrollment)
    return enrollment


async def record_reply(
    session: AsyncSession,
    organization_id: uuid.UUID,
    enrollment_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> SequenceEnrollment:
    """A candidate answered — stop the remaining steps if the sequence says so.

    This is the single most important stop condition in the product: nothing
    reads as more automated than a follow-up that arrives after a real reply.
    """
    enrollment = await get_enrollment(session, organization_id, enrollment_id)
    sequence = await get_sequence(session, organization_id, enrollment.sequence_id)
    stamp = _now(now)

    enrollment.replied_at = stamp
    if sequence.stop_on_reply:
        halt(
            enrollment, EnrollmentStatus.REPLIED, reason="Candidate replied", now=stamp
        )
    bump_stats(sequence, "replied")
    await session.commit()
    await session.refresh(enrollment)
    return enrollment


async def unsubscribe(
    session: AsyncSession,
    organization_id: uuid.UUID,
    enrollment_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> SequenceEnrollment:
    enrollment = await get_enrollment(session, organization_id, enrollment_id)
    halt(
        enrollment,
        EnrollmentStatus.UNSUBSCRIBED,
        reason="Candidate unsubscribed",
        now=now,
    )
    await session.commit()
    await session.refresh(enrollment)
    return enrollment


# --------------------------------------------------------------------------- #
# Opting out from a link in a message
# --------------------------------------------------------------------------- #
def unsubscribe_page_url(token: str | None) -> str | None:
    """The page a human lands on. Goes in the message body, for a human to read.

    A visible opt-out in the body is not the same requirement as the header
    one: the header serves the mail client, this serves the reader, and
    CAN-SPAM asks for the second regardless of whether the first is present.
    """
    if not token:
        return None
    return f"{settings.public_base_url.rstrip('/')}/outreach/unsubscribe/{token}"


def one_click_unsubscribe_url(token: str | None) -> str | None:
    """The endpoint a mail client POSTs to. Goes in ``List-Unsubscribe``.

    This must be the API and not the frontend. RFC 8058 one-click means the
    client sends a POST with no human involved and no page load, so pointing
    the header at a single-page app would advertise an opt-out that silently
    404s — and a candidate whose unsubscribe appears to be ignored reports the
    next message as spam, which is the outcome the header exists to avoid.
    """
    if not token:
        return None
    base = settings.api_base_url.rstrip("/")
    prefix = settings.api_v1_prefix.rstrip("/")
    return f"{base}{prefix}/outreach/unsubscribe/{token}"


@dataclass
class UnsubscribeResult:
    """What one opt-out did, for the confirmation page and for the audit log."""

    candidate_id: uuid.UUID
    organization_id: uuid.UUID
    organization_name: str | None
    enrollment_id: uuid.UUID | None
    # Enrollments this call actually stopped. Zero on a repeat click.
    stopped: int = 0
    # True when consent was already withdrawn before this call.
    already_unsubscribed: bool = False


async def message_by_tracking_token(
    session: AsyncSession, token: str
) -> OutreachMessage:
    """Resolve an opt-out token to the message it was minted for.

    Deliberately unscoped by tenant: the holder is a candidate with a link, not
    a user with an organization. The token is unique across the table, and the
    message it resolves to is what supplies the tenant for everything after.

    A bad token and a deleted one both surface as the same "not valid", so a
    guess cannot be distinguished from a near miss.
    """
    if not token:
        raise NotFoundError("This unsubscribe link is not valid")
    message = await session.scalar(
        select(OutreachMessage).where(
            OutreachMessage.tracking_token == token,
            OutreachMessage.deleted_at.is_(None),
        )
    )
    if message is None:
        raise NotFoundError("This unsubscribe link is not valid")
    return message


async def _active_enrollments_for(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> list[SequenceEnrollment]:
    result = await session.execute(
        scoped_select(SequenceEnrollment, organization_id).where(
            SequenceEnrollment.candidate_id == candidate_id,
            SequenceEnrollment.status.not_in(tuple(TERMINAL_ENROLLMENT_STATUSES)),
        )
    )
    return list(result.scalars().all())


async def unsubscribe_by_token(
    session: AsyncSession,
    token: str,
    *,
    now: datetime | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    source: str = "unsubscribe_link",
) -> UnsubscribeResult:
    """Act on a candidate clicking unsubscribe in one message.

    **The opt-out is per candidate, not per sequence.** Someone who asks to
    stop hearing from us means all of it. Honouring only the sequence the link
    came from would keep a candidate enrolled in two campaigns hearing from the
    other one, which is exactly the experience that earns a spam report.

    So the durable act is withdrawing ``EMAIL_COMMUNICATION`` consent, written
    as a new row because consent records are immutable. That alone stops future
    sends, since the dispatcher re-reads consent at send time — including for
    sequences this candidate has not been enrolled in yet. Halting the live
    enrollments on top of it is what makes the recruiter's pipeline view honest
    straight away rather than at each one's next send attempt.

    Withdrawal is scoped to email. It is not a blacklisting: a candidate who
    wants no more campaign mail has not asked the recruiter never to phone them
    about the role they applied for, and quietly widening it would destroy
    information the candidate never chose to give up.

    Idempotent, because a mail client that does not see a response will POST
    again, and because people click twice.
    """
    # Imported here: candidate_service imports this module for enrolment
    # filtering, so a module-level import would close the cycle.
    from app.services import candidate_service

    stamp = _now(now)
    message = await message_by_tracking_token(session, token)
    organization_id = message.organization_id
    candidate_id = message.candidate_id

    already = not await candidate_service.has_consent(
        session, organization_id, candidate_id, ConsentType.EMAIL_COMMUNICATION
    )
    if not already:
        session.add(
            CandidateConsent(
                organization_id=organization_id,
                candidate_id=candidate_id,
                consent_type=ConsentType.EMAIL_COMMUNICATION,
                status=ConsentStatus.WITHDRAWN,
                granted_at=stamp,
                withdrawn_at=stamp,
                source=source,
                ip_address=ip_address,
                user_agent=user_agent,
                evidence_json={"message_id": str(message.id)},
            )
        )

    live = await _active_enrollments_for(session, organization_id, candidate_id)
    for enrollment in live:
        halt(
            enrollment,
            EnrollmentStatus.UNSUBSCRIBED,
            reason="Candidate unsubscribed",
            now=stamp,
        )
        sequence = await get_scoped(
            session, OutreachSequence, enrollment.sequence_id, organization_id
        )
        if sequence is not None:
            bump_stats(sequence, "unsubscribed")

    organization = await session.get(Organization, organization_id)
    await session.commit()

    logger.info(
        "Candidate %s unsubscribed from email; %d enrollment(s) stopped",
        candidate_id,
        len(live),
    )
    return UnsubscribeResult(
        candidate_id=candidate_id,
        organization_id=organization_id,
        organization_name=organization.name if organization else None,
        enrollment_id=message.enrollment_id,
        stopped=len(live),
        already_unsubscribed=already,
    )


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def bump_stats(sequence: OutreachSequence, key: str, amount: int = 1) -> dict:
    """Increment one of a sequence's rolling counters.

    Reassigns the dict rather than mutating in place: a JSON column tracked by
    SQLAlchemy does not notice an in-place edit, and a counter that silently
    fails to persist is worse than no counter.
    """
    stats = dict(sequence.stats_json or {})
    stats[key] = int(stats.get(key, 0)) + amount
    sequence.stats_json = stats
    return stats


async def sequence_stats(
    session: AsyncSession, organization_id: uuid.UUID, sequence_id: uuid.UUID
) -> dict:
    """Live counts for one sequence, computed rather than trusted.

    The rolling counters on the row are cheap to read but drift whenever a send
    fails midway; these come from the message and enrollment tables, which is
    what a recruiter reporting on a campaign needs.
    """
    sequence = await get_sequence(session, organization_id, sequence_id)

    enrollment_rows = await session.execute(
        select(SequenceEnrollment.status, func.count())
        .where(
            SequenceEnrollment.organization_id == organization_id,
            SequenceEnrollment.sequence_id == sequence_id,
            SequenceEnrollment.deleted_at.is_(None),
        )
        .group_by(SequenceEnrollment.status)
    )
    by_status = {str(status): int(count) for status, count in enrollment_rows}

    message_rows = await session.execute(
        select(OutreachMessage.status, func.count())
        .join(
            SequenceEnrollment,
            SequenceEnrollment.id == OutreachMessage.enrollment_id,
        )
        .where(
            OutreachMessage.organization_id == organization_id,
            SequenceEnrollment.sequence_id == sequence_id,
            OutreachMessage.deleted_at.is_(None),
        )
        .group_by(OutreachMessage.status)
    )
    messages = {str(status): int(count) for status, count in message_rows}

    # Anything that left the box counts as sent, whatever happened to it after.
    delivered_or_later = (
        MessageStatus.SENT,
        MessageStatus.DELIVERED,
        MessageStatus.OPENED,
        MessageStatus.CLICKED,
        MessageStatus.REPLIED,
    )
    return {
        "sequence_id": str(sequence_id),
        "status": str(sequence.status),
        "enrollments": by_status,
        "enrollments_total": sum(by_status.values()),
        "messages": messages,
        "sent": sum(messages.get(str(s), 0) for s in delivered_or_later),
        "bounced": messages.get(str(MessageStatus.BOUNCED), 0),
        "failed": messages.get(str(MessageStatus.FAILED), 0),
        "queued": messages.get(str(MessageStatus.QUEUED), 0),
    }
