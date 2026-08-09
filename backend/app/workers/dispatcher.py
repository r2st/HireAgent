"""The outreach send loop: take what is due, render it, send it, move on.

This is where the sequence engine, the template renderer, the sender pool, and
the email gateway finally meet. Everything it does is shaped by one asymmetry:
a message that goes out late costs a little, and a message that goes out wrong
— to someone who withdrew consent, twice, or with ``Hi {{first_name}}`` in the
subject — costs the domain's reputation and the candidate's trust, neither of
which can be bought back.

**Consent is re-checked at send time, not trusted from enrolment.** A sequence
runs for weeks; the guard that ran when the recruiter clicked "enrol" says
nothing about today. This is the difference between a compliance story and a
compliance claim.

**The message row is written before the send, and reused after a failure.**
A crash between writing and sending leaves a queued row that the next pass
picks up; the reverse order would leave a candidate emailed with nothing to
show for it, and the pass after that would email them again.

**An unresolved placeholder stops the send.** ``render_message`` reports what
it could not fill rather than blanking it, and a non-empty ``missing`` is a
hard stop here — half-filled outreach identifies the sender as a bulk mailer to
exactly the person they were trying to impress.

**A failure that is ours pauses; a failure that is theirs ends.** A bounce is
terminal, because every later step would bounce too and spend reputation doing
it. A broken sender or a broken template pauses the enrollment with the reason
attached, so it is resumable once a human fixes the cause — failing it outright
would punish the candidate for our configuration error.
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
from app.db.tenancy import get_scoped, scoped_select
from app.integrations import email_gateway
from app.models.candidate import Candidate
from app.models.enums import (
    EnrollmentStatus,
    MessageStatus,
    OutreachChannel,
)
from app.models.job import Job
from app.models.organization import Organization
from app.models.outreach import (
    EmailAccount,
    MessageTemplate,
    OutreachMessage,
    OutreachSequence,
    SequenceEnrollment,
    SequenceStep,
)
from app.services import candidate_service, outreach_service, sender_service
from app.services import template_service as templates
from app.services.availability import load_zone, to_utc

logger = logging.getLogger(__name__)

# Channels this dispatcher can actually deliver. A sequence may legitimately
# mix in others; see _dispatch for why those advance rather than stall.
SENDABLE_CHANNELS = frozenset({OutreachChannel.EMAIL})

# How long to wait before retrying a send that failed for a transient reason.
RETRY_BACKOFF = timedelta(minutes=15)

# How long a capped sequence waits before looking again. A cap rolls at local
# midnight, so anything shorter just re-checks a decision that cannot change.
CAP_BACKOFF = timedelta(hours=1)


@dataclass
class SendOutcome:
    """What the dispatcher did with one enrollment, and why.

    ``outcome`` is a stable token meant for counting and for tests; ``detail``
    is the human sentence that ends up on the enrollment or in the log.
    """

    enrollment_id: uuid.UUID
    outcome: str
    detail: str | None = None
    message_id: uuid.UUID | None = None

    @property
    def sent(self) -> bool:
        return self.outcome == "sent"


@dataclass
class DispatchReport:
    """The tally for one pass, for the worker log and the admin endpoint."""

    considered: int = 0
    outcomes: list[SendOutcome] = field(default_factory=list)

    @property
    def sent(self) -> int:
        return sum(1 for o in self.outcomes if o.sent)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for outcome in self.outcomes:
            tally[outcome.outcome] = tally.get(outcome.outcome, 0) + 1
        return tally

    def to_dict(self) -> dict:
        return {
            "considered": self.considered,
            "sent": self.sent,
            "outcomes": self.counts(),
        }


def _now(value: datetime | None = None) -> datetime:
    return to_utc(value) if value is not None else datetime.now(UTC)


def _new_tracking_token() -> str:
    # 24 bytes of urlsafe base64 is 32 characters, inside the column's 64.
    return secrets.token_urlsafe(24)


def unsubscribe_url(token: str | None) -> str | None:
    """The one-click opt-out link that goes in every outbound message."""
    if not token:
        return None
    return f"{settings.public_base_url.rstrip('/')}/outreach/unsubscribe/{token}"


# --------------------------------------------------------------------------- #
# Daily caps
# --------------------------------------------------------------------------- #
def _local_day_start(sequence: OutreachSequence, now: datetime) -> datetime:
    """Midnight today in the sequence's own timezone, as UTC.

    The cap rolls with the recruiter's day for the same reason the send window
    is wall-clock: a campaign capped at fifty a day should not reset in the
    middle of their afternoon because the server keeps UTC.
    """
    zone = load_zone(sequence.timezone)
    local = to_utc(now).astimezone(zone)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


async def sent_today(
    session: AsyncSession, sequence: OutreachSequence, *, now: datetime | None = None
) -> int:
    """How many messages this sequence put on the wire during its local day.

    Counted from ``sent_at`` rather than ``created_at``: row creation is
    stamped by the database, so a cap keyed on it could not be reasoned about
    against an injected clock, and a row that was written but never sent — a
    retry still in flight, a dry run under the kill switch — has not spent any
    of the allowance the cap exists to ration.

    A bounce does count, because it reached the provider and cost the domain
    exactly what a delivery would have. That is the same call ``record_send``
    already makes for the mailbox's own daily allowance.
    """
    since = _local_day_start(sequence, _now(now))
    total = await session.scalar(
        select(func.count())
        .select_from(OutreachMessage)
        .join(
            SequenceEnrollment,
            SequenceEnrollment.id == OutreachMessage.enrollment_id,
        )
        .where(
            SequenceEnrollment.sequence_id == sequence.id,
            OutreachMessage.deleted_at.is_(None),
            OutreachMessage.sent_at.is_not(None),
            OutreachMessage.sent_at >= since,
        )
    )
    return int(total or 0)


# --------------------------------------------------------------------------- #
# Assembling one message
# --------------------------------------------------------------------------- #
@dataclass
class SendTargets:
    """Everything one send needs, gathered before anything is written."""

    sequence: OutreachSequence
    step: SequenceStep
    candidate: Candidate
    job: Job | None
    organization: Organization | None
    template: MessageTemplate | None


async def load_targets(
    session: AsyncSession, enrollment: SequenceEnrollment
) -> SendTargets | None:
    """Gather the rows this enrollment's next step renders against.

    Returns ``None`` when the sequence or the step is gone — an enrollment can
    outlive an edit that shortened the sequence under it.
    """
    organization_id = enrollment.organization_id
    sequence = await get_scoped(
        session, OutreachSequence, enrollment.sequence_id, organization_id
    )
    if sequence is None:
        return None
    step = outreach_service.step_at(sequence, enrollment.current_step)
    if step is None:
        return None

    candidate = await get_scoped(
        session, Candidate, enrollment.candidate_id, organization_id
    )
    if candidate is None:
        return None

    job = (
        await get_scoped(session, Job, sequence.job_id, organization_id)
        if sequence.job_id
        else None
    )
    template = (
        await get_scoped(session, MessageTemplate, step.template_id, organization_id)
        if step.template_id
        else None
    )
    organization = await session.get(Organization, organization_id)

    return SendTargets(
        sequence=sequence,
        step=step,
        candidate=candidate,
        job=job,
        organization=organization,
        template=template,
    )


def render_for(targets: SendTargets, sender: EmailAccount | None) -> templates.RenderedMessage:
    """Render the step's content, preferring its overrides over the template.

    A step carries overrides so one sequence can vary a shared template — an
    A/B subject line, a tweaked closing — without forking it.
    """
    subject = targets.step.subject_override or (
        targets.template.subject if targets.template else None
    )
    body = targets.step.body_override or (
        targets.template.body if targets.template else ""
    )
    # An HTML part only comes from the template: an inline override is plain
    # text, and pairing it with the template's HTML would send two different
    # messages in one envelope.
    body_html = (
        targets.template.body_html
        if targets.template and not targets.step.body_override
        else None
    )

    context = templates.build_context(
        candidate=targets.candidate,
        job=targets.job,
        organization=targets.organization,
        sender=sender,
    )
    return templates.render_message(
        subject=subject, body=body, body_html=body_html, context=context
    )


async def _pending_message(
    session: AsyncSession,
    enrollment: SequenceEnrollment,
    step: SequenceStep,
) -> OutreachMessage | None:
    """A queued message already written for this step, if the last pass died.

    Reusing it is what keeps a crash between "row written" and "mail sent" from
    turning into two emails on the next pass.
    """
    return await session.scalar(
        scoped_select(OutreachMessage, enrollment.organization_id)
        .where(
            OutreachMessage.enrollment_id == enrollment.id,
            OutreachMessage.step_id == step.id,
            OutreachMessage.status == MessageStatus.QUEUED,
        )
        .order_by(OutreachMessage.created_at.asc())
        .limit(1)
    )


# --------------------------------------------------------------------------- #
# Dispatching one enrollment
# --------------------------------------------------------------------------- #
def _pause(
    enrollment: SequenceEnrollment, reason: str, now: datetime
) -> None:
    """Stop this enrollment in a state a human can resume once they fix it."""
    outreach_service.halt(enrollment, EnrollmentStatus.PAUSED, reason=reason, now=now)


async def dispatch_one(
    session: AsyncSession,
    enrollment: SequenceEnrollment,
    *,
    now: datetime | None = None,
    cap_room: int | None = None,
) -> SendOutcome:
    """Send this enrollment's due step, or explain why it did not.

    Commits once at the end so the message row, the enrollment's new position,
    and the sender's counters move together: a partial write here would either
    re-send a message or lose the record of one.
    """
    stamp = _now(now)
    organization_id = enrollment.organization_id

    def outcome(name: str, detail: str | None = None, message=None) -> SendOutcome:
        return SendOutcome(
            enrollment_id=enrollment.id,
            outcome=name,
            detail=detail,
            message_id=message.id if message is not None else None,
        )

    targets = await load_targets(session, enrollment)
    if targets is None:
        # The sequence was shortened or the candidate removed underneath a
        # running enrollment; completing it is truer than leaving it due.
        outreach_service.halt(
            enrollment,
            EnrollmentStatus.COMPLETED,
            reason="Nothing left to send",
            now=stamp,
        )
        await session.commit()
        return outcome("nothing_to_send")

    sequence, step, candidate = targets.sequence, targets.step, targets.candidate

    # --- Guards that must hold now, not merely at enrolment ---
    if candidate.is_blacklisted:
        outreach_service.halt(
            enrollment,
            EnrollmentStatus.UNSUBSCRIBED,
            reason="Candidate is blacklisted",
            now=stamp,
        )
        await session.commit()
        return outcome("blacklisted")

    channel = OutreachChannel(step.channel)
    consent = outreach_service.CHANNEL_CONSENT.get(channel)
    if consent is not None and not await candidate_service.has_consent(
        session, organization_id, candidate.id, consent
    ):
        # Withdrawn since enrolment. Ending as UNSUBSCRIBED rather than pausing
        # is deliberate: this one must not be resumable by mistake.
        outreach_service.halt(
            enrollment,
            EnrollmentStatus.UNSUBSCRIBED,
            reason=f"No consent for {consent}",
            now=stamp,
        )
        await session.commit()
        return outcome("no_consent", str(consent))

    if channel not in SENDABLE_CHANNELS:
        # No transport for this channel yet. Advancing keeps the email steps of
        # a mixed sequence flowing instead of stalling the whole thing on one
        # WhatsApp step that nothing can deliver.
        logger.warning(
            "Skipping %s step %s: no transport for that channel", channel, step.id
        )
        outreach_service.advance(enrollment, sequence, now=stamp)
        await session.commit()
        return outcome("unsupported_channel", str(channel))

    address = (candidate.email or "").strip()
    if not address:
        _pause(enrollment, "Candidate has no email address", stamp)
        await session.commit()
        return outcome("no_address")

    if cap_room is not None and cap_room <= 0:
        enrollment.next_send_at = outreach_service.next_send_time(
            stamp + CAP_BACKOFF, sequence
        )
        await session.commit()
        return outcome("capped")

    # --- Pick a sender before rendering: {{sender_name}} is a variable ---
    sender = await sender_service.pick_sender(
        session,
        organization_id,
        allowed_ids=[uuid.UUID(i) for i in (sequence.sender_account_ids or [])],
        now=stamp,
    )
    if sender is None:
        # Every mailbox is capped or blocked. This is routine at the end of a
        # busy day, so it waits rather than failing the enrollment.
        enrollment.next_send_at = outreach_service.next_send_time(
            stamp + CAP_BACKOFF, sequence
        )
        await session.commit()
        return outcome("no_sender")

    rendered = render_for(targets, sender)
    if not rendered.ok:
        _pause(
            enrollment,
            f"Template is missing {', '.join(rendered.missing)}",
            stamp,
        )
        await session.commit()
        return outcome("unrendered", ", ".join(rendered.missing))

    # --- Write the row before sending, or reuse the one a crash left behind ---
    message = await _pending_message(session, enrollment, step)
    if message is None:
        message = OutreachMessage(
            organization_id=organization_id,
            enrollment_id=enrollment.id,
            candidate_id=candidate.id,
            step_id=step.id,
            channel=channel,
            status=MessageStatus.QUEUED,
            to_address=address,
            subject=rendered.subject,
            body=rendered.body,
            variant_group=step.variant_group,
            scheduled_at=enrollment.next_send_at,
            tracking_token=_new_tracking_token(),
        )
        session.add(message)
        await session.flush()
    message.email_account_id = sender.id

    if not settings.outreach_sending_enabled:
        # The kill switch keeps everything except the envelope: the rendered
        # message is recorded so staging can inspect exactly what would have
        # gone out, and the enrollment moves on so the pipeline stays testable.
        message.error = "Outreach sending is disabled"
        outreach_service.advance(enrollment, sequence, now=stamp)
        await session.commit()
        return outcome("suppressed", message.error, message)

    result = await _send(sender, message, rendered, address, candidate)
    sender_service.store_refreshed_token(sender, result.refreshed)

    if result.ok:
        message.status = MessageStatus.SENT
        message.sent_at = stamp
        message.error = None
        message.provider_message_id = result.provider_message_id
        sender_service.record_send(sender, ok=True, now=stamp)
        outreach_service.advance(enrollment, sequence, now=stamp)
        outreach_service.bump_stats(sequence, "sent")
        await session.commit()
        return outcome("sent", message=message)

    message.error = result.error
    sender_service.record_send(
        sender, ok=False, bounced=result.bounced, now=stamp, error=result.error
    )

    if result.bounced:
        # The address is the problem. Every further step would bounce too, and
        # each one would cost the sending domain a little more.
        message.status = MessageStatus.BOUNCED
        # Stamped as sent as well as bounced: it reached the provider, which is
        # what both daily allowances are rationing.
        message.sent_at = stamp
        message.bounced_at = stamp
        outreach_service.halt(
            enrollment,
            EnrollmentStatus.BOUNCED,
            reason=result.error or "Recipient rejected the message",
            now=stamp,
        )
        outreach_service.bump_stats(sequence, "bounced")
        await session.commit()
        return outcome("bounced", result.error, message)

    message.retry_count += 1
    exhausted = message.retry_count >= settings.outreach_max_send_attempts
    if result.retryable and not exhausted:
        # Leave the row QUEUED so the next pass reuses it rather than writing
        # a second one for the same step.
        enrollment.next_send_at = outreach_service.next_send_time(
            stamp + RETRY_BACKOFF, sequence
        )
        await session.commit()
        return outcome("retrying", result.error, message)

    message.status = MessageStatus.FAILED
    _pause(enrollment, result.error or "Sending failed", stamp)
    outreach_service.bump_stats(sequence, "failed")
    await session.commit()
    return outcome("failed", result.error, message)


async def _send(
    sender: EmailAccount,
    message: OutreachMessage,
    rendered: templates.RenderedMessage,
    address: str,
    candidate: Candidate,
) -> email_gateway.SendResult:
    """Hand one rendered message to its transport, converting a crash to a result.

    A transport that raises must look like a retryable failure rather than take
    down the pass: the next enrollment in the queue is unrelated to this one.
    """
    opt_out = unsubscribe_url(message.tracking_token)
    headers = {}
    if opt_out:
        # One-click opt-out is what keeps a sending domain out of the spam
        # folder, and it has to be a header — a link in the footer only helps
        # the candidates who scroll.
        headers["List-Unsubscribe"] = f"<{opt_out}>"
        headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

    outbound = email_gateway.OutboundEmail(
        to_email=address,
        to_name=candidate.full_name,
        subject=rendered.subject,
        text_body=rendered.body,
        html_body=rendered.body_html,
        headers=headers,
    )
    transport = email_gateway.get_transport(str(sender.provider))
    try:
        return await transport.send(sender_service.credentials_for(sender), outbound)
    except Exception as exc:  # noqa: BLE001 - one bad send must not stop the pass
        logger.exception("Transport %s raised sending %s", transport.name, message.id)
        return email_gateway.SendResult(ok=False, error=str(exc), retryable=True)


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
async def run_once(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 200,
) -> DispatchReport:
    """Dispatch every enrollment that is due, oldest first.

    Each one commits on its own, so a failure partway through leaves the sends
    that already happened recorded and the rest to be picked up next pass.
    """
    stamp = _now(now)
    due = await outreach_service.due_enrollments(
        session, now=stamp, organization_id=organization_id, limit=limit
    )
    report = DispatchReport(considered=len(due))

    # A sequence's remaining allowance is counted once and then decremented in
    # memory: re-querying per enrollment would be one round trip per candidate
    # to answer a question that only this loop is changing.
    room: dict[uuid.UUID, int | None] = {}

    for enrollment in due:
        sequence_id = enrollment.sequence_id
        if sequence_id not in room:
            sequence = await get_scoped(
                session, OutreachSequence, sequence_id, enrollment.organization_id
            )
            if sequence is None or sequence.daily_cap is None:
                room[sequence_id] = None
            else:
                room[sequence_id] = max(
                    0, sequence.daily_cap - await sent_today(session, sequence, now=stamp)
                )

        allowance = room[sequence_id]
        result = await dispatch_one(
            session, enrollment, now=stamp, cap_room=allowance
        )
        if allowance is not None and result.outcome not in ("capped",):
            room[sequence_id] = max(0, allowance - 1)
        report.outcomes.append(result)

    if due:
        logger.info(
            "Outreach pass: %d due, %d sent, %s",
            report.considered,
            report.sent,
            report.counts(),
        )
    return report
