"""Offer letters, with approval and self-serve e-signature (design §4.7).

A recruiter drafts an offer against one application — from a reusable letter
template or a one-off body — and it moves through three stages before a
candidate ever sees it: **drafted**, **submitted for approval**, and
**approved**. Only an approved offer can be sent, which is what makes
``offer:approve`` a distinct permission from ``offer:create`` — the split
exists so one person's draft needs another person's sign-off before it
becomes a commitment to pay someone money.

**The letter is rendered once, at send time is too late.** ``generate``
renders the template against the application's candidate, job, and
compensation immediately, the same way ``assessment_service`` snapshots a
paper at issue time: a template edited after an offer went out must not
retroactively change what the candidate was told. An unresolved
``{{variable}}`` is a hard stop here for the same reason ``template_service``
treats it as one everywhere else — it is worse for a candidate to receive a
broken placeholder in their compensation letter than for the recruiter to see
a validation error first.

**Acceptance is real without a provider configured.** ``esign_provider``,
``esign_envelope_id``, and ``esign_status`` exist for a DocuSign or Digio
integration this build does not wire up. Absent one, a candidate accepts by
typing their name, which is recorded in ``signed_by_name`` — an honest
self-serve confirmation, not a simulation of a provider that is not there.

**A decline never rejects the application.** Same principle as a failed
assessment: the machine records what the candidate said and stops. Only an
acceptance moves the pipeline card, to ``hired``.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import count_scoped, get_scoped, scoped_select
from app.models.application import Application
from app.models.candidate import Candidate
from app.models.enums import (
    STAGE_INDEX,
    ApplicationStatus,
    OfferStatus,
    PipelineStage,
)
from app.models.job import Job
from app.models.offer import OfferLetter, OfferTemplate
from app.models.organization import Organization
from app.schemas.common import PaginationParams
from app.services import application_service, template_service

logger = logging.getLogger(__name__)

# Statuses in play before a candidate has been sent a link.
RECRUITER_STATUSES = frozenset(
    {OfferStatus.DRAFT, OfferStatus.PENDING_APPROVAL, OfferStatus.APPROVED}
)
# Statuses the candidate's link still works for.
CANDIDATE_OPEN_STATUSES = frozenset({OfferStatus.SENT, OfferStatus.VIEWED})
# End of the line for a row — a new offer may be issued once one is here.
TERMINAL_STATUSES = frozenset(
    {
        OfferStatus.ACCEPTED,
        OfferStatus.DECLINED,
        OfferStatus.EXPIRED,
        OfferStatus.WITHDRAWN,
    }
)

CLOSED_APPLICATION_STATUSES = frozenset(
    {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}
)

MAX_BODY_CHARS = 20000


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def _validate_template_text(subject: str | None, body: str) -> None:
    if not (body or "").strip():
        raise ValidationError("A template needs a body")
    for part, label in ((subject, "subject"), (body, "body")):
        if template_service.has_malformed_placeholder(part):
            raise ValidationError(
                f"The {label} has an unclosed or malformed {{{{placeholder}}}}",
                details={"part": label},
            )


def _derive_variables(*parts: str | None) -> list[str]:
    seen: dict[str, None] = {}
    for part in parts:
        for name in template_service.find_variables(part):
            seen.setdefault(name, None)
    return list(seen)


async def create_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    name: str,
    body: str,
    subject: str | None = None,
    header_html: str | None = None,
    footer_html: str | None = None,
    is_default: bool = False,
) -> OfferTemplate:
    label = (name or "").strip()
    if not label:
        raise ValidationError("A template needs a name")
    _validate_template_text(subject, body)

    template = OfferTemplate(
        organization_id=organization_id,
        name=label,
        subject=subject,
        body=body,
        header_html=header_html,
        footer_html=footer_html,
        is_default=is_default,
        variables=_derive_variables(subject, body, header_html, footer_html),
    )
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def get_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> OfferTemplate:
    template = await get_scoped(session, OfferTemplate, template_id, organization_id)
    if template is None:
        raise NotFoundError("Offer template not found")
    return template


async def list_templates(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    is_active: bool | None = None,
    params: PaginationParams | None = None,
) -> tuple[list[OfferTemplate], int]:
    params = params or PaginationParams()
    filters: list[ColumnElement[bool]] = []
    if is_active is not None:
        filters.append(OfferTemplate.is_active.is_(is_active))

    stmt = scoped_select(OfferTemplate, organization_id)
    for condition in filters:
        stmt = stmt.where(condition)

    total = await count_scoped(session, OfferTemplate, organization_id, *filters)
    rows = await session.execute(
        stmt.order_by(OfferTemplate.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    return list(rows.scalars().all()), total


async def update_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    template_id: uuid.UUID,
    **changes: object,
) -> OfferTemplate:
    template = await get_template(session, organization_id, template_id)
    for attribute in ("name", "subject", "body", "header_html", "footer_html"):
        if attribute in changes and changes[attribute] is not None:
            setattr(template, attribute, changes[attribute])
    for flag in ("is_active", "is_default"):
        if flag in changes and changes[flag] is not None:
            setattr(template, flag, changes[flag])

    _validate_template_text(template.subject, template.body)
    template.variables = _derive_variables(
        template.subject, template.body, template.header_html, template.footer_html
    )
    await session.commit()
    await session.refresh(template)
    return template


async def delete_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> None:
    template = await get_template(session, organization_id, template_id)
    template.soft_delete()
    await session.commit()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _format_money(amount: float | None, currency: str) -> str | None:
    if amount is None:
        return None
    return f"{currency} {amount:,.2f}"


def _format_date(value: date | None) -> str | None:
    return value.strftime("%B %d, %Y") if value else None


def _format_benefits(benefits: dict | None) -> str | None:
    if not benefits:
        return None
    parts = []
    for key, value in benefits.items():
        label = str(key).replace("_", " ").strip()
        parts.append(f"{label}: {value}" if value not in (None, "", True) else label)
    return ", ".join(parts) or None


async def _load_offer_context(
    session: AsyncSession, organization_id: uuid.UUID, application: Application
) -> dict:
    organization = await session.get(Organization, organization_id)
    candidate = await get_scoped(
        session, Candidate, application.candidate_id, organization_id
    )
    job = await get_scoped(session, Job, application.job_id, organization_id)
    return template_service.build_context(
        candidate=candidate, job=job, organization=organization
    )


def _render_letter(
    *,
    subject: str | None,
    body: str,
    header_html: str | None,
    footer_html: str | None,
    context: dict,
) -> tuple[str, list[str]]:
    """Render a letter to one HTML/text blob, and the variables it could not fill.

    Header and footer are optional and rendered against the same context as
    the body, so a letterhead that references ``{{company_name}}`` works too.
    """
    rendered = template_service.render_message(
        subject=subject, body=body, body_html=None, context=context
    )
    header = template_service.render(header_html, context) if header_html else None
    footer = template_service.render(footer_html, context) if footer_html else None

    missing = list(rendered.missing)
    for part in (header, footer):
        if part is not None:
            for name in part.missing:
                if name not in missing:
                    missing.append(name)

    pieces = [p for p in (header.text if header else None, rendered.body) if p]
    if footer:
        pieces.append(footer.text)
    return "\n\n".join(pieces)[:MAX_BODY_CHARS], missing


# --------------------------------------------------------------------------- #
# Issuing (draft)
# --------------------------------------------------------------------------- #
async def generate(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    template_id: uuid.UUID | None = None,
    subject: str | None = None,
    body: str | None = None,
    job_title: str | None = None,
    salary_amount: float,
    salary_currency: str = "USD",
    salary_period: str = "annual",
    bonus_amount: float | None = None,
    equity: str | None = None,
    benefits: dict | None = None,
    start_date: date | None = None,
    expiry_date: date | None = None,
    reporting_manager: str | None = None,
    work_location: str | None = None,
    created_by_id: uuid.UUID | None = None,
) -> OfferLetter:
    """Draft an offer against one application. Nothing is sent yet."""
    application = await application_service.get_application(
        session, organization_id, application_id
    )
    if application.status in CLOSED_APPLICATION_STATUSES:
        raise ConflictError(
            f"Application is {application.status} and cannot be offered",
            details={"status": application.status},
        )

    outstanding = await session.scalar(
        scoped_select(OfferLetter, organization_id).where(
            OfferLetter.application_id == application_id,
            OfferLetter.status.notin_(sorted(TERMINAL_STATUSES)),
        )
    )
    if outstanding is not None:
        raise ConflictError(
            "This application already has an offer outstanding",
            details={"offer_id": str(outstanding.id), "status": outstanding.status},
        )

    template: OfferTemplate | None = None
    if template_id is not None:
        template = await get_template(session, organization_id, template_id)
        if not template.is_active:
            raise ConflictError(f"Offer template '{template.name}' is not active")
    elif not body:
        raise ValidationError("An offer needs either a template_id or a body")

    if salary_amount is None or salary_amount <= 0:
        raise ValidationError("An offer needs a salary_amount above zero")

    job = await get_scoped(session, Job, application.job_id, organization_id)
    resolved_title = job_title or (job.title if job else None)
    if not resolved_title:
        raise ValidationError("An offer needs a job_title")

    context = await _load_offer_context(session, organization_id, application)
    context.update(
        {
            "job_title": resolved_title,
            "salary": _format_money(salary_amount, salary_currency),
            "salary_amount": salary_amount,
            "salary_currency": salary_currency,
            "salary_period": salary_period,
            "bonus_amount": _format_money(bonus_amount, salary_currency),
            "equity": equity,
            "benefits": _format_benefits(benefits),
            "start_date": _format_date(start_date),
            "expiry_date": _format_date(expiry_date),
            "reporting_manager": reporting_manager,
            "work_location": work_location,
        }
    )

    # ``template`` and ``body`` are mutually guaranteed above: a missing
    # template_id took the ``elif not body`` branch, so exactly one is set.
    effective_body = template.body if template is not None else body or ""
    rendered_body, missing = _render_letter(
        subject=template.subject if template is not None else subject,
        body=effective_body,
        header_html=template.header_html if template is not None else None,
        footer_html=template.footer_html if template is not None else None,
        context=context,
    )
    if missing:
        raise ValidationError(
            "The offer letter references variables with no value",
            details={"missing": missing},
        )

    resolved_expiry = expiry_date or (
        datetime.now(UTC).date()
        + timedelta(days=settings.offer_default_expiry_days)
    )

    offer = OfferLetter(
        organization_id=organization_id,
        application_id=application_id,
        template_id=template.id if template is not None else None,
        status=OfferStatus.DRAFT,
        job_title=resolved_title,
        salary=_format_money(salary_amount, salary_currency),
        salary_amount=salary_amount,
        salary_currency=salary_currency,
        salary_period=salary_period,
        bonus_amount=bonus_amount,
        equity=equity,
        benefits_json=benefits or {},
        start_date=start_date,
        expiry_date=resolved_expiry,
        reporting_manager=reporting_manager,
        work_location=work_location,
        rendered_body=rendered_body,
        created_by_id=created_by_id,
    )
    session.add(offer)
    await session.commit()
    await session.refresh(offer)
    return offer


def _new_access_token() -> str:
    return secrets.token_urlsafe(32)


def invite_url(offer: OfferLetter) -> str | None:
    if not offer.access_token:
        return None
    base = settings.public_base_url.rstrip("/")
    return f"{base}/offer/{offer.access_token}"


# --------------------------------------------------------------------------- #
# Recruiter-side reads and workflow
# --------------------------------------------------------------------------- #
async def get_offer(
    session: AsyncSession, organization_id: uuid.UUID, offer_id: uuid.UUID
) -> OfferLetter:
    offer = await get_scoped(session, OfferLetter, offer_id, organization_id)
    if offer is None:
        raise NotFoundError("Offer not found")
    return offer


async def list_offers(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    application_id: uuid.UUID | None = None,
    status: OfferStatus | None = None,
    params: PaginationParams | None = None,
) -> tuple[list[OfferLetter], int]:
    params = params or PaginationParams()
    filters: list[ColumnElement[bool]] = []
    if application_id is not None:
        filters.append(OfferLetter.application_id == application_id)
    if status is not None:
        filters.append(OfferLetter.status == status)

    stmt = scoped_select(OfferLetter, organization_id)
    for condition in filters:
        stmt = stmt.where(condition)

    total = await count_scoped(session, OfferLetter, organization_id, *filters)
    rows = await session.execute(
        stmt.order_by(OfferLetter.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    return list(rows.scalars().all()), total


async def submit_for_approval(
    session: AsyncSession, organization_id: uuid.UUID, offer_id: uuid.UUID
) -> OfferLetter:
    offer = await get_offer(session, organization_id, offer_id)
    if offer.status != OfferStatus.DRAFT:
        raise ConflictError(
            f"An offer that is {offer.status} cannot be submitted for approval",
            details={"status": offer.status},
        )
    offer.status = OfferStatus.PENDING_APPROVAL
    await session.commit()
    await session.refresh(offer)
    return offer


async def approve(
    session: AsyncSession,
    organization_id: uuid.UUID,
    offer_id: uuid.UUID,
    *,
    approved_by_id: uuid.UUID | None = None,
) -> OfferLetter:
    offer = await get_offer(session, organization_id, offer_id)
    if offer.status != OfferStatus.PENDING_APPROVAL:
        raise ConflictError(
            f"An offer that is {offer.status} is not awaiting approval",
            details={"status": offer.status},
        )
    offer.status = OfferStatus.APPROVED
    offer.approved_by_id = approved_by_id
    offer.approved_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(offer)
    return offer


async def send(
    session: AsyncSession, organization_id: uuid.UUID, offer_id: uuid.UUID
) -> OfferLetter:
    """Mint the candidate's link and put the pipeline card at ``offered``."""
    offer = await get_offer(session, organization_id, offer_id)
    if offer.status != OfferStatus.APPROVED:
        raise ConflictError(
            f"An offer that is {offer.status} cannot be sent — it needs approval first",
            details={"status": offer.status},
        )
    offer.status = OfferStatus.SENT
    offer.access_token = _new_access_token()
    offer.sent_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(offer)

    await _move_stage(
        session, offer, PipelineStage.OFFERED, trigger="offer_sent", note="Offer sent"
    )
    await session.refresh(offer)
    return offer


async def withdraw(
    session: AsyncSession,
    organization_id: uuid.UUID,
    offer_id: uuid.UUID,
    *,
    reason: str | None = None,
) -> OfferLetter:
    offer = await get_offer(session, organization_id, offer_id)
    if offer.status in TERMINAL_STATUSES:
        raise ConflictError(f"An offer that is {offer.status} cannot be withdrawn")

    offer.status = OfferStatus.WITHDRAWN
    offer.access_token = None
    offer.responded_at = datetime.now(UTC)
    if reason:
        offer.decline_reason = reason
    await session.commit()
    await session.refresh(offer)
    return offer


async def _move_stage(
    session: AsyncSession,
    offer: OfferLetter,
    to_stage: PipelineStage,
    *,
    trigger: str,
    note: str,
) -> None:
    application = await session.get(Application, offer.application_id)
    if application is None or application.status in CLOSED_APPLICATION_STATUSES:
        return
    if STAGE_INDEX[PipelineStage(application.stage)] >= STAGE_INDEX[to_stage]:
        return
    await application_service.move_stage(
        session,
        offer.organization_id,
        application.id,
        to_stage,
        trigger=trigger,
        note=note,
        # An offer reaching this point is evidence in its own right; the card
        # should not be stuck behind an earlier stage's screening record.
        force=True,
    )


# --------------------------------------------------------------------------- #
# Candidate self-service (access token is the credential)
# --------------------------------------------------------------------------- #
async def get_by_access_token(session: AsyncSession, token: str) -> OfferLetter:
    if not token:
        raise NotFoundError("This offer link is not valid")

    offer = await session.scalar(
        select(OfferLetter).where(
            OfferLetter.access_token == token,
            OfferLetter.deleted_at.is_(None),
        )
    )
    if offer is None:
        raise NotFoundError("This offer link is not valid")
    if _is_expired(offer):
        raise NotFoundError("This offer link has expired")
    return offer


def _is_expired(offer: OfferLetter, *, today: date | None = None) -> bool:
    if offer.status == OfferStatus.EXPIRED:
        return True
    if offer.status not in CANDIDATE_OPEN_STATUSES:
        return False
    if offer.expiry_date is None:
        return False
    return offer.expiry_date < (today or datetime.now(UTC).date())


async def record_view(session: AsyncSession, token: str) -> OfferLetter:
    """Mark the offer viewed. Idempotent — reopening the link is not a new event."""
    offer = await get_by_access_token(session, token)
    if offer.status == OfferStatus.SENT:
        offer.status = OfferStatus.VIEWED
        offer.viewed_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(offer)
    return offer


async def accept(
    session: AsyncSession, token: str, *, signature_name: str
) -> OfferLetter:
    """Accept the offer via typed self-serve signature, and hire the candidate.

    No e-signature provider is configured in this build, so the typed name
    itself is the confirmation — see the module docstring.
    """
    offer = await get_by_access_token(session, token)
    if offer.status not in CANDIDATE_OPEN_STATUSES:
        raise ConflictError(
            f"This offer is {offer.status} and cannot be accepted",
            details={"status": offer.status},
        )
    name = (signature_name or "").strip()
    if not name:
        raise ValidationError("Type your full name to accept this offer")

    now = datetime.now(UTC)
    offer.status = OfferStatus.ACCEPTED
    offer.signed_by_name = name[:255]
    offer.esign_provider = offer.esign_provider or "self_serve"
    offer.esign_status = "completed"
    offer.signed_at = now
    offer.responded_at = now
    await session.commit()
    await session.refresh(offer)

    await _move_stage(
        session, offer, PipelineStage.HIRED, trigger="offer_accepted", note="Offer accepted"
    )
    await session.refresh(offer)
    return offer


async def decline(
    session: AsyncSession, token: str, *, reason: str | None = None
) -> OfferLetter:
    offer = await get_by_access_token(session, token)
    if offer.status not in CANDIDATE_OPEN_STATUSES:
        raise ConflictError(
            f"This offer is {offer.status} and cannot be declined",
            details={"status": offer.status},
        )
    offer.status = OfferStatus.DECLINED
    offer.decline_reason = reason
    offer.responded_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(offer)
    return offer


# --------------------------------------------------------------------------- #
# Public view assembly
# --------------------------------------------------------------------------- #
@dataclass
class OfferView:
    """What the candidate sees. Unlike an assessment, the compensation itself
    is exactly what the offer is — there is no answer key to withhold."""

    status: OfferStatus
    organization_name: str | None
    job_title: str
    candidate_name: str | None
    salary_amount: float | None
    salary_currency: str
    salary_period: str
    bonus_amount: float | None
    equity: str | None
    benefits_json: dict
    start_date: date | None
    expiry_date: date | None
    reporting_manager: str | None
    work_location: str | None
    rendered_body: str | None
    sent_at: datetime | None
    viewed_at: datetime | None
    responded_at: datetime | None


async def build_view(session: AsyncSession, offer: OfferLetter) -> OfferView:
    organization = await session.get(Organization, offer.organization_id)
    application = await get_scoped(
        session, Application, offer.application_id, offer.organization_id
    )
    candidate = None
    if application is not None:
        candidate = await get_scoped(
            session, Candidate, application.candidate_id, offer.organization_id
        )
    return OfferView(
        status=OfferStatus(offer.status),
        organization_name=organization.name if organization else None,
        job_title=offer.job_title,
        candidate_name=candidate.full_name if candidate else None,
        salary_amount=offer.salary_amount,
        salary_currency=offer.salary_currency,
        salary_period=offer.salary_period,
        bonus_amount=offer.bonus_amount,
        equity=offer.equity,
        benefits_json=offer.benefits_json or {},
        start_date=offer.start_date,
        expiry_date=offer.expiry_date,
        reporting_manager=offer.reporting_manager,
        work_location=offer.work_location,
        rendered_body=offer.rendered_body,
        sent_at=offer.sent_at,
        viewed_at=offer.viewed_at,
        responded_at=offer.responded_at,
    )


# --------------------------------------------------------------------------- #
# Expiry sweep
# --------------------------------------------------------------------------- #
async def expire_due(
    session: AsyncSession,
    *,
    today: date | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 500,
) -> list[OfferLetter]:
    """Close out offers whose deadline has passed without a response."""
    moment = today or datetime.now(UTC).date()
    stmt = select(OfferLetter).where(
        OfferLetter.deleted_at.is_(None),
        OfferLetter.status.in_(sorted(CANDIDATE_OPEN_STATUSES)),
        OfferLetter.expiry_date.is_not(None),
        OfferLetter.expiry_date < moment,
    )
    if organization_id is not None:
        stmt = stmt.where(OfferLetter.organization_id == organization_id)

    rows = list(
        (await session.execute(stmt.order_by(OfferLetter.expiry_date).limit(limit)))
        .scalars()
        .all()
    )
    for offer in rows:
        offer.status = OfferStatus.EXPIRED
        offer.access_token = None
    if rows:
        await session.commit()
        logger.info("Expired %d offer(s)", len(rows))
    return rows
