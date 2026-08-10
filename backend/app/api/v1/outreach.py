"""Outreach routes.

``router`` is the recruiter side: JWT-authenticated management of sequences,
their steps, and candidate enrollment (design §4.2, §6.1).

``public_router`` is the candidate side of outreach: unauthenticated, because
the only credential anyone holding it has is the token in the path. It mirrors
the booking routes — the API answers JSON at ``{api}/outreach/unsubscribe/…``
and the frontend renders a page at ``{public_base_url}/outreach/unsubscribe/…``
that calls it.

The one difference from booking is that a mail client, not a person, is the
primary caller. RFC 8058 one-click means Gmail POSTs to this path directly on
the candidate's behalf, so the POST has to work with no session, no page load,
no CSRF token, and no body.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, client_ip, get_session, require_permission
from app.core.errors import AppError, NotFoundError
from app.db.tenancy import get_scoped
from app.models.enums import (
    ConsentType,
    EnrollmentStatus,
    OutreachChannel,
    SequenceStatus,
)
from app.models.organization import Organization
from app.models.outreach import OutreachSequence, SequenceEnrollment
from app.schemas.common import MessageResponse
from app.schemas.outreach import UnsubscribeResponse, UnsubscribeView, mask_email
from app.schemas.outreach_sequence import (
    EnrollmentOut,
    EnrollmentPauseRequest,
    EnrollmentReportOut,
    EnrollmentSkip,
    EnrollRequest,
    MessageTemplateCreate,
    MessageTemplateOut,
    MessageTemplateUpdate,
    SequenceCreate,
    SequenceOut,
    SequenceStatsOut,
    SequenceUpdate,
    StepCreate,
    StepOut,
    StepReorder,
    StepUpdate,
)
from app.services import candidate_service, outreach_service, template_service

router = APIRouter(prefix="/outreach", tags=["outreach"])
public_router = APIRouter(prefix="/outreach", tags=["outreach"])


# --------------------------------------------------------------------------- #
# Message templates
# --------------------------------------------------------------------------- #
@router.post(
    "/templates", response_model=MessageTemplateOut, status_code=status.HTTP_201_CREATED
)
async def create_template(
    payload: MessageTemplateCreate,
    current: CurrentUser = Depends(require_permission("outreach:create")),
    session: AsyncSession = Depends(get_session),
) -> MessageTemplateOut:
    try:
        template = await template_service.create_template(
            session, current.organization_id, **payload.model_dump()
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageTemplateOut.model_validate(template)


@router.get("/templates", response_model=list[MessageTemplateOut])
async def list_templates(
    channel: OutreachChannel | None = Query(None),
    active_only: bool = Query(False),
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> list[MessageTemplateOut]:
    templates = await template_service.list_templates(
        session,
        current.organization_id,
        channel=channel,
        active_only=active_only,
    )
    return [MessageTemplateOut.model_validate(t) for t in templates]


@router.get("/templates/{template_id}", response_model=MessageTemplateOut)
async def get_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> MessageTemplateOut:
    try:
        template = await template_service.get_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageTemplateOut.model_validate(template)


@router.patch("/templates/{template_id}", response_model=MessageTemplateOut)
async def update_template(
    template_id: uuid.UUID,
    payload: MessageTemplateUpdate,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageTemplateOut:
    try:
        template = await template_service.update_template(
            session,
            current.organization_id,
            template_id,
            changes=payload.model_dump(exclude_unset=True),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageTemplateOut.model_validate(template)


@router.delete("/templates/{template_id}", response_model=MessageResponse)
async def delete_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await template_service.delete_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Template deleted")


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
@router.post(
    "/sequences", response_model=SequenceOut, status_code=status.HTTP_201_CREATED
)
async def create_sequence(
    payload: SequenceCreate,
    current: CurrentUser = Depends(require_permission("outreach:create")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.create_sequence(
            session, current.organization_id, **payload.model_dump()
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.get("/sequences", response_model=list[SequenceOut])
async def list_sequences(
    status_filter: SequenceStatus | None = Query(None, alias="status"),
    job_id: uuid.UUID | None = Query(None),
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> list[SequenceOut]:
    sequences = await outreach_service.list_sequences(
        session, current.organization_id, status=status_filter, job_id=job_id
    )
    return [SequenceOut.model_validate(s) for s in sequences]


@router.get("/sequences/{sequence_id}", response_model=SequenceOut)
async def get_sequence(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.get_sequence(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.patch("/sequences/{sequence_id}", response_model=SequenceOut)
async def update_sequence(
    sequence_id: uuid.UUID,
    payload: SequenceUpdate,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.update_sequence(
            session,
            current.organization_id,
            sequence_id,
            changes=payload.model_dump(exclude_unset=True),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.post("/sequences/{sequence_id}/activate", response_model=SequenceOut)
async def activate_sequence(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.activate(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.post("/sequences/{sequence_id}/pause", response_model=SequenceOut)
async def pause_sequence(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.pause(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.post("/sequences/{sequence_id}/complete", response_model=SequenceOut)
async def complete_sequence(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> SequenceOut:
    try:
        sequence = await outreach_service.complete_sequence(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceOut.model_validate(sequence)


@router.delete("/sequences/{sequence_id}", response_model=MessageResponse)
async def delete_sequence(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Archive a sequence. Enrollments already in flight are untouched."""
    try:
        await outreach_service.delete_sequence(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Sequence archived")


@router.get("/sequences/{sequence_id}/stats", response_model=SequenceStatsOut)
async def get_sequence_stats(
    sequence_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> SequenceStatsOut:
    try:
        stats = await outreach_service.sequence_stats(
            session, current.organization_id, sequence_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return SequenceStatsOut.model_validate(stats)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
@router.post(
    "/sequences/{sequence_id}/steps",
    response_model=StepOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_step(
    sequence_id: uuid.UUID,
    payload: StepCreate,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> StepOut:
    try:
        step = await outreach_service.add_step(
            session, current.organization_id, sequence_id, **payload.model_dump()
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return StepOut.model_validate(step)


@router.patch("/sequences/{sequence_id}/steps/{step_id}", response_model=StepOut)
async def update_step(
    sequence_id: uuid.UUID,
    step_id: uuid.UUID,
    payload: StepUpdate,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> StepOut:
    try:
        step = await outreach_service.get_step(session, current.organization_id, step_id)
        if step.sequence_id != sequence_id:
            raise NotFoundError("Step not found in this sequence")
        step = await outreach_service.update_step(
            session,
            current.organization_id,
            step_id,
            changes=payload.model_dump(exclude_unset=True),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return StepOut.model_validate(step)


@router.delete(
    "/sequences/{sequence_id}/steps/{step_id}", response_model=MessageResponse
)
async def delete_step(
    sequence_id: uuid.UUID,
    step_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        step = await outreach_service.get_step(session, current.organization_id, step_id)
        if step.sequence_id != sequence_id:
            raise NotFoundError("Step not found in this sequence")
        await outreach_service.delete_step(session, current.organization_id, step_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Step removed")


@router.post("/sequences/{sequence_id}/steps/reorder", response_model=list[StepOut])
async def reorder_steps(
    sequence_id: uuid.UUID,
    payload: StepReorder,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> list[StepOut]:
    try:
        steps = await outreach_service.reorder_steps(
            session, current.organization_id, sequence_id, payload.step_ids
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return [StepOut.model_validate(s) for s in steps]


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
@router.post(
    "/sequences/{sequence_id}/enroll",
    response_model=EnrollmentReportOut,
    status_code=status.HTTP_201_CREATED,
)
async def enroll_candidates(
    sequence_id: uuid.UUID,
    payload: EnrollRequest,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> EnrollmentReportOut:
    """Enroll candidates. Ineligible ones are skipped, not rejected wholesale."""
    try:
        report = await outreach_service.enroll(
            session, current.organization_id, sequence_id, payload.candidate_ids
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return EnrollmentReportOut(
        enrolled=[EnrollmentOut.model_validate(e) for e in report.enrolled],
        enrolled_count=report.enrolled_count,
        skipped=[EnrollmentSkip.model_validate(s) for s in report.skipped],
        skipped_count=report.skipped_count,
    )


@router.get("/sequences/{sequence_id}/enrollments", response_model=list[EnrollmentOut])
async def list_sequence_enrollments(
    sequence_id: uuid.UUID,
    enrollment_status: EnrollmentStatus | None = Query(None, alias="status"),
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> list[EnrollmentOut]:
    enrollments = await outreach_service.list_enrollments(
        session,
        current.organization_id,
        sequence_id=sequence_id,
        status=enrollment_status,
    )
    return [EnrollmentOut.model_validate(e) for e in enrollments]


@router.get("/enrollments/{enrollment_id}", response_model=EnrollmentOut)
async def get_enrollment(
    enrollment_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:read")),
    session: AsyncSession = Depends(get_session),
) -> EnrollmentOut:
    try:
        enrollment = await outreach_service.get_enrollment(
            session, current.organization_id, enrollment_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return EnrollmentOut.model_validate(enrollment)


@router.post("/enrollments/{enrollment_id}/pause", response_model=EnrollmentOut)
async def pause_enrollment(
    enrollment_id: uuid.UUID,
    payload: EnrollmentPauseRequest,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> EnrollmentOut:
    try:
        enrollment = await outreach_service.pause_enrollment(
            session, current.organization_id, enrollment_id, reason=payload.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return EnrollmentOut.model_validate(enrollment)


@router.post("/enrollments/{enrollment_id}/resume", response_model=EnrollmentOut)
async def resume_enrollment(
    enrollment_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("outreach:update")),
    session: AsyncSession = Depends(get_session),
) -> EnrollmentOut:
    try:
        enrollment = await outreach_service.resume_enrollment(
            session, current.organization_id, enrollment_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return EnrollmentOut.model_validate(enrollment)


@public_router.get("/unsubscribe/{token}", response_model=UnsubscribeView)
async def view_unsubscribe(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> UnsubscribeView:
    """Describe the subscription this link would end.

    Read-only on purpose. Some mail clients and security scanners prefetch
    every link in a message, so a GET that opted the candidate out would
    unsubscribe people who never clicked — and unlike a missed opt-out, that
    one is invisible to everybody until a recruiter asks where their pipeline
    went. The frontend page POSTs when the candidate confirms.
    """
    try:
        message = await outreach_service.message_by_tracking_token(session, token)
    except AppError as exc:
        raise exc.to_http() from exc

    organization = await session.get(Organization, message.organization_id)
    sequence = None
    if message.enrollment_id:
        enrollment = await get_scoped(
            session,
            SequenceEnrollment,
            message.enrollment_id,
            message.organization_id,
        )
        if enrollment is not None:
            sequence = await get_scoped(
                session,
                OutreachSequence,
                enrollment.sequence_id,
                message.organization_id,
            )

    # Asked of consent rather than of the enrollment: the enrollment may have
    # ended for an unrelated reason, and what this page is reporting is whether
    # we would still email them.
    subscribed = await candidate_service.has_consent(
        session,
        message.organization_id,
        message.candidate_id,
        ConsentType.EMAIL_COMMUNICATION,
    )

    return UnsubscribeView(
        organization_name=organization.name if organization else None,
        recipient=mask_email(message.to_address),
        sequence_name=sequence.name if sequence else None,
        already_unsubscribed=not subscribed,
    )


@public_router.post("/unsubscribe/{token}", response_model=UnsubscribeResponse)
async def confirm_unsubscribe(
    token: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> UnsubscribeResponse:
    """Record the opt-out. Stops email for this candidate, not just this campaign.

    Takes no body. A one-click client sends ``List-Unsubscribe=One-Click`` as a
    form field and nothing else, so requiring anything of the payload would
    make the header's own promise fail.
    """
    try:
        result = await outreach_service.unsubscribe_by_token(
            session,
            token,
            ip_address=client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return UnsubscribeResponse(
        organization_name=result.organization_name,
        already_unsubscribed=result.already_unsubscribed,
        sequences_stopped=result.stopped,
    )
