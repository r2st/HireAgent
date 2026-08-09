"""Assessment routes (design §4.4).

Two routers, two callers:

* ``router`` — recruiters and hiring managers, JWT-authenticated. Templates,
  issuing, results, and manual grading.
* ``public_router`` — the candidate, holding nothing but an invite token. These
  routes are unauthenticated by design; the token is the credential, and the
  global per-IP rate limit in ``main`` is what bounds guessing.

The public routes never echo the answer key, the marks, or the verdict. A
candidate sees the paper before they sit it and an acknowledgement afterwards —
the result belongs to the recruiter's conversation, not to a JSON response.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_session, require_permission
from app.core.errors import AppError
from app.db.tenancy import get_scoped
from app.models.application import Application
from app.models.assessment import Assessment
from app.models.candidate import Candidate
from app.models.enums import AssessmentStatus, AssessmentType
from app.models.job import Job
from app.models.organization import Organization
from app.schemas.assessment import (
    AssessmentDetail,
    AssessmentIssue,
    AssessmentOut,
    CancelRequest,
    IssuedAssessment,
    ManualGrade,
    PublicAssessmentView,
    SubmitRequest,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
)
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.services import assessment_service

router = APIRouter(prefix="/assessments", tags=["assessments"])
public_router = APIRouter(prefix="/assessment", tags=["assessment"])


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@router.post(
    "/templates", response_model=TemplateOut, status_code=status.HTTP_201_CREATED
)
async def create_template(
    payload: TemplateCreate,
    current: CurrentUser = Depends(require_permission("assessment:create")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    """Author a reusable paper."""
    try:
        template = await assessment_service.create_template(
            session,
            current.organization_id,
            name=payload.name,
            type=payload.type,
            description=payload.description,
            questions=payload.questions,
            duration_minutes=payload.duration_minutes,
            passing_score=payload.passing_score,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.get("/templates", response_model=Page[TemplateOut])
async def list_templates(
    type: AssessmentType | None = Query(None),
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    current: CurrentUser = Depends(require_permission("assessment:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[TemplateOut]:
    params = PaginationParams(page=page, page_size=page_size)
    templates, total = await assessment_service.list_templates(
        session, current.organization_id, type=type, is_active=is_active, params=params
    )
    return Page.build(
        [TemplateOut.model_validate(t) for t in templates], total, params
    )


@router.get("/templates/{template_id}", response_model=TemplateOut)
async def get_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("assessment:read")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    try:
        template = await assessment_service.get_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.patch("/templates/{template_id}", response_model=TemplateOut)
async def update_template(
    template_id: uuid.UUID,
    payload: TemplateUpdate,
    current: CurrentUser = Depends(require_permission("assessment:update")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    """Edit a template. Assessments already issued keep the paper they were sent."""
    try:
        template = await assessment_service.update_template(
            session,
            current.organization_id,
            template_id,
            **payload.model_dump(exclude_unset=True),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.delete("/templates/{template_id}", response_model=MessageResponse)
async def delete_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("assessment:delete")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await assessment_service.delete_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Assessment template deleted")


# --------------------------------------------------------------------------- #
# Issuing and results
# --------------------------------------------------------------------------- #
@router.post("", response_model=IssuedAssessment, status_code=status.HTTP_201_CREATED)
async def issue_assessment(
    payload: AssessmentIssue,
    current: CurrentUser = Depends(require_permission("assessment:create")),
    session: AsyncSession = Depends(get_session),
) -> IssuedAssessment:
    """Issue an assessment and mint the candidate's link.

    The link comes back in the response rather than being emailed here —
    delivery is the outreach engine's job, and a recruiter sending it by hand is
    a legitimate flow.
    """
    try:
        assessment = await assessment_service.issue(
            session,
            current.organization_id,
            payload.application_id,
            template_id=payload.template_id,
            questions=payload.questions,
            type=payload.type,
            duration_minutes=payload.duration_minutes,
            passing_score=payload.passing_score,
            expires_in_hours=payload.expires_in_hours,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return IssuedAssessment(
        assessment=AssessmentOut.model_validate(assessment),
        invite_url=assessment_service.invite_url(assessment),
    )


@router.get("", response_model=Page[AssessmentOut])
async def list_assessments(
    application_id: uuid.UUID | None = Query(None),
    status_filter: AssessmentStatus | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    current: CurrentUser = Depends(require_permission("assessment:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[AssessmentOut]:
    params = PaginationParams(page=page, page_size=page_size)
    assessments, total = await assessment_service.list_assessments(
        session,
        current.organization_id,
        application_id=application_id,
        status=status_filter,
        params=params,
    )
    return Page.build(
        [AssessmentOut.model_validate(a) for a in assessments], total, params
    )


@router.get("/{assessment_id}", response_model=AssessmentDetail)
async def get_assessment(
    assessment_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("assessment:read")),
    session: AsyncSession = Depends(get_session),
) -> AssessmentDetail:
    """One assessment, including the candidate's answers."""
    try:
        assessment = await assessment_service.get_assessment(
            session, current.organization_id, assessment_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    detail = AssessmentDetail.model_validate(assessment)
    detail.invite_url = assessment_service.invite_url(assessment)
    return detail


@router.post("/{assessment_id}/grade", response_model=AssessmentDetail)
async def grade_assessment(
    assessment_id: uuid.UUID,
    payload: ManualGrade,
    current: CurrentUser = Depends(require_permission("assessment:update")),
    session: AsyncSession = Depends(get_session),
) -> AssessmentDetail:
    """Mark the answers a machine would not, and settle the verdict."""
    try:
        assessment = await assessment_service.grade_manually(
            session,
            current.organization_id,
            assessment_id,
            grades=payload.grades,
            feedback=payload.feedback,
            changed_by_id=current.id,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    detail = AssessmentDetail.model_validate(assessment)
    detail.invite_url = assessment_service.invite_url(assessment)
    return detail


@router.post("/{assessment_id}/cancel", response_model=AssessmentOut)
async def cancel_assessment(
    assessment_id: uuid.UUID,
    payload: CancelRequest,
    current: CurrentUser = Depends(require_permission("assessment:update")),
    session: AsyncSession = Depends(get_session),
) -> AssessmentOut:
    """Withdraw an outstanding assessment and kill its link."""
    try:
        assessment = await assessment_service.cancel(
            session, current.organization_id, assessment_id, reason=payload.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return AssessmentOut.model_validate(assessment)


# --------------------------------------------------------------------------- #
# Candidate self-service (token is the credential)
# --------------------------------------------------------------------------- #
async def _public_view(
    session: AsyncSession, assessment: Assessment
) -> PublicAssessmentView:
    organization_id = assessment.organization_id
    organization = await session.get(Organization, organization_id)
    application = await get_scoped(
        session, Application, assessment.application_id, organization_id
    )
    candidate = job = None
    if application is not None:
        candidate = await get_scoped(
            session, Candidate, application.candidate_id, organization_id
        )
        job = await get_scoped(session, Job, application.job_id, organization_id)

    questions = list(assessment.questions_json or [])
    return PublicAssessmentView(
        status=AssessmentStatus(assessment.status),
        type=AssessmentType(assessment.type),
        organization_name=organization.name if organization else None,
        job_title=job.title if job else None,
        candidate_name=candidate.full_name if candidate else None,
        duration_minutes=assessment.duration_minutes,
        question_count=len(questions),
        questions=(
            assessment_service.candidate_view(questions)
            if assessment.status != AssessmentStatus.COMPLETED
            else []
        ),
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
        expires_at=assessment.expires_at,
    )


@public_router.get("/{token}", response_model=PublicAssessmentView)
async def view_assessment(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> PublicAssessmentView:
    """What the candidate sees when they open their invite link."""
    try:
        assessment = await assessment_service.get_by_invite_token(session, token)
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, assessment)


@public_router.post("/{token}/start", response_model=PublicAssessmentView)
async def start_assessment(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> PublicAssessmentView:
    """Start the clock. Safe to call again — a reload must not reset it."""
    try:
        assessment = await assessment_service.start(session, token)
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, assessment)


@public_router.post("/{token}/submit", response_model=PublicAssessmentView)
async def submit_assessment(
    token: str,
    payload: SubmitRequest,
    session: AsyncSession = Depends(get_session),
) -> PublicAssessmentView:
    """Hand in the paper. The response confirms receipt and nothing more."""
    try:
        assessment = await assessment_service.submit(
            session, token, payload.responses
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, assessment)
