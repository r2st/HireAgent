"""Application, screening, and pipeline-board routes (design §4.5, §6.1)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_session, require_permission
from app.core.errors import AppError
from app.models.enums import ApplicationStatus, CandidateSource, PipelineStage
from app.schemas.application import (
    ApplicationCreate,
    ApplicationDetail,
    ApplicationOut,
    ApplicationUpdate,
    AssignRequest,
    BoardCard,
    BoardColumn,
    BoardOut,
    BulkScreenRequest,
    BulkScreenResult,
    BulkStageMove,
    BulkStageResult,
    RankedCandidate,
    RejectRequest,
    ScreeningOut,
    ScreeningResult,
    ScreenRequest,
    StageEventOut,
    StageMove,
)
from app.schemas.candidate import CandidateSummary
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.schemas.job import JobSummary
from app.services import application_service, candidate_service

router = APIRouter(prefix="/applications", tags=["applications"])
# Design §6.1 nests the ranked shortlist and the board under the job.
job_router = APIRouter(prefix="/jobs", tags=["pipeline"])


@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
async def create_application(
    payload: ApplicationCreate,
    current: CurrentUser = Depends(require_permission("application:create")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    try:
        application = await application_service.create_application(
            session,
            current.organization_id,
            job_id=payload.job_id,
            candidate_id=payload.candidate_id,
            source=payload.source,
            source_detail=payload.source_detail,
            stage=payload.stage,
            assigned_to_id=payload.assigned_to_id,
            changed_by_id=current.id,
            metadata=payload.metadata,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.get("", response_model=Page[ApplicationOut])
async def list_applications(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    job_id: uuid.UUID | None = Query(None),
    candidate_id: uuid.UUID | None = Query(None),
    stage: PipelineStage | None = Query(None),
    application_status: ApplicationStatus | None = Query(None, alias="status"),
    source: CandidateSource | None = Query(None),
    min_score: float | None = Query(None, ge=0, le=100),
    assigned_to_id: uuid.UUID | None = Query(None),
    order_by: str = Query("applied_at", pattern="^(applied_at|score|stage)$"),
    current: CurrentUser = Depends(require_permission("application:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[ApplicationOut]:
    params = PaginationParams(page=page, page_size=page_size)
    rows, total = await application_service.list_applications(
        session,
        current.organization_id,
        params,
        job_id=job_id,
        candidate_id=candidate_id,
        stage=stage,
        status=application_status,
        source=source,
        min_score=min_score,
        assigned_to_id=assigned_to_id,
        order_by=order_by,
    )
    return Page[ApplicationOut].build(
        [ApplicationOut.model_validate(a) for a in rows], total, params
    )


@router.get("/{application_id}", response_model=ApplicationDetail)
async def get_application(
    application_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("application:read")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationDetail:
    from app.services import job_service

    try:
        application = await application_service.get_application(
            session, current.organization_id, application_id
        )
        candidate = await candidate_service.get_candidate(
            session, current.organization_id, application.candidate_id
        )
        job = await job_service.get_job(
            session, current.organization_id, application.job_id
        )
    except AppError as exc:
        raise exc.to_http() from exc

    screenings = await application_service.list_screenings(
        session, current.organization_id, application_id
    )
    detail = ApplicationDetail.model_validate(application)
    detail.candidate = CandidateSummary.model_validate(candidate)
    detail.job = JobSummary.model_validate(job)
    detail.latest_screening = (
        ScreeningOut.model_validate(screenings[0]) if screenings else None
    )
    return detail


@router.patch("/{application_id}", response_model=ApplicationOut)
async def update_application(
    application_id: uuid.UUID,
    payload: ApplicationUpdate,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    try:
        application = await application_service.get_application(
            session, current.organization_id, application_id
        )
    except AppError as exc:
        raise exc.to_http() from exc

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(application, field, value)
    await session.commit()
    await session.refresh(application)
    return ApplicationOut.model_validate(application)


@router.put("/{application_id}/stage", response_model=ApplicationOut)
async def move_stage(
    application_id: uuid.UUID,
    payload: StageMove,
    current: CurrentUser = Depends(require_permission("application:move")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    """Move a candidate to another pipeline stage."""
    try:
        application = await application_service.move_stage(
            session,
            current.organization_id,
            application_id,
            payload.stage,
            changed_by_id=current.id,
            note=payload.note,
            board_position=payload.board_position,
            force=payload.force,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.post("/bulk/stage", response_model=BulkStageResult)
async def bulk_move_stage(
    payload: BulkStageMove,
    current: CurrentUser = Depends(require_permission("application:move")),
    session: AsyncSession = Depends(get_session),
) -> BulkStageResult:
    """Move many candidates at once; per-application failures are reported."""
    moved, errors = await application_service.bulk_move_stage(
        session,
        current.organization_id,
        payload.application_ids,
        payload.stage,
        changed_by_id=current.id,
        note=payload.note,
        force=payload.force,
    )
    return BulkStageResult(
        total=len(payload.application_ids),
        moved=len(moved),
        failed=len(errors),
        applications=[ApplicationOut.model_validate(a) for a in moved],
        errors=errors,
    )


@router.post("/{application_id}/reject", response_model=ApplicationOut)
async def reject_application(
    application_id: uuid.UUID,
    payload: RejectRequest,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    try:
        application = await application_service.reject_application(
            session,
            current.organization_id,
            application_id,
            reason=payload.reason,
            changed_by_id=current.id,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.post("/{application_id}/withdraw", response_model=ApplicationOut)
async def withdraw_application(
    application_id: uuid.UUID,
    payload: RejectRequest,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    """Record that the candidate withdrew, rather than being rejected."""
    try:
        application = await application_service.withdraw_application(
            session, current.organization_id, application_id, reason=payload.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.post("/{application_id}/reopen", response_model=ApplicationOut)
async def reopen_application(
    application_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    try:
        application = await application_service.reopen_application(
            session, current.organization_id, application_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.put("/{application_id}/assignee", response_model=ApplicationOut)
async def assign_application(
    application_id: uuid.UUID,
    payload: AssignRequest,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> ApplicationOut:
    try:
        application = await application_service.assign(
            session, current.organization_id, application_id, payload.user_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ApplicationOut.model_validate(application)


@router.get("/{application_id}/events", response_model=list[StageEventOut])
async def list_stage_events(
    application_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("application:read")),
    session: AsyncSession = Depends(get_session),
) -> list[StageEventOut]:
    """The application's stage history, oldest first."""
    try:
        await application_service.get_application(
            session, current.organization_id, application_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    events = await application_service.list_stage_events(
        session, current.organization_id, application_id
    )
    return [StageEventOut.model_validate(e) for e in events]


@router.delete("/{application_id}", response_model=MessageResponse)
async def delete_application(
    application_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("application:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await application_service.delete_application(
            session, current.organization_id, application_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Application deleted")


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
@router.post("/{application_id}/screen", response_model=ScreeningResult)
async def screen_application(
    application_id: uuid.UUID,
    payload: ScreenRequest | None = None,
    current: CurrentUser = Depends(require_permission("screening:create")),
    session: AsyncSession = Depends(get_session),
) -> ScreeningResult:
    """Score one application against its job."""
    request = payload or ScreenRequest()
    try:
        screening, application = await application_service.screen(
            session,
            current.organization_id,
            application_id,
            changed_by_id=current.id,
            auto_advance=request.auto_advance,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ScreeningResult(
        screening=ScreeningOut.model_validate(screening),
        application=ApplicationOut.model_validate(application),
    )


@router.get("/{application_id}/screenings", response_model=list[ScreeningOut])
async def list_screenings(
    application_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("screening:read")),
    session: AsyncSession = Depends(get_session),
) -> list[ScreeningOut]:
    """Screening history, newest first — re-screening appends rather than replaces."""
    try:
        await application_service.get_application(
            session, current.organization_id, application_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    screenings = await application_service.list_screenings(
        session, current.organization_id, application_id
    )
    return [ScreeningOut.model_validate(s) for s in screenings]


# --------------------------------------------------------------------------- #
# Job-scoped pipeline views
# --------------------------------------------------------------------------- #
@job_router.get("/{job_id}/board", response_model=BoardOut)
async def get_board(
    job_id: uuid.UUID,
    include_inactive: bool = Query(False),
    per_stage_limit: int = Query(100, ge=1, le=500),
    current: CurrentUser = Depends(require_permission("application:read")),
    session: AsyncSession = Depends(get_session),
) -> BoardOut:
    """The Kanban board for one job, with every stage column present."""
    try:
        board = await application_service.get_board(
            session,
            current.organization_id,
            job_id,
            include_inactive=include_inactive,
            per_stage_limit=per_stage_limit,
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return BoardOut(
        job=JobSummary.model_validate(board["job"]),
        total=board["total"],
        columns=[
            BoardColumn(
                stage=column["stage"],
                total=column["total"],
                applications=[
                    BoardCard(
                        application=ApplicationOut.model_validate(card["application"]),
                        candidate=(
                            CandidateSummary.model_validate(card["candidate"])
                            if card["candidate"] is not None
                            else None
                        ),
                    )
                    for card in column["applications"]
                ],
            )
            for column in board["columns"]
        ],
    )


@job_router.get("/{job_id}/candidates", response_model=Page[RankedCandidate])
async def list_job_candidates(
    job_id: uuid.UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    stage: PipelineStage | None = Query(None),
    min_score: float | None = Query(None, ge=0, le=100),
    order_by: str = Query("score", pattern="^(applied_at|score|stage)$"),
    current: CurrentUser = Depends(require_permission("application:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[RankedCandidate]:
    """Candidates for a job with their scores, ranked best-first by default."""
    from app.services import job_service

    try:
        await job_service.get_job(session, current.organization_id, job_id)
    except AppError as exc:
        raise exc.to_http() from exc

    params = PaginationParams(page=page, page_size=page_size)
    applications, total = await application_service.list_applications(
        session,
        current.organization_id,
        params,
        job_id=job_id,
        stage=stage,
        min_score=min_score,
        order_by=order_by,
    )

    items: list[RankedCandidate] = []
    for application in applications:
        candidate = await candidate_service.get_candidate(
            session, current.organization_id, application.candidate_id
        )
        screenings = await application_service.list_screenings(
            session, current.organization_id, application.id
        )
        items.append(
            RankedCandidate(
                application=ApplicationOut.model_validate(application),
                candidate=CandidateSummary.model_validate(candidate),
                screening=(
                    ScreeningOut.model_validate(screenings[0]) if screenings else None
                ),
            )
        )
    return Page[RankedCandidate].build(items, total, params)


@job_router.post("/{job_id}/screen", response_model=BulkScreenResult)
async def bulk_screen(
    job_id: uuid.UUID,
    payload: BulkScreenRequest | None = None,
    current: CurrentUser = Depends(require_permission("screening:create")),
    session: AsyncSession = Depends(get_session),
) -> BulkScreenResult:
    """Screen a job's unscored applications in one pass."""
    request = payload or BulkScreenRequest()
    try:
        screenings, errors = await application_service.bulk_screen(
            session,
            current.organization_id,
            job_id,
            stage=request.stage,
            rescore=request.rescore,
            limit=request.limit,
            changed_by_id=current.id,
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return BulkScreenResult(
        total=len(screenings) + len(errors),
        succeeded=len(screenings),
        failed=len(errors),
        screenings=[ScreeningOut.model_validate(s) for s in screenings],
        errors=errors,
    )
