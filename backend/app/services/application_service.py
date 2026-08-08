"""Applications and the hiring pipeline (design §4.5).

An application is a candidate's candidacy for one job, and it is the unit the
Kanban board moves. Every transition writes a ``StageEvent``, which is what
makes time-in-stage and conversion analytics possible after the fact — the
board's current state alone cannot answer "how long did screening take".
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.integrations.openrouter import OpenRouterClient
from app.models.application import Application, Screening, StageEvent
from app.models.candidate import Candidate
from app.models.enums import (
    STAGE_INDEX,
    STAGE_ORDER,
    ApplicationStatus,
    CandidateSource,
    JobStatus,
    PipelineStage,
)
from app.models.job import Job
from app.schemas.common import PaginationParams
from app.services import candidate_service, job_service, scoring

logger = logging.getLogger(__name__)

# Stages a candidate cannot be in without having been screened first. Moving
# straight from "applied" to "offered" is almost always a mis-drag on the board.
STAGES_REQUIRING_SCREENING = frozenset(
    {PipelineStage.OFFERED, PipelineStage.HIRED}
)

# Spacing for manual board ordering, so a card can always be dropped between
# two neighbours without renumbering the column.
BOARD_POSITION_STEP = 1000.0


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
async def create_application(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    job_id: uuid.UUID,
    candidate_id: uuid.UUID,
    source: CandidateSource = CandidateSource.DIRECT,
    source_detail: str | None = None,
    stage: PipelineStage = PipelineStage.APPLIED,
    assigned_to_id: uuid.UUID | None = None,
    changed_by_id: uuid.UUID | None = None,
    metadata: dict | None = None,
) -> Application:
    """Apply a candidate to a job.

    Both sides are re-read through the tenant scope rather than trusted from
    the request, so an id belonging to another organization cannot be used to
    graft a foreign candidate onto a local job.
    """
    job = await job_service.get_job(session, organization_id, job_id)
    candidate = await candidate_service.get_candidate(
        session, organization_id, candidate_id
    )

    if job.status in {JobStatus.CLOSED, JobStatus.ARCHIVED}:
        raise ConflictError(f"Job '{job.title}' is {job.status} and is not accepting applications")
    if candidate.is_blacklisted:
        raise ConflictError("This candidate is blacklisted and cannot be applied to a job")

    existing = await session.scalar(
        scoped_select(Application, organization_id).where(
            Application.job_id == job_id,
            Application.candidate_id == candidate_id,
        )
    )
    if existing is not None:
        raise ConflictError(
            "This candidate has already been applied to this job",
            details={"application_id": str(existing.id), "stage": existing.stage},
        )

    now = datetime.now(UTC)
    application = Application(
        organization_id=organization_id,
        job_id=job_id,
        candidate_id=candidate_id,
        stage=stage,
        status=ApplicationStatus.ACTIVE,
        source=source,
        source_detail=source_detail,
        applied_at=now,
        stage_changed_at=now,
        assigned_to_id=assigned_to_id,
        board_position=await _next_board_position(session, organization_id, job_id, stage),
        metadata_json=metadata or {},
    )
    session.add(application)
    await session.flush()

    session.add(
        StageEvent(
            organization_id=organization_id,
            application_id=application.id,
            from_stage=None,
            to_stage=stage,
            changed_by_id=changed_by_id,
            trigger="import" if source == CandidateSource.IMPORT else "manual",
        )
    )
    candidate.last_activity_at = now

    await session.commit()
    await session.refresh(application)
    return application


async def get_application(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> Application:
    application = await get_scoped(session, Application, application_id, organization_id)
    if application is None:
        raise NotFoundError("Application not found")
    return application


async def list_applications(
    session: AsyncSession,
    organization_id: uuid.UUID,
    params: PaginationParams,
    *,
    job_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    stage: PipelineStage | None = None,
    status: ApplicationStatus | None = None,
    source: CandidateSource | None = None,
    min_score: float | None = None,
    assigned_to_id: uuid.UUID | None = None,
    order_by: str = "applied_at",
) -> tuple[list[Application], int]:
    stmt = scoped_select(Application, organization_id)
    count_stmt = (
        select(func.count())
        .select_from(Application)
        .where(
            Application.organization_id == organization_id,
            Application.deleted_at.is_(None),
        )
    )

    filters = []
    if job_id is not None:
        filters.append(Application.job_id == job_id)
    if candidate_id is not None:
        filters.append(Application.candidate_id == candidate_id)
    if stage is not None:
        filters.append(Application.stage == stage)
    if status is not None:
        filters.append(Application.status == status)
    if source is not None:
        filters.append(Application.source == source)
    if assigned_to_id is not None:
        filters.append(Application.assigned_to_id == assigned_to_id)
    if min_score is not None:
        filters.append(Application.score >= min_score)

    for f in filters:
        stmt = stmt.where(f)
        count_stmt = count_stmt.where(f)

    if order_by == "score":
        # Unscored applications sort last rather than as zero, so a fresh
        # application is not presented as a bad one.
        stmt = stmt.order_by(
            Application.score.is_(None), Application.score.desc(), Application.applied_at.desc()
        )
    elif order_by == "stage":
        stmt = stmt.order_by(Application.stage, Application.board_position)
    else:
        stmt = stmt.order_by(Application.applied_at.desc())

    total = int(await session.scalar(count_stmt) or 0)
    rows = list(
        (await session.execute(stmt.offset(params.offset).limit(params.page_size)))
        .scalars()
        .all()
    )
    return rows, total


# --------------------------------------------------------------------------- #
# Stage transitions (design §4.5)
# --------------------------------------------------------------------------- #
async def move_stage(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    to_stage: PipelineStage,
    *,
    changed_by_id: uuid.UUID | None = None,
    note: str | None = None,
    trigger: str = "manual",
    board_position: float | None = None,
    force: bool = False,
) -> Application:
    """Move an application to another stage and record the transition.

    Backwards moves are allowed — a candidate genuinely does get sent back for
    another interview round — but skipping screening on the way to an offer is
    refused unless the caller explicitly forces it.
    """
    application = await get_application(session, organization_id, application_id)

    if application.status != ApplicationStatus.ACTIVE:
        raise ConflictError(
            f"Application is {application.status} and cannot be moved",
            details={"status": application.status},
        )

    from_stage = PipelineStage(application.stage)
    if from_stage == to_stage and board_position is None:
        return application

    if not force and to_stage in STAGES_REQUIRING_SCREENING:
        screened = await session.scalar(
            select(func.count())
            .select_from(Screening)
            .where(
                Screening.organization_id == organization_id,
                Screening.application_id == application_id,
                Screening.deleted_at.is_(None),
            )
        )
        if not screened:
            raise ValidationError(
                f"Cannot move to '{to_stage}' before the candidate has been screened",
                details={"from_stage": from_stage, "to_stage": to_stage},
            )

    now = datetime.now(UTC)
    seconds_in_previous = None
    if application.stage_changed_at is not None:
        previous = application.stage_changed_at
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=UTC)
        seconds_in_previous = max(0, int((now - previous).total_seconds()))

    application.stage = to_stage
    application.stage_changed_at = now
    application.board_position = (
        board_position
        if board_position is not None
        else await _next_board_position(
            session, organization_id, application.job_id, to_stage
        )
    )

    if to_stage == PipelineStage.HIRED:
        application.status = ApplicationStatus.HIRED
        application.hired_at = now
    elif application.hired_at is not None:
        # Moved back out of hired: the hire did not stick.
        application.status = ApplicationStatus.ACTIVE
        application.hired_at = None

    session.add(
        StageEvent(
            organization_id=organization_id,
            application_id=application.id,
            from_stage=from_stage,
            to_stage=to_stage,
            changed_by_id=changed_by_id,
            trigger=trigger,
            note=note,
            seconds_in_previous_stage=seconds_in_previous,
        )
    )

    await session.commit()
    await session.refresh(application)
    return application


async def bulk_move_stage(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_ids: list[uuid.UUID],
    to_stage: PipelineStage,
    *,
    changed_by_id: uuid.UUID | None = None,
    note: str | None = None,
    force: bool = False,
) -> tuple[list[Application], list[dict]]:
    """Move many applications, reporting per-application failures.

    One card that cannot legally move must not abort the whole bulk action, so
    failures are collected and returned alongside the successes.
    """
    moved: list[Application] = []
    errors: list[dict] = []
    for application_id in application_ids:
        try:
            moved.append(
                await move_stage(
                    session,
                    organization_id,
                    application_id,
                    to_stage,
                    changed_by_id=changed_by_id,
                    note=note,
                    trigger="bulk",
                    force=force,
                )
            )
        except (NotFoundError, ConflictError, ValidationError) as exc:
            await session.rollback()
            errors.append({"application_id": str(application_id), "error": exc.message})
    return moved, errors


async def reject_application(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    reason: str | None = None,
    changed_by_id: uuid.UUID | None = None,
    trigger: str = "manual",
) -> Application:
    application = await get_application(session, organization_id, application_id)
    if application.status == ApplicationStatus.REJECTED:
        return application

    application.status = ApplicationStatus.REJECTED
    application.rejected_at = datetime.now(UTC)
    application.rejection_reason = reason

    session.add(
        StageEvent(
            organization_id=organization_id,
            application_id=application.id,
            from_stage=PipelineStage(application.stage),
            # The stage is preserved so analytics can see where candidates are
            # lost; only the status changes.
            to_stage=PipelineStage(application.stage),
            changed_by_id=changed_by_id,
            trigger=trigger if trigger != "manual" else "manual",
            note=reason,
        )
    )
    await session.commit()
    await session.refresh(application)
    return application


async def withdraw_application(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    reason: str | None = None,
) -> Application:
    """Mark that the candidate pulled out, as opposed to being rejected."""
    application = await get_application(session, organization_id, application_id)
    application.status = ApplicationStatus.WITHDRAWN
    application.rejection_reason = reason
    await session.commit()
    await session.refresh(application)
    return application


async def reopen_application(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> Application:
    application = await get_application(session, organization_id, application_id)
    application.status = ApplicationStatus.ACTIVE
    application.rejected_at = None
    application.rejection_reason = None
    await session.commit()
    await session.refresh(application)
    return application


async def assign(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    user_id: uuid.UUID | None,
) -> Application:
    application = await get_application(session, organization_id, application_id)
    application.assigned_to_id = user_id
    await session.commit()
    await session.refresh(application)
    return application


async def list_stage_events(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> list[StageEvent]:
    result = await session.execute(
        scoped_select(StageEvent, organization_id)
        .where(StageEvent.application_id == application_id)
        .order_by(StageEvent.created_at.asc())
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
async def screen(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    changed_by_id: uuid.UUID | None = None,
    auto_advance: bool = True,
    client: OpenRouterClient | None = None,
) -> tuple[Screening, Application]:
    """Score an application, then apply the job's automation thresholds."""
    application = await get_application(session, organization_id, application_id)
    job = await job_service.get_job(session, organization_id, application.job_id)
    candidate = await candidate_service.get_candidate(
        session, organization_id, application.candidate_id
    )

    screening = await scoring.screen_application(
        session, organization_id, application, job, candidate, client=client
    )
    await session.commit()
    await session.refresh(screening)
    await session.refresh(application)

    if auto_advance:
        application = await _apply_thresholds(
            session, organization_id, application, job, screening, changed_by_id
        )
    return screening, application


async def _apply_thresholds(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application: Application,
    job: Job,
    screening: Screening,
    changed_by_id: uuid.UUID | None,
) -> Application:
    """Advance or flag an application based on the job's thresholds (§4.1).

    A low-confidence screening never triggers automation: the design requires
    those to reach a human, and auto-rejecting on a parse the system does not
    trust is exactly the failure mode that erodes trust in the tool.
    """
    if screening.requires_human_review:
        logger.info(
            "Screening %s is below the confidence threshold; skipping automation",
            screening.id,
        )
        return application

    score = float(screening.overall_score)
    advance = job.auto_advance_threshold
    reject = job.auto_reject_threshold

    if advance is not None and score >= float(advance):
        if STAGE_INDEX[PipelineStage(application.stage)] < STAGE_INDEX[PipelineStage.SCREENED]:
            return await move_stage(
                session,
                organization_id,
                application.id,
                PipelineStage.SCREENED,
                changed_by_id=changed_by_id,
                trigger="auto_advance",
                note=f"Auto-advanced on score {score:.1f}",
            )
    elif reject is not None and score < float(reject):
        return await reject_application(
            session,
            organization_id,
            application.id,
            reason=f"Auto-flagged: score {score:.1f} is below the {float(reject):.1f} threshold",
            changed_by_id=changed_by_id,
            trigger="auto_reject",
        )
    return application


async def bulk_screen(
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    stage: PipelineStage | None = None,
    rescore: bool = False,
    limit: int = 200,
    changed_by_id: uuid.UUID | None = None,
    client: OpenRouterClient | None = None,
) -> tuple[list[Screening], list[dict]]:
    """Screen a job's applications in one pass.

    By default already-scored applications are skipped, so re-running after a
    batch of new applicants does not burn screening credits on old ones.
    """
    await job_service.get_job(session, organization_id, job_id)

    stmt = scoped_select(Application, organization_id).where(
        Application.job_id == job_id,
        Application.status == ApplicationStatus.ACTIVE,
    )
    if stage is not None:
        stmt = stmt.where(Application.stage == stage)
    if not rescore:
        stmt = stmt.where(Application.score.is_(None))

    applications = list(
        (await session.execute(stmt.order_by(Application.applied_at).limit(limit)))
        .scalars()
        .all()
    )

    screenings: list[Screening] = []
    errors: list[dict] = []
    for application in applications:
        try:
            screening, _ = await screen(
                session,
                organization_id,
                application.id,
                changed_by_id=changed_by_id,
                client=client,
            )
            screenings.append(screening)
        except (NotFoundError, ConflictError, ValidationError) as exc:
            await session.rollback()
            errors.append({"application_id": str(application.id), "error": exc.message})
    return screenings, errors


async def list_screenings(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> list[Screening]:
    result = await session.execute(
        scoped_select(Screening, organization_id)
        .where(Screening.application_id == application_id)
        .order_by(Screening.created_at.desc())
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Board (design §4.5)
# --------------------------------------------------------------------------- #
async def get_board(
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    include_inactive: bool = False,
    per_stage_limit: int = 100,
) -> dict:
    """Build the Kanban board for one job.

    Returns every stage, including the empty ones — a board that hides its
    empty columns cannot be dropped into.
    """
    job = await job_service.get_job(session, organization_id, job_id)

    stmt = scoped_select(Application, organization_id).where(
        Application.job_id == job_id
    )
    if not include_inactive:
        stmt = stmt.where(
            Application.status.in_(
                [ApplicationStatus.ACTIVE, ApplicationStatus.HIRED, ApplicationStatus.ON_HOLD]
            )
        )
    applications = list(
        (
            await session.execute(
                stmt.order_by(Application.board_position, Application.applied_at)
            )
        )
        .scalars()
        .all()
    )

    candidates = await _load_candidates(
        session, organization_id, {a.candidate_id for a in applications}
    )

    columns = []
    for stage in STAGE_ORDER:
        in_stage = [a for a in applications if a.stage == stage]
        columns.append(
            {
                "stage": stage,
                "total": len(in_stage),
                "applications": [
                    {"application": a, "candidate": candidates.get(a.candidate_id)}
                    for a in in_stage[:per_stage_limit]
                ],
            }
        )

    return {"job": job, "columns": columns, "total": len(applications)}


async def _load_candidates(
    session: AsyncSession, organization_id: uuid.UUID, ids: set[uuid.UUID]
) -> dict[uuid.UUID, Candidate]:
    """Fetch candidates for a board in one query rather than N."""
    if not ids:
        return {}
    rows = (
        (
            await session.execute(
                scoped_select(Candidate, organization_id).where(Candidate.id.in_(ids))
            )
        )
        .scalars()
        .all()
    )
    return {c.id: c for c in rows}


async def _next_board_position(
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    stage: PipelineStage,
) -> float:
    """Append position for a card entering a column."""
    highest = await session.scalar(
        select(func.max(Application.board_position)).where(
            Application.organization_id == organization_id,
            Application.job_id == job_id,
            Application.stage == stage,
            Application.deleted_at.is_(None),
        )
    )
    return float(highest or 0.0) + BOARD_POSITION_STEP


async def delete_application(
    session: AsyncSession, organization_id: uuid.UUID, application_id: uuid.UUID
) -> None:
    application = await get_application(session, organization_id, application_id)
    application.soft_delete()
    await session.commit()
