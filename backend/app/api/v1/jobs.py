"""Job management routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    get_current_organization,
    get_session,
    require_permission,
)
from app.core.errors import AppError
from app.models.enums import JobStatus
from app.models.organization import Organization
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.schemas.job import (
    JobCreate,
    JobOut,
    JobStatusChange,
    JobSummary,
    JobUpdate,
)
from app.services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobCreate,
    current: CurrentUser = Depends(require_permission("job:create")),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    """Create a job in draft status."""
    try:
        job = await job_service.create_job(
            session, current.organization_id, payload, created_by_id=current.id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return JobOut.model_validate(job)


@router.get("", response_model=Page[JobSummary])
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    job_status: JobStatus | None = Query(None, alias="status"),
    search: str | None = Query(None, max_length=200),
    department: str | None = Query(None, max_length=120),
    current: CurrentUser = Depends(require_permission("job:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[JobSummary]:
    params = PaginationParams(page=page, page_size=page_size)
    jobs, total = await job_service.list_jobs(
        session,
        current.organization_id,
        params,
        status=job_status,
        search=search,
        department=department,
    )
    return Page[JobSummary].build(
        [JobSummary.model_validate(j) for j in jobs], total, params
    )


@router.get("/{job_id}", response_model=JobOut)
async def get_job(
    job_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("job:read")),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    try:
        job = await job_service.get_job(session, current.organization_id, job_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return JobOut.model_validate(job)


@router.patch("/{job_id}", response_model=JobOut)
async def update_job(
    job_id: uuid.UUID,
    payload: JobUpdate,
    current: CurrentUser = Depends(require_permission("job:update")),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    try:
        job = await job_service.update_job(
            session, current.organization_id, job_id, payload
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return JobOut.model_validate(job)


@router.post(
    "/{job_id}/status",
    response_model=JobOut,
    dependencies=[Depends(require_permission("job:update"))],
)
async def change_job_status(
    job_id: uuid.UUID,
    payload: JobStatusChange,
    organization: Organization = Depends(get_current_organization),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    """Publish, pause, close, or archive a job."""
    try:
        job = await job_service.change_status(
            session, organization, job_id, payload.status
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return JobOut.model_validate(job)


@router.post(
    "/{job_id}/publish",
    response_model=JobOut,
    dependencies=[Depends(require_permission("job:publish"))],
)
async def publish_job(
    job_id: uuid.UUID,
    organization: Organization = Depends(get_current_organization),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    """Convenience alias for transitioning a job to published."""
    try:
        job = await job_service.change_status(
            session, organization, job_id, JobStatus.PUBLISHED
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return JobOut.model_validate(job)


@router.delete("/{job_id}", response_model=MessageResponse)
async def delete_job(
    job_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("job:delete")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await job_service.delete_job(session, current.organization_id, job_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Job deleted")
