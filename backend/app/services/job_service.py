"""Job lifecycle: create, update, publish, close, and plan-limit enforcement."""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, PlanLimitError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.models.enums import PLAN_LIMITS, JobStatus, PlanTier
from app.models.job import Job
from app.models.organization import Organization
from app.schemas.common import PaginationParams
from app.schemas.job import JobCreate, JobUpdate

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# Which status changes are legal. Closed/archived jobs are terminal except for
# an explicit reopen path back to draft.
ALLOWED_STATUS_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.DRAFT: {JobStatus.PUBLISHED, JobStatus.CLOSED, JobStatus.ARCHIVED},
    JobStatus.PUBLISHED: {JobStatus.PAUSED, JobStatus.CLOSED, JobStatus.ARCHIVED},
    JobStatus.PAUSED: {JobStatus.PUBLISHED, JobStatus.CLOSED, JobStatus.ARCHIVED},
    JobStatus.CLOSED: {JobStatus.ARCHIVED, JobStatus.DRAFT},
    JobStatus.ARCHIVED: {JobStatus.DRAFT},
}

# Statuses that consume an "active job" slot against the plan limit.
ACTIVE_STATUSES = {JobStatus.PUBLISHED, JobStatus.PAUSED}


def slugify(value: str) -> str:
    return _SLUG_STRIP.sub("-", value.lower()).strip("-") or "job"


async def _unique_slug(
    session: AsyncSession, organization_id: uuid.UUID, title: str
) -> str:
    """A slug unique within the organization."""
    base = slugify(title)[:200]
    candidate = base
    for _ in range(50):
        exists = await session.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.organization_id == organization_id, Job.slug == candidate)
        )
        if not exists:
            return candidate
        candidate = f"{base}-{secrets.token_hex(3)}"
    raise ConflictError("Could not allocate a unique job slug")


async def count_active_jobs(
    session: AsyncSession, organization_id: uuid.UUID, *, exclude: uuid.UUID | None = None
) -> int:
    stmt = (
        select(func.count())
        .select_from(Job)
        .where(
            Job.organization_id == organization_id,
            Job.deleted_at.is_(None),
            Job.status.in_(list(ACTIVE_STATUSES)),
        )
    )
    if exclude is not None:
        stmt = stmt.where(Job.id != exclude)
    return int(await session.scalar(stmt) or 0)


async def _enforce_active_job_limit(
    session: AsyncSession, organization: Organization, *, exclude: uuid.UUID | None
) -> None:
    """Block publishing beyond the plan's active-job allowance (design §7.1)."""
    limit = PLAN_LIMITS.get(PlanTier(organization.plan), {}).get("active_jobs")
    if limit is None:
        return
    current = await count_active_jobs(session, organization.id, exclude=exclude)
    if current >= limit:
        raise PlanLimitError(
            f"The {organization.plan} plan allows {limit} active jobs; "
            f"{current} are already active. Close a job or upgrade the plan.",
            details={"limit": limit, "current": current, "plan": str(organization.plan)},
        )


async def create_job(
    session: AsyncSession,
    organization_id: uuid.UUID,
    payload: JobCreate,
    *,
    created_by_id: uuid.UUID | None = None,
) -> Job:
    """Create a job in draft. Publishing is a separate, limit-checked step."""
    job = Job(
        organization_id=organization_id,
        title=payload.title.strip(),
        slug=await _unique_slug(session, organization_id, payload.title),
        department=payload.department,
        location=payload.location,
        work_mode=payload.work_mode,
        employment_type=payload.employment_type,
        seniority=payload.seniority,
        description=payload.description,
        status=JobStatus.DRAFT,
        openings=payload.openings,
        min_experience_years=payload.min_experience_years,
        max_experience_years=payload.max_experience_years,
        salary_min=payload.salary_min,
        salary_max=payload.salary_max,
        salary_currency=payload.salary_currency.upper(),
        requirements_json=payload.requirements.model_dump(mode="json"),
        scoring_weights=payload.scoring_weights.model_dump(mode="json"),
        auto_advance_threshold=payload.auto_advance_threshold,
        auto_reject_threshold=payload.auto_reject_threshold,
        hiring_manager_id=payload.hiring_manager_id,
        is_confidential=payload.is_confidential,
        created_by_id=created_by_id,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(
    session: AsyncSession, organization_id: uuid.UUID, job_id: uuid.UUID
) -> Job:
    job = await get_scoped(session, Job, job_id, organization_id)
    if job is None:
        raise NotFoundError("Job not found")
    return job


async def list_jobs(
    session: AsyncSession,
    organization_id: uuid.UUID,
    params: PaginationParams,
    *,
    status: JobStatus | None = None,
    search: str | None = None,
    department: str | None = None,
) -> tuple[list[Job], int]:
    stmt = scoped_select(Job, organization_id)
    count_stmt = (
        select(func.count())
        .select_from(Job)
        .where(Job.organization_id == organization_id, Job.deleted_at.is_(None))
    )

    if status is not None:
        stmt = stmt.where(Job.status == status)
        count_stmt = count_stmt.where(Job.status == status)
    if department:
        stmt = stmt.where(Job.department == department)
        count_stmt = count_stmt.where(Job.department == department)
    if search:
        pattern = f"%{search.strip()}%"
        predicate = or_(
            Job.title.ilike(pattern),
            Job.department.ilike(pattern),
            Job.location.ilike(pattern),
        )
        stmt = stmt.where(predicate)
        count_stmt = count_stmt.where(predicate)

    stmt = stmt.order_by(Job.created_at.desc()).offset(params.offset).limit(
        params.page_size
    )
    rows = list((await session.execute(stmt)).scalars().all())
    total = int(await session.scalar(count_stmt) or 0)
    return rows, total


async def update_job(
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    payload: JobUpdate,
) -> Job:
    job = await get_job(session, organization_id, job_id)
    data = payload.model_dump(exclude_unset=True)

    if "requirements" in data:
        requirements = data.pop("requirements")
        job.requirements_json = requirements if requirements is not None else {}
    if "scoring_weights" in data:
        weights = data.pop("scoring_weights")
        if weights is not None:
            job.scoring_weights = weights
    if "title" in data and data["title"]:
        job.title = data.pop("title").strip()
    if "salary_currency" in data and data["salary_currency"]:
        job.salary_currency = data.pop("salary_currency").upper()

    for field, value in data.items():
        setattr(job, field, value)

    _validate_job_ranges(job)
    await session.commit()
    await session.refresh(job)
    return job


def _validate_job_ranges(job: Job) -> None:
    """Re-check cross-field invariants after a partial update.

    A PATCH can move one side of a pair, so the schema-level check on create is
    not sufficient here.
    """
    if (
        job.min_experience_years is not None
        and job.max_experience_years is not None
        and job.min_experience_years > job.max_experience_years
    ):
        raise ValidationError("min_experience_years cannot exceed max_experience_years")
    if (
        job.salary_min is not None
        and job.salary_max is not None
        and job.salary_min > job.salary_max
    ):
        raise ValidationError("salary_min cannot exceed salary_max")
    if (
        job.auto_reject_threshold is not None
        and job.auto_advance_threshold is not None
        and job.auto_reject_threshold >= job.auto_advance_threshold
    ):
        raise ValidationError(
            "auto_reject_threshold must be below auto_advance_threshold"
        )


async def change_status(
    session: AsyncSession,
    organization: Organization,
    job_id: uuid.UUID,
    new_status: JobStatus,
) -> Job:
    """Move a job through its lifecycle, enforcing legal transitions."""
    job = await get_job(session, organization.id, job_id)
    current = JobStatus(job.status)

    if current == new_status:
        return job
    if new_status not in ALLOWED_STATUS_TRANSITIONS.get(current, set()):
        raise ValidationError(
            f"Cannot move a job from '{current}' to '{new_status}'",
            details={
                "allowed": sorted(
                    s.value for s in ALLOWED_STATUS_TRANSITIONS.get(current, set())
                )
            },
        )

    if new_status in ACTIVE_STATUSES and current not in ACTIVE_STATUSES:
        await _enforce_active_job_limit(session, organization, exclude=job.id)

    if new_status == JobStatus.PUBLISHED:
        _require_publishable(job)
        if job.published_at is None:
            job.published_at = datetime.now(UTC)
        job.closed_at = None
    elif new_status == JobStatus.CLOSED:
        job.closed_at = datetime.now(UTC)
    elif new_status == JobStatus.DRAFT:
        # Reopening starts a fresh publication cycle.
        job.published_at = None
        job.closed_at = None

    job.status = new_status
    await session.commit()
    await session.refresh(job)
    return job


def _require_publishable(job: Job) -> None:
    """A published job must carry enough detail to score candidates against."""
    missing: list[str] = []
    if not (job.description or "").strip():
        missing.append("description")
    requirements = job.requirements_json or {}
    if not requirements.get("required_skills"):
        missing.append("requirements.required_skills")
    if missing:
        raise ValidationError(
            "Job cannot be published until required fields are set",
            details={"missing": missing},
        )


async def delete_job(
    session: AsyncSession, organization_id: uuid.UUID, job_id: uuid.UUID
) -> None:
    """Soft delete — applications and history are preserved."""
    job = await get_job(session, organization_id, job_id)
    job.soft_delete()
    await session.commit()
