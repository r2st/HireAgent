"""Live pipeline conversion analytics (design §4.5, §6.1).

``AnalyticsSnapshot`` exists for nightly pre-aggregated rollups, but nothing
writes one yet — there is no scheduled job in this build to run that
aggregation. Rather than expose an endpoint backed by a table nobody
populates, this module answers the same question directly from
``applications`` and ``stage_events``: correct today, and a plain source of
truth to backfill the snapshot table against once a nightly job exists to
write it.

**Funnel counts are "ever reached", not "currently sitting at".** An
application that passed through ``screened`` on its way to ``offered`` still
counts toward ``screened``'s funnel — the same way a real hiring funnel
counts everyone who cleared a stage, not just the ones stalled there today.
``stage_counts`` answers the other question, of where live applications sit
right now.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.tenancy import scoped_select
from app.models.application import Application, StageEvent
from app.models.candidate import Candidate
from app.models.enums import STAGE_ORDER, ApplicationStatus, PipelineStage


@dataclass
class FunnelStep:
    stage: PipelineStage
    reached: int
    conversion_from_previous: float | None


@dataclass
class PipelineAnalytics:
    total_applications: int
    stage_counts: dict[str, int] = field(default_factory=dict)
    funnel: list[FunnelStep] = field(default_factory=list)
    hires: int = 0
    rejections: int = 0
    withdrawals: int = 0
    avg_time_to_hire_days: float | None = None
    avg_seconds_in_stage: dict[str, float] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)


async def pipeline_summary(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    job_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> PipelineAnalytics:
    base = scoped_select(Application, organization_id)
    if job_id is not None:
        base = base.where(Application.job_id == job_id)
    if since is not None:
        base = base.where(Application.created_at >= since)
    if until is not None:
        base = base.where(Application.created_at <= until)
    applications = list((await session.execute(base)).scalars().all())

    total = len(applications)
    if total == 0:
        return PipelineAnalytics(total_applications=0)

    app_ids = [a.id for a in applications]

    stage_counts: dict[str, int] = {}
    for application in applications:
        stage_counts[application.stage] = stage_counts.get(application.stage, 0) + 1

    hires = sum(1 for a in applications if a.status == ApplicationStatus.HIRED)
    rejections = sum(1 for a in applications if a.status == ApplicationStatus.REJECTED)
    withdrawals = sum(1 for a in applications if a.status == ApplicationStatus.WITHDRAWN)

    # "Ever reached" a stage: everyone starts at APPLIED with no event row, so
    # that count is every application in scope; later stages are counted from
    # the transitions that actually happened.
    reached_counts: dict[PipelineStage, int] = {PipelineStage.APPLIED: total}
    if len(STAGE_ORDER) > 1:
        rows = await session.execute(
            select(StageEvent.to_stage, func.count(func.distinct(StageEvent.application_id)))
            .where(
                StageEvent.organization_id == organization_id,
                StageEvent.application_id.in_(app_ids),
            )
            .group_by(StageEvent.to_stage)
        )
        for stage_value, count in rows.all():
            reached_counts[PipelineStage(stage_value)] = count

    funnel: list[FunnelStep] = []
    previous_count: int | None = None
    for stage in STAGE_ORDER:
        reached = reached_counts.get(stage, 0)
        conversion = (
            round(100.0 * reached / previous_count, 1)
            if previous_count
            else (100.0 if reached else None)
        )
        funnel.append(
            FunnelStep(stage=stage, reached=reached, conversion_from_previous=conversion)
        )
        previous_count = reached

    avg_seconds_rows = await session.execute(
        select(StageEvent.from_stage, func.avg(StageEvent.seconds_in_previous_stage))
        .where(
            StageEvent.organization_id == organization_id,
            StageEvent.application_id.in_(app_ids),
            StageEvent.from_stage.is_not(None),
            StageEvent.seconds_in_previous_stage.is_not(None),
        )
        .group_by(StageEvent.from_stage)
    )
    avg_seconds_in_stage = {
        str(stage): round(float(avg), 1)
        for stage, avg in avg_seconds_rows.all()
        if avg is not None
    }

    hired_ids = [a.id for a in applications if a.status == ApplicationStatus.HIRED]
    avg_time_to_hire_days: float | None = None
    if hired_ids:
        hire_events = await session.execute(
            select(
                StageEvent.application_id, func.min(StageEvent.created_at)
            )
            .where(
                StageEvent.organization_id == organization_id,
                StageEvent.application_id.in_(hired_ids),
                StageEvent.to_stage == PipelineStage.HIRED,
            )
            .group_by(StageEvent.application_id)
        )
        created_at_by_id = {a.id: a.created_at for a in applications}
        days: list[float] = []
        for application_id, hired_at in hire_events.all():
            started = created_at_by_id.get(application_id)
            if started is None or hired_at is None:
                continue
            days.append((hired_at - started).total_seconds() / 86400)
        if days:
            avg_time_to_hire_days = round(sum(days) / len(days), 1)

    by_source: dict[str, int] = {}
    if app_ids:
        source_rows = await session.execute(
            select(Candidate.source, func.count(func.distinct(Application.id)))
            .select_from(Application)
            .join(Candidate, Candidate.id == Application.candidate_id)
            .where(Application.id.in_(app_ids))
            .group_by(Candidate.source)
        )
        by_source = {str(source): count for source, count in source_rows.all()}

    return PipelineAnalytics(
        total_applications=total,
        stage_counts=stage_counts,
        funnel=funnel,
        hires=hires,
        rejections=rejections,
        withdrawals=withdrawals,
        avg_time_to_hire_days=avg_time_to_hire_days,
        avg_seconds_in_stage=avg_seconds_in_stage,
        by_source=by_source,
    )
