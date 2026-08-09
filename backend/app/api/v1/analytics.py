"""Pipeline analytics routes (design §6.1)."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_session, require_permission
from app.schemas.analytics import FunnelStepOut, PipelineAnalyticsOut
from app.services import analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pipeline", response_model=PipelineAnalyticsOut)
async def pipeline_analytics(
    job_id: uuid.UUID | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    current: CurrentUser = Depends(require_permission("analytics:read")),
    session: AsyncSession = Depends(get_session),
) -> PipelineAnalyticsOut:
    """Conversion funnel, time-in-stage, and source mix for the organization.

    Computed live from ``applications``/``stage_events`` rather than a
    pre-aggregated rollup — see the module docstring in ``analytics_service``.
    """
    summary = await analytics_service.pipeline_summary(
        session, current.organization_id, job_id=job_id, since=since, until=until
    )
    return PipelineAnalyticsOut(
        total_applications=summary.total_applications,
        stage_counts=summary.stage_counts,
        funnel=[
            FunnelStepOut(
                stage=step.stage,
                reached=step.reached,
                conversion_from_previous=step.conversion_from_previous,
            )
            for step in summary.funnel
        ],
        hires=summary.hires,
        rejections=summary.rejections,
        withdrawals=summary.withdrawals,
        avg_time_to_hire_days=summary.avg_time_to_hire_days,
        avg_seconds_in_stage=summary.avg_seconds_in_stage,
        by_source=summary.by_source,
    )
