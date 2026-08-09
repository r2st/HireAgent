"""Pipeline analytics response shapes (design §6.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.enums import PipelineStage


class FunnelStepOut(BaseModel):
    stage: PipelineStage
    reached: int
    conversion_from_previous: float | None = None


class PipelineAnalyticsOut(BaseModel):
    total_applications: int
    stage_counts: dict[str, int] = Field(default_factory=dict)
    funnel: list[FunnelStepOut] = Field(default_factory=list)
    hires: int = 0
    rejections: int = 0
    withdrawals: int = 0
    avg_time_to_hire_days: float | None = None
    avg_seconds_in_stage: dict[str, float] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
