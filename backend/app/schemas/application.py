"""Application, screening, and pipeline-board schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.enums import (
    ApplicationStatus,
    CandidateSource,
    PipelineStage,
)
from app.schemas.candidate import CandidateSummary
from app.schemas.common import ORMModel
from app.schemas.job import JobSummary


class ApplicationCreate(BaseModel):
    job_id: uuid.UUID
    candidate_id: uuid.UUID
    source: CandidateSource = CandidateSource.DIRECT
    source_detail: str | None = Field(default=None, max_length=255)
    stage: PipelineStage = PipelineStage.APPLIED
    assigned_to_id: uuid.UUID | None = None
    metadata: dict = Field(default_factory=dict)


class ApplicationUpdate(BaseModel):
    assigned_to_id: uuid.UUID | None = None
    source_detail: str | None = Field(default=None, max_length=255)


class StageMove(BaseModel):
    stage: PipelineStage
    note: str | None = Field(default=None, max_length=2000)
    board_position: float | None = Field(default=None, ge=0)
    # Skipping screening on the way to an offer requires an explicit override.
    force: bool = False


class BulkStageMove(BaseModel):
    application_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)
    stage: PipelineStage
    note: str | None = Field(default=None, max_length=2000)
    force: bool = False


class RejectRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class AssignRequest(BaseModel):
    user_id: uuid.UUID | None = None


class ScreenRequest(BaseModel):
    # When false, the job's auto-advance/auto-reject thresholds are not applied.
    auto_advance: bool = True


class BulkScreenRequest(BaseModel):
    stage: PipelineStage | None = None
    # Re-score applications that already have a score.
    rescore: bool = False
    limit: int = Field(default=200, ge=1, le=1000)


class ApplicationOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    job_id: uuid.UUID
    candidate_id: uuid.UUID
    stage: PipelineStage
    status: ApplicationStatus
    source: CandidateSource
    source_detail: str | None = None
    score: Decimal | None = None
    applied_at: datetime
    stage_changed_at: datetime | None = None
    hired_at: datetime | None = None
    rejected_at: datetime | None = None
    rejection_reason: str | None = None
    assigned_to_id: uuid.UUID | None = None
    board_position: Decimal
    created_at: datetime
    updated_at: datetime


class ApplicationDetail(ApplicationOut):
    """An application plus the records a reviewer needs alongside it."""

    candidate: CandidateSummary | None = None
    job: JobSummary | None = None
    latest_screening: "ScreeningOut | None" = None


class ScreeningOut(ORMModel):
    id: uuid.UUID
    application_id: uuid.UUID
    skill_match: Decimal
    experience_score: Decimal
    education_score: Decimal
    cultural_score: Decimal
    overall_score: Decimal
    weights_json: dict = Field(default_factory=dict)
    reasoning: str | None = None
    matched_skills: list = Field(default_factory=list)
    missing_skills: list = Field(default_factory=list)
    strengths: list = Field(default_factory=list)
    concerns: list = Field(default_factory=list)
    confidence: Decimal
    requires_human_review: bool
    model_used: str | None = None
    engine: str
    latency_ms: int | None = None
    created_at: datetime


class ScreeningResult(BaseModel):
    screening: ScreeningOut
    application: ApplicationOut


class BulkScreenResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    screenings: list[ScreeningOut] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class BulkStageResult(BaseModel):
    total: int
    moved: int
    failed: int
    applications: list[ApplicationOut] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class StageEventOut(ORMModel):
    id: uuid.UUID
    application_id: uuid.UUID
    from_stage: PipelineStage | None = None
    to_stage: PipelineStage
    changed_by_id: uuid.UUID | None = None
    trigger: str
    note: str | None = None
    seconds_in_previous_stage: int | None = None
    created_at: datetime


class BoardCard(BaseModel):
    """One Kanban card: the application plus enough candidate detail to read it."""

    application: ApplicationOut
    candidate: CandidateSummary | None = None


class BoardColumn(BaseModel):
    stage: PipelineStage
    total: int
    applications: list[BoardCard] = Field(default_factory=list)


class BoardOut(BaseModel):
    job: JobSummary
    columns: list[BoardColumn]
    total: int


class RankedCandidate(BaseModel):
    """A row of ``GET /jobs/{id}/candidates`` — the design's ranked shortlist."""

    application: ApplicationOut
    candidate: CandidateSummary
    screening: ScreeningOut | None = None


ApplicationDetail.model_rebuild()
