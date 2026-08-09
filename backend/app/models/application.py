"""Applications (the job↔candidate association) and AI screening results."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TenantBase
from app.db.types import JSONColumn, UTCDateTime
from app.models.enums import (
    ApplicationStatus,
    CandidateSource,
    PipelineStage,
)

if TYPE_CHECKING:
    from app.models.candidate import Candidate
    from app.models.job import Job


class Application(TenantBase):
    """One candidate's candidacy for one job — the unit the pipeline moves."""

    __tablename__ = "applications"
    __extra_table_args__ = (
        # A candidate may only apply to a given job once.
        Index("uq_applications_job_candidate", "job_id", "candidate_id", unique=True),
        Index("ix_applications_job_stage", "job_id", "stage", "deleted_at"),
        Index("ix_applications_org_stage", "organization_id", "stage", "deleted_at"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    stage: Mapped[PipelineStage] = mapped_column(
        String(24), default=PipelineStage.APPLIED, nullable=False, index=True
    )
    status: Mapped[ApplicationStatus] = mapped_column(
        String(24), default=ApplicationStatus.ACTIVE, nullable=False, index=True
    )
    source: Mapped[CandidateSource] = mapped_column(
        String(32), default=CandidateSource.DIRECT, nullable=False, index=True
    )
    source_detail: Mapped[str | None] = mapped_column(String(255))

    # Denormalised copy of the latest screening's overall score, so the pipeline
    # board can sort without joining screenings.
    score: Mapped[float | None] = mapped_column(Numeric(5, 2), index=True)

    applied_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    stage_changed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    hired_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    rejected_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    rejection_reason: Mapped[str | None] = mapped_column(Text)

    assigned_to_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # Manual ordering within a Kanban column.
    board_position: Mapped[float] = mapped_column(
        Numeric(12, 4), default=0, nullable=False
    )
    metadata_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)

    job: Mapped[Job] = relationship(back_populates="applications", lazy="noload")
    candidate: Mapped[Candidate] = relationship(
        back_populates="applications", lazy="noload"
    )
    screenings: Mapped[list[Screening]] = relationship(
        back_populates="application", cascade="all, delete-orphan", lazy="noload"
    )
    stage_events: Mapped[list[StageEvent]] = relationship(
        back_populates="application", cascade="all, delete-orphan", lazy="noload"
    )

    @property
    def is_open(self) -> bool:
        return self.status == ApplicationStatus.ACTIVE and self.deleted_at is None


class Screening(TenantBase):
    """An AI screening run scoring one application against its job.

    Screenings are append-only: re-screening writes a new row so score history
    is preserved and results stay reproducible/auditable.
    """

    __tablename__ = "screenings"
    __extra_table_args__ = (
        Index("ix_screenings_application", "application_id", "created_at"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Component scores, each 0-100 (design §2.2 step 6).
    skill_match: Mapped[float] = mapped_column(Numeric(5, 2), default=0, nullable=False)
    experience_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=0, nullable=False
    )
    education_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=0, nullable=False
    )
    cultural_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=0, nullable=False
    )
    overall_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=0, nullable=False, index=True
    )

    # Weights actually applied, snapshotted so an old score stays explainable
    # after the job's weights are edited.
    weights_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text)
    matched_skills: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    missing_skills: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    strengths: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    concerns: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)

    # Below 0.70 the result is surfaced for human review (design §5).
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), default=0, nullable=False)
    requires_human_review: Mapped[bool] = mapped_column(
        default=False, nullable=False
    )
    model_used: Mapped[str | None] = mapped_column(String(120))
    # "llm" when the model answered, "heuristic" on fallback.
    engine: Mapped[str] = mapped_column(String(24), default="llm", nullable=False)
    latency_ms: Mapped[int | None] = mapped_column()

    application: Mapped[Application] = relationship(
        back_populates="screenings", lazy="noload"
    )


class StageEvent(TenantBase):
    """Audit trail of pipeline stage transitions.

    Powers time-in-stage and conversion analytics (design §4.5, §6.1).
    """

    __tablename__ = "stage_events"
    __extra_table_args__ = (
        Index("ix_stage_events_application", "application_id", "created_at"),
        Index("ix_stage_events_org_to", "organization_id", "to_stage", "created_at"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_stage: Mapped[PipelineStage | None] = mapped_column(String(24))
    to_stage: Mapped[PipelineStage] = mapped_column(String(24), nullable=False)
    changed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # "manual", "auto_advance", "auto_reject", "bulk", "import"
    trigger: Mapped[str] = mapped_column(String(32), default="manual", nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    # Seconds spent in ``from_stage``, computed at transition time.
    seconds_in_previous_stage: Mapped[int | None] = mapped_column()

    application: Mapped[Application] = relationship(
        back_populates="stage_events", lazy="noload"
    )
