"""Skill assessments sent to candidates."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TenantBase
from app.db.types import EncryptedJSON, JSONColumn, UTCDateTime
from app.models.enums import AssessmentStatus, AssessmentType


class AssessmentTemplate(TenantBase):
    """A reusable assessment definition."""

    __tablename__ = "assessment_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[AssessmentType] = mapped_column(
        String(32), default=AssessmentType.MCQ, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)
    # [{"id": ..., "prompt": ..., "type": ..., "options": [...],
    #   "expected": ..., "weight": ...}]
    questions_json: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    duration_minutes: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    passing_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=60, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Assessment(TenantBase):
    """An assessment instance issued to one application."""

    __tablename__ = "assessments"
    __extra_table_args__ = (
        Index("ix_assessments_application", "application_id", "created_at"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("assessment_templates.id", ondelete="SET NULL")
    )
    type: Mapped[AssessmentType] = mapped_column(
        String(32), default=AssessmentType.MCQ, nullable=False
    )
    status: Mapped[AssessmentStatus] = mapped_column(
        String(24), default=AssessmentStatus.PENDING, nullable=False, index=True
    )

    # The paper exactly as it was issued. Snapshotted from the template rather
    # than read through ``template_id`` at grading time: editing a template must
    # not silently rewrite the questions somebody is part-way through answering,
    # nor change the mark of one already sat.
    questions_json: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    passing_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=60, nullable=False
    )
    duration_minutes: Mapped[int | None] = mapped_column(Integer)

    score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    max_score: Mapped[float] = mapped_column(Numeric(5, 2), default=100, nullable=False)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    # Candidate answers are personal data — encrypted at rest.
    responses_json: Mapped[dict | None] = mapped_column(EncryptedJSON)
    # Per-question breakdown produced by the grader.
    breakdown_json: Mapped[dict | None] = mapped_column(JSONColumn)
    ai_feedback: Mapped[str | None] = mapped_column(Text)

    # Single-use link the candidate follows to take the assessment.
    invite_token: Mapped[str | None] = mapped_column(
        String(64), unique=True, index=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    time_spent_seconds: Mapped[int | None] = mapped_column(Integer)

    template: Mapped[AssessmentTemplate | None] = relationship(lazy="noload")
