"""Job postings and their scoring configuration."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

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
from app.db.types import JSONColumn, UTCDateTime
from app.models.enums import (
    EmploymentType,
    JobStatus,
    SeniorityLevel,
    WorkMode,
)

if TYPE_CHECKING:
    from app.models.application import Application
    from app.models.organization import Organization

# Design §2.2 step 6: default candidate scoring weights.
DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "skills": 0.40,
    "experience": 0.30,
    "education": 0.15,
    "cultural": 0.15,
}


class Job(TenantBase):
    """An open position.

    ``requirements_json`` holds the structured requirement set the scoring
    engine matches against; ``scoring_weights`` lets each job override the
    platform defaults (design §4.1).
    """

    __tablename__ = "jobs"
    __extra_table_args__ = (
        Index("ix_jobs_org_status", "organization_id", "status"),
        Index("uq_jobs_org_slug", "organization_id", "slug", unique=True),
    )

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(280), nullable=False)
    department: Mapped[str | None] = mapped_column(String(120))
    location: Mapped[str | None] = mapped_column(String(255))
    work_mode: Mapped[WorkMode] = mapped_column(
        String(16), default=WorkMode.ONSITE, nullable=False
    )
    employment_type: Mapped[EmploymentType] = mapped_column(
        String(24), default=EmploymentType.FULL_TIME, nullable=False
    )
    seniority: Mapped[SeniorityLevel | None] = mapped_column(String(24))
    description: Mapped[str | None] = mapped_column(Text)

    status: Mapped[JobStatus] = mapped_column(
        String(24), default=JobStatus.DRAFT, nullable=False, index=True
    )
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    openings: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    min_experience_years: Mapped[float | None] = mapped_column(Numeric(4, 1))
    max_experience_years: Mapped[float | None] = mapped_column(Numeric(4, 1))
    salary_min: Mapped[float | None] = mapped_column(Numeric(14, 2))
    salary_max: Mapped[float | None] = mapped_column(Numeric(14, 2))
    salary_currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)

    # {"required_skills": [...], "preferred_skills": [...],
    #  "education": {...}, "certifications": [...], "responsibilities": [...]}
    requirements_json: Mapped[dict] = mapped_column(
        JSONColumn, default=dict, nullable=False
    )
    scoring_weights: Mapped[dict] = mapped_column(
        JSONColumn, default=lambda: dict(DEFAULT_SCORING_WEIGHTS), nullable=False
    )
    # Applications scoring at or above this advance automatically (design §4.1).
    auto_advance_threshold: Mapped[float | None] = mapped_column(Numeric(5, 2))
    # Applications below this are flagged for rejection review.
    auto_reject_threshold: Mapped[float | None] = mapped_column(Numeric(5, 2))

    hiring_manager_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    # Latest bias-analysis result for the description (design §4.9).
    bias_report_json: Mapped[dict | None] = mapped_column(JSONColumn)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    organization: Mapped[Organization] = relationship(
        back_populates="jobs", lazy="noload"
    )
    applications: Mapped[list[Application]] = relationship(
        back_populates="job", lazy="noload"
    )

    @property
    def effective_weights(self) -> dict[str, float]:
        """Job weights merged over the defaults and normalised to sum to 1."""
        merged = dict(DEFAULT_SCORING_WEIGHTS) | dict(self.scoring_weights or {})
        merged = {k: float(v) for k, v in merged.items() if float(v) >= 0}
        total = sum(merged.values())
        if total <= 0:
            return dict(DEFAULT_SCORING_WEIGHTS)
        return {k: v / total for k, v in merged.items()}
