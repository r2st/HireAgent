"""Daily rollups and job-board postings."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import TenantBase
from app.db.types import JSONColumn
from app.models.enums import JobBoard, PostingStatus


class AnalyticsSnapshot(TenantBase):
    """A per-day, per-job rollup of funnel counters.

    Live queries against applications/stage_events stay correct but get slow as
    history grows; this table is written nightly (and on demand) so dashboards
    read pre-aggregated rows.
    """

    __tablename__ = "analytics"
    __extra_table_args__ = (
        Index("uq_analytics_org_job_date", "organization_id", "job_id", "date", unique=True),
        Index("ix_analytics_org_date", "organization_id", "date"),
    )

    # NULL job_id means "all jobs" — the organization-wide rollup for that day.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)

    applications: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    screenings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    interviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    assessments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    offers: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hires: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    messages_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    messages_opened: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    messages_replied: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    avg_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    avg_time_to_hire_days: Mapped[float | None] = mapped_column(Numeric(8, 2))
    # Per-source counts, e.g. {"naukri": {"applications": 12, "hires": 1}}
    source_breakdown: Mapped[dict] = mapped_column(
        JSONColumn, default=dict, nullable=False
    )
    stage_counts: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)


class JobBoardPosting(TenantBase):
    """A job's presence on one external board (design §4.6)."""

    __tablename__ = "job_board_postings"
    __extra_table_args__ = (
        Index("uq_posting_job_board", "job_id", "board", unique=True),
        Index("ix_postings_org_board", "organization_id", "board", "status"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    board: Mapped[JobBoard] = mapped_column(String(24), nullable=False)
    status: Mapped[PostingStatus] = mapped_column(
        String(24), default=PostingStatus.PENDING, nullable=False, index=True
    )
    external_id: Mapped[str | None] = mapped_column(String(255), index=True)
    external_url: Mapped[str | None] = mapped_column(String(1000))

    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Watermark for incremental application imports.
    last_import_cursor: Mapped[str | None] = mapped_column(String(255))
    imported_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    views: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Board-specific fields (category ids, industry codes, etc.).
    payload_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
