"""Interviews, interviewer participation, and connected calendar accounts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
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
from app.db.types import EncryptedText, JSONColumn
from app.models.enums import InterviewStatus, InterviewType

if TYPE_CHECKING:
    from app.models.application import Application


class CalendarAccount(TenantBase):
    """A connected Google or Outlook calendar used for availability and booking."""

    __tablename__ = "calendar_accounts"
    __extra_table_args__ = (
        Index("uq_calendar_user_provider", "user_id", "provider", "email", unique=True),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)  # google | outlook
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    calendar_id: Mapped[str | None] = mapped_column(String(255))

    # OAuth material is credentials, not display data — always encrypted.
    access_token: Mapped[str | None] = mapped_column(EncryptedText)
    refresh_token: Mapped[str | None] = mapped_column(EncryptedText)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sync_error: Mapped[str | None] = mapped_column(Text)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Working hours used when proposing slots, e.g.
    # {"mon": [["09:00","17:00"]], ...} in ``timezone``.
    working_hours: Mapped[dict] = mapped_column(
        JSONColumn, default=dict, nullable=False
    )
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)


class Interview(TenantBase):
    """A scheduled interview for an application."""

    __tablename__ = "interviews"
    __extra_table_args__ = (
        Index("ix_interviews_application", "application_id", "scheduled_at"),
        Index("ix_interviews_org_scheduled", "organization_id", "scheduled_at", "status"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[InterviewType] = mapped_column(
        String(24), default=InterviewType.VIDEO, nullable=False
    )
    status: Mapped[InterviewStatus] = mapped_column(
        String(24), default=InterviewStatus.PENDING, nullable=False, index=True
    )
    round_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))

    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    duration_minutes: Mapped[int] = mapped_column(Integer, default=45, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    location: Mapped[str | None] = mapped_column(String(500))
    meeting_url: Mapped[str | None] = mapped_column(String(1000))

    # Slots offered to the candidate before they picked one:
    # [{"start": iso, "end": iso}, ...]
    proposed_slots: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    # Opaque token letting the candidate self-serve booking/reschedule without
    # an account (design §4.3).
    booking_token: Mapped[str | None] = mapped_column(
        String(64), unique=True, index=True
    )
    booking_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    calendar_account_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("calendar_accounts.id", ondelete="SET NULL")
    )
    external_event_id: Mapped[str | None] = mapped_column(String(255))

    reminder_24h_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    reminder_1h_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancellation_reason: Mapped[str | None] = mapped_column(Text)
    rescheduled_from_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("interviews.id", ondelete="SET NULL")
    )

    # --- Async video screening analysis (design §4.4) ---
    recording_url: Mapped[str | None] = mapped_column(String(1000))
    transcript: Mapped[str | None] = mapped_column(EncryptedText)
    analysis_json: Mapped[dict | None] = mapped_column(JSONColumn)
    ai_score: Mapped[float | None] = mapped_column(Numeric(5, 2))

    notes: Mapped[str | None] = mapped_column(Text)

    application: Mapped[Application] = relationship(lazy="noload")
    participants: Mapped[list[InterviewParticipant]] = relationship(
        back_populates="interview", cascade="all, delete-orphan", lazy="selectin"
    )


class InterviewParticipant(TenantBase):
    """An interviewer assigned to an interview, plus their feedback."""

    __tablename__ = "interview_participants"
    __extra_table_args__ = (
        Index("uq_participant", "interview_id", "user_id", unique=True),
    )

    interview_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_organizer: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    response_status: Mapped[str] = mapped_column(
        String(24), default="pending", nullable=False
    )  # pending | accepted | declined | tentative

    # Scorecard filled in after the interview.
    rating: Mapped[float | None] = mapped_column(Numeric(4, 2))
    recommendation: Mapped[str | None] = mapped_column(String(32))
    feedback: Mapped[str | None] = mapped_column(Text)
    feedback_submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    scorecard_json: Mapped[dict | None] = mapped_column(JSONColumn)

    interview: Mapped[Interview] = relationship(back_populates="participants")
