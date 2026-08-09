"""Multi-channel outreach: sender accounts, templates, sequences, and messages.

Models the TalentPing-style engine described in design §2.2 and §4.2 — sender
rotation with warm-up, multi-step sequences, and per-message delivery tracking.
"""

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
from app.db.types import EncryptedText, JSONColumn, UTCDateTime
from app.models.enums import (
    EmailProvider,
    EnrollmentStatus,
    MessageStatus,
    OutreachChannel,
    SequenceStatus,
    WarmupStatus,
)

if TYPE_CHECKING:
    from app.models.candidate import Candidate


class EmailAccount(TenantBase):
    """A sender mailbox used for candidate outreach.

    Multiple accounts per org enable the sender rotation and warm-up that keep
    deliverability high (design §2.2).
    """

    __tablename__ = "email_accounts"
    __extra_table_args__ = (
        Index("uq_email_accounts_org_email", "organization_id", "email", unique=True),
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    provider: Mapped[EmailProvider] = mapped_column(
        String(24), default=EmailProvider.SMTP, nullable=False
    )

    # SMTP/IMAP connection details; the password and OAuth tokens are encrypted.
    smtp_host: Mapped[str | None] = mapped_column(String(255))
    smtp_port: Mapped[int | None] = mapped_column(Integer)
    smtp_username: Mapped[str | None] = mapped_column(String(320))
    smtp_password: Mapped[str | None] = mapped_column(EncryptedText)
    imap_host: Mapped[str | None] = mapped_column(String(255))
    imap_port: Mapped[int | None] = mapped_column(Integer)
    oauth_access_token: Mapped[str | None] = mapped_column(EncryptedText)
    oauth_refresh_token: Mapped[str | None] = mapped_column(EncryptedText)
    oauth_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    warmup_status: Mapped[WarmupStatus] = mapped_column(
        String(24), default=WarmupStatus.NOT_STARTED, nullable=False
    )
    warmup_started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # Ramps up during warm-up; the scheduler will not exceed it.
    daily_limit: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    sent_today: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sent_today_date: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # 0-100; drops on bounces/spam complaints and gates rotation eligibility.
    reputation_score: Mapped[float] = mapped_column(
        Numeric(5, 2), default=100, nullable=False
    )
    bounce_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    complaint_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    @property
    def is_sendable(self) -> bool:
        """Whether the rotation may pick this account right now."""
        return (
            self.is_active
            and self.deleted_at is None
            and self.warmup_status in (WarmupStatus.WARMING, WarmupStatus.READY)
            and float(self.reputation_score) >= 50
            and self.sent_today < self.daily_limit
        )


class MessageTemplate(TenantBase):
    """A reusable message body with ``{{variable}}`` placeholders."""

    __tablename__ = "message_templates"
    __extra_table_args__ = (
        Index("ix_templates_org_channel", "organization_id", "channel", "deleted_at"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    channel: Mapped[OutreachChannel] = mapped_column(
        String(24), default=OutreachChannel.EMAIL, nullable=False
    )
    subject: Mapped[str | None] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text, nullable=False)
    body_html: Mapped[str | None] = mapped_column(Text)
    # Declared placeholders, for editor autocomplete and validation.
    variables: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    # WhatsApp Business requires pre-approved template names.
    provider_template_name: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(80))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class OutreachSequence(TenantBase):
    """A multi-step, multi-channel campaign targeting candidates."""

    __tablename__ = "outreach_sequences"
    __extra_table_args__ = (
        Index("ix_sequences_org_status", "organization_id", "status", "deleted_at"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[SequenceStatus] = mapped_column(
        String(24), default=SequenceStatus.DRAFT, nullable=False, index=True
    )
    # Sender pool for rotation; empty means "any sendable account in the org".
    sender_account_ids: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    # Rolling counters: sent, delivered, opened, replied, bounced.
    stats_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)

    # Send-window guard rails so outreach lands during business hours.
    send_window_start_hour: Mapped[int] = mapped_column(
        Integer, default=9, nullable=False
    )
    send_window_end_hour: Mapped[int] = mapped_column(
        Integer, default=18, nullable=False
    )
    send_on_weekends: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    # A reply halts all remaining steps for that candidate (design §4.2).
    stop_on_reply: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    daily_cap: Mapped[int | None] = mapped_column(Integer)

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    steps: Mapped[list[SequenceStep]] = relationship(
        back_populates="sequence",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="SequenceStep.step_order",
    )
    enrollments: Mapped[list[SequenceEnrollment]] = relationship(
        back_populates="sequence", cascade="all, delete-orphan", lazy="noload"
    )


class SequenceStep(TenantBase):
    """One step of a sequence: wait N days, then send via a channel."""

    __tablename__ = "sequence_steps"
    __extra_table_args__ = (
        Index("uq_step_order", "sequence_id", "step_order", unique=True),
    )

    sequence_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("outreach_sequences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[OutreachChannel] = mapped_column(
        String(24), default=OutreachChannel.EMAIL, nullable=False
    )
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("message_templates.id", ondelete="SET NULL")
    )
    # Delay measured from the completion of the previous step.
    delay_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    delay_hours: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # A/B testing of subject lines and bodies (design §4.2).
    variant_group: Mapped[str | None] = mapped_column(String(16))
    subject_override: Mapped[str | None] = mapped_column(String(500))
    body_override: Mapped[str | None] = mapped_column(Text)

    sequence: Mapped[OutreachSequence] = relationship(back_populates="steps")


class SequenceEnrollment(TenantBase):
    """One candidate's progress through one sequence."""

    __tablename__ = "sequence_enrollments"
    __extra_table_args__ = (
        Index("uq_enrollment", "sequence_id", "candidate_id", unique=True),
        Index("ix_enrollments_due", "status", "next_send_at"),
    )

    sequence_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("outreach_sequences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL")
    )
    status: Mapped[EnrollmentStatus] = mapped_column(
        String(24), default=EnrollmentStatus.ACTIVE, nullable=False, index=True
    )
    current_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_send_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, index=True
    )
    enrolled_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    replied_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    paused_reason: Mapped[str | None] = mapped_column(String(255))

    sequence: Mapped[OutreachSequence] = relationship(back_populates="enrollments")
    candidate: Mapped[Candidate] = relationship(lazy="noload")
    messages: Mapped[list[OutreachMessage]] = relationship(
        back_populates="enrollment", cascade="all, delete-orphan", lazy="noload"
    )


class OutreachMessage(TenantBase):
    """A single rendered message and its delivery lifecycle."""

    __tablename__ = "outreach_messages"
    __extra_table_args__ = (
        Index("ix_messages_enrollment", "enrollment_id", "created_at"),
        Index("ix_messages_org_status", "organization_id", "status", "created_at"),
    )

    enrollment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sequence_enrollments.id", ondelete="CASCADE"),
        index=True,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("sequence_steps.id", ondelete="SET NULL")
    )
    email_account_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("email_accounts.id", ondelete="SET NULL")
    )

    channel: Mapped[OutreachChannel] = mapped_column(
        String(24), default=OutreachChannel.EMAIL, nullable=False
    )
    status: Mapped[MessageStatus] = mapped_column(
        String(24), default=MessageStatus.QUEUED, nullable=False, index=True
    )
    # Rendered content is candidate-identifying; keep it encrypted.
    to_address: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    subject: Mapped[str | None] = mapped_column(EncryptedText)
    body: Mapped[str | None] = mapped_column(EncryptedText)
    variant_group: Mapped[str | None] = mapped_column(String(16))

    scheduled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    delivered_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    opened_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    clicked_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    replied_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    bounced_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    provider_message_id: Mapped[str | None] = mapped_column(String(500), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Correlates opens/clicks/replies back to this message.
    tracking_token: Mapped[str | None] = mapped_column(
        String(64), unique=True, index=True
    )

    enrollment: Mapped[SequenceEnrollment | None] = relationship(
        back_populates="messages"
    )
