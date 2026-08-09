"""Offer letter templates and generated offers."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
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
from app.models.enums import OfferStatus


class OfferTemplate(TenantBase):
    """A letter body with ``{{variable}}`` placeholders."""

    __tablename__ = "offer_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Letterhead / footer wrapped around the body when rendering to PDF.
    header_html: Mapped[str | None] = mapped_column(Text)
    footer_html: Mapped[str | None] = mapped_column(Text)
    variables: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class OfferLetter(TenantBase):
    """A generated offer for one application."""

    __tablename__ = "offer_letters"
    __extra_table_args__ = (
        Index("ix_offers_application", "application_id", "created_at"),
        Index("ix_offers_org_status", "organization_id", "status", "deleted_at"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("offer_templates.id", ondelete="SET NULL")
    )
    status: Mapped[OfferStatus] = mapped_column(
        String(24), default=OfferStatus.DRAFT, nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # --- Compensation (encrypted: sensitive to both candidate and employer) ---
    job_title: Mapped[str] = mapped_column(String(255), nullable=False)
    salary: Mapped[str | None] = mapped_column(EncryptedText)
    salary_amount: Mapped[float | None] = mapped_column(Numeric(14, 2))
    salary_currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    salary_period: Mapped[str] = mapped_column(
        String(16), default="annual", nullable=False
    )
    bonus_amount: Mapped[float | None] = mapped_column(Numeric(14, 2))
    equity: Mapped[str | None] = mapped_column(String(255))
    benefits_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)

    start_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    reporting_manager: Mapped[str | None] = mapped_column(String(255))
    work_location: Mapped[str | None] = mapped_column(String(255))

    # Rendered letter, kept so a re-render after a template edit cannot change
    # what the candidate was actually sent.
    rendered_body: Mapped[str | None] = mapped_column(EncryptedText)
    document_path: Mapped[str | None] = mapped_column(String(1000))

    # --- Approval and e-signature (design §4.7) ---
    approved_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    viewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    signed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    responded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    decline_reason: Mapped[str | None] = mapped_column(Text)

    esign_provider: Mapped[str | None] = mapped_column(String(32))  # docusign | digio
    esign_envelope_id: Mapped[str | None] = mapped_column(String(255), index=True)
    esign_status: Mapped[str | None] = mapped_column(String(40))
    access_token: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)

    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    template: Mapped[OfferTemplate | None] = relationship(lazy="noload")
