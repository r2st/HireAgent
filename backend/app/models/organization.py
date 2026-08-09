"""Tenants and their users."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import JSONColumn, UTCDateTime
from app.models.enums import OrganizationType, PlanTier, UserRole

if TYPE_CHECKING:
    from app.models.job import Job


class Organization(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A hiring company, recruitment agency, or staffing firm.

    The root of the tenant tree: every other tenant-owned row points here via
    ``organization_id``.
    """

    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    type: Mapped[OrganizationType] = mapped_column(
        String(32), default=OrganizationType.COMPANY, nullable=False
    )
    industry: Mapped[str | None] = mapped_column(String(120))
    size: Mapped[str | None] = mapped_column(String(40))
    website: Mapped[str | None] = mapped_column(String(255))
    country: Mapped[str | None] = mapped_column(String(2))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)

    plan: Mapped[PlanTier] = mapped_column(
        String(32), default=PlanTier.STARTUP, nullable=False
    )
    screening_credits_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    screening_credits_reset_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime
    )

    # Free-form per-tenant configuration: scoring weight defaults, pipeline
    # customisations, notification preferences.
    settings_json: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)

    # Design §8.1: configurable retention, default 24 months from last activity.
    data_retention_months: Mapped[int] = mapped_column(
        Integer, default=24, nullable=False
    )

    users: Mapped[list[User]] = relationship(
        back_populates="organization", cascade="all, delete-orphan", lazy="selectin"
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="organization", lazy="noload")


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A recruiter, hiring manager, interviewer, or admin."""

    __tablename__ = "users"
    __table_args__ = (
        # Email is unique per tenant, not globally: the same person may work
        # with more than one organization.
        Index("uq_users_org_email", "organization_id", "email", unique=True),
        Index("ix_users_org_active", "organization_id", "deleted_at"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        String(32), default=UserRole.RECRUITER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # Custom per-user permission overrides on top of the role defaults.
    permissions_json: Mapped[dict] = mapped_column(
        JSONColumn, default=dict, nullable=False
    )

    organization: Mapped[Organization] = relationship(back_populates="users")
