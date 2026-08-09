"""Declarative base and the mixins every table in HireAgent shares."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import ForeignKey, Index, Uuid, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
)

from app.db.types import UTCDateTime


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Root declarative class."""

    @declared_attr.directive
    def __tablename__(cls) -> str:  # noqa: N805
        # CamelCase -> snake_case, naive pluralisation. Models that need a
        # different name set __tablename__ explicitly.
        name = cls.__name__
        out = [name[0].lower()]
        for ch in name[1:]:
            out.append("_" + ch.lower() if ch.isupper() else ch)
        return "".join(out) + "s"


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        server_default=func.now(),
        onupdate=utcnow,
        nullable=False,
    )


class SoftDeleteMixin:
    """Soft delete only — rows are never physically removed.

    Every read path must filter ``deleted_at IS NULL``; use the helpers in
    ``app.db.tenancy`` rather than hand-writing the predicate.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True, index=True
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        if self.deleted_at is None:
            self.deleted_at = utcnow()


class OrganizationScopedMixin:
    """Adds the tenant discriminator.

    Every tenant-owned table carries ``organization_id`` and every query must
    filter on it (design §3.1).
    """

    @declared_attr
    def organization_id(cls) -> Mapped[uuid.UUID]:  # noqa: N805
        return mapped_column(
            Uuid(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )

    # Models add their own constraints/indexes here; the org+active composite
    # below is appended automatically so subclasses never have to repeat it.
    __extra_table_args__: tuple = ()

    @declared_attr.directive
    def __table_args__(cls) -> tuple:  # noqa: N805
        # Composite index matching the shape of nearly every query:
        # "rows for this org that are not deleted".
        table_name = getattr(cls, "__tablename__", type(cls).__name__.lower())
        return (
            *cls.__extra_table_args__,
            Index(f"ix_{table_name}_org_active", "organization_id", "deleted_at"),
        )


class TenantBase(
    Base,
    UUIDPrimaryKeyMixin,
    OrganizationScopedMixin,
    TimestampMixin,
    SoftDeleteMixin,
):
    """Base for all tenant-owned tables."""

    __abstract__ = True
