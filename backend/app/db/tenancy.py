"""Tenant-scoped query helpers.

Multi-tenancy in HireAgent is enforced in one place: every read and write of a
tenant-owned table goes through these helpers, which pin ``organization_id``
and exclude soft-deleted rows. Handlers should never build a bare
``select(Model)`` against a tenant table.
"""

from __future__ import annotations

import uuid
from typing import TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement

from app.db.base import TenantBase

ModelT = TypeVar("ModelT", bound=TenantBase)


class TenantScopeError(RuntimeError):
    """Raised when a query would cross a tenant boundary."""


def scoped_select(
    model: type[ModelT],
    organization_id: uuid.UUID,
    *,
    include_deleted: bool = False,
) -> Select:
    """``SELECT`` restricted to one organization's live rows."""
    stmt = select(model).where(model.organization_id == organization_id)
    if not include_deleted:
        stmt = stmt.where(model.deleted_at.is_(None))
    return stmt


async def get_scoped(
    session: AsyncSession,
    model: type[ModelT],
    entity_id: uuid.UUID,
    organization_id: uuid.UUID,
    *,
    include_deleted: bool = False,
) -> ModelT | None:
    """Fetch one row by id, but only if it belongs to ``organization_id``.

    Returns ``None`` for both "does not exist" and "belongs to another tenant"
    so callers cannot use the API to probe for the existence of other orgs'
    records.
    """
    stmt = scoped_select(model, organization_id, include_deleted=include_deleted).where(
        model.id == entity_id
    )
    result = await session.execute(stmt.limit(1))
    return result.scalar_one_or_none()


async def count_scoped(
    session: AsyncSession,
    model: type[ModelT],
    organization_id: uuid.UUID,
    *extra_filters: ColumnElement[bool],
    include_deleted: bool = False,
) -> int:
    stmt = (
        select(func.count())
        .select_from(model)
        .where(model.organization_id == organization_id)
    )
    if not include_deleted:
        stmt = stmt.where(model.deleted_at.is_(None))
    for f in extra_filters:
        stmt = stmt.where(f)
    return int((await session.execute(stmt)).scalar_one())


def assert_same_org(entity: TenantBase, organization_id: uuid.UUID) -> None:
    """Guard for objects loaded via relationships rather than a scoped query."""
    if entity.organization_id != organization_id:
        raise TenantScopeError(
            f"{type(entity).__name__} {entity.id} belongs to another organization"
        )
