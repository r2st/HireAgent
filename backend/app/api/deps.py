"""Shared FastAPI dependencies: authentication, tenancy, and authorization.

``get_current_user`` is the single place a request acquires its tenant
identity. Route handlers receive a ``CurrentUser`` and must pass
``current.organization_id`` into every query.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AuthenticationError, PermissionDeniedError
from app.core.permissions import has_permission
from app.core.security import TokenError, decode_token
from app.db.session import get_db
from app.models.enums import UserRole
from app.models.organization import Organization, User

# auto_error=False so a missing header raises our own 401 shape.
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated principal and its tenant context."""

    id: uuid.UUID
    organization_id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    permissions_json: dict

    def can(self, permission: str) -> bool:
        return has_permission(self.role, permission, self.permissions_json)

    def require(self, permission: str) -> None:
        if not self.can(permission):
            raise PermissionDeniedError(
                f"Role '{self.role}' lacks permission '{permission}'"
            ).to_http()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_session),
) -> CurrentUser:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token").to_http()

    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise AuthenticationError(str(exc)).to_http() from exc

    try:
        user_id = uuid.UUID(payload["sub"])
        org_id = uuid.UUID(payload["org"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AuthenticationError("Malformed token claims").to_http() from exc

    # Re-read the user rather than trusting the token's role claim: a
    # deactivated user or a demoted role must take effect before the token
    # expires.
    result = await session.execute(
        select(User).where(
            User.id == user_id,
            User.organization_id == org_id,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise AuthenticationError("User no longer exists").to_http()
    if not user.is_active:
        raise AuthenticationError("User account is disabled").to_http()

    return CurrentUser(
        id=user.id,
        organization_id=user.organization_id,
        email=user.email,
        full_name=user.full_name,
        role=UserRole(user.role),
        permissions_json=dict(user.permissions_json or {}),
    )


async def get_current_organization(
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Organization:
    result = await session.execute(
        select(Organization).where(
            Organization.id == current.organization_id,
            Organization.deleted_at.is_(None),
        )
    )
    org = result.scalar_one_or_none()
    if org is None:
        raise AuthenticationError("Organization not found").to_http()
    return org


def require_permission(permission: str):
    """Dependency factory gating a route on a single permission."""

    async def _dep(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        current.require(permission)
        return current

    return _dep


def require_roles(*roles: UserRole):
    """Dependency factory gating a route on membership in a role set."""

    async def _dep(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current.role not in roles:
            raise PermissionDeniedError(
                f"Requires one of: {', '.join(r.value for r in roles)}"
            ).to_http()
        return current

    return _dep


def client_ip(request: Request) -> str | None:
    """Best-effort client IP, honouring a reverse proxy's X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None
