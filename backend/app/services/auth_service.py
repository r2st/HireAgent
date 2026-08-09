"""Registration, login, token refresh, and user administration."""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from app.core.security import (
    TokenError,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.config import settings
from app.models.enums import UserRole
from app.models.organization import Organization, User
from app.schemas.auth import (
    InviteUserRequest,
    LoginRequest,
    RegisterRequest,
    TokenPair,
    UpdateUserRequest,
)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _SLUG_STRIP.sub("-", value.lower()).strip("-")
    return slug or "org"


async def _unique_org_slug(session: AsyncSession, name: str) -> str:
    """Derive a slug from ``name``, suffixing until it is free."""
    base = slugify(name)[:100]
    candidate = base
    for _ in range(50):
        exists = await session.scalar(
            select(func.count())
            .select_from(Organization)
            .where(Organization.slug == candidate)
        )
        if not exists:
            return candidate
        candidate = f"{base}-{secrets.token_hex(3)}"
    raise ConflictError("Could not allocate a unique organization slug")


def issue_tokens(user: User) -> TokenPair:
    access = create_token(
        subject=user.id,
        organization_id=user.organization_id,
        role=str(user.role),
        token_type="access",
    )
    refresh = create_token(
        subject=user.id,
        organization_id=user.organization_id,
        role=str(user.role),
        token_type="refresh",
    )
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


async def register(
    session: AsyncSession, payload: RegisterRequest
) -> tuple[User, Organization, TokenPair]:
    """Create an organization plus its first admin user.

    The registering user is always an admin — there is nobody else to grant
    them access.
    """
    email = payload.email.lower().strip()

    organization = Organization(
        name=payload.organization_name.strip(),
        slug=await _unique_org_slug(session, payload.organization_name),
        type=payload.organization_type,
        industry=payload.industry,
        settings_json={},
        data_retention_months=settings.default_retention_months,
    )
    session.add(organization)
    await session.flush()

    user = User(
        organization_id=organization.id,
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name.strip(),
        role=UserRole.ADMIN,
        is_active=True,
        permissions_json={},
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    await session.refresh(organization)

    return user, organization, issue_tokens(user)


async def authenticate(
    session: AsyncSession, payload: LoginRequest
) -> tuple[User, Organization, TokenPair]:
    """Verify credentials and mint a token pair.

    Emails are unique per organization, not globally, so a user belonging to
    several organizations must disambiguate with ``organization_slug``.
    """
    email = payload.email.lower().strip()

    stmt = select(User).where(User.email == email, User.deleted_at.is_(None))
    if payload.organization_slug:
        stmt = stmt.join(Organization, Organization.id == User.organization_id).where(
            Organization.slug == payload.organization_slug.lower().strip(),
            Organization.deleted_at.is_(None),
        )
    users = list((await session.execute(stmt)).scalars().all())

    if len(users) > 1:
        raise ValidationError(
            "This email belongs to multiple organizations; "
            "supply organization_slug to disambiguate."
        )

    user = users[0] if users else None

    # Always run a hash comparison, even when no user matched, so response
    # timing does not reveal whether an account exists.
    hashed = user.hashed_password if user else hash_password("invalid-placeholder")
    password_ok = verify_password(payload.password, hashed)

    if user is None or not password_ok:
        raise AuthenticationError("Invalid email or password")
    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    organization = await session.get(Organization, user.organization_id)
    if organization is None or organization.deleted_at is not None:
        raise AuthenticationError("Organization is no longer active")

    user.last_login_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(user)

    return user, organization, issue_tokens(user)


async def refresh_tokens(session: AsyncSession, refresh_token: str) -> TokenPair:
    try:
        payload = decode_token(refresh_token, expected_type="refresh")
    except TokenError as exc:
        raise AuthenticationError(str(exc)) from exc

    try:
        user_id = uuid.UUID(payload["sub"])
        org_id = uuid.UUID(payload["org"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AuthenticationError("Malformed refresh token") from exc

    user = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.organization_id == org_id,
            User.deleted_at.is_(None),
        )
    )
    if user is None or not user.is_active:
        raise AuthenticationError("Refresh token is no longer valid")
    return issue_tokens(user)


async def create_user(
    session: AsyncSession, organization_id: uuid.UUID, payload: InviteUserRequest
) -> User:
    email = payload.email.lower().strip()
    existing = await session.scalar(
        select(func.count())
        .select_from(User)
        .where(
            User.organization_id == organization_id,
            User.email == email,
            User.deleted_at.is_(None),
        )
    )
    if existing:
        raise ConflictError(f"A user with email {email} already exists")

    user = User(
        organization_id=organization_id,
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name.strip(),
        role=payload.role,
        is_active=True,
        permissions_json={},
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user(
    session: AsyncSession, organization_id: uuid.UUID, user_id: uuid.UUID
) -> User:
    """Fetch one user within a tenant.

    ``User`` predates ``TenantBase`` — it hangs off ``Organization`` directly —
    so it cannot use ``get_scoped``; the org filter is spelled out instead.
    """
    user = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.organization_id == organization_id,
            User.deleted_at.is_(None),
        )
    )
    if user is None:
        raise NotFoundError("User not found")
    return user


async def get_users(
    session: AsyncSession, organization_id: uuid.UUID, user_ids: list[uuid.UUID]
) -> list[User]:
    """Fetch several users in one query, preserving the caller's ordering.

    Ids belonging to another tenant simply do not come back, so callers should
    compare lengths rather than assuming a full result.
    """
    if not user_ids:
        return []
    rows = (
        (
            await session.execute(
                select(User).where(
                    User.id.in_(user_ids),
                    User.organization_id == organization_id,
                    User.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    by_id = {u.id: u for u in rows}
    return [by_id[i] for i in user_ids if i in by_id]


async def list_users(
    session: AsyncSession, organization_id: uuid.UUID
) -> list[User]:
    result = await session.execute(
        select(User)
        .where(User.organization_id == organization_id, User.deleted_at.is_(None))
        .order_by(User.created_at)
    )
    return list(result.scalars().all())


async def _count_active_admins(
    session: AsyncSession, organization_id: uuid.UUID, *, exclude: uuid.UUID
) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.organization_id == organization_id,
                User.role == UserRole.ADMIN,
                User.is_active.is_(True),
                User.deleted_at.is_(None),
                User.id != exclude,
            )
        )
        or 0
    )


async def update_user(
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: UpdateUserRequest,
) -> User:
    user = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.organization_id == organization_id,
            User.deleted_at.is_(None),
        )
    )
    if user is None:
        raise NotFoundError("User not found")

    # Guard against an organization locking itself out of admin access.
    demoting = payload.role is not None and payload.role != UserRole.ADMIN
    deactivating = payload.is_active is False
    if user.role == UserRole.ADMIN and (demoting or deactivating):
        if await _count_active_admins(session, organization_id, exclude=user_id) == 0:
            raise PermissionDeniedError(
                "Cannot remove the last active admin of the organization"
            )

    if payload.full_name is not None:
        user.full_name = payload.full_name.strip()
    if payload.role is not None:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active

    await session.commit()
    await session.refresh(user)
    return user


async def deactivate_user(
    session: AsyncSession, organization_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Soft delete — the row is retained for audit history."""
    user = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.organization_id == organization_id,
            User.deleted_at.is_(None),
        )
    )
    if user is None:
        raise NotFoundError("User not found")
    if (
        user.role == UserRole.ADMIN
        and await _count_active_admins(session, organization_id, exclude=user_id) == 0
    ):
        raise PermissionDeniedError(
            "Cannot remove the last active admin of the organization"
        )

    user.soft_delete()
    user.is_active = False
    await session.commit()


async def change_password(
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
    current_password: str,
    new_password: str,
) -> None:
    user = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.organization_id == organization_id,
            User.deleted_at.is_(None),
        )
    )
    if user is None:
        raise NotFoundError("User not found")
    if not verify_password(current_password, user.hashed_password):
        raise AuthenticationError("Current password is incorrect")

    user.hashed_password = hash_password(new_password)
    await session.commit()
