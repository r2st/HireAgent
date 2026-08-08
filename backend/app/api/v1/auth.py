"""Authentication and user-management routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_current_user, get_session, require_permission
from app.core.errors import AppError, ValidationError
from app.schemas.auth import (
    AuthResponse,
    ChangePasswordRequest,
    InviteUserRequest,
    LoginRequest,
    OrganizationOut,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UpdateUserRequest,
    UserOut,
)
from app.schemas.common import MessageResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest, session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    """Create a new organization and its first admin user."""
    try:
        user, org, tokens = await auth_service.register(session, payload)
    except AppError as exc:
        raise exc.to_http() from exc
    return AuthResponse(
        user=UserOut.model_validate(user),
        organization=OrganizationOut.model_validate(org),
        tokens=tokens,
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest, session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    try:
        user, org, tokens = await auth_service.authenticate(session, payload)
    except AppError as exc:
        raise exc.to_http() from exc
    return AuthResponse(
        user=UserOut.model_validate(user),
        organization=OrganizationOut.model_validate(org),
        tokens=tokens,
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    payload: RefreshRequest, session: AsyncSession = Depends(get_session)
) -> TokenPair:
    try:
        return await auth_service.refresh_tokens(session, payload.refresh_token)
    except AppError as exc:
        raise exc.to_http() from exc


@router.get("/me", response_model=UserOut)
async def me(
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    users = await auth_service.list_users(session, current.organization_id)
    user = next(u for u in users if u.id == current.id)
    return UserOut.model_validate(user)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    payload: ChangePasswordRequest,
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await auth_service.change_password(
            session,
            current.organization_id,
            current.id,
            payload.current_password,
            payload.new_password,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Password updated")


# --------------------------------------------------------------------------- #
# User administration (admin only)
# --------------------------------------------------------------------------- #
@router.get("/users", response_model=list[UserOut])
async def list_users(
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[UserOut]:
    users = await auth_service.list_users(session, current.organization_id)
    return [UserOut.model_validate(u) for u in users]


@router.post(
    "/users", response_model=UserOut, status_code=status.HTTP_201_CREATED
)
async def invite_user(
    payload: InviteUserRequest,
    current: CurrentUser = Depends(require_permission("user:create")),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    try:
        user = await auth_service.create_user(
            session, current.organization_id, payload
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return UserOut.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID,
    payload: UpdateUserRequest,
    current: CurrentUser = Depends(require_permission("user:update")),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    try:
        user = await auth_service.update_user(
            session, current.organization_id, user_id, payload
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return UserOut.model_validate(user)


@router.delete("/users/{user_id}", response_model=MessageResponse)
async def deactivate_user(
    user_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("user:delete")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    if user_id == current.id:
        raise ValidationError("You cannot remove your own account").to_http()
    try:
        await auth_service.deactivate_user(session, current.organization_id, user_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="User removed")
