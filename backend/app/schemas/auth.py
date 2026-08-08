"""Authentication and user schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models.enums import OrganizationType, PlanTier, UserRole
from app.schemas.common import ORMModel


def _validate_password_strength(v: str) -> str:
    if len(v) < 10:
        raise ValueError("Password must be at least 10 characters")
    if len(v.encode("utf-8")) > 72:
        # bcrypt truncates beyond 72 bytes; reject rather than silently ignore.
        raise ValueError("Password must be at most 72 bytes")
    if not any(c.isalpha() for c in v):
        raise ValueError("Password must contain a letter")
    if not any(c.isdigit() for c in v):
        raise ValueError("Password must contain a digit")
    return v


class RegisterRequest(BaseModel):
    """Creates an organization and its first admin user in one call."""

    organization_name: str = Field(min_length=2, max_length=255)
    organization_type: OrganizationType = OrganizationType.COMPANY
    industry: str | None = Field(default=None, max_length=120)
    full_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str

    _check_password = field_validator("password")(_validate_password_strength)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    # Required only when the same email exists in more than one organization.
    organization_slug: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime


class OrganizationOut(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    type: OrganizationType
    industry: str | None = None
    size: str | None = None
    plan: PlanTier
    timezone: str
    data_retention_months: int
    created_at: datetime


class AuthResponse(BaseModel):
    user: UserOut
    organization: OrganizationOut
    tokens: TokenPair


class InviteUserRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    role: UserRole = UserRole.RECRUITER
    password: str

    _check_password = field_validator("password")(_validate_password_strength)


class UpdateUserRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    role: UserRole | None = None
    is_active: bool | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    _check_password = field_validator("new_password")(_validate_password_strength)
