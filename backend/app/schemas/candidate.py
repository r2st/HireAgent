"""Candidate, resume, and consent schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import (
    CandidateSource,
    ConsentStatus,
    ConsentType,
    ResumeParseStatus,
    SeniorityLevel,
)
from app.schemas.common import ORMModel


class ConsentGrant(BaseModel):
    consent_type: ConsentType
    granted: bool = True
    policy_version: str | None = Field(default=None, max_length=40)
    source: str = Field(default="api", max_length=80)


class CandidateCreate(BaseModel):
    full_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=255)
    current_company: str | None = Field(default=None, max_length=255)
    current_role: str | None = Field(default=None, max_length=255)
    experience_years: float | None = Field(default=None, ge=0, le=60)
    seniority: SeniorityLevel | None = None
    notice_period_days: int | None = Field(default=None, ge=0, le=365)
    expected_salary: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    skills: list[str] = Field(default_factory=list)
    linkedin_url: str | None = Field(default=None, max_length=500)
    github_url: str | None = Field(default=None, max_length=500)
    portfolio_url: str | None = Field(default=None, max_length=500)
    source: CandidateSource = CandidateSource.DIRECT
    source_detail: str | None = Field(default=None, max_length=255)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    # GDPR/DPDP: consent is captured at the point the record is created.
    consents: list[ConsentGrant] = Field(default_factory=list)


class CandidateUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=255)
    current_company: str | None = Field(default=None, max_length=255)
    current_role: str | None = Field(default=None, max_length=255)
    experience_years: float | None = Field(default=None, ge=0, le=60)
    seniority: SeniorityLevel | None = None
    notice_period_days: int | None = Field(default=None, ge=0, le=365)
    expected_salary: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    skills: list[str] | None = None
    linkedin_url: str | None = Field(default=None, max_length=500)
    github_url: str | None = Field(default=None, max_length=500)
    portfolio_url: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = None
    notes: str | None = None
    is_blacklisted: bool | None = None


class CandidateOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    full_name: str
    email: str
    phone: str | None = None
    location: str | None = None
    current_company: str | None = None
    current_role: str | None = None
    experience_years: Decimal | None = None
    seniority: SeniorityLevel | None = None
    notice_period_days: int | None = None
    expected_salary: Decimal | None = None
    salary_currency: str | None = None
    skills_json: list = Field(default_factory=list)
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None
    source: CandidateSource
    source_detail: str | None = None
    tags: list = Field(default_factory=list)
    is_blacklisted: bool
    last_activity_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CandidateSummary(ORMModel):
    """Compact row for list views and pipeline cards."""

    id: uuid.UUID
    full_name: str
    email: str
    current_company: str | None = None
    current_role: str | None = None
    experience_years: Decimal | None = None
    skills_json: list = Field(default_factory=list)
    source: CandidateSource
    created_at: datetime


class ResumeOut(ORMModel):
    id: uuid.UUID
    candidate_id: uuid.UUID
    original_filename: str
    content_type: str
    file_size: int
    parse_status: ResumeParseStatus
    parse_error: str | None = None
    parsed_at: datetime | None = None
    parser_model: str | None = None
    parse_confidence: Decimal | None = None
    skills_extracted: list = Field(default_factory=list)
    is_primary: bool
    created_at: datetime


class ResumeDetail(ResumeOut):
    """Includes the decrypted parse payload — only for single-record reads."""

    parsed_json: dict | None = None


class ResumeUploadResult(BaseModel):
    candidate: CandidateOut
    resume: ResumeOut
    # True when this upload matched an existing candidate by email.
    is_existing_candidate: bool
    warnings: list[str] = Field(default_factory=list)


class BulkUploadResult(BaseModel):
    """Outcome of a bulk resume upload (design §6.1 bulk-upload endpoint)."""

    total: int
    succeeded: int
    failed: int
    results: list[ResumeUploadResult] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class ConsentOut(ORMModel):
    id: uuid.UUID
    candidate_id: uuid.UUID
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: datetime
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None
    source: str
    policy_version: str | None = None
    created_at: datetime
