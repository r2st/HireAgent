"""Job schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    EmploymentType,
    JobStatus,
    SeniorityLevel,
    WorkMode,
)
from app.schemas.common import ORMModel


class SkillRequirement(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    # Weight relative to other skills in the same list.
    weight: float = Field(default=1.0, ge=0, le=10)
    min_years: float | None = Field(default=None, ge=0, le=60)


class EducationRequirement(BaseModel):
    degree: str | None = Field(default=None, max_length=120)
    field_of_study: str | None = Field(default=None, max_length=200)
    # When false, a strong skill match can compensate for a missing degree
    # (design §4.1: skill-based matching over credential matching).
    is_mandatory: bool = False


class JobRequirements(BaseModel):
    """The structured requirement set the scoring engine matches against."""

    required_skills: list[SkillRequirement] = Field(default_factory=list)
    preferred_skills: list[SkillRequirement] = Field(default_factory=list)
    education: EducationRequirement | None = None
    certifications: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class ScoringWeights(BaseModel):
    """Per-job overrides of the default 40/30/15/15 split (design §2.2)."""

    skills: float = Field(default=0.40, ge=0, le=1)
    experience: float = Field(default=0.30, ge=0, le=1)
    education: float = Field(default=0.15, ge=0, le=1)
    cultural: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def _must_not_be_all_zero(self) -> "ScoringWeights":
        if self.skills + self.experience + self.education + self.cultural <= 0:
            raise ValueError("At least one scoring weight must be greater than zero")
        return self


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=255)
    department: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=255)
    work_mode: WorkMode = WorkMode.ONSITE
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    seniority: SeniorityLevel | None = None
    description: str | None = None
    openings: int = Field(default=1, ge=1, le=10_000)

    min_experience_years: float | None = Field(default=None, ge=0, le=60)
    max_experience_years: float | None = Field(default=None, ge=0, le=60)
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    salary_currency: str = Field(default="USD", min_length=3, max_length=3)

    requirements: JobRequirements = Field(default_factory=JobRequirements)
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    auto_advance_threshold: float | None = Field(default=None, ge=0, le=100)
    auto_reject_threshold: float | None = Field(default=None, ge=0, le=100)
    hiring_manager_id: uuid.UUID | None = None
    is_confidential: bool = False

    @model_validator(mode="after")
    def _check_ranges(self) -> "JobCreate":
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.min_experience_years > self.max_experience_years
        ):
            raise ValueError(
                "min_experience_years cannot exceed max_experience_years"
            )
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_min > self.salary_max
        ):
            raise ValueError("salary_min cannot exceed salary_max")
        if (
            self.auto_reject_threshold is not None
            and self.auto_advance_threshold is not None
            and self.auto_reject_threshold >= self.auto_advance_threshold
        ):
            raise ValueError(
                "auto_reject_threshold must be below auto_advance_threshold"
            )
        return self


class JobUpdate(BaseModel):
    """All fields optional — only supplied keys are applied."""

    title: str | None = Field(default=None, min_length=2, max_length=255)
    department: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=255)
    work_mode: WorkMode | None = None
    employment_type: EmploymentType | None = None
    seniority: SeniorityLevel | None = None
    description: str | None = None
    openings: int | None = Field(default=None, ge=1, le=10_000)
    min_experience_years: float | None = Field(default=None, ge=0, le=60)
    max_experience_years: float | None = Field(default=None, ge=0, le=60)
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    requirements: JobRequirements | None = None
    scoring_weights: ScoringWeights | None = None
    auto_advance_threshold: float | None = Field(default=None, ge=0, le=100)
    auto_reject_threshold: float | None = Field(default=None, ge=0, le=100)
    hiring_manager_id: uuid.UUID | None = None
    is_confidential: bool | None = None


class JobOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    title: str
    slug: str
    department: str | None = None
    location: str | None = None
    work_mode: WorkMode
    employment_type: EmploymentType
    seniority: SeniorityLevel | None = None
    description: str | None = None
    status: JobStatus
    openings: int
    min_experience_years: Decimal | None = None
    max_experience_years: Decimal | None = None
    salary_min: Decimal | None = None
    salary_max: Decimal | None = None
    salary_currency: str
    requirements_json: dict
    scoring_weights: dict
    auto_advance_threshold: Decimal | None = None
    auto_reject_threshold: Decimal | None = None
    hiring_manager_id: uuid.UUID | None = None
    is_confidential: bool
    bias_report_json: dict | None = None
    published_at: datetime | None = None
    closed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class JobSummary(ORMModel):
    """Lightweight row for list views and the pipeline board header."""

    id: uuid.UUID
    title: str
    slug: str
    department: str | None = None
    location: str | None = None
    status: JobStatus
    employment_type: EmploymentType
    openings: int
    created_at: datetime
    published_at: datetime | None = None


class JobStatusChange(BaseModel):
    status: JobStatus
    reason: str | None = None
