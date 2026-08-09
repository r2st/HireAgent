"""Assessment template, issue, and candidate-facing schemas (design §4.4)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.models.enums import AssessmentStatus, AssessmentType
from app.schemas.common import ORMModel


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
class TemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    type: AssessmentType = AssessmentType.MCQ
    description: str | None = Field(default=None, max_length=4000)
    # Shape is enforced by ``assessment_service.normalise_questions`` so the
    # rules live next to the grader that depends on them.
    questions: list = Field(min_length=1)
    duration_minutes: int = Field(default=60, ge=1, le=1440)
    passing_score: float = Field(default=60, ge=0, le=100)


class TemplateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    questions: list | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=1440)
    passing_score: float | None = Field(default=None, ge=0, le=100)
    is_active: bool | None = None


class TemplateOut(ORMModel):
    id: uuid.UUID
    name: str
    type: AssessmentType
    description: str | None = None
    questions_json: list = Field(default_factory=list)
    duration_minutes: int
    passing_score: Decimal
    is_active: bool
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# Issuing and grading
# --------------------------------------------------------------------------- #
class AssessmentIssue(BaseModel):
    application_id: uuid.UUID
    template_id: uuid.UUID | None = None
    # A one-off paper for this candidate, instead of a stored template.
    questions: list | None = None
    type: AssessmentType | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=1440)
    passing_score: float | None = Field(default=None, ge=0, le=100)
    expires_in_hours: int | None = Field(default=None, ge=1, le=2160)

    @model_validator(mode="after")
    def _needs_a_paper(self) -> AssessmentIssue:
        if self.template_id is None and not self.questions:
            raise ValueError("Provide either template_id or questions")
        return self


class ManualGrade(BaseModel):
    """A reviewer's marks, 0-100 per question id."""

    grades: dict[str, float] = Field(min_length=1)
    feedback: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _in_range(self) -> ManualGrade:
        for question_id, score in self.grades.items():
            if not 0 <= score <= 100:
                raise ValueError(
                    f"Mark for '{question_id}' must be between 0 and 100"
                )
        return self


class CancelRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class AssessmentOut(ORMModel):
    """The recruiter's view: everything except the candidate's raw answers."""

    id: uuid.UUID
    organization_id: uuid.UUID
    application_id: uuid.UUID
    template_id: uuid.UUID | None = None
    type: AssessmentType
    status: AssessmentStatus
    questions_json: list = Field(default_factory=list)
    score: Decimal | None = None
    max_score: Decimal
    passing_score: Decimal
    passed: bool | None = None
    breakdown_json: dict | None = None
    ai_feedback: str | None = None
    duration_minutes: int | None = None
    sent_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    time_spent_seconds: int | None = None
    created_at: datetime


class AssessmentDetail(AssessmentOut):
    """Adds the answers themselves, and the link while it is still live."""

    responses_json: dict | None = None
    invite_url: str | None = None


class IssuedAssessment(BaseModel):
    assessment: AssessmentOut
    invite_url: str | None = None


# --------------------------------------------------------------------------- #
# Candidate-facing
# --------------------------------------------------------------------------- #
class SubmitRequest(BaseModel):
    responses: dict = Field(default_factory=dict)


class PublicAssessmentView(BaseModel):
    """What the candidate sees. Never carries the answer key or a mark.

    The score is deliberately absent: whether an assessment is a pass is a
    hiring decision the recruiter delivers, not something a candidate should
    read off an API response seconds after submitting.
    """

    status: AssessmentStatus
    type: AssessmentType
    organization_name: str | None = None
    job_title: str | None = None
    candidate_name: str | None = None
    duration_minutes: int | None = None
    question_count: int
    # Withheld once the paper is submitted — there is nothing left to answer.
    questions: list = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
