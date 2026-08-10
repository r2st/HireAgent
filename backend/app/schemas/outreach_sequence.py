"""Recruiter-facing outreach sequence schemas (design §4.2, §6.1).

Companion to ``app/schemas/outreach.py``, which stays reserved for the
unauthenticated candidate-facing opt-out. Everything here is behind a JWT.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.models.enums import EnrollmentStatus, OutreachChannel, SequenceStatus
from app.schemas.common import ORMModel


class MessageTemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    channel: OutreachChannel = OutreachChannel.EMAIL
    subject: str | None = Field(default=None, max_length=500)
    body: str = Field(min_length=1)
    body_html: str | None = None
    category: str | None = Field(default=None, max_length=80)
    # WhatsApp Business requires pre-approved template names (design §2.3).
    provider_template_name: str | None = Field(default=None, max_length=255)


class MessageTemplateUpdate(BaseModel):
    """All fields optional — only supplied keys are applied."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    subject: str | None = Field(default=None, max_length=500)
    body: str | None = Field(default=None, min_length=1)
    body_html: str | None = None
    category: str | None = Field(default=None, max_length=80)
    provider_template_name: str | None = Field(default=None, max_length=255)
    is_active: bool | None = None


class MessageTemplateOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    channel: OutreachChannel
    subject: str | None = None
    body: str
    body_html: str | None = None
    variables: list
    provider_template_name: str | None = None
    category: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SequenceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    job_id: uuid.UUID | None = None
    sender_account_ids: list[uuid.UUID] = Field(default_factory=list)
    timezone: str = Field(default="UTC", max_length=64)
    send_window_start_hour: int = Field(default=9, ge=0, le=23)
    send_window_end_hour: int = Field(default=18, ge=1, le=24)
    send_on_weekends: bool = False
    stop_on_reply: bool = True
    daily_cap: int | None = Field(default=None, ge=1)


class SequenceUpdate(BaseModel):
    """All fields optional — only supplied keys are applied."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    job_id: uuid.UUID | None = None
    sender_account_ids: list[uuid.UUID] | None = None
    timezone: str | None = Field(default=None, max_length=64)
    send_window_start_hour: int | None = Field(default=None, ge=0, le=23)
    send_window_end_hour: int | None = Field(default=None, ge=1, le=24)
    send_on_weekends: bool | None = None
    stop_on_reply: bool | None = None
    daily_cap: int | None = Field(default=None, ge=1)


class StepCreate(BaseModel):
    template_id: uuid.UUID | None = None
    channel: OutreachChannel = OutreachChannel.EMAIL
    delay_days: int = Field(default=0, ge=0)
    delay_hours: int = Field(default=0, ge=0)
    subject_override: str | None = Field(default=None, max_length=500)
    body_override: str | None = None
    variant_group: str | None = Field(default=None, max_length=16)
    # Position to insert at; omitted means "append".
    step_order: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _needs_content(self) -> "StepCreate":
        if self.template_id is None and not (self.body_override or "").strip():
            raise ValueError("A step needs either a template or a body override")
        return self


class StepUpdate(BaseModel):
    template_id: uuid.UUID | None = None
    delay_days: int | None = Field(default=None, ge=0)
    delay_hours: int | None = Field(default=None, ge=0)
    subject_override: str | None = Field(default=None, max_length=500)
    body_override: str | None = None
    variant_group: str | None = Field(default=None, max_length=16)


class StepReorder(BaseModel):
    step_ids: list[uuid.UUID] = Field(min_length=1)


class StepOut(ORMModel):
    id: uuid.UUID
    sequence_id: uuid.UUID
    step_order: int
    channel: OutreachChannel
    template_id: uuid.UUID | None = None
    delay_days: int
    delay_hours: int
    subject_override: str | None = None
    body_override: str | None = None
    variant_group: str | None = None


class SequenceOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    job_id: uuid.UUID | None = None
    status: SequenceStatus
    sender_account_ids: list
    stats_json: dict
    send_window_start_hour: int
    send_window_end_hour: int
    send_on_weekends: bool
    timezone: str
    stop_on_reply: bool
    daily_cap: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    steps: list[StepOut] = Field(default_factory=list)


class EnrollRequest(BaseModel):
    candidate_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)


class EnrollmentSkip(BaseModel):
    candidate_id: uuid.UUID
    reason: str
    detail: str


class EnrollmentOut(ORMModel):
    id: uuid.UUID
    sequence_id: uuid.UUID
    candidate_id: uuid.UUID
    application_id: uuid.UUID | None = None
    status: EnrollmentStatus
    current_step: int
    next_send_at: datetime | None = None
    enrolled_at: datetime
    completed_at: datetime | None = None
    replied_at: datetime | None = None
    paused_reason: str | None = None


class EnrollmentReportOut(BaseModel):
    enrolled: list[EnrollmentOut]
    enrolled_count: int
    skipped: list[EnrollmentSkip]
    skipped_count: int


class EnrollmentPauseRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=255)


class SequenceStatsOut(BaseModel):
    sequence_id: uuid.UUID
    status: SequenceStatus
    enrollments: dict[str, int]
    enrollments_total: int
    messages: dict[str, int]
    sent: int
    bounced: int
    failed: int
    queued: int
