"""Offer template, generation, and candidate-facing schemas (design §4.7)."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.models.enums import OfferStatus
from app.schemas.common import ORMModel


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
class TemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    subject: str | None = Field(default=None, max_length=500)
    body: str = Field(min_length=1)
    header_html: str | None = None
    footer_html: str | None = None
    is_default: bool = False


class TemplateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    subject: str | None = Field(default=None, max_length=500)
    body: str | None = Field(default=None, min_length=1)
    header_html: str | None = None
    footer_html: str | None = None
    is_active: bool | None = None
    is_default: bool | None = None


class TemplateOut(ORMModel):
    id: uuid.UUID
    name: str
    subject: str | None = None
    body: str
    header_html: str | None = None
    footer_html: str | None = None
    variables: list = Field(default_factory=list)
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# Generating and workflow
# --------------------------------------------------------------------------- #
class OfferGenerate(BaseModel):
    application_id: uuid.UUID
    template_id: uuid.UUID | None = None
    # A one-off letter, instead of a stored template.
    subject: str | None = Field(default=None, max_length=500)
    body: str | None = None
    job_title: str | None = Field(default=None, max_length=255)
    salary_amount: float = Field(gt=0)
    salary_currency: str = Field(default="USD", min_length=3, max_length=3)
    salary_period: str = Field(default="annual", max_length=16)
    bonus_amount: float | None = Field(default=None, ge=0)
    equity: str | None = Field(default=None, max_length=255)
    benefits: dict | None = None
    start_date: date | None = None
    expiry_date: date | None = None
    reporting_manager: str | None = Field(default=None, max_length=255)
    work_location: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def _needs_a_letter(self) -> OfferGenerate:
        if self.template_id is None and not (self.body or "").strip():
            raise ValueError("Provide either template_id or body")
        return self


class WithdrawRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class AcceptRequest(BaseModel):
    """The candidate's typed name, standing in for a signature (see
    ``offer_service`` for why — no e-signature provider is configured)."""

    signature_name: str = Field(min_length=1, max_length=255)


class DeclineRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class OfferOut(ORMModel):
    """The recruiter's view of an offer."""

    id: uuid.UUID
    organization_id: uuid.UUID
    application_id: uuid.UUID
    template_id: uuid.UUID | None = None
    status: OfferStatus
    version: int
    job_title: str
    salary_amount: Decimal | None = None
    salary_currency: str
    salary_period: str
    bonus_amount: Decimal | None = None
    equity: str | None = None
    benefits_json: dict = Field(default_factory=dict)
    start_date: date | None = None
    expiry_date: date | None = None
    reporting_manager: str | None = None
    work_location: str | None = None
    approved_by_id: uuid.UUID | None = None
    approved_at: datetime | None = None
    sent_at: datetime | None = None
    viewed_at: datetime | None = None
    signed_at: datetime | None = None
    responded_at: datetime | None = None
    signed_by_name: str | None = None
    decline_reason: str | None = None
    created_at: datetime


class OfferDetail(OfferOut):
    """Adds the rendered letter and the candidate's link while it is live."""

    rendered_body: str | None = None
    invite_url: str | None = None


# --------------------------------------------------------------------------- #
# Candidate-facing
# --------------------------------------------------------------------------- #
class PublicOfferView(BaseModel):
    """What the candidate sees. The compensation *is* the offer, so unlike an
    assessment link there is nothing here to withhold."""

    status: OfferStatus
    organization_name: str | None = None
    job_title: str
    candidate_name: str | None = None
    salary_amount: float | None = None
    salary_currency: str
    salary_period: str
    bonus_amount: float | None = None
    equity: str | None = None
    benefits_json: dict = Field(default_factory=dict)
    start_date: date | None = None
    expiry_date: date | None = None
    reporting_manager: str | None = None
    work_location: str | None = None
    rendered_body: str | None = None
    sent_at: datetime | None = None
    viewed_at: datetime | None = None
    responded_at: datetime | None = None
