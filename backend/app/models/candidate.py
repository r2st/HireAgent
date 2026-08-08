"""Candidate profiles, resumes, and consent records.

Candidate PII is encrypted at the column level (design §8.1). Because
ciphertext cannot be searched, each searchable encrypted field is paired with a
blind index (a keyed hash) used for equality lookup and deduplication.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TenantBase
from app.db.types import EncryptedJSON, EncryptedText, JSONColumn
from app.models.enums import (
    CandidateSource,
    ConsentStatus,
    ConsentType,
    ResumeParseStatus,
    SeniorityLevel,
)

if TYPE_CHECKING:
    from app.models.application import Application


class Candidate(TenantBase):
    """A person in the talent pool."""

    __tablename__ = "candidates"
    __extra_table_args__ = (
        # Deduplication key: one candidate per email per organization.
        Index("uq_candidates_org_email_idx", "organization_id", "email_index", unique=True),
        Index("ix_candidates_org_phone", "organization_id", "phone_index"),
    )

    # --- Encrypted PII + searchable blind indexes ---
    full_name: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    email: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    email_index: Mapped[str] = mapped_column(String(64), nullable=False)
    phone: Mapped[str | None] = mapped_column(EncryptedText)
    phone_index: Mapped[str | None] = mapped_column(String(64))
    location: Mapped[str | None] = mapped_column(EncryptedText)

    # --- Non-sensitive profile data (searchable, filterable) ---
    current_company: Mapped[str | None] = mapped_column(String(255))
    current_role: Mapped[str | None] = mapped_column(String(255))
    experience_years: Mapped[float | None] = mapped_column(Numeric(4, 1), index=True)
    seniority: Mapped[SeniorityLevel | None] = mapped_column(String(24))
    notice_period_days: Mapped[int | None] = mapped_column(Integer)
    expected_salary: Mapped[float | None] = mapped_column(Numeric(14, 2))
    salary_currency: Mapped[str | None] = mapped_column(String(3))

    # Normalised skill tags, e.g. [{"name": "React", "proficiency": "advanced"}]
    skills_json: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    linkedin_url: Mapped[str | None] = mapped_column(String(500))
    github_url: Mapped[str | None] = mapped_column(String(500))
    portfolio_url: Mapped[str | None] = mapped_column(String(500))

    source: Mapped[CandidateSource] = mapped_column(
        String(32), default=CandidateSource.DIRECT, nullable=False, index=True
    )
    source_detail: Mapped[str | None] = mapped_column(String(255))
    referred_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    tags: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    notes: Mapped[str | None] = mapped_column(EncryptedText)

    # Drives retention sweeps (design §8.1: N months from last activity).
    last_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    is_blacklisted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    resumes: Mapped[list[Resume]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="noload"
    )
    applications: Mapped[list[Application]] = relationship(
        back_populates="candidate", lazy="noload"
    )
    consents: Mapped[list[CandidateConsent]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="noload"
    )

    @property
    def skill_names(self) -> list[str]:
        out: list[str] = []
        for s in self.skills_json or []:
            if isinstance(s, dict) and s.get("name"):
                out.append(str(s["name"]))
            elif isinstance(s, str):
                out.append(s)
        return out


class Resume(TenantBase):
    """An uploaded resume and its parsed output.

    Both the raw file reference and the structured parse are kept so the
    pipeline never has to re-run extraction to read a field (design §3.1).
    """

    __tablename__ = "resumes"
    __extra_table_args__ = (
        Index("ix_resumes_candidate", "candidate_id", "deleted_at"),
    )

    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Detects duplicate uploads without decrypting anything.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    parse_status: Mapped[ResumeParseStatus] = mapped_column(
        String(24), default=ResumeParseStatus.PENDING, nullable=False, index=True
    )
    parse_error: Mapped[str | None] = mapped_column(Text)
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parser_model: Mapped[str | None] = mapped_column(String(120))
    parse_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))

    # Encrypted: full extracted text and the structured LLM output.
    raw_text: Mapped[str | None] = mapped_column(EncryptedText)
    parsed_json: Mapped[dict | None] = mapped_column(EncryptedJSON)
    # Skill tags are duplicated here unencrypted so matching can run in SQL.
    skills_extracted: Mapped[list] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    embedding_id: Mapped[str | None] = mapped_column(String(120))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    candidate: Mapped[Candidate] = relationship(back_populates="resumes", lazy="noload")


class CandidateConsent(TenantBase):
    """An immutable consent record (design §8.2).

    Withdrawal is recorded by writing a new row rather than mutating the
    existing one, preserving a full audit trail.
    """

    __tablename__ = "candidate_consents"
    __extra_table_args__ = (
        Index(
            "ix_consents_candidate_type",
            "candidate_id",
            "consent_type",
            "granted_at",
        ),
    )

    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    consent_type: Mapped[ConsentType] = mapped_column(String(40), nullable=False)
    status: Mapped[ConsentStatus] = mapped_column(
        String(24), default=ConsentStatus.GRANTED, nullable=False
    )
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Provenance for auditability.
    source: Mapped[str] = mapped_column(String(80), default="api", nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    policy_version: Mapped[str | None] = mapped_column(String(40))
    evidence_json: Mapped[dict | None] = mapped_column(JSONColumn)

    candidate: Mapped[Candidate] = relationship(back_populates="consents", lazy="noload")
