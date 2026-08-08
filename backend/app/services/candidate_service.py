"""Candidate records, resume ingestion, and consent tracking."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import blind_index
from app.db.tenancy import get_scoped, scoped_select
from app.models.candidate import Candidate, CandidateConsent, Resume
from app.models.enums import (
    CandidateSource,
    ConsentStatus,
    ConsentType,
    ResumeParseStatus,
)
from app.schemas.candidate import (
    CandidateCreate,
    CandidateUpdate,
    ConsentGrant,
)
from app.schemas.common import PaginationParams
from app.services import storage as storage_module
from app.services.resume_parser import ParsedResume, parse_resume
from app.services.skill_taxonomy import normalize_skills
from app.services.text_extraction import ExtractionError, extract_text

logger = logging.getLogger(__name__)

# Parses below this confidence are surfaced for human review (design §5).
LOW_CONFIDENCE_THRESHOLD = 0.70


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
async def find_by_email(
    session: AsyncSession, organization_id: uuid.UUID, email: str
) -> Candidate | None:
    """Look a candidate up by email using the blind index.

    The email column is encrypted and therefore not searchable; the blind index
    is a keyed hash of the normalised address that supports equality lookup.
    """
    return await session.scalar(
        scoped_select(Candidate, organization_id).where(
            Candidate.email_index == blind_index(email)
        )
    )


async def create_candidate(
    session: AsyncSession,
    organization_id: uuid.UUID,
    payload: CandidateCreate,
    *,
    referred_by_id: uuid.UUID | None = None,
) -> Candidate:
    email = payload.email.lower().strip()
    if await find_by_email(session, organization_id, email) is not None:
        raise ConflictError(f"A candidate with email {email} already exists")

    candidate = Candidate(
        organization_id=organization_id,
        full_name=payload.full_name.strip(),
        email=email,
        email_index=blind_index(email),
        phone=payload.phone,
        phone_index=blind_index(payload.phone) if payload.phone else None,
        location=payload.location,
        current_company=payload.current_company,
        current_role=payload.current_role,
        experience_years=payload.experience_years,
        seniority=payload.seniority,
        notice_period_days=payload.notice_period_days,
        expected_salary=payload.expected_salary,
        salary_currency=(
            payload.salary_currency.upper() if payload.salary_currency else None
        ),
        skills_json=normalize_skills(payload.skills),
        linkedin_url=payload.linkedin_url,
        github_url=payload.github_url,
        portfolio_url=payload.portfolio_url,
        source=payload.source,
        source_detail=payload.source_detail,
        referred_by_id=referred_by_id,
        tags=payload.tags,
        notes=payload.notes,
        last_activity_at=datetime.now(UTC),
    )
    session.add(candidate)
    await session.flush()

    for grant in payload.consents:
        session.add(_build_consent(organization_id, candidate.id, grant))

    await session.commit()
    await session.refresh(candidate)
    return candidate


async def get_candidate(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> Candidate:
    candidate = await get_scoped(session, Candidate, candidate_id, organization_id)
    if candidate is None:
        raise NotFoundError("Candidate not found")
    return candidate


async def list_candidates(
    session: AsyncSession,
    organization_id: uuid.UUID,
    params: PaginationParams,
    *,
    search: str | None = None,
    skill: str | None = None,
    min_experience: float | None = None,
    max_experience: float | None = None,
    source: CandidateSource | None = None,
) -> tuple[list[Candidate], int]:
    """List candidates with filters.

    Note: ``search`` only covers unencrypted columns (company, role). Name and
    email are encrypted and cannot be matched with a LIKE; use the exact-match
    email lookup for those.
    """
    stmt = scoped_select(Candidate, organization_id)
    count_stmt = (
        select(func.count())
        .select_from(Candidate)
        .where(
            Candidate.organization_id == organization_id,
            Candidate.deleted_at.is_(None),
        )
    )

    filters = []
    if search:
        pattern = f"%{search.strip()}%"
        filters.append(
            or_(
                Candidate.current_company.ilike(pattern),
                Candidate.current_role.ilike(pattern),
            )
        )
    if source is not None:
        filters.append(Candidate.source == source)
    if min_experience is not None:
        filters.append(Candidate.experience_years >= min_experience)
    if max_experience is not None:
        filters.append(Candidate.experience_years <= max_experience)

    for f in filters:
        stmt = stmt.where(f)
        count_stmt = count_stmt.where(f)

    stmt = stmt.order_by(Candidate.created_at.desc())

    if skill:
        # skills_json is JSON in SQLite and JSONB in PostgreSQL, so filter in
        # Python rather than writing two dialect-specific queries.
        rows = list((await session.execute(stmt)).scalars().all())
        wanted = skill.strip().lower()
        rows = [c for c in rows if any(s.lower() == wanted for s in c.skill_names)]
        total = len(rows)
        start = params.offset
        return rows[start : start + params.page_size], total

    stmt = stmt.offset(params.offset).limit(params.page_size)
    rows = list((await session.execute(stmt)).scalars().all())
    total = int(await session.scalar(count_stmt) or 0)
    return rows, total


async def update_candidate(
    session: AsyncSession,
    organization_id: uuid.UUID,
    candidate_id: uuid.UUID,
    payload: CandidateUpdate,
) -> Candidate:
    candidate = await get_candidate(session, organization_id, candidate_id)
    data = payload.model_dump(exclude_unset=True)

    if "email" in data and data["email"]:
        email = str(data.pop("email")).lower().strip()
        if email != candidate.email:
            clash = await find_by_email(session, organization_id, email)
            if clash is not None and clash.id != candidate.id:
                raise ConflictError(f"A candidate with email {email} already exists")
            candidate.email = email
            candidate.email_index = blind_index(email)
    if "phone" in data:
        phone = data.pop("phone")
        candidate.phone = phone
        candidate.phone_index = blind_index(phone) if phone else None
    if "skills" in data:
        skills = data.pop("skills")
        candidate.skills_json = normalize_skills(skills or [])
    if "salary_currency" in data and data["salary_currency"]:
        candidate.salary_currency = str(data.pop("salary_currency")).upper()

    for field, value in data.items():
        setattr(candidate, field, value)

    candidate.last_activity_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(candidate)
    return candidate


async def delete_candidate(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> None:
    """Soft delete the candidate and their resumes."""
    candidate = await get_candidate(session, organization_id, candidate_id)
    candidate.soft_delete()

    resumes = (
        (
            await session.execute(
                scoped_select(Resume, organization_id).where(
                    Resume.candidate_id == candidate_id
                )
            )
        )
        .scalars()
        .all()
    )
    for resume in resumes:
        resume.soft_delete()
    await session.commit()


# --------------------------------------------------------------------------- #
# Consent (design §8.2)
# --------------------------------------------------------------------------- #
def _build_consent(
    organization_id: uuid.UUID,
    candidate_id: uuid.UUID,
    grant: ConsentGrant,
    *,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> CandidateConsent:
    return CandidateConsent(
        organization_id=organization_id,
        candidate_id=candidate_id,
        consent_type=grant.consent_type,
        status=ConsentStatus.GRANTED if grant.granted else ConsentStatus.WITHDRAWN,
        granted_at=datetime.now(UTC),
        withdrawn_at=None if grant.granted else datetime.now(UTC),
        source=grant.source,
        ip_address=ip_address,
        user_agent=user_agent,
        policy_version=grant.policy_version,
    )


async def record_consent(
    session: AsyncSession,
    organization_id: uuid.UUID,
    candidate_id: uuid.UUID,
    grant: ConsentGrant,
    *,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> CandidateConsent:
    """Append a consent record.

    Records are immutable: withdrawing consent writes a new row rather than
    editing the grant, so the history stays auditable.
    """
    await get_candidate(session, organization_id, candidate_id)
    consent = _build_consent(
        organization_id,
        candidate_id,
        grant,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.add(consent)
    await session.commit()
    await session.refresh(consent)
    return consent


async def list_consents(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> list[CandidateConsent]:
    result = await session.execute(
        scoped_select(CandidateConsent, organization_id)
        .where(CandidateConsent.candidate_id == candidate_id)
        .order_by(CandidateConsent.granted_at.desc())
    )
    return list(result.scalars().all())


async def has_consent(
    session: AsyncSession,
    organization_id: uuid.UUID,
    candidate_id: uuid.UUID,
    consent_type: ConsentType,
) -> bool:
    """Whether the most recent record for this type is an active grant."""
    latest = await session.scalar(
        scoped_select(CandidateConsent, organization_id)
        .where(
            CandidateConsent.candidate_id == candidate_id,
            CandidateConsent.consent_type == consent_type,
        )
        .order_by(CandidateConsent.granted_at.desc())
        .limit(1)
    )
    if latest is None or latest.status != ConsentStatus.GRANTED:
        return False
    if latest.expires_at is not None and latest.expires_at < datetime.now(UTC):
        return False
    return True


# --------------------------------------------------------------------------- #
# Resume ingestion
# --------------------------------------------------------------------------- #
async def ingest_resume(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    data: bytes,
    filename: str,
    content_type: str | None,
    source: CandidateSource = CandidateSource.DIRECT,
    candidate_id: uuid.UUID | None = None,
    consent_granted: bool = True,
) -> tuple[Candidate, Resume, bool, list[str]]:
    """Extract, parse, and attach a resume, creating the candidate if needed.

    Returns ``(candidate, resume, is_existing_candidate, warnings)``.
    """
    warnings: list[str] = []

    text, _fmt = extract_text(data, content_type=content_type, filename=filename)
    parsed = await parse_resume(text)

    if parsed.confidence < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Low parse confidence ({parsed.confidence:.2f}); "
            "review the extracted fields before screening."
        )

    candidate, is_existing = await _resolve_candidate(
        session,
        organization_id,
        parsed,
        source=source,
        candidate_id=candidate_id,
        filename=filename,
    )

    if consent_granted:
        # Storing a parsed resume requires processing consent (design §8.2).
        session.add(
            _build_consent(
                organization_id,
                candidate.id,
                ConsentGrant(
                    consent_type=ConsentType.RESUME_PROCESSING,
                    granted=True,
                    source="resume_upload",
                ),
            )
        )
    else:
        warnings.append(
            "Resume processing consent was not recorded; "
            "outreach will be blocked until consent is captured."
        )

    storage = storage_module.get_storage()
    file_path = storage.save(organization_id, filename, data)

    # A new resume becomes primary; earlier ones are demoted.
    existing_resumes = (
        (
            await session.execute(
                scoped_select(Resume, organization_id).where(
                    Resume.candidate_id == candidate.id
                )
            )
        )
        .scalars()
        .all()
    )
    for previous in existing_resumes:
        previous.is_primary = False

    resume = Resume(
        organization_id=organization_id,
        candidate_id=candidate.id,
        file_path=file_path,
        original_filename=storage_module.sanitize_filename(filename),
        content_type=content_type or "application/octet-stream",
        file_size=len(data),
        content_hash=storage_module.content_hash(data),
        parse_status=ResumeParseStatus.PARSED,
        parsed_at=datetime.now(UTC),
        parser_model=parsed.model,
        parse_confidence=parsed.confidence,
        raw_text=text,
        parsed_json=parsed.to_dict(),
        skills_extracted=parsed.skills,
        is_primary=True,
    )
    session.add(resume)
    await session.commit()
    await session.refresh(candidate)
    await session.refresh(resume)
    return candidate, resume, is_existing, warnings


async def _resolve_candidate(
    session: AsyncSession,
    organization_id: uuid.UUID,
    parsed: ParsedResume,
    *,
    source: CandidateSource,
    candidate_id: uuid.UUID | None,
    filename: str,
) -> tuple[Candidate, bool]:
    """Find the candidate this resume belongs to, or create one.

    Matching is by email, which is the only field reliable enough to
    deduplicate on. A resume with no extractable email cannot be attached to a
    new candidate, since the record would be unreachable for outreach.
    """
    if candidate_id is not None:
        return await get_candidate(session, organization_id, candidate_id), True

    if parsed.email:
        existing = await find_by_email(session, organization_id, parsed.email)
        if existing is not None:
            _enrich_from_resume(existing, parsed)
            existing.last_activity_at = datetime.now(UTC)
            await session.flush()
            return existing, True

    if not parsed.email:
        raise ValidationError(
            "No email address could be extracted from the resume. "
            "Create the candidate manually, or upload against an existing candidate.",
            details={"filename": filename},
        )

    candidate = Candidate(
        organization_id=organization_id,
        full_name=parsed.full_name or parsed.email.split("@")[0],
        email=parsed.email,
        email_index=blind_index(parsed.email),
        phone=parsed.phone,
        phone_index=blind_index(parsed.phone) if parsed.phone else None,
        location=parsed.location,
        current_company=parsed.current_company,
        current_role=parsed.current_role,
        experience_years=parsed.total_experience_years,
        skills_json=parsed.skills,
        linkedin_url=parsed.linkedin_url,
        github_url=parsed.github_url,
        portfolio_url=parsed.portfolio_url,
        source=source,
        source_detail=storage_module.sanitize_filename(filename),
        tags=[],
        last_activity_at=datetime.now(UTC),
    )
    session.add(candidate)
    await session.flush()
    return candidate, False


def _enrich_from_resume(candidate: Candidate, parsed: ParsedResume) -> None:
    """Fill blank profile fields from a newly parsed resume.

    Existing values are never overwritten: a recruiter's manual correction
    should outrank a machine parse.
    """
    if not candidate.phone and parsed.phone:
        candidate.phone = parsed.phone
        candidate.phone_index = blind_index(parsed.phone)
    for attr, value in (
        ("location", parsed.location),
        ("current_company", parsed.current_company),
        ("current_role", parsed.current_role),
        ("experience_years", parsed.total_experience_years),
        ("linkedin_url", parsed.linkedin_url),
        ("github_url", parsed.github_url),
        ("portfolio_url", parsed.portfolio_url),
    ):
        if getattr(candidate, attr) in (None, "") and value not in (None, ""):
            setattr(candidate, attr, value)

    if parsed.skills:
        # Union of known and newly parsed skills.
        merged = normalize_skills([*(candidate.skills_json or []), *parsed.skills])
        candidate.skills_json = merged


async def list_resumes(
    session: AsyncSession, organization_id: uuid.UUID, candidate_id: uuid.UUID
) -> list[Resume]:
    result = await session.execute(
        scoped_select(Resume, organization_id)
        .where(Resume.candidate_id == candidate_id)
        .order_by(Resume.created_at.desc())
    )
    return list(result.scalars().all())


async def get_resume(
    session: AsyncSession, organization_id: uuid.UUID, resume_id: uuid.UUID
) -> Resume:
    resume = await get_scoped(session, Resume, resume_id, organization_id)
    if resume is None:
        raise NotFoundError("Resume not found")
    return resume


async def bulk_ingest(
    session: AsyncSession,
    organization_id: uuid.UUID,
    files: list[tuple[str, str | None, bytes]],
    *,
    source: CandidateSource = CandidateSource.DIRECT,
) -> tuple[list[tuple[Candidate, Resume, bool, list[str]]], list[dict]]:
    """Ingest many resumes, isolating per-file failures.

    One malformed file must not abort the batch, so each file is committed
    independently and failures are collected.
    """
    succeeded: list[tuple[Candidate, Resume, bool, list[str]]] = []
    errors: list[dict] = []

    for filename, content_type, data in files:
        try:
            result = await ingest_resume(
                session,
                organization_id,
                data=data,
                filename=filename,
                content_type=content_type,
                source=source,
            )
            succeeded.append(result)
        except (ExtractionError, ValidationError, ConflictError) as exc:
            await session.rollback()
            errors.append({"filename": filename, "error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive
            await session.rollback()
            logger.exception("Unexpected failure ingesting %s", filename)
            errors.append({"filename": filename, "error": str(exc)})

    return succeeded, errors
