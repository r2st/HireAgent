"""Candidate, resume, and consent routes."""

from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, client_ip, get_session, require_permission
from app.core.config import settings
from app.core.errors import AppError, ValidationError
from app.models.enums import CandidateSource
from app.schemas.candidate import (
    BulkUploadResult,
    CandidateCreate,
    CandidateOut,
    CandidateSummary,
    CandidateUpdate,
    ConsentGrant,
    ConsentOut,
    ResumeDetail,
    ResumeOut,
    ResumeUploadResult,
)
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.services import candidate_service

router = APIRouter(prefix="/candidates", tags=["candidates"])

# Guards the bulk endpoint so one request cannot pin a worker indefinitely.
MAX_BULK_FILES = 50


@router.post("", response_model=CandidateOut, status_code=status.HTTP_201_CREATED)
async def create_candidate(
    payload: CandidateCreate,
    current: CurrentUser = Depends(require_permission("candidate:create")),
    session: AsyncSession = Depends(get_session),
) -> CandidateOut:
    try:
        candidate = await candidate_service.create_candidate(
            session, current.organization_id, payload
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return CandidateOut.model_validate(candidate)


@router.get("", response_model=Page[CandidateSummary])
async def list_candidates(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str | None = Query(None, max_length=200),
    skill: str | None = Query(None, max_length=120),
    min_experience: float | None = Query(None, ge=0, le=60),
    max_experience: float | None = Query(None, ge=0, le=60),
    source: CandidateSource | None = Query(None),
    current: CurrentUser = Depends(require_permission("candidate:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[CandidateSummary]:
    params = PaginationParams(page=page, page_size=page_size)
    rows, total = await candidate_service.list_candidates(
        session,
        current.organization_id,
        params,
        search=search,
        skill=skill,
        min_experience=min_experience,
        max_experience=max_experience,
        source=source,
    )
    return Page[CandidateSummary].build(
        [CandidateSummary.model_validate(c) for c in rows], total, params
    )


@router.get("/{candidate_id}", response_model=CandidateOut)
async def get_candidate(
    candidate_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("candidate:read")),
    session: AsyncSession = Depends(get_session),
) -> CandidateOut:
    try:
        candidate = await candidate_service.get_candidate(
            session, current.organization_id, candidate_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return CandidateOut.model_validate(candidate)


@router.patch("/{candidate_id}", response_model=CandidateOut)
async def update_candidate(
    candidate_id: uuid.UUID,
    payload: CandidateUpdate,
    current: CurrentUser = Depends(require_permission("candidate:update")),
    session: AsyncSession = Depends(get_session),
) -> CandidateOut:
    try:
        candidate = await candidate_service.update_candidate(
            session, current.organization_id, candidate_id, payload
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return CandidateOut.model_validate(candidate)


@router.delete("/{candidate_id}", response_model=MessageResponse)
async def delete_candidate(
    candidate_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("candidate:delete")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await candidate_service.delete_candidate(
            session, current.organization_id, candidate_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Candidate deleted")


# --------------------------------------------------------------------------- #
# Resumes
# --------------------------------------------------------------------------- #
@router.post(
    "/upload-resume",
    response_model=ResumeUploadResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_id: uuid.UUID | None = Form(None),
    source: CandidateSource = Form(CandidateSource.DIRECT),
    consent_granted: bool = Form(True),
    current: CurrentUser = Depends(require_permission("resume:create")),
    session: AsyncSession = Depends(get_session),
) -> ResumeUploadResult:
    """Upload and parse one resume, creating or enriching the candidate."""
    data = await _read_upload(file)
    try:
        candidate, resume, is_existing, warnings = (
            await candidate_service.ingest_resume(
                session,
                current.organization_id,
                data=data,
                filename=file.filename or "resume",
                content_type=file.content_type,
                source=source,
                candidate_id=candidate_id,
                consent_granted=consent_granted,
            )
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return ResumeUploadResult(
        candidate=CandidateOut.model_validate(candidate),
        resume=ResumeOut.model_validate(resume),
        is_existing_candidate=is_existing,
        warnings=warnings,
    )


@router.post(
    "/bulk-upload",
    response_model=BulkUploadResult,
    status_code=status.HTTP_207_MULTI_STATUS,
)
async def bulk_upload_resumes(
    files: list[UploadFile] = File(...),
    source: CandidateSource = Form(CandidateSource.DIRECT),
    current: CurrentUser = Depends(require_permission("resume:create")),
    session: AsyncSession = Depends(get_session),
) -> BulkUploadResult:
    """Upload many resumes; per-file failures are reported, not fatal."""
    if len(files) > MAX_BULK_FILES:
        raise ValidationError(
            f"At most {MAX_BULK_FILES} files can be uploaded per request",
            details={"received": len(files)},
        ).to_http()

    payloads: list[tuple[str, str | None, bytes]] = []
    errors: list[dict] = []
    for upload in files:
        try:
            payloads.append(
                (upload.filename or "resume", upload.content_type, await _read_upload(upload))
            )
        except Exception as exc:
            errors.append({"filename": upload.filename, "error": str(exc)})

    succeeded, ingest_errors = await candidate_service.bulk_ingest(
        session, current.organization_id, payloads, source=source
    )
    errors.extend(ingest_errors)

    return BulkUploadResult(
        total=len(files),
        succeeded=len(succeeded),
        failed=len(errors),
        results=[
            ResumeUploadResult(
                candidate=CandidateOut.model_validate(candidate),
                resume=ResumeOut.model_validate(resume),
                is_existing_candidate=is_existing,
                warnings=warnings,
            )
            for candidate, resume, is_existing, warnings in succeeded
        ],
        errors=errors,
    )


@router.get("/{candidate_id}/resumes", response_model=list[ResumeOut])
async def list_resumes(
    candidate_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("resume:read")),
    session: AsyncSession = Depends(get_session),
) -> list[ResumeOut]:
    try:
        await candidate_service.get_candidate(
            session, current.organization_id, candidate_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    resumes = await candidate_service.list_resumes(
        session, current.organization_id, candidate_id
    )
    return [ResumeOut.model_validate(r) for r in resumes]


@router.get("/resumes/{resume_id}", response_model=ResumeDetail)
async def get_resume(
    resume_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("resume:read")),
    session: AsyncSession = Depends(get_session),
) -> ResumeDetail:
    """Full parse payload — decrypted on read."""
    try:
        resume = await candidate_service.get_resume(
            session, current.organization_id, resume_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ResumeDetail.model_validate(resume)


# --------------------------------------------------------------------------- #
# Consent (design §8.2)
# --------------------------------------------------------------------------- #
@router.post(
    "/{candidate_id}/consents",
    response_model=ConsentOut,
    status_code=status.HTTP_201_CREATED,
)
async def record_consent(
    candidate_id: uuid.UUID,
    payload: ConsentGrant,
    request: Request,
    current: CurrentUser = Depends(require_permission("candidate:update")),
    session: AsyncSession = Depends(get_session),
) -> ConsentOut:
    try:
        consent = await candidate_service.record_consent(
            session,
            current.organization_id,
            candidate_id,
            payload,
            ip_address=client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ConsentOut.model_validate(consent)


@router.get("/{candidate_id}/consents", response_model=list[ConsentOut])
async def list_consents(
    candidate_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("candidate:read")),
    session: AsyncSession = Depends(get_session),
) -> list[ConsentOut]:
    try:
        await candidate_service.get_candidate(
            session, current.organization_id, candidate_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    consents = await candidate_service.list_consents(
        session, current.organization_id, candidate_id
    )
    return [ConsentOut.model_validate(c) for c in consents]


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, enforcing the size limit before anything touches disk."""
    data = await upload.read()
    if not data:
        raise ValidationError(
            f"Uploaded file '{upload.filename}' is empty"
        ).to_http()
    if len(data) > settings.max_upload_bytes:
        raise ValidationError(
            f"'{upload.filename}' exceeds the "
            f"{settings.max_upload_bytes // (1024 * 1024)}MB limit",
            details={"size": len(data)},
        ).to_http()
    return data
