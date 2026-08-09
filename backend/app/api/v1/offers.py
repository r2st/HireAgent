"""Offer routes (design §4.7).

Two routers, two callers, the same split as assessments and interviews:

* ``router`` — recruiters and hiring managers, JWT-authenticated. Templates,
  drafting, approval, and sending.
* ``public_router`` — the candidate, holding nothing but the access token from
  their offer link. Unauthenticated by design; the token is the credential.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_session, require_permission
from app.core.errors import AppError
from app.models.enums import OfferStatus
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.schemas.offer import (
    AcceptRequest,
    DeclineRequest,
    OfferDetail,
    OfferGenerate,
    OfferOut,
    PublicOfferView,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
    WithdrawRequest,
)
from app.services import offer_service

router = APIRouter(prefix="/offers", tags=["offers"])
public_router = APIRouter(prefix="/offer", tags=["offer"])


def _detail(offer) -> OfferDetail:
    detail = OfferDetail.model_validate(offer)
    detail.invite_url = offer_service.invite_url(offer)
    return detail


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@router.post(
    "/templates", response_model=TemplateOut, status_code=status.HTTP_201_CREATED
)
async def create_template(
    payload: TemplateCreate,
    current: CurrentUser = Depends(require_permission("offer:create")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    try:
        template = await offer_service.create_template(
            session,
            current.organization_id,
            name=payload.name,
            subject=payload.subject,
            body=payload.body,
            header_html=payload.header_html,
            footer_html=payload.footer_html,
            is_default=payload.is_default,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.get("/templates", response_model=Page[TemplateOut])
async def list_templates(
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    current: CurrentUser = Depends(require_permission("offer:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[TemplateOut]:
    params = PaginationParams(page=page, page_size=page_size)
    templates, total = await offer_service.list_templates(
        session, current.organization_id, is_active=is_active, params=params
    )
    return Page.build([TemplateOut.model_validate(t) for t in templates], total, params)


@router.get("/templates/{template_id}", response_model=TemplateOut)
async def get_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:read")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    try:
        template = await offer_service.get_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.patch("/templates/{template_id}", response_model=TemplateOut)
async def update_template(
    template_id: uuid.UUID,
    payload: TemplateUpdate,
    current: CurrentUser = Depends(require_permission("offer:update")),
    session: AsyncSession = Depends(get_session),
) -> TemplateOut:
    try:
        template = await offer_service.update_template(
            session,
            current.organization_id,
            template_id,
            **payload.model_dump(exclude_unset=True),
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return TemplateOut.model_validate(template)


@router.delete("/templates/{template_id}", response_model=MessageResponse)
async def delete_template(
    template_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:delete")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await offer_service.delete_template(
            session, current.organization_id, template_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Offer template deleted")


# --------------------------------------------------------------------------- #
# Drafting, approval, and sending
# --------------------------------------------------------------------------- #
@router.post("", response_model=OfferDetail, status_code=status.HTTP_201_CREATED)
async def generate_offer(
    payload: OfferGenerate,
    current: CurrentUser = Depends(require_permission("offer:create")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    """Draft an offer against an application. Nothing is sent to anyone yet."""
    try:
        offer = await offer_service.generate(
            session,
            current.organization_id,
            payload.application_id,
            template_id=payload.template_id,
            subject=payload.subject,
            body=payload.body,
            job_title=payload.job_title,
            salary_amount=payload.salary_amount,
            salary_currency=payload.salary_currency,
            salary_period=payload.salary_period,
            bonus_amount=payload.bonus_amount,
            equity=payload.equity,
            benefits=payload.benefits,
            start_date=payload.start_date,
            expiry_date=payload.expiry_date,
            reporting_manager=payload.reporting_manager,
            work_location=payload.work_location,
            created_by_id=current.id,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


@router.get("", response_model=Page[OfferOut])
async def list_offers(
    application_id: uuid.UUID | None = Query(None),
    status_filter: OfferStatus | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    current: CurrentUser = Depends(require_permission("offer:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[OfferOut]:
    params = PaginationParams(page=page, page_size=page_size)
    offers, total = await offer_service.list_offers(
        session,
        current.organization_id,
        application_id=application_id,
        status=status_filter,
        params=params,
    )
    return Page.build([OfferOut.model_validate(o) for o in offers], total, params)


@router.get("/{offer_id}", response_model=OfferDetail)
async def get_offer(
    offer_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:read")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    try:
        offer = await offer_service.get_offer(session, current.organization_id, offer_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


@router.post("/{offer_id}/submit", response_model=OfferDetail)
async def submit_offer(
    offer_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:update")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    """Send a draft for approval."""
    try:
        offer = await offer_service.submit_for_approval(
            session, current.organization_id, offer_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


@router.post("/{offer_id}/approve", response_model=OfferDetail)
async def approve_offer(
    offer_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:approve")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    try:
        offer = await offer_service.approve(
            session, current.organization_id, offer_id, approved_by_id=current.id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


@router.post("/{offer_id}/send", response_model=OfferDetail)
async def send_offer(
    offer_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("offer:update")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    """Mint the candidate's link. Requires the offer to already be approved."""
    try:
        offer = await offer_service.send(session, current.organization_id, offer_id)
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


@router.post("/{offer_id}/withdraw", response_model=OfferDetail)
async def withdraw_offer(
    offer_id: uuid.UUID,
    payload: WithdrawRequest,
    current: CurrentUser = Depends(require_permission("offer:update")),
    session: AsyncSession = Depends(get_session),
) -> OfferDetail:
    try:
        offer = await offer_service.withdraw(
            session, current.organization_id, offer_id, reason=payload.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _detail(offer)


# --------------------------------------------------------------------------- #
# Candidate self-service (access token is the credential)
# --------------------------------------------------------------------------- #
async def _public_view(session: AsyncSession, offer) -> PublicOfferView:
    view = await offer_service.build_view(session, offer)
    return PublicOfferView(**asdict(view))


@public_router.get("/{token}", response_model=PublicOfferView)
async def view_offer(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> PublicOfferView:
    """What the candidate sees when they open their offer link."""
    try:
        offer = await offer_service.record_view(session, token)
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, offer)


@public_router.post("/{token}/accept", response_model=PublicOfferView)
async def accept_offer(
    token: str,
    payload: AcceptRequest,
    session: AsyncSession = Depends(get_session),
) -> PublicOfferView:
    try:
        offer = await offer_service.accept(
            session, token, signature_name=payload.signature_name
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, offer)


@public_router.post("/{token}/decline", response_model=PublicOfferView)
async def decline_offer(
    token: str,
    payload: DeclineRequest,
    session: AsyncSession = Depends(get_session),
) -> PublicOfferView:
    try:
        offer = await offer_service.decline(session, token, reason=payload.reason)
    except AppError as exc:
        raise exc.to_http() from exc
    return await _public_view(session, offer)
