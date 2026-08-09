"""Outreach routes.

``public_router`` is the candidate side of outreach: unauthenticated, because
the only credential anyone holding it has is the token in the path. It mirrors
the booking routes — the API answers JSON at ``{api}/outreach/unsubscribe/…``
and the frontend renders a page at ``{public_base_url}/outreach/unsubscribe/…``
that calls it.

The one difference from booking is that a mail client, not a person, is the
primary caller. RFC 8058 one-click means Gmail POSTs to this path directly on
the candidate's behalf, so the POST has to work with no session, no page load,
no CSRF token, and no body.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import client_ip, get_session
from app.core.errors import AppError
from app.db.tenancy import get_scoped
from app.models.enums import ConsentType
from app.models.organization import Organization
from app.models.outreach import OutreachSequence, SequenceEnrollment
from app.schemas.outreach import UnsubscribeResponse, UnsubscribeView, mask_email
from app.services import candidate_service, outreach_service

public_router = APIRouter(prefix="/outreach", tags=["outreach"])


@public_router.get("/unsubscribe/{token}", response_model=UnsubscribeView)
async def view_unsubscribe(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> UnsubscribeView:
    """Describe the subscription this link would end.

    Read-only on purpose. Some mail clients and security scanners prefetch
    every link in a message, so a GET that opted the candidate out would
    unsubscribe people who never clicked — and unlike a missed opt-out, that
    one is invisible to everybody until a recruiter asks where their pipeline
    went. The frontend page POSTs when the candidate confirms.
    """
    try:
        message = await outreach_service.message_by_tracking_token(session, token)
    except AppError as exc:
        raise exc.to_http() from exc

    organization = await session.get(Organization, message.organization_id)
    sequence = None
    if message.enrollment_id:
        enrollment = await get_scoped(
            session,
            SequenceEnrollment,
            message.enrollment_id,
            message.organization_id,
        )
        if enrollment is not None:
            sequence = await get_scoped(
                session,
                OutreachSequence,
                enrollment.sequence_id,
                message.organization_id,
            )

    # Asked of consent rather than of the enrollment: the enrollment may have
    # ended for an unrelated reason, and what this page is reporting is whether
    # we would still email them.
    subscribed = await candidate_service.has_consent(
        session,
        message.organization_id,
        message.candidate_id,
        ConsentType.EMAIL_COMMUNICATION,
    )

    return UnsubscribeView(
        organization_name=organization.name if organization else None,
        recipient=mask_email(message.to_address),
        sequence_name=sequence.name if sequence else None,
        already_unsubscribed=not subscribed,
    )


@public_router.post("/unsubscribe/{token}", response_model=UnsubscribeResponse)
async def confirm_unsubscribe(
    token: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> UnsubscribeResponse:
    """Record the opt-out. Stops email for this candidate, not just this campaign.

    Takes no body. A one-click client sends ``List-Unsubscribe=One-Click`` as a
    form field and nothing else, so requiring anything of the payload would
    make the header's own promise fail.
    """
    try:
        result = await outreach_service.unsubscribe_by_token(
            session,
            token,
            ip_address=client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return UnsubscribeResponse(
        organization_name=result.organization_name,
        already_unsubscribed=result.already_unsubscribed,
        sequences_stopped=result.stopped,
    )
