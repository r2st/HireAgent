"""Interview scheduling, calendar, and candidate booking routes (design §4.3, §6.1).

Three routers live here because they have three different callers:

* ``router`` — recruiters and hiring managers, JWT-authenticated.
* ``calendar_router`` — interviewers connecting their own calendars.
* ``public_router`` — the candidate, holding nothing but a booking token. These
  routes are deliberately unauthenticated; the token is the credential, and
  the global IP rate limit in ``main`` is what bounds guessing.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_current_user, get_session, require_permission
from app.core.errors import AppError, PermissionDeniedError
from app.db.tenancy import get_scoped
from app.models.application import Application
from app.models.enums import InterviewStatus, InterviewType, UserRole
from app.models.job import Job
from app.models.organization import Organization
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.schemas.interview import (
    AvailabilityOut,
    AvailabilityQuery,
    BookingCancelRequest,
    BookingSlotChoice,
    CalendarAccountConnect,
    CalendarAccountOut,
    CancelRequest,
    CompleteRequest,
    FeedbackRequest,
    FeedbackSummaryOut,
    InterviewDetail,
    InterviewerAvailabilityOut,
    InterviewOut,
    ParticipantOut,
    ProposeSlotsRequest,
    PublicBookingView,
    RescheduleRequest,
    RespondRequest,
    ScheduleInterviewRequest,
    SchedulingResponse,
    SlotOut,
    WorkingHoursUpdate,
)
from app.services import calendar_service, interview_service
from app.services.availability import Interval, to_utc

router = APIRouter(prefix="/interviews", tags=["interviews"])
calendar_router = APIRouter(prefix="/calendar", tags=["calendar"])
public_router = APIRouter(prefix="/booking", tags=["booking"])

# Interview states a candidate may still act on from their booking link.
_CANDIDATE_ACTIONABLE = {
    InterviewStatus.PENDING,
    InterviewStatus.SCHEDULED,
    InterviewStatus.CONFIRMED,
}


def _slots(intervals: list[Interval]) -> list[SlotOut]:
    return [SlotOut(start=i.start, end=i.end) for i in intervals]


def _response(result: interview_service.SchedulingResult) -> SchedulingResponse:
    return SchedulingResponse(
        interview=InterviewOut.model_validate(result.interview),
        slots=_slots(result.slots),
        warnings=result.warnings,
    )


# --------------------------------------------------------------------------- #
# Calendar accounts
# --------------------------------------------------------------------------- #
@calendar_router.post(
    "/accounts", response_model=CalendarAccountOut, status_code=status.HTTP_201_CREATED
)
async def connect_calendar(
    payload: CalendarAccountConnect,
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> CalendarAccountOut:
    """Connect a Google or Outlook calendar.

    Anyone may connect their own; connecting on behalf of a colleague is an
    admin action, since it means storing OAuth tokens against another user.
    """
    target_user_id = payload.user_id or current.id
    if target_user_id != current.id and current.role != UserRole.ADMIN:
        raise PermissionDeniedError(
            "Only an admin can connect a calendar for another user"
        ).to_http()

    try:
        account = await calendar_service.connect_account(
            session,
            current.organization_id,
            target_user_id,
            provider=payload.provider,
            email=payload.email,
            access_token=payload.access_token,
            refresh_token=payload.refresh_token,
            token_expires_at=payload.token_expires_at,
            calendar_id=payload.calendar_id,
            scopes=payload.scopes,
            working_hours=payload.working_hours,
            timezone=payload.timezone,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return CalendarAccountOut.model_validate(account)


@calendar_router.get("/accounts", response_model=list[CalendarAccountOut])
async def list_calendars(
    user_id: uuid.UUID | None = Query(None),
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[CalendarAccountOut]:
    """Connected calendars. Non-admins only ever see their own."""
    scope = user_id if current.role == UserRole.ADMIN else current.id
    accounts = await calendar_service.list_accounts(
        session, current.organization_id, user_id=scope
    )
    return [CalendarAccountOut.model_validate(a) for a in accounts]


@calendar_router.patch("/accounts/{account_id}", response_model=CalendarAccountOut)
async def update_calendar(
    account_id: uuid.UUID,
    payload: WorkingHoursUpdate,
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> CalendarAccountOut:
    """Set the working hours slot proposals are drawn from."""
    try:
        account = await calendar_service.get_account(
            session, current.organization_id, account_id
        )
        if account.user_id != current.id and current.role != UserRole.ADMIN:
            raise PermissionDeniedError(
                "Only an admin can change another user's calendar settings"
            ).to_http()
        account = await calendar_service.update_working_hours(
            session,
            current.organization_id,
            account_id,
            working_hours=payload.working_hours,
            timezone=payload.timezone,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return CalendarAccountOut.model_validate(account)


@calendar_router.delete("/accounts/{account_id}", response_model=MessageResponse)
async def disconnect_calendar(
    account_id: uuid.UUID,
    current: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Disconnect a calendar and discard its stored OAuth tokens."""
    try:
        account = await calendar_service.get_account(
            session, current.organization_id, account_id
        )
        if account.user_id != current.id and current.role != UserRole.ADMIN:
            raise PermissionDeniedError(
                "Only an admin can disconnect another user's calendar"
            ).to_http()
        await calendar_service.disconnect_account(
            session, current.organization_id, account_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Calendar disconnected")


# --------------------------------------------------------------------------- #
# Availability and scheduling
# --------------------------------------------------------------------------- #
@router.post("/availability", response_model=AvailabilityOut)
async def check_availability(
    payload: AvailabilityQuery,
    current: CurrentUser = Depends(require_permission("interview:create")),
    session: AsyncSession = Depends(get_session),
) -> AvailabilityOut:
    """Slots every listed interviewer can make, without creating anything."""
    try:
        slots, availability = await interview_service.find_available_slots(
            session,
            current.organization_id,
            payload.interviewer_ids,
            duration_minutes=payload.duration_minutes,
            window_start=payload.window_start,
            window_end=payload.window_end,
            granularity_minutes=payload.granularity_minutes,
            buffer_minutes=payload.buffer_minutes,
            min_notice_hours=payload.min_notice_hours,
            limit=payload.limit,
            per_day=payload.per_day,
            default_timezone=payload.timezone,
        )
    except AppError as exc:
        raise exc.to_http() from exc

    return AvailabilityOut(
        slots=_slots(slots),
        interviewers=[
            InterviewerAvailabilityOut(
                user_id=entry.user_id,  # type: ignore[arg-type]
                calendar_synced=entry.calendar_synced,
                error=entry.error,
                free_intervals=_slots(entry.free),
            )
            for entry in availability
        ],
        warnings=[
            f"Calendar for interviewer {entry.user_id} is not synced "
            f"({entry.error or 'unknown reason'})"
            for entry in availability
            if not entry.calendar_synced
        ],
    )


@router.post(
    "/schedule", response_model=SchedulingResponse, status_code=status.HTTP_201_CREATED
)
async def schedule_interview(
    payload: ScheduleInterviewRequest,
    current: CurrentUser = Depends(require_permission("interview:create")),
    session: AsyncSession = Depends(get_session),
) -> SchedulingResponse:
    """Book an interview at a known time (design §6.1)."""
    try:
        result = await interview_service.schedule_interview(
            session,
            current.organization_id,
            payload.application_id,
            scheduled_at=payload.scheduled_at,
            interviewer_ids=payload.interviewer_ids,
            type=payload.type,
            duration_minutes=payload.duration_minutes,
            round_number=payload.round_number,
            title=payload.title,
            location=payload.location,
            meeting_url=payload.meeting_url,
            timezone=payload.timezone,
            notes=payload.notes,
            organizer_id=payload.organizer_id,
            allow_conflicts=payload.allow_conflicts,
            create_calendar_event=payload.create_calendar_event,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _response(result)


@router.post(
    "/propose", response_model=SchedulingResponse, status_code=status.HTTP_201_CREATED
)
async def propose_slots(
    payload: ProposeSlotsRequest,
    current: CurrentUser = Depends(require_permission("interview:create")),
    session: AsyncSession = Depends(get_session),
) -> SchedulingResponse:
    """Offer the candidate a set of times and hand back their booking link."""
    try:
        result = await interview_service.propose_slots(
            session,
            current.organization_id,
            payload.application_id,
            interviewer_ids=payload.interviewer_ids,
            type=payload.type,
            duration_minutes=payload.duration_minutes,
            round_number=payload.round_number,
            title=payload.title,
            location=payload.location,
            timezone=payload.timezone,
            window_start=payload.window_start,
            window_end=payload.window_end,
            granularity_minutes=payload.granularity_minutes,
            buffer_minutes=payload.buffer_minutes,
            min_notice_hours=payload.min_notice_hours,
            slot_count=payload.slot_count,
            organizer_id=payload.organizer_id,
            require_consent=payload.require_consent,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _response(result)


# --------------------------------------------------------------------------- #
# Reading interviews
# --------------------------------------------------------------------------- #
@router.get("", response_model=Page[InterviewOut])
async def list_interviews(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    application_id: uuid.UUID | None = Query(None),
    candidate_id: uuid.UUID | None = Query(None),
    job_id: uuid.UUID | None = Query(None),
    interviewer_id: uuid.UUID | None = Query(None),
    interview_status: InterviewStatus | None = Query(None, alias="status"),
    interview_type: InterviewType | None = Query(None, alias="type"),
    upcoming_only: bool = Query(False),
    mine: bool = Query(False, description="Only interviews I am a participant in"),
    current: CurrentUser = Depends(require_permission("interview:read")),
    session: AsyncSession = Depends(get_session),
) -> Page[InterviewOut]:
    params = PaginationParams(page=page, page_size=page_size)
    rows, total = await interview_service.list_interviews(
        session,
        current.organization_id,
        params,
        application_id=application_id,
        candidate_id=candidate_id,
        job_id=job_id,
        interviewer_id=current.id if mine else interviewer_id,
        status=interview_status,
        type=interview_type,
        upcoming_only=upcoming_only,
    )
    return Page[InterviewOut].build(
        [InterviewOut.model_validate(i) for i in rows], total, params
    )


@router.get("/{interview_id}", response_model=InterviewDetail)
async def get_interview(
    interview_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("interview:read")),
    session: AsyncSession = Depends(get_session),
) -> InterviewDetail:
    try:
        interview = await interview_service.get_interview(
            session, current.organization_id, interview_id
        )
    except AppError as exc:
        raise exc.to_http() from exc

    participants = await interview_service.list_participants(
        session, current.organization_id, interview_id
    )
    detail = InterviewDetail.model_validate(interview)
    detail.participants = [ParticipantOut.model_validate(p) for p in participants]
    detail.feedback_summary = await interview_service.feedback_summary(
        session, current.organization_id, interview_id
    )
    detail.booking_url = interview_service.booking_url(interview)
    return detail


@router.get("/{interview_id}/participants", response_model=list[ParticipantOut])
async def list_participants(
    interview_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("interview:read")),
    session: AsyncSession = Depends(get_session),
) -> list[ParticipantOut]:
    try:
        await interview_service.get_interview(
            session, current.organization_id, interview_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    participants = await interview_service.list_participants(
        session, current.organization_id, interview_id
    )
    return [ParticipantOut.model_validate(p) for p in participants]


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@router.post("/{interview_id}/reschedule", response_model=SchedulingResponse)
async def reschedule_interview(
    interview_id: uuid.UUID,
    payload: RescheduleRequest,
    current: CurrentUser = Depends(require_permission("interview:update")),
    session: AsyncSession = Depends(get_session),
) -> SchedulingResponse:
    """Move an interview, superseding it with a linked replacement."""
    try:
        interview = await interview_service.get_interview(
            session, current.organization_id, interview_id
        )
        proposed: list[Interval] | None = None
        if payload.propose_new_slots:
            participants = await interview_service.list_participants(
                session, current.organization_id, interview_id
            )
            proposed, _ = await interview_service.find_available_slots(
                session,
                current.organization_id,
                [p.user_id for p in participants],
                duration_minutes=interview.duration_minutes,
                window_start=payload.window_start,
                window_end=payload.window_end,
                limit=payload.slot_count,
                default_timezone=interview.timezone,
                exclude_interview_id=interview_id,
            )

        result = await interview_service.reschedule(
            session,
            current.organization_id,
            interview_id,
            scheduled_at=payload.scheduled_at,
            proposed_slots=proposed,
            reason=payload.reason,
            allow_conflicts=payload.allow_conflicts,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _response(result)


@router.post("/{interview_id}/cancel", response_model=SchedulingResponse)
async def cancel_interview(
    interview_id: uuid.UUID,
    payload: CancelRequest,
    current: CurrentUser = Depends(require_permission("interview:update")),
    session: AsyncSession = Depends(get_session),
) -> SchedulingResponse:
    try:
        result = await interview_service.cancel(
            session, current.organization_id, interview_id, reason=payload.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return _response(result)


@router.post("/{interview_id}/complete", response_model=InterviewOut)
async def complete_interview(
    interview_id: uuid.UUID,
    payload: CompleteRequest | None = None,
    current: CurrentUser = Depends(require_permission("interview:update")),
    session: AsyncSession = Depends(get_session),
) -> InterviewOut:
    """Mark the interview as held and advance the pipeline card."""
    request = payload or CompleteRequest()
    try:
        interview = await interview_service.complete(
            session,
            current.organization_id,
            interview_id,
            notes=request.notes,
            advance_application=request.advance_application,
            changed_by_id=current.id,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return InterviewOut.model_validate(interview)


@router.post("/{interview_id}/no-show", response_model=InterviewOut)
async def mark_no_show(
    interview_id: uuid.UUID,
    payload: CancelRequest | None = None,
    current: CurrentUser = Depends(require_permission("interview:update")),
    session: AsyncSession = Depends(get_session),
) -> InterviewOut:
    request = payload or CancelRequest()
    try:
        interview = await interview_service.mark_no_show(
            session, current.organization_id, interview_id, notes=request.reason
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return InterviewOut.model_validate(interview)


@router.delete("/{interview_id}", response_model=MessageResponse)
async def delete_interview(
    interview_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("interview:update")),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    try:
        await interview_service.delete_interview(
            session, current.organization_id, interview_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Interview deleted")


# --------------------------------------------------------------------------- #
# Interviewer actions
# --------------------------------------------------------------------------- #
@router.post("/{interview_id}/respond", response_model=ParticipantOut)
async def respond_to_invite(
    interview_id: uuid.UUID,
    payload: RespondRequest,
    current: CurrentUser = Depends(require_permission("interview:read")),
    session: AsyncSession = Depends(get_session),
) -> ParticipantOut:
    """Accept, decline, or tentatively accept your own invitation."""
    try:
        participant = await interview_service.respond(
            session,
            current.organization_id,
            interview_id,
            current.id,
            payload.response_status,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ParticipantOut.model_validate(participant)


@router.post("/{interview_id}/feedback", response_model=ParticipantOut)
async def submit_feedback(
    interview_id: uuid.UUID,
    payload: FeedbackRequest,
    current: CurrentUser = Depends(require_permission("interview:feedback")),
    session: AsyncSession = Depends(get_session),
) -> ParticipantOut:
    """Leave your scorecard. Feedback is always attributed to the caller."""
    try:
        participant = await interview_service.submit_feedback(
            session,
            current.organization_id,
            interview_id,
            current.id,
            rating=payload.rating,
            recommendation=payload.recommendation,
            feedback=payload.feedback,
            scorecard=payload.scorecard,
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return ParticipantOut.model_validate(participant)


@router.get("/{interview_id}/feedback", response_model=FeedbackSummaryOut)
async def get_feedback_summary(
    interview_id: uuid.UUID,
    current: CurrentUser = Depends(require_permission("interview:read")),
    session: AsyncSession = Depends(get_session),
) -> FeedbackSummaryOut:
    try:
        await interview_service.get_interview(
            session, current.organization_id, interview_id
        )
    except AppError as exc:
        raise exc.to_http() from exc
    summary = await interview_service.feedback_summary(
        session, current.organization_id, interview_id
    )
    return FeedbackSummaryOut(**summary)


# --------------------------------------------------------------------------- #
# Candidate self-service (design §4.3)
# --------------------------------------------------------------------------- #
async def _booking_view(
    session: AsyncSession, interview
) -> PublicBookingView:
    """Assemble the candidate-facing view of one interview."""
    organization = await session.get(Organization, interview.organization_id)
    application = await get_scoped(
        session, Application, interview.application_id, interview.organization_id
    )
    job = None
    if application is not None:
        job = await get_scoped(session, Job, application.job_id, interview.organization_id)

    slots = _slots(interview_service.proposed_intervals(interview))
    open_for_action = interview.status in _CANDIDATE_ACTIONABLE
    return PublicBookingView(
        interview_id=interview.id,
        organization_name=organization.name if organization else "",
        job_title=job.title if job else None,
        type=interview.type,
        status=interview.status,
        duration_minutes=interview.duration_minutes,
        timezone=interview.timezone,
        scheduled_at=interview.scheduled_at,
        location=interview.location,
        meeting_url=interview.meeting_url,
        proposed_slots=slots,
        can_book=open_for_action and interview.scheduled_at is None and bool(slots),
        can_reschedule=open_for_action and bool(slots),
        can_cancel=open_for_action,
    )


@public_router.get("/{token}", response_model=PublicBookingView)
async def view_booking(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> PublicBookingView:
    """What the candidate sees when they open their booking link."""
    try:
        interview = await interview_service.get_by_booking_token(session, token)
    except AppError as exc:
        raise exc.to_http() from exc
    return await _booking_view(session, interview)


@public_router.post("/{token}/book", response_model=PublicBookingView)
async def book_slot(
    token: str,
    payload: BookingSlotChoice,
    session: AsyncSession = Depends(get_session),
) -> PublicBookingView:
    """Book one of the offered slots."""
    try:
        result = await interview_service.book_slot(
            session, token, to_utc(payload.slot_start)
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return await _booking_view(session, result.interview)


@public_router.post("/{token}/reschedule", response_model=PublicBookingView)
async def reschedule_booking(
    token: str,
    payload: BookingSlotChoice,
    session: AsyncSession = Depends(get_session),
) -> PublicBookingView:
    """Move to a different offered slot."""
    try:
        result = await interview_service.reschedule_by_token(
            session, token, to_utc(payload.slot_start)
        )
    except AppError as exc:
        raise exc.to_http() from exc
    return await _booking_view(session, result.interview)


@public_router.post("/{token}/cancel", response_model=MessageResponse)
async def cancel_booking(
    token: str,
    payload: BookingCancelRequest | None = None,
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Cancel the interview from the candidate side."""
    request = payload or BookingCancelRequest()
    try:
        await interview_service.cancel_by_token(session, token, reason=request.reason)
    except AppError as exc:
        raise exc.to_http() from exc
    return MessageResponse(message="Interview cancelled")
