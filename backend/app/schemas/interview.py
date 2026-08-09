"""Interview scheduling, calendar, and scorecard schemas (design §4.3)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from app.models.enums import InterviewStatus, InterviewType
from app.schemas.common import ORMModel
from app.services.availability import DAY_KEYS, parse_hhmm

# Feedback verbs, ordered strong-no to strong-yes.
RECOMMENDATIONS = ("strong_no", "no", "neutral", "yes", "strong_yes")
RESPONSE_STATUSES = ("pending", "accepted", "declined", "tentative")


def _validate_working_hours(value: dict | None) -> dict | None:
    """Reject a malformed working-hours map at the edge.

    Bad hours would otherwise be silently skipped during slot expansion and
    show up as "no availability", which is a maddening thing to debug.
    """
    if value is None:
        return None

    cleaned: dict[str, list[list[str]]] = {}
    for day, ranges in value.items():
        key = str(day).lower()[:3]
        if key not in DAY_KEYS:
            raise ValueError(f"Unknown day '{day}'; expected one of {list(DAY_KEYS)}")
        if not isinstance(ranges, list):
            raise ValueError(f"Ranges for '{key}' must be a list")
        day_ranges: list[list[str]] = []
        for entry in ranges:
            if not isinstance(entry, list | tuple) or len(entry) != 2:
                raise ValueError(
                    f"Each range for '{key}' must be a [start, end] pair"
                )
            start, end = parse_hhmm(str(entry[0])), parse_hhmm(str(entry[1]))
            if end <= start:
                raise ValueError(
                    f"Range {entry} for '{key}' does not end after it starts"
                )
            day_ranges.append([str(entry[0]), str(entry[1])])
        cleaned[key] = day_ranges
    return cleaned


# --------------------------------------------------------------------------- #
# Slots
# --------------------------------------------------------------------------- #
class SlotOut(BaseModel):
    start: datetime
    end: datetime


# --------------------------------------------------------------------------- #
# Calendar accounts
# --------------------------------------------------------------------------- #
class CalendarAccountConnect(BaseModel):
    """OAuth material captured after the user authorises HireAgent."""

    user_id: uuid.UUID | None = Field(
        default=None,
        description="Defaults to the authenticated user; admins may connect for others.",
    )
    provider: str = Field(pattern="^(google|outlook)$")
    email: str = Field(max_length=320)
    access_token: str | None = None
    refresh_token: str | None = None
    token_expires_at: datetime | None = None
    calendar_id: str | None = Field(default=None, max_length=255)
    scopes: list[str] = Field(default_factory=list)
    working_hours: dict | None = None
    timezone: str = Field(default="UTC", max_length=64)

    @field_validator("working_hours")
    @classmethod
    def _hours(cls, v: dict | None) -> dict | None:
        return _validate_working_hours(v)


class WorkingHoursUpdate(BaseModel):
    working_hours: dict | None = None
    timezone: str | None = Field(default=None, max_length=64)

    @field_validator("working_hours")
    @classmethod
    def _hours(cls, v: dict | None) -> dict | None:
        return _validate_working_hours(v)


class CalendarAccountOut(ORMModel):
    """A connected calendar. Deliberately never includes the OAuth tokens."""

    id: uuid.UUID
    user_id: uuid.UUID
    provider: str
    email: str
    calendar_id: str | None = None
    scopes: list = Field(default_factory=list)
    is_active: bool
    sync_error: str | None = None
    last_synced_at: datetime | None = None
    working_hours: dict = Field(default_factory=dict)
    timezone: str
    created_at: datetime


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
class AvailabilityQuery(BaseModel):
    interviewer_ids: list[uuid.UUID] = Field(min_length=1, max_length=20)
    duration_minutes: int = Field(default=45, ge=5, le=480)
    window_start: datetime | None = None
    window_end: datetime | None = None
    granularity_minutes: int = Field(default=30, ge=5, le=120)
    buffer_minutes: int = Field(default=0, ge=0, le=120)
    min_notice_hours: int | None = Field(default=None, ge=0, le=720)
    limit: int = Field(default=10, ge=1, le=50)
    per_day: int = Field(default=3, ge=1, le=20)
    timezone: str = Field(default="UTC", max_length=64)


class InterviewerAvailabilityOut(BaseModel):
    user_id: uuid.UUID
    calendar_synced: bool
    error: str | None = None
    free_intervals: list[SlotOut] = Field(default_factory=list)


class AvailabilityOut(BaseModel):
    slots: list[SlotOut] = Field(default_factory=list)
    interviewers: list[InterviewerAvailabilityOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
class ProposeSlotsRequest(BaseModel):
    application_id: uuid.UUID
    interviewer_ids: list[uuid.UUID] = Field(min_length=1, max_length=20)
    type: InterviewType = InterviewType.VIDEO
    duration_minutes: int = Field(default=45, ge=5, le=480)
    round_number: int | None = Field(default=None, ge=1, le=20)
    title: str | None = Field(default=None, max_length=255)
    location: str | None = Field(default=None, max_length=500)
    timezone: str = Field(default="UTC", max_length=64)
    window_start: datetime | None = None
    window_end: datetime | None = None
    granularity_minutes: int = Field(default=30, ge=5, le=120)
    buffer_minutes: int = Field(default=0, ge=0, le=120)
    min_notice_hours: int | None = Field(default=None, ge=0, le=720)
    slot_count: int = Field(default=5, ge=1, le=20)
    organizer_id: uuid.UUID | None = None
    # Design §8.2: sending slots is candidate contact, so consent is required.
    # Only an explicit opt-out skips the check (e.g. slots relayed by phone).
    require_consent: bool = True


class ScheduleInterviewRequest(BaseModel):
    """Direct booking — design §6.1 ``POST /api/v1/interviews/schedule``."""

    application_id: uuid.UUID
    scheduled_at: datetime
    interviewer_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    type: InterviewType = InterviewType.VIDEO
    duration_minutes: int = Field(default=45, ge=5, le=480)
    round_number: int | None = Field(default=None, ge=1, le=20)
    title: str | None = Field(default=None, max_length=255)
    location: str | None = Field(default=None, max_length=500)
    meeting_url: str | None = Field(default=None, max_length=1000)
    timezone: str = Field(default="UTC", max_length=64)
    notes: str | None = Field(default=None, max_length=5000)
    organizer_id: uuid.UUID | None = None
    allow_conflicts: bool = False
    create_calendar_event: bool = True


class RescheduleRequest(BaseModel):
    scheduled_at: datetime | None = None
    propose_new_slots: bool = False
    slot_count: int = Field(default=5, ge=1, le=20)
    window_start: datetime | None = None
    window_end: datetime | None = None
    reason: str | None = Field(default=None, max_length=2000)
    allow_conflicts: bool = False


class CancelRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class CompleteRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=5000)
    # Move the application to "interviewed" if it is not already past it.
    advance_application: bool = True


class RespondRequest(BaseModel):
    response_status: str = Field(pattern="^(pending|accepted|declined|tentative)$")


class FeedbackRequest(BaseModel):
    rating: float | None = Field(default=None, ge=0, le=10)
    recommendation: str | None = Field(
        default=None, pattern="^(strong_no|no|neutral|yes|strong_yes)$"
    )
    feedback: str | None = Field(default=None, max_length=10000)
    scorecard: dict | None = None


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class ParticipantOut(ORMModel):
    id: uuid.UUID
    interview_id: uuid.UUID
    user_id: uuid.UUID
    is_organizer: bool
    response_status: str
    rating: Decimal | None = None
    recommendation: str | None = None
    feedback: str | None = None
    feedback_submitted_at: datetime | None = None
    scorecard_json: dict | None = None


class InterviewOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    application_id: uuid.UUID
    type: InterviewType
    status: InterviewStatus
    round_number: int
    title: str | None = None
    scheduled_at: datetime | None = None
    duration_minutes: int
    timezone: str
    location: str | None = None
    meeting_url: str | None = None
    proposed_slots: list = Field(default_factory=list)
    booking_expires_at: datetime | None = None
    calendar_account_id: uuid.UUID | None = None
    external_event_id: str | None = None
    reminder_24h_sent_at: datetime | None = None
    reminder_1h_sent_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancellation_reason: str | None = None
    rescheduled_from_id: uuid.UUID | None = None
    ai_score: Decimal | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class InterviewDetail(InterviewOut):
    participants: list[ParticipantOut] = Field(default_factory=list)
    feedback_summary: dict | None = None
    # The candidate-facing link. Only ever returned to authenticated staff.
    booking_url: str | None = None


class SchedulingResponse(BaseModel):
    """An interview plus the soft failures that did not block it."""

    interview: InterviewOut
    slots: list[SlotOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FeedbackSummaryOut(BaseModel):
    participants: int
    submitted: int
    average_rating: float | None = None
    recommendations: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Candidate-facing (token-authenticated, unauthenticated routes)
# --------------------------------------------------------------------------- #
class BookingSlotChoice(BaseModel):
    slot_start: datetime


class BookingCancelRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class PublicBookingView(BaseModel):
    """What a candidate holding a booking link may see.

    Scoped tightly on purpose: the token is a bearer credential that may be
    forwarded or logged, so this exposes the meeting details and nothing about
    the pipeline, the interviewers, or the candidate's own record.
    """

    interview_id: uuid.UUID
    organization_name: str
    job_title: str | None = None
    type: InterviewType
    status: InterviewStatus
    duration_minutes: int
    timezone: str
    scheduled_at: datetime | None = None
    location: str | None = None
    meeting_url: str | None = None
    proposed_slots: list[SlotOut] = Field(default_factory=list)
    can_book: bool
    can_reschedule: bool
    can_cancel: bool
