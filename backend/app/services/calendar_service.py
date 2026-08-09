"""Connected calendars and the availability they feed (design §4.3).

This module owns ``CalendarAccount`` rows and turns them into the free-time
intervals the scheduler intersects. It is the seam between the pure interval
maths in ``availability`` and the vendor APIs in ``integrations.calendar``.

Two rules shape the availability it returns:

* **A failed calendar read is not an empty calendar.** If the vendor cannot be
  reached the result is flagged ``calendar_synced=False`` and falls back to
  working hours, so a recruiter is told the slots are unverified rather than
  being handed slots that quietly ignore a full day of meetings.
* **Interviews HireAgent booked always count as busy**, whether or not they
  made it onto an external calendar. Otherwise disabling calendar sync would
  let the product double-book its own interviews.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError
from app.db.tenancy import get_scoped, scoped_select
from app.integrations import calendar as calendar_api
from app.integrations.calendar import CalendarCredentials, TokenRefresh
from app.models.enums import InterviewStatus
from app.models.interview import CalendarAccount, Interview, InterviewParticipant
from app.services import auth_service
from app.services.availability import (
    Interval,
    ParticipantAvailability,
    expand_working_hours,
    merge,
    subtract,
    to_utc,
)

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = frozenset({calendar_api.GOOGLE, calendar_api.OUTLOOK})

# Interview states that still occupy the interviewer's diary.
BLOCKING_STATUSES = (
    InterviewStatus.PENDING,
    InterviewStatus.SCHEDULED,
    InterviewStatus.CONFIRMED,
)


# --------------------------------------------------------------------------- #
# Account management
# --------------------------------------------------------------------------- #
async def connect_account(
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    provider: str,
    email: str,
    access_token: str | None = None,
    refresh_token: str | None = None,
    token_expires_at: datetime | None = None,
    calendar_id: str | None = None,
    scopes: list[str] | None = None,
    working_hours: dict | None = None,
    timezone: str = "UTC",
) -> CalendarAccount:
    """Connect (or reconnect) a calendar for a user.

    Reconnecting the same provider+email updates the existing row instead of
    creating a second one, because the unique index would reject the duplicate
    and a user re-authorising is the normal way a lapsed refresh token is
    replaced.
    """
    provider = (provider or "").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConflictError(
            f"Unsupported calendar provider '{provider}'",
            details={"supported": sorted(SUPPORTED_PROVIDERS)},
        )

    # Raises NotFoundError if the user belongs to another tenant.
    await auth_service.get_user(session, organization_id, user_id)

    existing = await session.scalar(
        scoped_select(CalendarAccount, organization_id).where(
            CalendarAccount.user_id == user_id,
            CalendarAccount.provider == provider,
            CalendarAccount.email == email,
        )
    )

    account = existing or CalendarAccount(
        organization_id=organization_id,
        user_id=user_id,
        provider=provider,
        email=email,
    )
    account.access_token = access_token
    account.refresh_token = refresh_token
    account.token_expires_at = token_expires_at
    account.calendar_id = calendar_id
    account.scopes = scopes or []
    account.is_active = True
    # A fresh authorisation clears whatever went wrong last time.
    account.sync_error = None
    if working_hours is not None:
        account.working_hours = working_hours
    account.timezone = timezone or "UTC"

    if existing is None:
        session.add(account)
    await session.commit()
    await session.refresh(account)
    return account


async def list_accounts(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    user_id: uuid.UUID | None = None,
) -> list[CalendarAccount]:
    stmt = scoped_select(CalendarAccount, organization_id)
    if user_id is not None:
        stmt = stmt.where(CalendarAccount.user_id == user_id)
    result = await session.execute(stmt.order_by(CalendarAccount.created_at))
    return list(result.scalars().all())


async def get_account(
    session: AsyncSession, organization_id: uuid.UUID, account_id: uuid.UUID
) -> CalendarAccount:
    account = await get_scoped(session, CalendarAccount, account_id, organization_id)
    if account is None:
        raise NotFoundError("Calendar account not found")
    return account


async def update_working_hours(
    session: AsyncSession,
    organization_id: uuid.UUID,
    account_id: uuid.UUID,
    *,
    working_hours: dict | None = None,
    timezone: str | None = None,
) -> CalendarAccount:
    account = await get_account(session, organization_id, account_id)
    if working_hours is not None:
        account.working_hours = working_hours
    if timezone is not None:
        account.timezone = timezone
    await session.commit()
    await session.refresh(account)
    return account


async def disconnect_account(
    session: AsyncSession, organization_id: uuid.UUID, account_id: uuid.UUID
) -> None:
    """Soft-delete the connection and drop the stored OAuth material.

    The tokens are cleared rather than merely orphaned: a disconnected account
    should not leave usable credentials sitting in the database.
    """
    account = await get_account(session, organization_id, account_id)
    account.is_active = False
    account.access_token = None
    account.refresh_token = None
    account.token_expires_at = None
    account.soft_delete()
    await session.commit()


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
async def participant_availability(
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_ids: list[uuid.UUID],
    window: Interval,
    *,
    default_timezone: str = "UTC",
    default_working_hours: dict | None = None,
    exclude_interview_id: uuid.UUID | None = None,
) -> list[ParticipantAvailability]:
    """Free time for each user across ``window``."""
    accounts_by_user = await _accounts_by_user(session, organization_id, user_ids)
    booked = await _booked_intervals(
        session,
        organization_id,
        user_ids,
        window,
        exclude_interview_id=exclude_interview_id,
    )

    out: list[ParticipantAvailability] = []
    for user_id in user_ids:
        accounts = accounts_by_user.get(user_id, [])
        free, synced, error = await _free_for_user(
            session,
            accounts,
            window,
            default_timezone=default_timezone,
            default_working_hours=default_working_hours,
        )
        free = subtract(free, booked.get(user_id, []))
        out.append(
            ParticipantAvailability(
                user_id=user_id, free=free, calendar_synced=synced, error=error
            )
        )
    return out


async def _free_for_user(
    session: AsyncSession,
    accounts: list[CalendarAccount],
    window: Interval,
    *,
    default_timezone: str,
    default_working_hours: dict | None,
) -> tuple[list[Interval], bool, str | None]:
    """Working hours minus external busy time, for one user's accounts."""
    if not accounts:
        # No connected calendar is not an error: the interviewer is simply
        # schedulable during the organization's default hours.
        free = expand_working_hours(default_working_hours, default_timezone, window)
        return free, False, "No connected calendar"

    # The first connected account defines the working week; every account
    # contributes busy time, since a meeting on either calendar is a conflict.
    primary = accounts[0]
    free = expand_working_hours(
        primary.working_hours or default_working_hours,
        primary.timezone or default_timezone,
        window,
    )

    busy: list[Interval] = []
    synced = False
    errors: list[str] = []
    for account in accounts:
        result = await _fetch_busy(session, account, window)
        if result.synced:
            synced = True
        elif result.error:
            errors.append(f"{account.provider}: {result.error}")
        for start, end in result.blocks:
            interval = _safe_interval(start, end)
            if interval is not None:
                busy.append(interval)

    return subtract(free, busy), synced, "; ".join(errors) or None


async def _fetch_busy(
    session: AsyncSession, account: CalendarAccount, window: Interval
) -> calendar_api.BusyResult:
    """Read one account's busy blocks, recording the outcome on the row."""
    provider = calendar_api.get_provider(account.provider)
    credentials = CalendarCredentials(
        provider=account.provider,
        email=account.email,
        access_token=account.access_token,
        refresh_token=account.refresh_token,
        expires_at=account.token_expires_at,
        calendar_id=account.calendar_id,
    )

    try:
        result = await provider.fetch_busy(credentials, window.start, window.end)
    except Exception as exc:  # noqa: BLE001 - a provider bug must not break scheduling
        logger.exception("Calendar provider %s raised", account.provider)
        result = calendar_api.BusyResult(synced=False, error=str(exc))

    _persist_refresh(account, result.refreshed)
    if result.synced:
        account.last_synced_at = datetime.now(UTC)
        account.sync_error = None
    else:
        account.sync_error = result.error
    # Flushed, not committed: the caller decides the transaction boundary, and
    # a read-only slot lookup should not be what commits unrelated pending work.
    await session.flush()
    return result


def _persist_refresh(account: CalendarAccount, refreshed: TokenRefresh | None) -> None:
    if refreshed is None:
        return
    account.access_token = refreshed.access_token
    account.token_expires_at = refreshed.expires_at


def _safe_interval(start: datetime, end: datetime) -> Interval | None:
    """Build an interval from provider data, discarding nonsense."""
    try:
        return Interval(to_utc(start), to_utc(end))
    except ValueError:
        logger.warning("Discarding invalid busy block %s..%s", start, end)
        return None


async def _accounts_by_user(
    session: AsyncSession, organization_id: uuid.UUID, user_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[CalendarAccount]]:
    if not user_ids:
        return {}
    rows = (
        (
            await session.execute(
                scoped_select(CalendarAccount, organization_id)
                .where(
                    CalendarAccount.user_id.in_(user_ids),
                    CalendarAccount.is_active.is_(True),
                )
                .order_by(CalendarAccount.created_at)
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[uuid.UUID, list[CalendarAccount]] = {}
    for account in rows:
        grouped.setdefault(account.user_id, []).append(account)
    return grouped


async def _booked_intervals(
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_ids: list[uuid.UUID],
    window: Interval,
    *,
    exclude_interview_id: uuid.UUID | None = None,
) -> dict[uuid.UUID, list[Interval]]:
    """Interviews HireAgent has already booked, per interviewer."""
    if not user_ids:
        return {}

    stmt = (
        scoped_select(Interview, organization_id)
        .join(
            InterviewParticipant,
            InterviewParticipant.interview_id == Interview.id,
        )
        .where(
            InterviewParticipant.user_id.in_(user_ids),
            InterviewParticipant.deleted_at.is_(None),
            Interview.status.in_(list(BLOCKING_STATUSES)),
            Interview.scheduled_at.is_not(None),
            Interview.scheduled_at < window.end,
        )
        .add_columns(InterviewParticipant.user_id)
    )
    if exclude_interview_id is not None:
        stmt = stmt.where(Interview.id != exclude_interview_id)

    booked: dict[uuid.UUID, list[Interval]] = {}
    for interview, user_id in (await session.execute(stmt)).all():
        interval = interview_interval(interview)
        if interval is None or interval.end <= window.start:
            continue
        booked.setdefault(user_id, []).append(interval)
    return {user_id: merge(items) for user_id, items in booked.items()}


def interview_interval(interview: Interview) -> Interval | None:
    """The time an interview occupies, or ``None`` if it is not yet scheduled."""
    if interview.scheduled_at is None:
        return None
    start = to_utc(interview.scheduled_at)
    return Interval(start, start + timedelta(minutes=interview.duration_minutes or 45))
