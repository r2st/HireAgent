"""Sender mailboxes: warm-up, rotation, and reputation (design §2.2, §4.2).

Cold-sending a thousand messages from a brand-new mailbox is the fastest way to
get a domain blocked, so a sender ramps: it starts at a small daily allowance
and climbs a fixed step per day until it reaches the target, at which point it
is READY. The ramp is computed from ``warmup_started_at`` rather than
accumulated by a nightly job, so a worker that misses a night does not stall a
mailbox at yesterday's allowance.

Rotation picks the least recently used sendable account. "Sendable" is the
model's own predicate — active, warmed, in good standing, under today's cap —
which keeps the rule in one place and the query honest about what it filters.

Reputation is the brake. Bounces and complaints cost far more than a send earns
back, because a mailbox that keeps bouncing needs to stop long before a human
notices.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.integrations import email_gateway as email_api
from app.models.enums import EmailProvider, WarmupStatus
from app.models.outreach import EmailAccount

logger = logging.getLogger(__name__)

# What a bounce and a spam complaint cost a sender's 0-100 reputation. A
# complaint is far more damaging than a bounce: bad addresses happen, but a
# recipient pressing "spam" is the signal mailbox providers actually act on.
BOUNCE_PENALTY = 5.0
COMPLAINT_PENALTY = 20.0
# A clean send earns a little back, so one bad day does not sideline a mailbox
# forever — but recovery is deliberately slower than damage.
SEND_REWARD = 0.1


def _today(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(UTC).date()


def _clamp_reputation(value: float) -> float:
    return max(0.0, min(100.0, round(value, 2)))


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #
def warmup_limit(account: EmailAccount, now: datetime | None = None) -> int:
    """The daily allowance the ramp says this account should have today.

    Day 0 (the day warm-up starts) gets the initial allowance; each further day
    adds one increment, capped at the target.
    """
    initial = max(1, settings.warmup_initial_daily_limit)
    target = max(initial, settings.warmup_target_daily_limit)
    if account.warmup_started_at is None:
        return initial

    started = account.warmup_started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    days = (_today(now) - started.astimezone(UTC).date()).days
    if days < 0:
        # A start date in the future means warm-up has not begun.
        return initial
    return min(target, initial + days * max(0, settings.warmup_daily_increment))


def apply_warmup(account: EmailAccount, now: datetime | None = None) -> bool:
    """Move an account along its ramp. Returns whether anything changed.

    Only WARMING accounts ramp: a paused or blocked mailbox keeps whatever
    allowance it had, and a READY one is already at the target.
    """
    if account.warmup_status != WarmupStatus.WARMING:
        return False

    limit = warmup_limit(account, now)
    changed = False
    if account.daily_limit != limit:
        account.daily_limit = limit
        changed = True
    if limit >= max(1, settings.warmup_target_daily_limit):
        account.warmup_status = WarmupStatus.READY
        changed = True
    return changed


def start_warmup(account: EmailAccount, now: datetime | None = None) -> EmailAccount:
    """Begin (or restart) the ramp for an account."""
    account.warmup_status = WarmupStatus.WARMING
    account.warmup_started_at = now or datetime.now(UTC)
    account.daily_limit = max(1, settings.warmup_initial_daily_limit)
    apply_warmup(account, now)
    return account


# --------------------------------------------------------------------------- #
# Daily counters
# --------------------------------------------------------------------------- #
def roll_daily_counter(account: EmailAccount, now: datetime | None = None) -> bool:
    """Zero ``sent_today`` when the UTC date has turned over.

    Done lazily on read rather than by a midnight job: the counter is only ever
    consulted while deciding whether to send, so rolling it there means there is
    no window in which a stale count blocks a mailbox.
    """
    today = _today(now)
    stamped = account.sent_today_date
    if stamped is not None:
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=UTC)
        if stamped.astimezone(UTC).date() == today:
            return False
    account.sent_today = 0
    account.sent_today_date = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
    return True


def remaining_today(account: EmailAccount, now: datetime | None = None) -> int:
    """How many more messages this account may send today."""
    roll_daily_counter(account, now)
    return max(0, account.daily_limit - account.sent_today)


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
async def sendable_accounts(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    allowed_ids: list[uuid.UUID] | None = None,
    now: datetime | None = None,
) -> list[EmailAccount]:
    """Every account eligible to send right now, least recently used first.

    The coarse filters run in SQL; warm-up ramping and the daily-counter roll
    are per-row mutations, so eligibility is finished in Python against the
    freshly rolled state.
    """
    stmt = scoped_select(EmailAccount, organization_id).where(
        EmailAccount.is_active.is_(True),
        EmailAccount.warmup_status.in_(
            [WarmupStatus.WARMING.value, WarmupStatus.READY.value]
        ),
    )
    if allowed_ids:
        stmt = stmt.where(EmailAccount.id.in_(allowed_ids))
    # NULLs first: a never-used account should be tried before any used one.
    stmt = stmt.order_by(EmailAccount.last_used_at.asc().nullsfirst())

    accounts = list((await session.execute(stmt)).scalars().all())
    eligible: list[EmailAccount] = []
    for account in accounts:
        apply_warmup(account, now)
        roll_daily_counter(account, now)
        if float(account.reputation_score) < settings.sender_min_reputation:
            continue
        if account.sent_today >= account.daily_limit:
            continue
        eligible.append(account)
    return eligible


async def pick_sender(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    allowed_ids: list[uuid.UUID] | None = None,
    now: datetime | None = None,
) -> EmailAccount | None:
    """The next account rotation should use, or ``None`` if the org is capped.

    Falls back to the whole org when the sequence's own sender pool is exhausted
    but other mailboxes are free — a campaign should not stall because one of
    its two nominated senders hit its cap, and the pool is a preference rather
    than a security boundary.
    """
    if allowed_ids:
        preferred = await sendable_accounts(
            session, organization_id, allowed_ids=allowed_ids, now=now
        )
        if preferred:
            return preferred[0]
    accounts = await sendable_accounts(session, organization_id, now=now)
    return accounts[0] if accounts else None


def record_send(
    account: EmailAccount,
    *,
    ok: bool,
    bounced: bool = False,
    complained: bool = False,
    now: datetime | None = None,
    error: str | None = None,
) -> EmailAccount:
    """Fold one send's outcome into the account's counters and reputation.

    A failed attempt still counts against the daily allowance when it reached
    the provider: the mailbox's reputation was spent either way, and retrying
    it against the same account within the same day is exactly what a bounce
    loop looks like from outside.
    """
    stamp = now or datetime.now(UTC)
    roll_daily_counter(account, stamp)

    account.sent_today += 1
    account.last_used_at = stamp
    reputation = float(account.reputation_score)

    if ok:
        account.total_sent += 1
        account.last_error = None
        reputation += SEND_REWARD
    else:
        account.last_error = error
    if bounced:
        account.bounce_count += 1
        reputation -= BOUNCE_PENALTY
    if complained:
        account.complaint_count += 1
        reputation -= COMPLAINT_PENALTY

    account.reputation_score = _clamp_reputation(reputation)
    if account.reputation_score < settings.sender_min_reputation:
        # Pull it from rotation loudly rather than letting it sit at a
        # reputation the rotation filter would silently skip.
        account.warmup_status = WarmupStatus.BLOCKED
        logger.warning(
            "Email account %s blocked: reputation %.2f below %.2f",
            account.id,
            account.reputation_score,
            settings.sender_min_reputation,
        )
    return account


def credentials_for(account: EmailAccount) -> email_api.EmailCredentials:
    """Project an account row onto the gateway's credential object."""
    return email_api.EmailCredentials(
        provider=str(account.provider),
        email=account.email,
        display_name=account.display_name,
        smtp_host=account.smtp_host,
        smtp_port=account.smtp_port,
        smtp_username=account.smtp_username or account.email,
        smtp_password=account.smtp_password,
        access_token=account.oauth_access_token,
        refresh_token=account.oauth_refresh_token,
        expires_at=account.oauth_expires_at,
    )


def store_refreshed_token(
    account: EmailAccount, refreshed: email_api.TokenRefresh | None
) -> None:
    """Persist a token the gateway renewed mid-send."""
    if refreshed is None:
        return
    account.oauth_access_token = refreshed.access_token
    account.oauth_expires_at = refreshed.expires_at


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
async def create_account(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    email: str,
    provider: EmailProvider = EmailProvider.SMTP,
    display_name: str | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_username: str | None = None,
    smtp_password: str | None = None,
    imap_host: str | None = None,
    imap_port: int | None = None,
    oauth_access_token: str | None = None,
    oauth_refresh_token: str | None = None,
    oauth_expires_at: datetime | None = None,
    start_warmup_now: bool = True,
    now: datetime | None = None,
) -> EmailAccount:
    address = (email or "").strip().lower()
    if not address:
        raise ValidationError("An email address is required")

    existing = await session.scalar(
        scoped_select(EmailAccount, organization_id).where(
            EmailAccount.email == address
        )
    )
    if existing is not None:
        raise ConflictError(f"Email account {address} already exists")

    if provider in (EmailProvider.SMTP, EmailProvider.SES) and not smtp_host:
        raise ValidationError(f"{provider} accounts require an SMTP host")

    account = EmailAccount(
        organization_id=organization_id,
        email=address,
        display_name=display_name,
        provider=provider,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        imap_host=imap_host,
        imap_port=imap_port,
        oauth_access_token=oauth_access_token,
        oauth_refresh_token=oauth_refresh_token,
        oauth_expires_at=oauth_expires_at,
    )
    if start_warmup_now:
        start_warmup(account, now)
    else:
        account.daily_limit = max(1, settings.warmup_initial_daily_limit)

    session.add(account)
    await session.commit()
    await session.refresh(account)
    return account


async def get_account(
    session: AsyncSession, organization_id: uuid.UUID, account_id: uuid.UUID
) -> EmailAccount:
    account = await get_scoped(session, EmailAccount, account_id, organization_id)
    if account is None:
        raise NotFoundError("Email account not found")
    return account


async def list_accounts(
    session: AsyncSession, organization_id: uuid.UUID, *, now: datetime | None = None
) -> list[EmailAccount]:
    stmt = scoped_select(EmailAccount, organization_id).order_by(
        EmailAccount.created_at.asc()
    )
    accounts = list((await session.execute(stmt)).scalars().all())
    for account in accounts:
        apply_warmup(account, now)
        roll_daily_counter(account, now)
    return accounts


async def update_account(
    session: AsyncSession,
    organization_id: uuid.UUID,
    account_id: uuid.UUID,
    *,
    changes: dict,
) -> EmailAccount:
    account = await get_account(session, organization_id, account_id)

    for field_name, value in changes.items():
        if value is None or not hasattr(account, field_name):
            continue
        setattr(account, field_name, value)

    # Restarting warm-up is a state transition, not a field assignment: the
    # ramp has to be re-anchored or the account jumps straight to whatever
    # allowance its old start date implies.
    status = changes.get("warmup_status")
    if status == WarmupStatus.WARMING and account.warmup_started_at is None:
        start_warmup(account)

    await session.commit()
    await session.refresh(account)
    return account


async def delete_account(
    session: AsyncSession, organization_id: uuid.UUID, account_id: uuid.UUID
) -> None:
    account = await get_account(session, organization_id, account_id)
    account.deleted_at = datetime.now(UTC)
    account.is_active = False
    await session.commit()


async def due_for_warmup(
    session: AsyncSession, *, now: datetime | None = None, limit: int = 500
) -> list[EmailAccount]:
    """Warming accounts whose allowance is behind the ramp.

    The sweep exists so a dashboard shows the right number without waiting for
    the account's next send to roll it forward.
    """
    stmt = (
        select(EmailAccount)
        .where(
            EmailAccount.deleted_at.is_(None),
            EmailAccount.is_active.is_(True),
            EmailAccount.warmup_status == WarmupStatus.WARMING.value,
        )
        .limit(limit)
    )
    accounts = list((await session.execute(stmt)).scalars().all())
    return [a for a in accounts if a.daily_limit != warmup_limit(a, now)]


async def sweep_warmup(
    session: AsyncSession, *, now: datetime | None = None, limit: int = 500
) -> int:
    """Advance every lagging warm-up ramp. Returns how many moved."""
    accounts = await due_for_warmup(session, now=now, limit=limit)
    changed = sum(1 for account in accounts if apply_warmup(account, now))
    if changed:
        await session.commit()
    return changed


def next_reset_at(now: datetime | None = None) -> datetime:
    """Midnight UTC after ``now`` — when daily allowances refresh."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    return datetime.combine(
        stamp.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
