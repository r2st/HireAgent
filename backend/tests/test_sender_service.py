"""Sender mailboxes: warm-up, rotation, and reputation (design §2.2, §4.2).

Deliverability is the one thing outreach cannot buy back once spent, so the
rules here are pinned hard: a mailbox ramps on a schedule derived from its start
date (not from how often a worker ran), rotation never hands out an account that
is over its cap, and a bouncing sender leaves the pool before a human notices.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.integrations.oauth import TokenRefresh
from app.models.enums import EmailProvider, WarmupStatus
from app.models.organization import Organization
from app.models.outreach import EmailAccount
from app.services import sender_service as svc

# A Monday, so nothing here accidentally depends on a weekend rule.
NOW = datetime(2027, 3, 8, 12, 0, tzinfo=UTC)


async def make_org(session, name: str = "Acme") -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


def account(org: Organization, **fields) -> EmailAccount:
    """An in-memory account with sane defaults; nothing is persisted."""
    defaults = dict(
        organization_id=org.id,
        email=f"sender-{uuid.uuid4().hex[:8]}@acme.test",
        provider=EmailProvider.SMTP,
        smtp_host="smtp.acme.test",
        warmup_status=WarmupStatus.READY,
        daily_limit=50,
        sent_today=0,
        reputation_score=100,
        bounce_count=0,
        complaint_count=0,
        total_sent=0,
        is_active=True,
    )
    defaults.update(fields)
    return EmailAccount(**defaults)


async def persist(session, *accounts: EmailAccount) -> None:
    session.add_all(accounts)
    await session.commit()


# --------------------------------------------------------------------------- #
# Warm-up ramp
# --------------------------------------------------------------------------- #
class TestWarmupLimit:
    def test_an_unstarted_ramp_gets_the_initial_allowance(self, org_stub) -> None:
        acct = account(org_stub, warmup_started_at=None)
        assert svc.warmup_limit(acct, NOW) == settings.warmup_initial_daily_limit

    def test_day_zero_gets_the_initial_allowance(self, org_stub) -> None:
        acct = account(org_stub, warmup_started_at=NOW)
        assert svc.warmup_limit(acct, NOW) == settings.warmup_initial_daily_limit

    def test_each_further_day_adds_one_increment(self, org_stub) -> None:
        acct = account(org_stub, warmup_started_at=NOW - timedelta(days=3))
        expected = (
            settings.warmup_initial_daily_limit + 3 * settings.warmup_daily_increment
        )
        assert svc.warmup_limit(acct, NOW) == expected

    def test_the_ramp_is_capped_at_the_target(self, org_stub) -> None:
        acct = account(org_stub, warmup_started_at=NOW - timedelta(days=9999))
        assert svc.warmup_limit(acct, NOW) == settings.warmup_target_daily_limit

    def test_the_ramp_derives_from_the_start_date_not_from_worker_runs(
        self, org_stub
    ) -> None:
        # The point of computing rather than accumulating: a worker that missed
        # a week still lands the account on the allowance the calendar implies.
        acct = account(org_stub, warmup_started_at=NOW - timedelta(days=7))
        missed_a_week = svc.warmup_limit(acct, NOW)
        assert missed_a_week == (
            settings.warmup_initial_daily_limit + 7 * settings.warmup_daily_increment
        )

    def test_a_future_start_date_has_not_begun_ramping(self, org_stub) -> None:
        acct = account(org_stub, warmup_started_at=NOW + timedelta(days=2))
        assert svc.warmup_limit(acct, NOW) == settings.warmup_initial_daily_limit

    def test_a_naive_start_date_is_read_as_utc(self, org_stub) -> None:
        # SQLite hands back naive datetimes on some paths; the ramp must not
        # throw when it subtracts one.
        acct = account(org_stub, warmup_started_at=datetime(2027, 3, 5, 12, 0))
        assert svc.warmup_limit(acct, NOW) == (
            settings.warmup_initial_daily_limit + 3 * settings.warmup_daily_increment
        )


class TestApplyWarmup:
    def test_a_warming_account_moves_to_todays_allowance(self, org_stub) -> None:
        acct = account(
            org_stub,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=2),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        assert svc.apply_warmup(acct, NOW) is True
        assert acct.daily_limit == (
            settings.warmup_initial_daily_limit + 2 * settings.warmup_daily_increment
        )

    def test_an_account_already_at_todays_allowance_is_unchanged(
        self, org_stub
    ) -> None:
        acct = account(
            org_stub,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW,
            daily_limit=settings.warmup_initial_daily_limit,
        )
        assert svc.apply_warmup(acct, NOW) is False

    def test_reaching_the_target_promotes_the_account_to_ready(self, org_stub) -> None:
        acct = account(
            org_stub,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=500),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        assert svc.apply_warmup(acct, NOW) is True
        assert acct.warmup_status == WarmupStatus.READY
        assert acct.daily_limit == settings.warmup_target_daily_limit

    @pytest.mark.parametrize(
        "status",
        [
            WarmupStatus.NOT_STARTED,
            WarmupStatus.PAUSED,
            WarmupStatus.BLOCKED,
            WarmupStatus.READY,
        ],
    )
    def test_only_warming_accounts_ramp(self, org_stub, status) -> None:
        # A paused or blocked mailbox keeps the allowance it had: ramping one
        # would quietly undo the intervention that paused it.
        acct = account(
            org_stub,
            warmup_status=status,
            warmup_started_at=NOW - timedelta(days=30),
            daily_limit=7,
        )
        assert svc.apply_warmup(acct, NOW) is False
        assert acct.daily_limit == 7
        assert acct.warmup_status == status


class TestStartWarmup:
    def test_it_anchors_the_ramp_and_resets_the_allowance(self, org_stub) -> None:
        acct = account(org_stub, warmup_status=WarmupStatus.NOT_STARTED, daily_limit=99)
        svc.start_warmup(acct, NOW)
        assert acct.warmup_status == WarmupStatus.WARMING
        assert acct.warmup_started_at == NOW
        assert acct.daily_limit == settings.warmup_initial_daily_limit

    def test_restarting_re_anchors_an_old_ramp(self, org_stub) -> None:
        # Without re-anchoring, an account restarted after a long pause would
        # jump straight back to its pre-pause allowance.
        acct = account(
            org_stub,
            warmup_status=WarmupStatus.PAUSED,
            warmup_started_at=NOW - timedelta(days=300),
            daily_limit=settings.warmup_target_daily_limit,
        )
        svc.start_warmup(acct, NOW)
        assert acct.daily_limit == settings.warmup_initial_daily_limit
        assert acct.warmup_status == WarmupStatus.WARMING


# --------------------------------------------------------------------------- #
# Daily counters
# --------------------------------------------------------------------------- #
class TestDailyCounter:
    def test_a_fresh_account_is_stamped_with_todays_date(self, org_stub) -> None:
        acct = account(org_stub, sent_today=4, sent_today_date=None)
        assert svc.roll_daily_counter(acct, NOW) is True
        assert acct.sent_today == 0
        assert acct.sent_today_date is not None
        assert acct.sent_today_date.date() == NOW.date()

    def test_the_same_utc_day_leaves_the_count_alone(self, org_stub) -> None:
        acct = account(org_stub, sent_today=4, sent_today_date=NOW - timedelta(hours=6))
        assert svc.roll_daily_counter(acct, NOW) is False
        assert acct.sent_today == 4

    def test_a_new_utc_day_zeroes_the_count(self, org_stub) -> None:
        acct = account(org_stub, sent_today=40, sent_today_date=NOW - timedelta(days=1))
        assert svc.roll_daily_counter(acct, NOW) is True
        assert acct.sent_today == 0

    def test_a_naive_stamp_is_read_as_utc(self, org_stub) -> None:
        acct = account(org_stub, sent_today=4, sent_today_date=datetime(2027, 3, 8, 1))
        assert svc.roll_daily_counter(acct, NOW) is False
        assert acct.sent_today == 4

    def test_remaining_today_rolls_before_it_answers(self, org_stub) -> None:
        # Yesterday's exhausted account has its full allowance again today, and
        # must not have to wait for a nightly job to learn that.
        acct = account(
            org_stub,
            daily_limit=20,
            sent_today=20,
            sent_today_date=NOW - timedelta(days=1),
        )
        assert svc.remaining_today(acct, NOW) == 20

    def test_remaining_today_never_goes_negative(self, org_stub) -> None:
        acct = account(org_stub, daily_limit=5, sent_today=9, sent_today_date=NOW)
        assert svc.remaining_today(acct, NOW) == 0


def test_next_reset_at_is_the_following_midnight_utc() -> None:
    assert svc.next_reset_at(NOW) == datetime(2027, 3, 9, tzinfo=UTC)


def test_next_reset_at_from_a_non_utc_zone_uses_the_utc_day() -> None:
    # 23:00 UTC-5 is 04:00 UTC the next day, so the reset is that day's end.
    stamp = datetime(2027, 3, 8, 23, 0, tzinfo=UTC) + timedelta(hours=5)
    assert svc.next_reset_at(stamp) == datetime(2027, 3, 10, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
class TestSendableAccounts:
    async def test_least_recently_used_comes_first(self, session) -> None:
        org = await make_org(session)
        stale = account(org, last_used_at=NOW - timedelta(hours=5))
        recent = account(org, last_used_at=NOW - timedelta(minutes=1))
        await persist(session, recent, stale)

        order = await svc.sendable_accounts(session, org.id, now=NOW)
        assert [a.id for a in order] == [stale.id, recent.id]

    async def test_a_never_used_account_outranks_every_used_one(self, session) -> None:
        org = await make_org(session)
        used = account(org, last_used_at=NOW - timedelta(days=30))
        virgin = account(org, last_used_at=None)
        await persist(session, used, virgin)

        order = await svc.sendable_accounts(session, org.id, now=NOW)
        assert [a.id for a in order] == [virgin.id, used.id]

    async def test_an_inactive_account_is_excluded(self, session) -> None:
        org = await make_org(session)
        await persist(session, account(org, is_active=False))
        assert await svc.sendable_accounts(session, org.id, now=NOW) == []

    async def test_a_soft_deleted_account_is_excluded(self, session) -> None:
        org = await make_org(session)
        await persist(session, account(org, deleted_at=NOW))
        assert await svc.sendable_accounts(session, org.id, now=NOW) == []

    @pytest.mark.parametrize(
        "status",
        [WarmupStatus.NOT_STARTED, WarmupStatus.PAUSED, WarmupStatus.BLOCKED],
    )
    async def test_only_warming_and_ready_accounts_may_send(
        self, session, status
    ) -> None:
        org = await make_org(session)
        await persist(session, account(org, warmup_status=status))
        assert await svc.sendable_accounts(session, org.id, now=NOW) == []

    async def test_a_low_reputation_account_is_pulled_from_rotation(
        self, session
    ) -> None:
        org = await make_org(session)
        await persist(
            session, account(org, reputation_score=settings.sender_min_reputation - 1)
        )
        assert await svc.sendable_accounts(session, org.id, now=NOW) == []

    async def test_an_account_exactly_at_the_reputation_floor_still_sends(
        self, session
    ) -> None:
        org = await make_org(session)
        acct = account(org, reputation_score=settings.sender_min_reputation)
        await persist(session, acct)
        assert [
            a.id for a in await svc.sendable_accounts(session, org.id, now=NOW)
        ] == [acct.id]

    async def test_an_account_at_its_daily_cap_is_excluded(self, session) -> None:
        org = await make_org(session)
        await persist(
            session, account(org, daily_limit=5, sent_today=5, sent_today_date=NOW)
        )
        assert await svc.sendable_accounts(session, org.id, now=NOW) == []

    async def test_yesterdays_cap_does_not_block_today(self, session) -> None:
        org = await make_org(session)
        acct = account(
            org, daily_limit=5, sent_today=5, sent_today_date=NOW - timedelta(days=1)
        )
        await persist(session, acct)
        eligible = await svc.sendable_accounts(session, org.id, now=NOW)
        assert [a.id for a in eligible] == [acct.id]
        assert acct.sent_today == 0

    async def test_eligibility_is_judged_after_the_ramp_moves(self, session) -> None:
        # Stored allowance says 10 and today's sends are 12, but the ramp says
        # this account is entitled to more than that by now.
        org = await make_org(session)
        acct = account(
            org,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=4),
            daily_limit=settings.warmup_initial_daily_limit,
            sent_today=settings.warmup_initial_daily_limit + 2,
            sent_today_date=NOW,
        )
        await persist(session, acct)
        eligible = await svc.sendable_accounts(session, org.id, now=NOW)
        assert [a.id for a in eligible] == [acct.id]

    async def test_another_tenants_accounts_are_invisible(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        await persist(session, account(theirs))
        assert await svc.sendable_accounts(session, mine.id, now=NOW) == []

    async def test_allowed_ids_narrows_the_pool(self, session) -> None:
        org = await make_org(session)
        wanted = account(org, last_used_at=NOW)
        other = account(org, last_used_at=None)
        await persist(session, wanted, other)

        eligible = await svc.sendable_accounts(
            session, org.id, allowed_ids=[wanted.id], now=NOW
        )
        assert [a.id for a in eligible] == [wanted.id]


@pytest.mark.anyio
class TestPickSender:
    async def test_it_returns_the_least_recently_used_account(self, session) -> None:
        org = await make_org(session)
        stale = account(org, last_used_at=NOW - timedelta(days=1))
        recent = account(org, last_used_at=NOW)
        await persist(session, stale, recent)
        picked = await svc.pick_sender(session, org.id, now=NOW)
        assert picked is not None and picked.id == stale.id

    async def test_it_prefers_the_sequences_own_pool(self, session) -> None:
        org = await make_org(session)
        pooled = account(org, last_used_at=NOW)
        outsider = account(org, last_used_at=None)
        await persist(session, pooled, outsider)

        picked = await svc.pick_sender(
            session, org.id, allowed_ids=[pooled.id], now=NOW
        )
        assert picked is not None and picked.id == pooled.id

    async def test_an_exhausted_pool_falls_back_to_the_org(self, session) -> None:
        # The pool is a preference, not a security boundary: a campaign should
        # not stall because its two nominated senders hit their caps.
        org = await make_org(session)
        capped = account(org, daily_limit=1, sent_today=1, sent_today_date=NOW)
        spare = account(org)
        await persist(session, capped, spare)

        picked = await svc.pick_sender(
            session, org.id, allowed_ids=[capped.id], now=NOW
        )
        assert picked is not None and picked.id == spare.id

    async def test_a_fully_capped_org_yields_nothing(self, session) -> None:
        org = await make_org(session)
        await persist(
            session, account(org, daily_limit=2, sent_today=2, sent_today_date=NOW)
        )
        assert await svc.pick_sender(session, org.id, now=NOW) is None

    async def test_an_org_with_no_accounts_yields_nothing(self, session) -> None:
        org = await make_org(session)
        assert await svc.pick_sender(session, org.id, now=NOW) is None


# --------------------------------------------------------------------------- #
# Reputation
# --------------------------------------------------------------------------- #
class TestRecordSend:
    def test_a_clean_send_advances_the_counters(self, org_stub) -> None:
        acct = account(org_stub, sent_today_date=NOW, total_sent=3, reputation_score=90)
        svc.record_send(acct, ok=True, now=NOW)
        assert acct.sent_today == 1
        assert acct.total_sent == 4
        assert acct.last_used_at == NOW
        assert float(acct.reputation_score) == pytest.approx(90 + svc.SEND_REWARD)

    def test_a_clean_send_clears_a_stale_error(self, org_stub) -> None:
        acct = account(org_stub, sent_today_date=NOW, last_error="yesterday's timeout")
        svc.record_send(acct, ok=True, now=NOW)
        assert acct.last_error is None

    def test_a_failure_still_spends_the_daily_allowance(self, org_stub) -> None:
        # It reached the provider, so the mailbox's reputation was spent either
        # way; retrying it against the same account today is a bounce loop.
        acct = account(org_stub, sent_today_date=NOW)
        svc.record_send(acct, ok=False, now=NOW, error="SMTP 421")
        assert acct.sent_today == 1
        assert acct.total_sent == 0
        assert acct.last_error == "SMTP 421"

    def test_a_bounce_costs_reputation_and_is_counted(self, org_stub) -> None:
        acct = account(org_stub, sent_today_date=NOW, reputation_score=90)
        svc.record_send(acct, ok=False, bounced=True, now=NOW)
        assert acct.bounce_count == 1
        assert float(acct.reputation_score) == pytest.approx(90 - svc.BOUNCE_PENALTY)

    def test_a_complaint_costs_far_more_than_a_bounce(self, org_stub) -> None:
        assert svc.COMPLAINT_PENALTY > svc.BOUNCE_PENALTY
        acct = account(org_stub, sent_today_date=NOW, reputation_score=90)
        svc.record_send(acct, ok=False, complained=True, now=NOW)
        assert acct.complaint_count == 1
        assert float(acct.reputation_score) == pytest.approx(90 - svc.COMPLAINT_PENALTY)

    def test_recovery_is_slower_than_damage(self, org_stub) -> None:
        assert svc.SEND_REWARD < svc.BOUNCE_PENALTY
        acct = account(org_stub, sent_today_date=NOW, reputation_score=99.95)
        svc.record_send(acct, ok=True, now=NOW)
        assert float(acct.reputation_score) == 100.0

    def test_reputation_is_clamped_at_zero(self, org_stub) -> None:
        acct = account(org_stub, sent_today_date=NOW, reputation_score=5)
        svc.record_send(acct, ok=False, bounced=True, complained=True, now=NOW)
        assert float(acct.reputation_score) == 0.0

    def test_falling_below_the_floor_blocks_the_account(self, org_stub) -> None:
        # Blocked rather than merely filtered out: the state should be visible
        # to a human, not implied by a query nobody reads.
        acct = account(
            org_stub,
            sent_today_date=NOW,
            reputation_score=settings.sender_min_reputation + 1,
        )
        svc.record_send(acct, ok=False, bounced=True, now=NOW)
        assert acct.warmup_status == WarmupStatus.BLOCKED

    def test_staying_at_the_floor_does_not_block(self, org_stub) -> None:
        acct = account(
            org_stub,
            sent_today_date=NOW,
            reputation_score=settings.sender_min_reputation + svc.BOUNCE_PENALTY,
        )
        svc.record_send(acct, ok=False, bounced=True, now=NOW)
        assert float(acct.reputation_score) == settings.sender_min_reputation
        assert acct.warmup_status != WarmupStatus.BLOCKED

    def test_it_rolls_a_stale_daily_counter_first(self, org_stub) -> None:
        acct = account(org_stub, sent_today=99, sent_today_date=NOW - timedelta(days=2))
        svc.record_send(acct, ok=True, now=NOW)
        assert acct.sent_today == 1


# --------------------------------------------------------------------------- #
# Credential projection
# --------------------------------------------------------------------------- #
class TestCredentials:
    def test_it_projects_the_row_onto_the_gateway_object(self, org_stub) -> None:
        acct = account(
            org_stub,
            email="Sender@Acme.test",
            display_name="Acme Talent",
            provider=EmailProvider.GMAIL,
            oauth_access_token="at",
            oauth_refresh_token="rt",
            oauth_expires_at=NOW,
        )
        creds = svc.credentials_for(acct)
        assert creds.provider == EmailProvider.GMAIL
        assert creds.display_name == "Acme Talent"
        assert (creds.access_token, creds.refresh_token) == ("at", "rt")
        assert creds.expires_at == NOW

    def test_a_blank_smtp_username_defaults_to_the_address(self, org_stub) -> None:
        acct = account(org_stub, email="me@acme.test", smtp_username=None)
        assert svc.credentials_for(acct).smtp_username == "me@acme.test"

    def test_a_refreshed_token_is_written_back(self, org_stub) -> None:
        acct = account(org_stub, oauth_access_token="old", oauth_expires_at=NOW)
        later = NOW + timedelta(hours=1)
        svc.store_refreshed_token(acct, TokenRefresh("new", later))
        assert acct.oauth_access_token == "new"
        assert acct.oauth_expires_at == later

    def test_no_refresh_leaves_the_stored_token_alone(self, org_stub) -> None:
        acct = account(org_stub, oauth_access_token="old", oauth_expires_at=NOW)
        svc.store_refreshed_token(acct, None)
        assert acct.oauth_access_token == "old"
        assert acct.oauth_expires_at == NOW


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
class TestCreateAccount:
    async def test_it_lowercases_and_trims_the_address(self, session) -> None:
        org = await make_org(session)
        acct = await svc.create_account(
            session, org.id, email="  Talent@Acme.TEST ", smtp_host="smtp.acme.test"
        )
        assert acct.email == "talent@acme.test"

    async def test_it_starts_the_ramp_by_default(self, session) -> None:
        org = await make_org(session)
        acct = await svc.create_account(
            session, org.id, email="a@acme.test", smtp_host="smtp.acme.test", now=NOW
        )
        assert acct.warmup_status == WarmupStatus.WARMING
        assert acct.daily_limit == settings.warmup_initial_daily_limit

    async def test_warmup_can_be_deferred(self, session) -> None:
        org = await make_org(session)
        acct = await svc.create_account(
            session,
            org.id,
            email="b@acme.test",
            smtp_host="smtp.acme.test",
            start_warmup_now=False,
        )
        assert acct.warmup_status == WarmupStatus.NOT_STARTED
        assert acct.warmup_started_at is None

    async def test_a_blank_address_is_rejected(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError):
            await svc.create_account(session, org.id, email="   ")

    async def test_a_duplicate_address_conflicts(self, session) -> None:
        org = await make_org(session)
        await svc.create_account(
            session, org.id, email="dup@acme.test", smtp_host="smtp.acme.test"
        )
        with pytest.raises(ConflictError):
            await svc.create_account(
                session, org.id, email="DUP@acme.test", smtp_host="smtp.acme.test"
            )

    async def test_the_same_address_may_exist_in_another_tenant(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        for org in (mine, theirs):
            await svc.create_account(
                session, org.id, email="shared@acme.test", smtp_host="smtp.acme.test"
            )
        assert len(await svc.list_accounts(session, mine.id)) == 1

    @pytest.mark.parametrize("provider", [EmailProvider.SMTP, EmailProvider.SES])
    async def test_smtp_style_providers_require_a_host(self, session, provider) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError, match="SMTP host"):
            await svc.create_account(
                session, org.id, email="c@acme.test", provider=provider
            )

    async def test_oauth_providers_need_no_host(self, session) -> None:
        org = await make_org(session)
        acct = await svc.create_account(
            session,
            org.id,
            email="d@acme.test",
            provider=EmailProvider.GMAIL,
            oauth_refresh_token="rt",
        )
        assert acct.provider == EmailProvider.GMAIL

    async def test_secrets_survive_the_encrypted_round_trip(self, session) -> None:
        org = await make_org(session)
        created = await svc.create_account(
            session,
            org.id,
            email="e@acme.test",
            smtp_host="smtp.acme.test",
            smtp_password="hunter2",
        )
        session.expunge_all()
        loaded = await svc.get_account(session, org.id, created.id)
        assert loaded.smtp_password == "hunter2"


@pytest.mark.anyio
class TestAccountLookup:
    async def test_a_missing_account_is_not_found(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(NotFoundError):
            await svc.get_account(session, org.id, uuid.uuid4())

    async def test_another_tenants_account_is_not_found(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        acct = account(theirs)
        await persist(session, acct)
        with pytest.raises(NotFoundError):
            await svc.get_account(session, mine.id, acct.id)

    async def test_listing_refreshes_the_ramp_it_shows(self, session) -> None:
        # A dashboard should show today's real allowance without waiting for
        # the account's next send to roll it forward.
        org = await make_org(session)
        acct = account(
            org,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=2),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        await persist(session, acct)
        listed = await svc.list_accounts(session, org.id, now=NOW)
        assert listed[0].daily_limit > settings.warmup_initial_daily_limit

    async def test_listing_hides_deleted_accounts(self, session) -> None:
        org = await make_org(session)
        acct = account(org)
        await persist(session, acct)
        await svc.delete_account(session, org.id, acct.id)
        assert await svc.list_accounts(session, org.id) == []


@pytest.mark.anyio
class TestUpdateAccount:
    async def test_it_applies_supplied_fields(self, session) -> None:
        org = await make_org(session)
        acct = account(org, display_name="Old")
        await persist(session, acct)
        updated = await svc.update_account(
            session, org.id, acct.id, changes={"display_name": "New"}
        )
        assert updated.display_name == "New"

    async def test_none_means_leave_alone_not_clear(self, session) -> None:
        org = await make_org(session)
        acct = account(org, display_name="Keep")
        await persist(session, acct)
        updated = await svc.update_account(
            session, org.id, acct.id, changes={"display_name": None}
        )
        assert updated.display_name == "Keep"

    async def test_unknown_fields_are_ignored(self, session) -> None:
        org = await make_org(session)
        acct = account(org)
        await persist(session, acct)
        updated = await svc.update_account(
            session, org.id, acct.id, changes={"not_a_column": "x"}
        )
        assert not hasattr(updated, "not_a_column")

    async def test_switching_to_warming_anchors_a_fresh_ramp(self, session) -> None:
        org = await make_org(session)
        acct = account(org, warmup_status=WarmupStatus.NOT_STARTED, daily_limit=99)
        await persist(session, acct)
        updated = await svc.update_account(
            session,
            org.id,
            acct.id,
            changes={"warmup_status": WarmupStatus.WARMING},
        )
        assert updated.warmup_started_at is not None
        assert updated.daily_limit == settings.warmup_initial_daily_limit

    async def test_an_existing_ramp_is_not_re_anchored(self, session) -> None:
        org = await make_org(session)
        started = NOW - timedelta(days=3)
        acct = account(
            org, warmup_status=WarmupStatus.PAUSED, warmup_started_at=started
        )
        await persist(session, acct)
        updated = await svc.update_account(
            session,
            org.id,
            acct.id,
            changes={"warmup_status": WarmupStatus.WARMING},
        )
        assert updated.warmup_started_at == started


@pytest.mark.anyio
class TestDeleteAccount:
    async def test_deletion_is_soft_and_deactivates(self, session) -> None:
        org = await make_org(session)
        acct = account(org)
        await persist(session, acct)
        await svc.delete_account(session, org.id, acct.id)
        assert acct.deleted_at is not None
        assert acct.is_active is False

    async def test_a_deleted_account_leaves_the_rotation(self, session) -> None:
        org = await make_org(session)
        acct = account(org)
        await persist(session, acct)
        await svc.delete_account(session, org.id, acct.id)
        assert await svc.pick_sender(session, org.id, now=NOW) is None


# --------------------------------------------------------------------------- #
# Warm-up sweep
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
class TestWarmupSweep:
    async def test_it_finds_only_accounts_behind_their_ramp(self, session) -> None:
        org = await make_org(session)
        behind = account(
            org,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=3),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        current = account(
            org,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW,
            daily_limit=settings.warmup_initial_daily_limit,
        )
        await persist(session, behind, current)
        due = await svc.due_for_warmup(session, now=NOW)
        assert [a.id for a in due] == [behind.id]

    async def test_it_ignores_accounts_that_are_not_warming(self, session) -> None:
        org = await make_org(session)
        await persist(
            session,
            account(
                org,
                warmup_status=WarmupStatus.PAUSED,
                warmup_started_at=NOW - timedelta(days=30),
                daily_limit=1,
            ),
        )
        assert await svc.due_for_warmup(session, now=NOW) == []

    async def test_it_ignores_deleted_and_inactive_accounts(self, session) -> None:
        org = await make_org(session)
        common = dict(
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=3),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        await persist(
            session,
            account(org, deleted_at=NOW, **common),
            account(org, is_active=False, **common),
        )
        assert await svc.due_for_warmup(session, now=NOW) == []

    async def test_the_sweep_crosses_tenants(self, session) -> None:
        # Warm-up is infrastructure, not a tenant-facing query: one nightly
        # pass advances every org's ramps.
        first = await make_org(session, "First")
        second = await make_org(session, "Second")
        common = dict(
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=2),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        await persist(session, account(first, **common), account(second, **common))
        assert await svc.sweep_warmup(session, now=NOW) == 2

    async def test_the_sweep_persists_the_new_allowance(self, session) -> None:
        org = await make_org(session)
        acct = account(
            org,
            warmup_status=WarmupStatus.WARMING,
            warmup_started_at=NOW - timedelta(days=2),
            daily_limit=settings.warmup_initial_daily_limit,
        )
        await persist(session, acct)
        await svc.sweep_warmup(session, now=NOW)
        session.expunge_all()

        reloaded = await svc.get_account(session, org.id, acct.id)
        assert reloaded.daily_limit == (
            settings.warmup_initial_daily_limit + 2 * settings.warmup_daily_increment
        )

    async def test_a_second_sweep_is_a_no_op(self, session) -> None:
        org = await make_org(session)
        await persist(
            session,
            account(
                org,
                warmup_status=WarmupStatus.WARMING,
                warmup_started_at=NOW - timedelta(days=2),
                daily_limit=settings.warmup_initial_daily_limit,
            ),
        )
        assert await svc.sweep_warmup(session, now=NOW) == 1
        assert await svc.sweep_warmup(session, now=NOW) == 0

    async def test_nothing_to_do_sweeps_zero(self, session) -> None:
        await make_org(session)
        assert await svc.sweep_warmup(session, now=NOW) == 0


def test_today_defaults_to_the_current_utc_date() -> None:
    assert svc._today() == datetime.now(UTC).date()
    assert isinstance(svc._today(NOW), date)


@pytest.fixture
def org_stub() -> Organization:
    """An unsaved organization, for the pure-function tests that need an id."""
    return Organization(id=uuid.uuid4(), name="Acme", slug="acme")
