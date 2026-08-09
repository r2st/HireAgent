"""The outreach send loop: guards, rendering, delivery, and what each failure does.

Most of this file is about the asymmetry the dispatcher is built around. Sending
late costs a little; sending wrong costs the domain's reputation and the
candidate's trust. So the interesting cases here are not the happy path — they
are the ones where something has changed underneath a running sequence: consent
withdrawn last week, a mailbox blocked this morning, a template edited into a
broken state, a step deleted out from under an enrollment that still points at
it.

The retry tests deserve particular attention. A message row is written before
the send and reused after a transient failure, which is what stops a crash
between "row written" and "mail sent" from becoming two identical emails.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import settings
from app.core.security import blind_index
from app.integrations import email_gateway
from app.models.candidate import Candidate, CandidateConsent
from app.models.enums import (
    ConsentStatus,
    ConsentType,
    EmailProvider,
    EnrollmentStatus,
    MessageStatus,
    OutreachChannel,
    WarmupStatus,
)
from app.models.job import Job
from app.models.organization import Organization
from app.models.outreach import EmailAccount, MessageTemplate, OutreachMessage
from app.services import outreach_service
from app.workers import dispatcher as svc

__all__ = ["FakeTransport", "grant", "make_account", "make_candidate", "make_org"]

NOW = datetime(2027, 3, 8, 12, 0, tzinfo=UTC)  # A Monday, inside the send window.


# --------------------------------------------------------------------------- #
# A transport that answers from a script
# --------------------------------------------------------------------------- #
class FakeTransport(email_gateway.EmailTransport):
    """Stands in for a vendor. Records what it was handed, replies from a queue.

    A queued ``Exception`` is raised rather than returned, which is how a
    transport blowing up mid-pass is simulated.
    """

    name = "smtp"

    def __init__(self, *results: object) -> None:
        self.results = list(results) or [
            email_gateway.SendResult(ok=True, provider_message_id="msg-1")
        ]
        self.sent: list[email_gateway.OutboundEmail] = []
        self.credentials: list[email_gateway.EmailCredentials] = []

    @property
    def is_configured(self) -> bool:
        return True

    async def send(self, credentials, message):
        self.sent.append(message)
        self.credentials.append(credentials)
        # The last entry repeats, so a test need not count the calls a loop makes.
        nxt = self.results[0] if len(self.results) == 1 else self.results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def transport() -> FakeTransport:
    fake = FakeTransport()
    email_gateway.set_transports({"smtp": fake, "ses": fake})
    return fake


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
async def make_org(session, name: str = "Acme") -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


async def make_candidate(session, org, **fields) -> Candidate:
    email = fields.pop("email", f"cand-{uuid.uuid4().hex[:8]}@example.test")
    candidate = Candidate(
        organization_id=org.id,
        full_name=fields.pop("full_name", "Grace Hopper"),
        email=email,
        email_index=blind_index(email) if email else f"none-{uuid.uuid4().hex[:8]}",
        **fields,
    )
    session.add(candidate)
    await session.commit()
    return candidate


async def grant(
    session, org, candidate, consent_type=ConsentType.EMAIL_COMMUNICATION, **fields
) -> CandidateConsent:
    consent = CandidateConsent(
        organization_id=org.id,
        candidate_id=candidate.id,
        consent_type=consent_type,
        status=fields.pop("status", ConsentStatus.GRANTED),
        granted_at=fields.pop("granted_at", NOW),
    )
    session.add(consent)
    await session.commit()
    return consent


async def make_account(session, org, **fields) -> EmailAccount:
    defaults = dict(
        organization_id=org.id,
        email=f"sender-{uuid.uuid4().hex[:8]}@acme.test",
        display_name="Acme Talent",
        provider=EmailProvider.SMTP,
        smtp_host="smtp.acme.test",
        warmup_status=WarmupStatus.READY,
        daily_limit=50,
        sent_today=0,
        reputation_score=100,
        is_active=True,
    )
    defaults.update(fields)
    account = EmailAccount(**defaults)
    session.add(account)
    await session.commit()
    return account


@pytest.fixture
async def world(session, transport, monkeypatch) -> dict:
    """One org, one sender, a two-step sequence, one enrolled candidate."""
    monkeypatch.setattr(settings, "outreach_sending_enabled", True)
    org = await make_org(session)
    account = await make_account(session, org)
    candidate = await make_candidate(session, org)
    await grant(session, org, candidate)

    sequence = await outreach_service.create_sequence(
        session, org.id, name="Backend outreach"
    )
    await outreach_service.add_step(
        session, org.id, sequence.id, body_override="Hello there", subject_override="Hi"
    )
    await outreach_service.add_step(
        session,
        org.id,
        sequence.id,
        body_override="Following up",
        subject_override="Re: Hi",
        delay_days=2,
    )
    await outreach_service.activate(session, org.id, sequence.id, now=NOW)
    report = await outreach_service.enroll(
        session, org.id, sequence.id, [candidate.id], now=NOW
    )

    await session.refresh(sequence)
    return {
        "org": org,
        "account": account,
        "candidate": candidate,
        "sequence": sequence,
        "enrollment": report.enrolled[0],
        "transport": transport,
    }


async def messages_for(session, enrollment) -> list[OutreachMessage]:
    from sqlalchemy import select

    result = await session.execute(
        select(OutreachMessage)
        .where(OutreachMessage.enrollment_id == enrollment.id)
        .order_by(OutreachMessage.created_at.asc())
    )
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
class TestSuccessfulSend:
    async def test_the_step_is_sent_and_recorded(self, session, world) -> None:
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "sent"

        message = (await messages_for(session, world["enrollment"]))[0]
        assert message.status == MessageStatus.SENT
        assert message.sent_at == NOW
        assert message.provider_message_id == "msg-1"
        assert message.error is None
        assert message.email_account_id == world["account"].id

    async def test_the_enrollment_moves_to_the_next_step(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        enrollment = world["enrollment"]
        assert enrollment.current_step == 1
        assert enrollment.status == EnrollmentStatus.ACTIVE
        assert enrollment.next_send_at == NOW + timedelta(days=2)

    async def test_the_last_step_completes_the_enrollment(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        later = NOW + timedelta(days=2)
        await svc.dispatch_one(session, world["enrollment"], now=later)

        assert world["enrollment"].status == EnrollmentStatus.COMPLETED
        assert world["enrollment"].next_send_at is None

    async def test_the_recipient_and_sender_are_addressed_properly(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        outbound = world["transport"].sent[0]
        assert outbound.to_email == world["candidate"].email
        assert outbound.to_name == "Grace Hopper"
        assert outbound.subject == "Hi"
        assert outbound.text_body == "Hello there"

    async def test_the_senders_counters_and_reputation_move(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        account = world["account"]
        assert account.sent_today == 1
        assert account.total_sent == 1
        assert account.last_used_at == NOW

    async def test_the_sequence_counter_moves(self, session, world) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        sequence = await outreach_service.get_sequence(
            session, world["org"].id, world["sequence"].id
        )
        assert sequence.stats_json["sent"] == 1

    async def test_every_message_carries_a_one_click_opt_out(
        self, session, world
    ) -> None:
        """A footer link only reaches the candidates who scroll; a header always does."""
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        message = (await messages_for(session, world["enrollment"]))[0]
        outbound = world["transport"].sent[0]

        assert message.tracking_token
        assert outbound.headers["List-Unsubscribe"] == (
            f"<{outreach_service.one_click_unsubscribe_url(message.tracking_token)}>"
        )
        assert outbound.headers["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"

    async def test_the_opt_out_header_points_at_the_api_not_the_frontend(
        self, session, world
    ) -> None:
        """One-click is a POST from the mail client; an SPA cannot answer it."""
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        header = world["transport"].sent[0].headers["List-Unsubscribe"]

        assert settings.api_base_url in header
        assert settings.api_v1_prefix in header

    async def test_the_body_can_carry_a_human_opt_out_link(
        self, session, world
    ) -> None:
        """CAN-SPAM asks for a visible opt-out, which the header is not."""
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Hello. Opt out: {{unsubscribe_url}}"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)

        message = (await messages_for(session, world["enrollment"]))[0]
        page = outreach_service.unsubscribe_page_url(message.tracking_token)
        assert page is not None
        assert world["transport"].sent[0].text_body == f"Hello. Opt out: {page}"
        # The reader's link goes to the page, not to the machine endpoint.
        assert page.startswith(settings.public_base_url)

    async def test_each_message_gets_its_own_tracking_token(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        await svc.dispatch_one(
            session, world["enrollment"], now=NOW + timedelta(days=2)
        )
        tokens = {m.tracking_token for m in await messages_for(session, world["enrollment"])}
        assert len(tokens) == 2


class TestRetryKeepsTheSameOptOutLink:
    async def test_a_retry_reuses_the_rows_token(
        self, session, world, transport
    ) -> None:
        """Otherwise the body on record cites a link the candidate never got."""
        transport.results = [
            email_gateway.SendResult(ok=False, error="timeout", retryable=True),
            email_gateway.SendResult(ok=True, provider_message_id="msg-2"),
        ]
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Opt out: {{unsubscribe_url}}"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        await svc.dispatch_one(
            session, world["enrollment"], now=NOW + svc.RETRY_BACKOFF
        )

        rows = await messages_for(session, world["enrollment"])
        assert len(rows) == 1
        sent_bodies = {m.text_body for m in transport.sent}
        assert len(sent_bodies) == 1
        assert rows[0].body == sent_bodies.pop()

    async def test_a_retry_re_renders_so_the_record_matches_what_was_sent(
        self, session, world, transport
    ) -> None:
        """A template fixed between attempts must not leave a stale row behind."""
        transport.results = [
            email_gateway.SendResult(ok=False, error="timeout", retryable=True),
            email_gateway.SendResult(ok=True, provider_message_id="msg-2"),
        ]
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Corrected copy"},
        )
        await svc.dispatch_one(
            session, world["enrollment"], now=NOW + svc.RETRY_BACKOFF
        )

        rows = await messages_for(session, world["enrollment"])
        assert rows[0].body == "Corrected copy"
        assert transport.sent[-1].text_body == "Corrected copy"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class TestRendering:
    async def _template_step(self, session, world, **template_fields) -> None:
        """Replace the sequence's steps with a single template-backed one."""
        org, sequence = world["org"], world["sequence"]
        for step in list(sequence.steps):
            await outreach_service.delete_step(session, org.id, step.id)

        template = MessageTemplate(
            organization_id=org.id,
            name="Intro",
            channel=OutreachChannel.EMAIL,
            **template_fields,
        )
        session.add(template)
        await session.commit()
        await outreach_service.add_step(
            session, org.id, sequence.id, template_id=template.id
        )
        await session.refresh(sequence)

    async def test_a_template_supplies_the_content(self, session, world) -> None:
        await self._template_step(
            session, world, subject="Hello {{first_name}}", body="From {{sender_name}}"
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)

        outbound = world["transport"].sent[0]
        assert outbound.subject == "Hello Grace"
        assert outbound.text_body == "From Acme Talent"

    async def test_the_html_part_comes_from_the_template(
        self, session, world
    ) -> None:
        await self._template_step(
            session,
            world,
            subject="Hi",
            body="plain",
            body_html="<p>{{candidate_name}}</p>",
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["transport"].sent[0].html_body == "<p>Grace Hopper</p>"

    async def test_an_inline_override_wins_over_the_template(
        self, session, world
    ) -> None:
        """A step overrides a shared template so one sequence can vary it."""
        await self._template_step(
            session, world, subject="Template subject", body="Template body"
        )
        step = world["sequence"].steps[0]
        await outreach_service.update_step(
            session,
            world["org"].id,
            step.id,
            changes={"subject_override": "Step subject", "body_override": "Step body"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)

        outbound = world["transport"].sent[0]
        assert outbound.subject == "Step subject"
        assert outbound.text_body == "Step body"

    async def test_an_override_body_suppresses_the_templates_html(
        self, session, world
    ) -> None:
        """Otherwise the envelope would carry two different messages."""
        await self._template_step(
            session, world, subject="Hi", body="plain", body_html="<p>template</p>"
        )
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Step body"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["transport"].sent[0].html_body is None

    async def test_job_variables_are_available_when_the_sequence_has_one(
        self, session, world
    ) -> None:
        job = Job(
            organization_id=world["org"].id,
            title="Staff Engineer",
            slug=f"se-{uuid.uuid4().hex[:6]}",
        )
        session.add(job)
        await session.commit()
        await outreach_service.update_sequence(
            session, world["org"].id, world["sequence"].id, changes={"job_id": job.id}
        )
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "About the {{job_title}} role"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["transport"].sent[0].text_body == "About the Staff Engineer role"

    async def test_an_unresolved_variable_stops_the_send(
        self, session, world
    ) -> None:
        """Half-filled outreach is worse than none: it announces the bulk mailer."""
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Hi {{nickname}}"},
        )
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "unrendered"
        assert world["transport"].sent == []
        assert await messages_for(session, world["enrollment"]) == []

    async def test_an_unrendered_step_pauses_so_it_can_be_fixed_and_resumed(
        self, session, world
    ) -> None:
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Hi {{nickname}}"},
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW)

        enrollment = world["enrollment"]
        assert enrollment.status == EnrollmentStatus.PAUSED
        assert enrollment.paused_reason is not None
        assert "nickname" in enrollment.paused_reason
        assert enrollment.current_step == 0

    async def test_a_fallback_keeps_an_optional_variable_from_stopping_it(
        self, session, world
    ) -> None:
        await outreach_service.update_step(
            session,
            world["org"].id,
            world["sequence"].steps[0].id,
            changes={"body_override": "Hi {{nickname|there}}"},
        )
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "sent"
        assert world["transport"].sent[0].text_body == "Hi there"


# --------------------------------------------------------------------------- #
# Guards that must hold at send time, not merely at enrolment
# --------------------------------------------------------------------------- #
class TestSendTimeGuards:
    async def test_consent_withdrawn_since_enrolment_stops_the_sequence(
        self, session, world
    ) -> None:
        """A sequence runs for weeks; the check made at enrolment is stale."""
        await grant(
            session,
            world["org"],
            world["candidate"],
            status=ConsentStatus.WITHDRAWN,
            granted_at=NOW + timedelta(hours=1),
        )
        result = await svc.dispatch_one(
            session, world["enrollment"], now=NOW + timedelta(hours=2)
        )

        assert result.outcome == "no_consent"
        assert world["enrollment"].status == EnrollmentStatus.UNSUBSCRIBED
        assert world["transport"].sent == []

    async def test_an_unsubscribed_enrollment_is_not_resumable(
        self, session, world
    ) -> None:
        """Ending rather than pausing is what keeps this one from being undone."""
        await grant(
            session,
            world["org"],
            world["candidate"],
            status=ConsentStatus.WITHDRAWN,
            granted_at=NOW + timedelta(hours=1),
        )
        await svc.dispatch_one(session, world["enrollment"], now=NOW + timedelta(hours=2))

        from app.core.errors import ConflictError

        with pytest.raises(ConflictError):
            await outreach_service.resume_enrollment(
                session, world["org"].id, world["enrollment"].id, now=NOW
            )

    async def test_a_blacklisting_since_enrolment_stops_the_sequence(
        self, session, world
    ) -> None:
        world["candidate"].is_blacklisted = True
        await session.commit()

        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "blacklisted"
        assert world["enrollment"].status == EnrollmentStatus.UNSUBSCRIBED
        assert world["transport"].sent == []

    async def test_a_candidate_who_lost_their_address_pauses(
        self, session, world
    ) -> None:
        world["candidate"].email = "  "
        await session.commit()

        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "no_address"
        assert world["enrollment"].status == EnrollmentStatus.PAUSED

    async def test_a_step_deleted_underneath_the_enrollment_completes_it(
        self, session, world
    ) -> None:
        """An enrollment can outlive an edit that shortened its sequence."""
        for step in list(world["sequence"].steps):
            await outreach_service.delete_step(session, world["org"].id, step.id)

        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "nothing_to_send"
        assert world["enrollment"].status == EnrollmentStatus.COMPLETED

    async def test_a_channel_with_no_transport_is_stepped_over(
        self, session, world
    ) -> None:
        """A mixed sequence keeps delivering its email steps."""
        await grant(
            session,
            world["org"],
            world["candidate"],
            ConsentType.WHATSAPP_COMMUNICATION,
        )
        step = world["sequence"].steps[0]
        step.channel = OutreachChannel.WHATSAPP
        await session.commit()

        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "unsupported_channel"
        assert world["enrollment"].current_step == 1
        assert world["transport"].sent == []


# --------------------------------------------------------------------------- #
# The kill switch
# --------------------------------------------------------------------------- #
class TestSendingDisabled:
    async def test_nothing_leaves_the_box(self, session, world, monkeypatch) -> None:
        monkeypatch.setattr(settings, "outreach_sending_enabled", False)
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "suppressed"
        assert world["transport"].sent == []

    async def test_the_rendered_message_is_still_recorded(
        self, session, world, monkeypatch
    ) -> None:
        """Which is what makes a staging environment worth having."""
        monkeypatch.setattr(settings, "outreach_sending_enabled", False)
        await svc.dispatch_one(session, world["enrollment"], now=NOW)

        message = (await messages_for(session, world["enrollment"]))[0]
        assert message.status == MessageStatus.QUEUED
        assert message.body == "Hello there"
        assert message.error == "Outreach sending is disabled"

    async def test_the_pipeline_keeps_moving(
        self, session, world, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "outreach_sending_enabled", False)
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["enrollment"].current_step == 1

    async def test_no_reputation_is_spent(self, session, world, monkeypatch) -> None:
        monkeypatch.setattr(settings, "outreach_sending_enabled", False)
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["account"].sent_today == 0
        assert world["account"].total_sent == 0


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("bouncing")
class TestBounce:
    @pytest.fixture
    def bouncing(self, transport) -> FakeTransport:
        transport.results = [
            email_gateway.SendResult(
                ok=False, error="550 no such user", retryable=False, bounced=True
            )
        ]
        return transport

    async def test_a_bounce_ends_the_enrollment(
        self, session, world
    ) -> None:
        """Every later step would bounce too, spending reputation each time."""
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "bounced"
        assert world["enrollment"].status == EnrollmentStatus.BOUNCED
        assert world["enrollment"].next_send_at is None

    async def test_the_message_records_the_bounce(
        self, session, world
    ) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        message = (await messages_for(session, world["enrollment"]))[0]
        assert message.status == MessageStatus.BOUNCED
        assert message.bounced_at == NOW
        assert message.error == "550 no such user"

    async def test_the_sender_is_penalised(self, session, world) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert world["account"].bounce_count == 1
        assert float(world["account"].reputation_score) < 100


@pytest.mark.usefixtures("flaky")
class TestRetry:
    @pytest.fixture
    def flaky(self, transport) -> FakeTransport:
        transport.results = [
            email_gateway.SendResult(ok=False, error="timeout", retryable=True)
        ]
        return transport

    async def test_a_transient_failure_reschedules_rather_than_fails(
        self, session, world
    ) -> None:
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "retrying"
        assert world["enrollment"].status == EnrollmentStatus.ACTIVE
        assert world["enrollment"].current_step == 0
        assert world["enrollment"].next_send_at == NOW + svc.RETRY_BACKOFF

    async def test_the_queued_row_is_reused_rather_than_duplicated(
        self, session, world
    ) -> None:
        """Otherwise every retry leaves another row claiming a message went out."""
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        await svc.dispatch_one(
            session, world["enrollment"], now=NOW + svc.RETRY_BACKOFF
        )

        rows = await messages_for(session, world["enrollment"])
        assert len(rows) == 1
        assert rows[0].retry_count == 2

    async def test_attempts_are_bounded(self, session, world) -> None:
        stamp = NOW
        for _ in range(settings.outreach_max_send_attempts):
            result = await svc.dispatch_one(session, world["enrollment"], now=stamp)
            stamp += svc.RETRY_BACKOFF

        assert result.outcome == "failed"
        rows = await messages_for(session, world["enrollment"])
        assert len(rows) == 1
        assert rows[0].status == MessageStatus.FAILED

    async def test_an_exhausted_message_pauses_the_enrollment(
        self, session, world
    ) -> None:
        """Resumable, because a dead mailbox is our problem, not the candidate's."""
        stamp = NOW
        for _ in range(settings.outreach_max_send_attempts):
            await svc.dispatch_one(session, world["enrollment"], now=stamp)
            stamp += svc.RETRY_BACKOFF

        assert world["enrollment"].status == EnrollmentStatus.PAUSED
        assert world["enrollment"].paused_reason == "timeout"

    async def test_a_transport_that_raises_is_treated_as_transient(
        self, session, world, transport
    ) -> None:
        """The next enrollment in the queue is unrelated to this one."""
        transport.results = [RuntimeError("connection reset")]
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "retrying"
        assert world["enrollment"].status == EnrollmentStatus.ACTIVE

    async def test_a_permanent_failure_pauses_immediately(
        self, session, world, transport
    ) -> None:
        transport.results = [
            email_gateway.SendResult(
                ok=False, error="sender rejected", retryable=False
            )
        ]
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)

        assert result.outcome == "failed"
        assert world["enrollment"].status == EnrollmentStatus.PAUSED
        assert world["enrollment"].paused_reason == "sender rejected"

    async def test_a_later_step_writes_its_own_message(
        self, session, world, transport
    ) -> None:
        """A SENT row must never be picked up as the pending one for a later step."""
        transport.results = [email_gateway.SendResult(ok=True, provider_message_id="ok")]
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        await svc.dispatch_one(
            session, world["enrollment"], now=NOW + timedelta(days=2)
        )
        rows = await messages_for(session, world["enrollment"])
        assert len(rows) == 2
        assert {r.step_id for r in rows} == {s.id for s in world["sequence"].steps}


class TestNoSender:
    async def test_a_capped_org_waits_instead_of_failing(
        self, session, world
    ) -> None:
        """Routine at the end of a busy day; it must not end the enrollment."""
        world["account"].sent_today = world["account"].daily_limit
        world["account"].sent_today_date = NOW
        await session.commit()

        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "no_sender"
        assert world["enrollment"].status == EnrollmentStatus.ACTIVE
        assert world["enrollment"].next_send_at == NOW + svc.CAP_BACKOFF
        assert await messages_for(session, world["enrollment"]) == []

    async def test_a_blocked_mailbox_is_not_used(self, session, world) -> None:
        world["account"].warmup_status = WarmupStatus.BLOCKED
        await session.commit()
        result = await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert result.outcome == "no_sender"

    async def test_the_sequences_own_pool_is_preferred(
        self, session, world
    ) -> None:
        chosen = await make_account(session, world["org"], display_name="Chosen")
        world["sequence"].sender_account_ids = [str(chosen.id)]
        await session.commit()

        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        message = (await messages_for(session, world["enrollment"]))[0]
        assert message.email_account_id == chosen.id


# --------------------------------------------------------------------------- #
# Daily caps
# --------------------------------------------------------------------------- #
class TestSentToday:
    async def test_it_counts_this_sequences_messages(self, session, world) -> None:
        assert await svc.sent_today(session, world["sequence"], now=NOW) == 0
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        assert await svc.sent_today(session, world["sequence"], now=NOW) == 1

    async def test_yesterdays_messages_do_not_count(self, session, world) -> None:
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        tomorrow = NOW + timedelta(days=1)
        assert await svc.sent_today(session, world["sequence"], now=tomorrow) == 0

    async def test_the_day_boundary_follows_the_sequences_timezone(
        self, session, world
    ) -> None:
        """A cap should not roll over in the middle of the recruiter's afternoon."""
        sequence = world["sequence"]
        sequence.timezone = "Asia/Kolkata"
        await session.commit()

        # 19:00 UTC on the 8th is already 00:30 on the 9th in Kolkata.
        start = svc._local_day_start(sequence, datetime(2027, 3, 8, 19, 0, tzinfo=UTC))
        assert start == datetime(2027, 3, 8, 18, 30, tzinfo=UTC)


class TestDailyCap:
    async def test_a_capped_sequence_stops_after_its_allowance(
        self, session, world
    ) -> None:
        second = await make_candidate(session, world["org"])
        await grant(session, world["org"], second)
        await outreach_service.enroll(
            session, world["org"].id, world["sequence"].id, [second.id], now=NOW
        )
        await outreach_service.update_sequence(
            session, world["org"].id, world["sequence"].id, changes={"daily_cap": 1}
        )

        report = await svc.run_once(session, now=NOW)
        assert report.considered == 2
        assert report.counts() == {"sent": 1, "capped": 1}

    async def test_a_capped_enrollment_is_rescheduled_not_lost(
        self, session, world
    ) -> None:
        await outreach_service.update_sequence(
            session, world["org"].id, world["sequence"].id, changes={"daily_cap": 1}
        )
        # Consume the day's allowance with a message from an earlier pass.
        await svc.dispatch_one(session, world["enrollment"], now=NOW)
        world["enrollment"].current_step = 0
        world["enrollment"].status = EnrollmentStatus.ACTIVE
        world["enrollment"].next_send_at = NOW
        await session.commit()

        report = await svc.run_once(session, now=NOW)
        assert report.counts() == {"capped": 1}
        assert world["enrollment"].next_send_at == NOW + svc.CAP_BACKOFF

    async def test_an_uncapped_sequence_is_not_counted(
        self, session, world
    ) -> None:
        second = await make_candidate(session, world["org"])
        await grant(session, world["org"], second)
        await outreach_service.enroll(
            session, world["org"].id, world["sequence"].id, [second.id], now=NOW
        )
        report = await svc.run_once(session, now=NOW)
        assert report.sent == 2


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
class TestRunOnce:
    @pytest.mark.usefixtures("world")
    async def test_a_due_enrollment_is_dispatched(self, session) -> None:
        report = await svc.run_once(session, now=NOW)
        assert report.considered == 1
        assert report.sent == 1

    async def test_nothing_due_does_nothing(self, session, world) -> None:
        report = await svc.run_once(session, now=NOW - timedelta(days=1))
        assert report.considered == 0
        assert world["transport"].sent == []

    async def test_a_paused_sequence_sends_nothing(self, session, world) -> None:
        await outreach_service.pause(session, world["org"].id, world["sequence"].id)
        report = await svc.run_once(session, now=NOW)
        assert report.considered == 0

    async def test_one_bad_enrollment_does_not_stop_the_pass(
        self, session, world
    ) -> None:
        """A blacklisted candidate ahead of a good one must not shadow it.

        Blacklisted after enrolling, because enrolment refuses them outright —
        this is the state that only arises while a sequence is running.
        """
        second = await make_candidate(session, world["org"])
        await grant(session, world["org"], second)
        await outreach_service.enroll(
            session,
            world["org"].id,
            world["sequence"].id,
            [second.id],
            now=NOW - timedelta(hours=1),
        )
        second.is_blacklisted = True
        await session.commit()

        report = await svc.run_once(session, now=NOW)
        assert report.considered == 2
        assert report.counts() == {"blacklisted": 1, "sent": 1}

    @pytest.mark.usefixtures("world")
    async def test_the_pass_can_be_narrowed_to_one_org(
        self, session
    ) -> None:
        other = await make_org(session, "Rival")
        report = await svc.run_once(session, now=NOW, organization_id=other.id)
        assert report.considered == 0

    async def test_the_limit_is_honoured(self, session, world) -> None:
        second = await make_candidate(session, world["org"])
        await grant(session, world["org"], second)
        await outreach_service.enroll(
            session, world["org"].id, world["sequence"].id, [second.id], now=NOW
        )
        report = await svc.run_once(session, now=NOW, limit=1)
        assert report.considered == 1

    @pytest.mark.usefixtures("world")
    async def test_the_report_summarises_the_pass(self, session) -> None:
        report = await svc.run_once(session, now=NOW)
        assert report.to_dict() == {
            "considered": 1,
            "sent": 1,
            "outcomes": {"sent": 1},
        }
