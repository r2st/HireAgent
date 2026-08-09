"""Candidate opt-out: the link in the message, and what clicking it stops.

The unsubscribe path is the one part of outreach a candidate drives, and the
only one where getting it wrong is both a legal problem and the fastest way to
lose a sending domain. Three properties carry most of this file.

**The opt-out is per candidate, not per campaign.** A candidate enrolled in two
sequences who unsubscribes from one and keeps hearing from the other has been
ignored, whatever the database says.

**GET does not unsubscribe.** Mail clients and security scanners prefetch
links. A mutating GET opts out people who never clicked, and — unlike a missed
opt-out, which the candidate notices — that failure is silent until a recruiter
asks where their pipeline went.

**The header link and the reader's link are different URLs.** RFC 8058
one-click is a POST from the mail client with no browser involved, so the
header has to point at the API; the sentence in the body is for a human and
points at the frontend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.errors import NotFoundError
from app.models.candidate import CandidateConsent
from app.models.enums import ConsentStatus, ConsentType, EnrollmentStatus
from app.models.outreach import OutreachMessage
from app.schemas.outreach import mask_email
from app.services import candidate_service, outreach_service
from app.workers import dispatcher

# Reused rather than re-declared: these build the same world the dispatcher
# tests do, and a second copy would drift from it.
from tests.test_dispatcher import (
    FakeTransport,
    make_account,
    make_candidate,
    make_org,
)
from tests.test_dispatcher import grant as _grant_at_dispatcher_now

# A Monday inside the send window, and deliberately in the *past*, unlike the
# rest of the suite. The HTTP endpoints have no injectable clock — a one-click
# POST arrives from a mail client, which has no opinion about what time the
# test thinks it is — so they stamp the real one. Consent is resolved by taking
# the latest record, and a fixture that granted consent in 2027 would leave
# every opt-out looking older than the grant it was meant to supersede.
NOW = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)


async def grant(session, org, candidate, consent_type=ConsentType.EMAIL_COMMUNICATION):
    """Grant consent on *this* module's clock.

    The borrowed helper defaults ``granted_at`` to the dispatcher suite's own
    NOW, which is a different year. Consent is resolved by taking the latest
    record, so inheriting that default would silently date every grant after
    the withdrawal meant to supersede it and make these tests assert nothing.
    """
    return await _grant_at_dispatcher_now(
        session, org, candidate, consent_type, granted_at=NOW
    )


@pytest.fixture
def transport(monkeypatch) -> FakeTransport:
    from app.integrations import email_gateway

    monkeypatch.setattr(settings, "outreach_sending_enabled", True)
    fake = FakeTransport()
    email_gateway.set_transports({"smtp": fake, "ses": fake})
    return fake


async def build_sequence(session, org, *, name: str = "Backend outreach"):
    """Two steps, so an enrollment is still live after the first message.

    A single-step sequence completes the moment it sends, which would make
    "unsubscribing stops the enrollment" vacuously true.
    """
    sequence = await outreach_service.create_sequence(session, org.id, name=name)
    await outreach_service.add_step(
        session, org.id, sequence.id, subject_override="Hi", body_override="Hello there"
    )
    await outreach_service.add_step(
        session,
        org.id,
        sequence.id,
        subject_override="Re: Hi",
        body_override="Following up",
        delay_days=2,
    )
    await outreach_service.activate(session, org.id, sequence.id, now=NOW)
    return sequence


@pytest.fixture
async def mailed(session, transport) -> dict:
    """One candidate who has been sent one message, and so holds one token."""
    org = await make_org(session)
    await make_account(session, org)
    candidate = await make_candidate(
        session, org, email="grace.hopper@example.test", full_name="Grace Hopper"
    )
    await grant(session, org, candidate)
    sequence = await build_sequence(session, org)
    report = await outreach_service.enroll(
        session, org.id, sequence.id, [candidate.id], now=NOW
    )
    enrollment = report.enrolled[0]

    result = await dispatcher.dispatch_one(session, enrollment, now=NOW)
    assert result.outcome == "sent"
    message = await session.get(OutreachMessage, result.message_id)

    return {
        "org": org,
        "candidate": candidate,
        "sequence": sequence,
        "enrollment": enrollment,
        "message": message,
        "token": message.tracking_token,
        "transport": transport,
    }


async def consent_rows(session, candidate_id) -> list[CandidateConsent]:
    result = await session.execute(
        select(CandidateConsent)
        .where(CandidateConsent.candidate_id == candidate_id)
        .order_by(CandidateConsent.granted_at.asc())
    )
    return list(result.scalars().all())


async def withdrawals(session, candidate_id) -> list[CandidateConsent]:
    """Selected by status rather than by position.

    The grant and the withdrawal deliberately share a ``granted_at`` in most of
    these tests, so "the last row" is not a well-defined thing to ask for.
    """
    rows = await consent_rows(session, candidate_id)
    return [r for r in rows if r.status == ConsentStatus.WITHDRAWN]


# --------------------------------------------------------------------------- #
# The two links
# --------------------------------------------------------------------------- #
class TestLinks:
    def test_the_readers_link_points_at_the_frontend(self) -> None:
        url = outreach_service.unsubscribe_page_url("abc123")
        assert url == f"{settings.public_base_url}/outreach/unsubscribe/abc123"

    def test_the_mail_clients_link_points_at_the_api(self) -> None:
        """An SPA cannot answer the POST that one-click sends."""
        url = outreach_service.one_click_unsubscribe_url("abc123")
        assert url == (
            f"{settings.api_base_url}{settings.api_v1_prefix}"
            "/outreach/unsubscribe/abc123"
        )

    def test_they_are_not_the_same_url(self) -> None:
        assert outreach_service.unsubscribe_page_url(
            "abc123"
        ) != outreach_service.one_click_unsubscribe_url("abc123")

    @pytest.mark.parametrize("token", [None, ""])
    def test_no_token_means_no_link(self, token) -> None:
        assert outreach_service.unsubscribe_page_url(token) is None
        assert outreach_service.one_click_unsubscribe_url(token) is None

    def test_a_trailing_slash_on_the_base_does_not_double_up(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "api_base_url", "https://api.acme.test/")
        monkeypatch.setattr(settings, "public_base_url", "https://acme.test/")
        assert "//outreach" not in str(
            outreach_service.unsubscribe_page_url("t")
        )
        assert "test//api" not in str(
            outreach_service.one_click_unsubscribe_url("t")
        )


# --------------------------------------------------------------------------- #
# Resolving the token
# --------------------------------------------------------------------------- #
class TestTokenLookup:
    async def test_a_real_token_resolves_to_its_message(
        self, session, mailed
    ) -> None:
        found = await outreach_service.message_by_tracking_token(
            session, mailed["token"]
        )
        assert found.id == mailed["message"].id

    async def test_the_lookup_crosses_no_tenant_boundary_of_its_own(
        self, session, mailed
    ) -> None:
        """The holder is a candidate with a link; the message supplies the tenant."""
        found = await outreach_service.message_by_tracking_token(
            session, mailed["token"]
        )
        assert found.organization_id == mailed["org"].id

    @pytest.mark.parametrize("token", ["", "not-a-real-token"])
    async def test_a_bad_token_is_not_found(self, session, token) -> None:
        with pytest.raises(NotFoundError):
            await outreach_service.message_by_tracking_token(session, token)

    async def test_a_deleted_message_is_not_found(self, session, mailed) -> None:
        mailed["message"].deleted_at = NOW
        await session.commit()
        with pytest.raises(NotFoundError):
            await outreach_service.message_by_tracking_token(session, mailed["token"])


# --------------------------------------------------------------------------- #
# What unsubscribing does
# --------------------------------------------------------------------------- #
class TestUnsubscribeByToken:
    async def test_email_consent_is_withdrawn(self, session, mailed) -> None:
        await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=NOW + timedelta(hours=1)
        )
        assert not await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.EMAIL_COMMUNICATION,
        )

    async def test_the_withdrawal_is_appended_not_edited(
        self, session, mailed
    ) -> None:
        """Consent records are an audit trail; the original grant must survive."""
        rows = await consent_rows(session, mailed["candidate"].id)
        assert len(rows) == 1

        await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=NOW + timedelta(hours=1)
        )
        rows = await consent_rows(session, mailed["candidate"].id)
        assert sorted(r.status for r in rows) == [
            ConsentStatus.GRANTED,
            ConsentStatus.WITHDRAWN,
        ]

    async def test_a_withdrawal_beats_a_grant_of_the_same_instant(
        self, session, mailed
    ) -> None:
        """granted_at is a supplied timestamp, so the two really can collide.

        With no tie-break the winner is whichever row the database returns
        first, and half the time that is the grant — which means emailing
        someone who opted out.
        """
        original = (await consent_rows(session, mailed["candidate"].id))[0]
        await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=original.granted_at
        )

        rows = await consent_rows(session, mailed["candidate"].id)
        assert len({r.granted_at for r in rows}) == 1  # the collision is real
        assert not await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.EMAIL_COMMUNICATION,
        )

    async def test_the_enrollment_ends_as_unsubscribed(self, session, mailed) -> None:
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        await session.refresh(mailed["enrollment"])
        assert mailed["enrollment"].status == EnrollmentStatus.UNSUBSCRIBED
        assert mailed["enrollment"].next_send_at is None

    async def test_every_other_sequence_stops_too(self, session, mailed) -> None:
        """Unsubscribing from one campaign and hearing from another is being ignored."""
        other = await build_sequence(session, mailed["org"], name="Second campaign")
        report = await outreach_service.enroll(
            session, mailed["org"].id, other.id, [mailed["candidate"].id], now=NOW
        )
        second = report.enrolled[0]

        result = await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=NOW
        )
        await session.refresh(second)
        assert second.status == EnrollmentStatus.UNSUBSCRIBED
        assert result.stopped == 2

    async def test_another_candidate_is_untouched(self, session, mailed) -> None:
        bystander = await make_candidate(session, mailed["org"])
        await grant(session, mailed["org"], bystander)
        report = await outreach_service.enroll(
            session,
            mailed["org"].id,
            mailed["sequence"].id,
            [bystander.id],
            now=NOW,
        )

        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        await session.refresh(report.enrolled[0])
        assert report.enrolled[0].status == EnrollmentStatus.ACTIVE
        assert await candidate_service.has_consent(
            session, mailed["org"].id, bystander.id, ConsentType.EMAIL_COMMUNICATION
        )

    async def test_only_the_email_channel_is_withdrawn(
        self, session, mailed
    ) -> None:
        """Opting out of campaign mail is not a request never to be phoned."""
        await grant(
            session,
            mailed["org"],
            mailed["candidate"],
            ConsentType.SMS_COMMUNICATION,
        )
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)

        assert await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.SMS_COMMUNICATION,
        )

    async def test_the_candidate_is_not_blacklisted(self, session, mailed) -> None:
        """A stronger, recruiter-driven state the candidate never asked for."""
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        await session.refresh(mailed["candidate"])
        assert mailed["candidate"].is_blacklisted is False

    async def test_the_sequence_counter_moves(self, session, mailed) -> None:
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        sequence = await outreach_service.get_sequence(
            session, mailed["org"].id, mailed["sequence"].id
        )
        assert sequence.stats_json["unsubscribed"] == 1

    async def test_provenance_is_recorded(self, session, mailed) -> None:
        """Proving a candidate opted out is the point of keeping the row."""
        await outreach_service.unsubscribe_by_token(
            session,
            mailed["token"],
            now=NOW,
            ip_address="203.0.113.7",
            user_agent="Mozilla/5.0",
            source="one_click",
        )
        withdrawal = (await withdrawals(session, mailed["candidate"].id))[0]
        assert withdrawal.source == "one_click"
        assert withdrawal.ip_address == "203.0.113.7"
        assert withdrawal.user_agent == "Mozilla/5.0"
        assert withdrawal.withdrawn_at == NOW
        assert withdrawal.evidence_json == {"message_id": str(mailed["message"].id)}

    async def test_the_result_names_the_organization(self, session, mailed) -> None:
        result = await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=NOW
        )
        assert result.organization_name == "Acme"
        assert result.candidate_id == mailed["candidate"].id
        assert result.enrollment_id == mailed["enrollment"].id


class TestIdempotence:
    async def test_a_second_click_still_succeeds(self, session, mailed) -> None:
        """People click twice, and a client that missed the response retries."""
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        again = await outreach_service.unsubscribe_by_token(
            session, mailed["token"], now=NOW + timedelta(minutes=1)
        )
        assert again.already_unsubscribed is True
        assert again.stopped == 0

    async def test_it_does_not_pile_up_withdrawal_rows(
        self, session, mailed
    ) -> None:
        for _ in range(3):
            await outreach_service.unsubscribe_by_token(
                session, mailed["token"], now=NOW
            )
        assert len(await withdrawals(session, mailed["candidate"].id)) == 1


class TestTheDispatcherHonoursIt:
    async def test_no_further_message_goes_out(self, session, mailed) -> None:
        """The end-to-end claim: the loop re-reads consent, so this actually stops it."""
        other = await build_sequence(session, mailed["org"], name="Second campaign")
        await outreach_service.enroll(
            session, mailed["org"].id, other.id, [mailed["candidate"].id], now=NOW
        )
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)

        sent_before = len(mailed["transport"].sent)
        report = await dispatcher.run_once(session, now=NOW + timedelta(days=1))
        assert len(mailed["transport"].sent) == sent_before
        assert report.sent == 0

    async def test_a_new_enrolment_is_refused_afterwards(
        self, session, mailed
    ) -> None:
        """The durable fact is the consent row, so it outlives these enrollments."""
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        fresh = await build_sequence(session, mailed["org"], name="Later campaign")
        report = await outreach_service.enroll(
            session, mailed["org"].id, fresh.id, [mailed["candidate"].id], now=NOW
        )
        assert report.enrolled == []
        assert report.skipped


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
class TestMaskEmail:
    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            ("grace.hopper@example.test", "g***r@example.test"),
            ("ab@example.test", "a***@example.test"),
            ("a@example.test", "a***@example.test"),
        ],
    )
    def test_it_keeps_the_shape_and_drops_the_middle(self, address, expected) -> None:
        assert mask_email(address) == expected

    @pytest.mark.parametrize("address", [None, "", "not-an-address"])
    def test_anything_unusable_masks_to_nothing(self, address) -> None:
        assert mask_email(address) is None


# --------------------------------------------------------------------------- #
# The endpoints
# --------------------------------------------------------------------------- #
def url(token: str) -> str:
    return f"{settings.api_v1_prefix}/outreach/unsubscribe/{token}"


class TestViewEndpoint:
    async def test_it_describes_the_subscription(self, client, mailed) -> None:
        resp = await client.get(url(mailed["token"]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["organization_name"] == "Acme"
        assert body["sequence_name"] == "Backend outreach"
        assert body["already_unsubscribed"] is False

    async def test_the_address_is_masked(self, client, mailed) -> None:
        """The token travelled by email and may be sitting in a forwarded thread."""
        resp = await client.get(url(mailed["token"]))
        assert resp.json()["recipient"] == "g***r@example.test"
        assert "grace.hopper" not in resp.text

    async def test_it_needs_no_credentials(self, client, mailed) -> None:
        resp = await client.get(url(mailed["token"]))
        assert resp.status_code == 200

    async def test_looking_does_not_unsubscribe(
        self, session, client, mailed
    ) -> None:
        """Scanners and mail clients prefetch links; a mutating GET opts out ghosts."""
        await client.get(url(mailed["token"]))

        assert await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.EMAIL_COMMUNICATION,
        )
        await session.refresh(mailed["enrollment"])
        assert mailed["enrollment"].status == EnrollmentStatus.ACTIVE

    async def test_it_reports_an_opt_out_already_on_file(
        self, session, client, mailed
    ) -> None:
        await outreach_service.unsubscribe_by_token(session, mailed["token"], now=NOW)
        body = (await client.get(url(mailed["token"]))).json()
        assert body["already_unsubscribed"] is True

    async def test_an_unknown_token_is_a_404(self, client) -> None:
        resp = await client.get(url(uuid.uuid4().hex))
        assert resp.status_code == 404


class TestConfirmEndpoint:
    async def test_it_records_the_opt_out(self, session, client, mailed) -> None:
        resp = await client.post(url(mailed["token"]))
        assert resp.status_code == 200, resp.text
        assert resp.json()["sequences_stopped"] == 1

        assert not await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.EMAIL_COMMUNICATION,
        )

    async def test_it_accepts_the_one_click_form_body(
        self, session, client, mailed
    ) -> None:
        """What RFC 8058 actually puts on the wire, from a client with no session."""
        resp = await client.post(
            url(mailed["token"]), data={"List-Unsubscribe": "One-Click"}
        )
        assert resp.status_code == 200, resp.text
        assert not await candidate_service.has_consent(
            session,
            mailed["org"].id,
            mailed["candidate"].id,
            ConsentType.EMAIL_COMMUNICATION,
        )

    async def test_it_needs_no_credentials_and_no_body(
        self, client, mailed
    ) -> None:
        resp = await client.post(url(mailed["token"]))
        assert resp.status_code == 200

    async def test_a_repeat_post_is_not_an_error(self, client, mailed) -> None:
        """A client that did not see the first response will send another."""
        await client.post(url(mailed["token"]))
        resp = await client.post(url(mailed["token"]))
        assert resp.status_code == 200
        assert resp.json()["already_unsubscribed"] is True
        assert resp.json()["sequences_stopped"] == 0

    async def test_the_provenance_of_a_web_click_is_captured(
        self, session, client, mailed
    ) -> None:
        await client.post(
            url(mailed["token"]),
            headers={"user-agent": "TestClient/1.0", "x-forwarded-for": "203.0.113.9"},
        )
        withdrawal = (await withdrawals(session, mailed["candidate"].id))[0]
        assert withdrawal.user_agent == "TestClient/1.0"
        assert withdrawal.ip_address == "203.0.113.9"

    async def test_an_unknown_token_is_a_404(self, client) -> None:
        resp = await client.post(url(uuid.uuid4().hex))
        assert resp.status_code == 404

    async def test_a_404_leaks_nothing_about_which_tokens_exist(
        self, client
    ) -> None:
        """A near miss must not be distinguishable from a wild guess."""
        missing = (await client.post(url(uuid.uuid4().hex))).json()
        empty = (await client.post(url("x"))).json()
        assert missing["detail"]["message"] == empty["detail"]["message"]
