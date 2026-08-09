"""Outreach sequences: send windows, steps, enrollment, and progression.

Three things in this module are worth guarding hard, and most of the file is
about them.

The **send window** must snap, never skip. A step that comes due outside
business hours has to move forward to the next open moment; dropping it leaves
a candidate hearing the first and last message of a story and none of the
middle. The window is wall-clock, so the DST cases here are not academic — the
whole point of "we send at nine" is that it stays nine.

**Step order is an index other rows point at.** An enrollment remembers where it
is as an integer, so a hole or a duplicate in ``step_order`` silently sends the
wrong message to a real person. Every mutation that can reshuffle the order is
exercised, including inserting into the middle, which is where the interesting
constraint collisions live.

**Enrolment filters rather than refuses.** A recruiter handing over fifty
candidates gets forty-four enrolled and six explained, not a 422 because one of
them opted out.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.security import blind_index
from app.models.application import Application
from app.models.candidate import Candidate, CandidateConsent
from app.models.enums import (
    ApplicationStatus,
    ConsentStatus,
    ConsentType,
    EnrollmentStatus,
    MessageStatus,
    OutreachChannel,
    PipelineStage,
    SequenceStatus,
)
from app.models.job import Job
from app.models.organization import Organization
from app.models.outreach import (
    MessageTemplate,
    OutreachMessage,
    OutreachSequence,
    SequenceEnrollment,
)
from app.services import outreach_service as svc

# A Monday at midday UTC, so nothing here depends on a weekend rule by accident.
NOW = datetime(2027, 3, 8, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
async def make_org(session, name: str = "Acme") -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


@pytest.fixture
async def org(session) -> Organization:
    return await make_org(session)


@pytest.fixture
async def other_org(session) -> Organization:
    return await make_org(session, "Rival")


def sequence_stub(**fields) -> OutreachSequence:
    """An unsaved sequence, for the pure send-window functions."""
    defaults = dict(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        name="Stub",
        timezone="UTC",
        send_window_start_hour=9,
        send_window_end_hour=18,
        send_on_weekends=False,
    )
    defaults.update(fields)
    return OutreachSequence(**defaults)


async def make_candidate(session, org: Organization, **fields) -> Candidate:
    email = fields.pop("email", f"cand-{uuid.uuid4().hex[:8]}@example.test")
    candidate = Candidate(
        organization_id=org.id,
        full_name=fields.pop("full_name", "Grace Hopper"),
        email=email,
        email_index=blind_index(email) if email else "",
        **fields,
    )
    session.add(candidate)
    await session.commit()
    return candidate


async def grant(
    session,
    org: Organization,
    candidate: Candidate,
    consent_type: ConsentType = ConsentType.EMAIL_COMMUNICATION,
    *,
    status: ConsentStatus = ConsentStatus.GRANTED,
    granted_at: datetime | None = None,
) -> CandidateConsent:
    consent = CandidateConsent(
        organization_id=org.id,
        candidate_id=candidate.id,
        consent_type=consent_type,
        status=status,
        granted_at=granted_at or NOW,
    )
    session.add(consent)
    await session.commit()
    return consent


async def make_template(
    session,
    org: Organization,
    *,
    channel: OutreachChannel = OutreachChannel.EMAIL,
    name: str = "Intro",
) -> MessageTemplate:
    template = MessageTemplate(
        organization_id=org.id,
        name=name,
        channel=channel,
        subject="Hello",
        body="Hi there",
    )
    session.add(template)
    await session.commit()
    return template


async def make_sequence(session, org: Organization, **fields) -> OutreachSequence:
    return await svc.create_sequence(
        session, org.id, name=fields.pop("name", "Backend outreach"), **fields
    )


async def with_steps(
    session, org: Organization, count: int = 2, **fields
) -> OutreachSequence:
    """A sequence carrying ``count`` email steps, the first with no delay."""
    sequence = await make_sequence(session, org, **fields)
    for index in range(count):
        await svc.add_step(
            session,
            org.id,
            sequence.id,
            body_override=f"Message {index}",
            delay_days=0 if index == 0 else 2,
        )
    await session.refresh(sequence)
    return sequence


def orders(sequence: OutreachSequence) -> list[int]:
    return sorted(step.step_order for step in sequence.steps)


def bodies_in_order(sequence: OutreachSequence) -> list[str | None]:
    return [
        step.body_override
        for step in sorted(sequence.steps, key=lambda s: s.step_order)
    ]


# --------------------------------------------------------------------------- #
# Send windows
# --------------------------------------------------------------------------- #
class TestDayIsAllowed:
    def test_weekdays_are_allowed_by_default(self) -> None:
        sequence = sequence_stub()
        assert svc.day_is_allowed(datetime(2027, 3, 8).date(), sequence) is True

    def test_weekends_are_refused_by_default(self) -> None:
        sequence = sequence_stub()
        assert svc.day_is_allowed(datetime(2027, 3, 13).date(), sequence) is False
        assert svc.day_is_allowed(datetime(2027, 3, 14).date(), sequence) is False

    def test_opting_into_weekends_allows_them(self) -> None:
        sequence = sequence_stub(send_on_weekends=True)
        assert svc.day_is_allowed(datetime(2027, 3, 13).date(), sequence) is True


class TestIsWithinWindow:
    def test_midday_on_a_weekday_is_inside(self) -> None:
        assert svc.is_within_window(NOW, sequence_stub()) is True

    def test_the_opening_hour_is_inside(self) -> None:
        moment = datetime(2027, 3, 8, 9, 0, tzinfo=UTC)
        assert svc.is_within_window(moment, sequence_stub()) is True

    def test_the_closing_hour_is_outside(self) -> None:
        """The window is half-open, so 18:00 belongs to the evening."""
        moment = datetime(2027, 3, 8, 18, 0, tzinfo=UTC)
        assert svc.is_within_window(moment, sequence_stub()) is False

    def test_before_the_window_opens_is_outside(self) -> None:
        moment = datetime(2027, 3, 8, 3, 0, tzinfo=UTC)
        assert svc.is_within_window(moment, sequence_stub()) is False

    def test_a_weekend_is_outside_whatever_the_hour(self) -> None:
        moment = datetime(2027, 3, 13, 12, 0, tzinfo=UTC)
        assert svc.is_within_window(moment, sequence_stub()) is False

    def test_the_window_is_read_in_the_sequences_own_timezone(self) -> None:
        """Kolkata is UTC+5:30, so a 9-18 window closes at 12:30 UTC."""
        sequence = sequence_stub(timezone="Asia/Kolkata")
        assert svc.is_within_window(datetime(2027, 3, 8, 12, 0, tzinfo=UTC), sequence)
        assert not svc.is_within_window(
            datetime(2027, 3, 8, 13, 0, tzinfo=UTC), sequence
        )

    def test_a_naive_moment_is_read_as_utc(self) -> None:
        assert svc.is_within_window(datetime(2027, 3, 8, 12, 0), sequence_stub()) is True

    def test_an_end_hour_of_24_runs_to_midnight(self) -> None:
        sequence = sequence_stub(send_window_start_hour=0, send_window_end_hour=24)
        assert svc.is_within_window(datetime(2027, 3, 8, 23, 59, tzinfo=UTC), sequence)


class TestNextSendTime:
    def test_a_moment_inside_the_window_is_returned_unchanged(self) -> None:
        assert svc.next_send_time(NOW, sequence_stub()) == NOW

    def test_before_the_window_waits_for_it_to_open(self) -> None:
        early = datetime(2027, 3, 8, 3, 0, tzinfo=UTC)
        assert svc.next_send_time(early, sequence_stub()) == datetime(
            2027, 3, 8, 9, 0, tzinfo=UTC
        )

    def test_after_the_window_rolls_to_the_next_morning(self) -> None:
        late = datetime(2027, 3, 8, 22, 0, tzinfo=UTC)
        assert svc.next_send_time(late, sequence_stub()) == datetime(
            2027, 3, 9, 9, 0, tzinfo=UTC
        )

    def test_a_weekend_rolls_forward_to_monday(self) -> None:
        saturday = datetime(2027, 3, 13, 12, 0, tzinfo=UTC)
        assert svc.next_send_time(saturday, sequence_stub()) == datetime(
            2027, 3, 15, 9, 0, tzinfo=UTC
        )

    def test_weekend_sending_keeps_saturday(self) -> None:
        saturday = datetime(2027, 3, 13, 12, 0, tzinfo=UTC)
        sequence = sequence_stub(send_on_weekends=True)
        assert svc.next_send_time(saturday, sequence) == saturday

    def test_the_window_opens_at_local_nine_not_utc_nine(self) -> None:
        sequence = sequence_stub(timezone="Asia/Kolkata")
        early = datetime(2027, 3, 8, 0, 0, tzinfo=UTC)  # 05:30 IST
        assert svc.next_send_time(early, sequence) == datetime(
            2027, 3, 8, 3, 30, tzinfo=UTC
        )  # 09:00 IST

    def test_crossing_a_dst_boundary_still_lands_at_local_nine(self) -> None:
        """Spring forward is on 14 March 2027 in New York.

        Computed by absolute arithmetic — "Saturday plus two days plus nine
        hours" — this lands at 14:00 UTC, an hour late. The window is wall
        clock, so it has to be 09:00 EDT, which is 13:00 UTC.
        """
        sequence = sequence_stub(timezone="America/New_York")
        saturday_evening = datetime(2027, 3, 14, 1, 0, tzinfo=UTC)  # Sat 20:00 EST
        assert svc.next_send_time(saturday_evening, sequence) == datetime(
            2027, 3, 15, 13, 0, tzinfo=UTC
        )

    def test_an_unknown_timezone_degrades_to_utc_rather_than_raising(self) -> None:
        sequence = sequence_stub(timezone="Mars/Olympus_Mons")
        assert svc.next_send_time(NOW, sequence) == NOW

    def test_a_naive_input_is_read_as_utc(self) -> None:
        assert svc.next_send_time(datetime(2027, 3, 8, 3, 0), sequence_stub()) == (
            datetime(2027, 3, 8, 9, 0, tzinfo=UTC)
        )


class TestStepDueAt:
    def _step(self, **fields):
        from app.models.outreach import SequenceStep

        defaults = dict(
            organization_id=uuid.uuid4(),
            sequence_id=uuid.uuid4(),
            step_order=0,
            delay_days=0,
            delay_hours=0,
        )
        defaults.update(fields)
        return SequenceStep(**defaults)

    def test_no_delay_means_as_soon_as_the_window_allows(self) -> None:
        assert svc.step_due_at(self._step(), after=NOW, sequence=sequence_stub()) == NOW

    def test_the_delay_is_applied_before_the_window(self) -> None:
        """Two days means two days, not "two days rounded down to an opening"."""
        step = self._step(delay_days=2)
        due = svc.step_due_at(step, after=NOW, sequence=sequence_stub())
        assert due == datetime(2027, 3, 10, 12, 0, tzinfo=UTC)

    def test_a_delay_landing_at_night_is_pushed_into_the_morning(self) -> None:
        step = self._step(delay_hours=14)  # 02:00 the next day
        due = svc.step_due_at(step, after=NOW, sequence=sequence_stub())
        assert due == datetime(2027, 3, 9, 9, 0, tzinfo=UTC)

    def test_a_delay_landing_on_a_weekend_is_pushed_to_monday(self) -> None:
        step = self._step(delay_days=5)  # Saturday
        due = svc.step_due_at(step, after=NOW, sequence=sequence_stub())
        assert due == datetime(2027, 3, 15, 9, 0, tzinfo=UTC)

    def test_a_negative_delay_is_treated_as_none(self) -> None:
        step = self._step(delay_days=-3, delay_hours=-1)
        assert svc.step_due_at(step, after=NOW, sequence=sequence_stub()) == NOW


# --------------------------------------------------------------------------- #
# Sequence CRUD
# --------------------------------------------------------------------------- #
class TestCreateSequence:
    async def test_a_new_sequence_starts_as_a_draft(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        assert sequence.status == SequenceStatus.DRAFT
        assert sequence.started_at is None
        assert sequence.steps == []

    async def test_counters_start_at_zero_rather_than_absent(
        self, session, org
    ) -> None:
        """A dashboard should read 0, not blank, before anything has happened."""
        sequence = await make_sequence(session, org)
        assert sequence.stats_json == dict.fromkeys(svc.STAT_KEYS, 0)

    async def test_the_name_is_trimmed(self, session, org) -> None:
        sequence = await make_sequence(session, org, name="  Sourcing  ")
        assert sequence.name == "Sourcing"

    async def test_a_blank_name_is_refused(self, session, org) -> None:
        with pytest.raises(ValidationError):
            await make_sequence(session, org, name="   ")

    async def test_sender_ids_are_stored_as_strings_for_json(
        self, session, org
    ) -> None:
        sender = uuid.uuid4()
        sequence = await make_sequence(session, org, sender_account_ids=[sender])
        assert sequence.sender_account_ids == [str(sender)]

    @pytest.mark.parametrize(
        "start,end",
        [(-1, 18), (24, 25), (9, 0), (9, 25), (18, 18), (19, 18)],
    )
    async def test_an_impossible_window_is_refused(
        self, session, org, start, end
    ) -> None:
        with pytest.raises(ValidationError):
            await make_sequence(
                session, org, send_window_start_hour=start, send_window_end_hour=end
            )

    async def test_an_unknown_timezone_is_refused_at_authoring_time(
        self, session, org
    ) -> None:
        """load_zone falls back to UTC at send time, which would hide the typo."""
        with pytest.raises(ValidationError):
            await make_sequence(session, org, timezone="Mars/Olympus_Mons")

    async def test_a_daily_cap_below_one_is_refused(self, session, org) -> None:
        with pytest.raises(ValidationError):
            await make_sequence(session, org, daily_cap=0)


class TestReadSequences:
    async def test_a_sequence_from_another_org_is_not_found(
        self, session, org, other_org
    ) -> None:
        sequence = await make_sequence(session, org)
        with pytest.raises(NotFoundError):
            await svc.get_sequence(session, other_org.id, sequence.id)

    async def test_listing_is_scoped_to_the_org(
        self, session, org, other_org
    ) -> None:
        await make_sequence(session, org)
        await make_sequence(session, other_org)
        assert len(await svc.list_sequences(session, org.id)) == 1

    async def test_listing_filters_by_status(self, session, org) -> None:
        await with_steps(session, org, 1, name="Live")
        live = (await svc.list_sequences(session, org.id))[0]
        await svc.activate(session, org.id, live.id, now=NOW)
        await make_sequence(session, org, name="Draft")

        found = await svc.list_sequences(session, org.id, status=SequenceStatus.ACTIVE)
        assert [s.name for s in found] == ["Live"]

    async def test_listing_filters_by_job(self, session, org) -> None:
        job_id = uuid.uuid4()
        await make_sequence(session, org, name="For the job", job_id=job_id)
        await make_sequence(session, org, name="Generic")
        found = await svc.list_sequences(session, org.id, job_id=job_id)
        assert [s.name for s in found] == ["For the job"]

    async def test_a_deleted_sequence_leaves_the_list(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        await svc.delete_sequence(session, org.id, sequence.id, now=NOW)
        assert await svc.list_sequences(session, org.id) == []


class TestUpdateSequence:
    async def test_fields_are_written_through(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        updated = await svc.update_sequence(
            session,
            org.id,
            sequence.id,
            changes={"name": "Renamed", "timezone": "Asia/Kolkata"},
        )
        assert updated.name == "Renamed"
        assert updated.timezone == "Asia/Kolkata"

    async def test_absent_keys_are_left_alone(self, session, org) -> None:
        sequence = await make_sequence(session, org, timezone="Asia/Kolkata")
        updated = await svc.update_sequence(
            session, org.id, sequence.id, changes={"name": "Renamed"}
        )
        assert updated.timezone == "Asia/Kolkata"

    async def test_turning_a_flag_off_is_not_mistaken_for_absent(
        self, session, org
    ) -> None:
        """False is a value; only None means "leave it alone"."""
        sequence = await make_sequence(session, org, stop_on_reply=True)
        updated = await svc.update_sequence(
            session, org.id, sequence.id, changes={"stop_on_reply": False}
        )
        assert updated.stop_on_reply is False

    async def test_an_edit_that_breaks_the_window_is_refused(
        self, session, org
    ) -> None:
        sequence = await make_sequence(session, org)
        with pytest.raises(ValidationError):
            await svc.update_sequence(
                session, org.id, sequence.id, changes={"send_window_start_hour": 20}
            )

    async def test_an_edit_to_an_unknown_timezone_is_refused(
        self, session, org
    ) -> None:
        sequence = await make_sequence(session, org)
        with pytest.raises(ValidationError):
            await svc.update_sequence(
                session, org.id, sequence.id, changes={"timezone": "Nowhere/Special"}
            )


class TestLifecycle:
    async def test_activation_stamps_the_start(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        activated = await svc.activate(session, org.id, sequence.id, now=NOW)
        assert activated.status == SequenceStatus.ACTIVE
        assert activated.started_at == NOW

    async def test_a_sequence_with_no_steps_cannot_run(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        with pytest.raises(ValidationError):
            await svc.activate(session, org.id, sequence.id, now=NOW)

    async def test_activating_twice_is_a_no_op(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.activate(session, org.id, sequence.id, now=NOW)
        again = await svc.activate(
            session, org.id, sequence.id, now=NOW + timedelta(days=1)
        )
        assert again.started_at == NOW

    async def test_resuming_keeps_the_original_start(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.activate(session, org.id, sequence.id, now=NOW)
        await svc.pause(session, org.id, sequence.id)
        resumed = await svc.activate(
            session, org.id, sequence.id, now=NOW + timedelta(days=3)
        )
        assert resumed.started_at == NOW

    async def test_an_archived_sequence_cannot_be_activated(
        self, session, org
    ) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.delete_sequence(session, org.id, sequence.id, now=NOW)
        with pytest.raises(NotFoundError):
            await svc.activate(session, org.id, sequence.id, now=NOW)

    async def test_pausing_keeps_the_sequence_readable(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.activate(session, org.id, sequence.id, now=NOW)
        paused = await svc.pause(session, org.id, sequence.id)
        assert paused.status == SequenceStatus.PAUSED

    async def test_a_completed_sequence_cannot_be_paused(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.complete_sequence(session, org.id, sequence.id, now=NOW)
        with pytest.raises(ConflictError):
            await svc.pause(session, org.id, sequence.id)

    async def test_completion_stamps_the_end(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        done = await svc.complete_sequence(session, org.id, sequence.id, now=NOW)
        assert done.status == SequenceStatus.COMPLETED
        assert done.completed_at == NOW

    async def test_deletion_is_soft_and_archives(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.delete_sequence(session, org.id, sequence.id, now=NOW)
        with pytest.raises(NotFoundError):
            await svc.get_sequence(session, org.id, sequence.id)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
class TestAddStep:
    async def test_steps_are_numbered_from_zero(self, session, org) -> None:
        sequence = await with_steps(session, org, 3)
        assert orders(sequence) == [0, 1, 2]

    async def test_a_step_needs_something_to_say(self, session, org) -> None:
        """A step with neither template nor body would send nothing silently."""
        sequence = await make_sequence(session, org)
        with pytest.raises(ValidationError):
            await svc.add_step(session, org.id, sequence.id)
        with pytest.raises(ValidationError):
            await svc.add_step(session, org.id, sequence.id, body_override="   ")

    async def test_a_template_satisfies_the_content_requirement(
        self, session, org
    ) -> None:
        sequence = await make_sequence(session, org)
        template = await make_template(session, org)
        step = await svc.add_step(
            session, org.id, sequence.id, template_id=template.id
        )
        assert step.template_id == template.id

    async def test_a_template_from_another_org_is_not_found(
        self, session, org, other_org
    ) -> None:
        sequence = await make_sequence(session, org)
        template = await make_template(session, other_org)
        with pytest.raises(NotFoundError):
            await svc.add_step(
                session, org.id, sequence.id, template_id=template.id
            )

    async def test_a_template_for_another_channel_is_refused(
        self, session, org
    ) -> None:
        sequence = await make_sequence(session, org)
        template = await make_template(session, org, channel=OutreachChannel.WHATSAPP)
        with pytest.raises(ValidationError):
            await svc.add_step(
                session,
                org.id,
                sequence.id,
                template_id=template.id,
                channel=OutreachChannel.EMAIL,
            )

    async def test_a_negative_delay_is_refused(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        with pytest.raises(ValidationError):
            await svc.add_step(
                session, org.id, sequence.id, body_override="Hi", delay_days=-1
            )

    async def test_the_step_count_is_capped(self, session, org) -> None:
        sequence = await with_steps(session, org, svc.MAX_STEPS)
        with pytest.raises(ValidationError):
            await svc.add_step(session, org.id, sequence.id, body_override="One more")

    @pytest.mark.parametrize("position", [0, 1, 2, 3])
    async def test_inserting_at_any_position_renumbers_cleanly(
        self, session, org, position
    ) -> None:
        """Every position, because the collision this guards is order-dependent.

        ``(sequence_id, step_order)`` is unique and a flush emits its UPDATEs in
        whatever order it likes, so a staging slot that overlaps the range being
        rewritten fails for some insert positions and not others — and for a
        given position, not on every run.
        """
        sequence = await with_steps(session, org, 3)
        await svc.add_step(
            session, org.id, sequence.id, body_override="New", step_order=position
        )
        await session.refresh(sequence)

        expected = ["Message 0", "Message 1", "Message 2"]
        expected.insert(position, "New")
        assert bodies_in_order(sequence) == expected
        assert orders(sequence) == [0, 1, 2, 3]

    async def test_a_position_past_the_end_appends(self, session, org) -> None:
        sequence = await with_steps(session, org, 2)
        await svc.add_step(
            session, org.id, sequence.id, body_override="Last", step_order=99
        )
        await session.refresh(sequence)
        assert bodies_in_order(sequence)[-1] == "Last"
        assert orders(sequence) == [0, 1, 2]

    async def test_a_negative_position_lands_at_the_front(
        self, session, org
    ) -> None:
        sequence = await with_steps(session, org, 2)
        await svc.add_step(
            session, org.id, sequence.id, body_override="Front", step_order=-5
        )
        await session.refresh(sequence)
        assert bodies_in_order(sequence)[0] == "Front"


class TestUpdateStep:
    async def test_a_delay_can_be_changed(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        step = sequence.steps[0]
        updated = await svc.update_step(
            session, org.id, step.id, changes={"delay_days": 4}
        )
        assert updated.delay_days == 4

    async def test_an_edit_may_not_empty_the_step(self, session, org) -> None:
        sequence = await make_sequence(session, org)
        template = await make_template(session, org)
        step = await svc.add_step(
            session, org.id, sequence.id, template_id=template.id
        )
        step.template_id = None
        with pytest.raises(ValidationError):
            await svc.update_step(
                session, org.id, step.id, changes={"body_override": "  "}
            )

    async def test_an_edit_to_a_negative_delay_is_refused(
        self, session, org
    ) -> None:
        sequence = await with_steps(session, org, 1)
        with pytest.raises(ValidationError):
            await svc.update_step(
                session, org.id, sequence.steps[0].id, changes={"delay_hours": -2}
            )

    async def test_a_step_from_another_org_is_not_found(
        self, session, org, other_org
    ) -> None:
        sequence = await with_steps(session, org, 1)
        with pytest.raises(NotFoundError):
            await svc.get_step(session, other_org.id, sequence.steps[0].id)


class TestDeleteStep:
    async def test_deleting_closes_the_gap(self, session, org) -> None:
        """Enrollments index by position, so a hole would skip a message."""
        sequence = await with_steps(session, org, 3)
        middle = sorted(sequence.steps, key=lambda s: s.step_order)[1]
        await svc.delete_step(session, org.id, middle.id)
        await session.refresh(sequence)
        assert orders(sequence) == [0, 1]
        assert bodies_in_order(sequence) == ["Message 0", "Message 2"]

    async def test_deleting_the_last_step_is_allowed(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        await svc.delete_step(session, org.id, sequence.steps[0].id)
        await session.refresh(sequence)
        assert sequence.steps == []


class TestReorderSteps:
    async def test_the_order_is_rewritten(self, session, org) -> None:
        sequence = await with_steps(session, org, 3)
        ordered = sorted(sequence.steps, key=lambda s: s.step_order)
        reversed_ids = [step.id for step in reversed(ordered)]

        result = await svc.reorder_steps(session, org.id, sequence.id, reversed_ids)
        assert [step.id for step in result] == reversed_ids
        assert [step.step_order for step in result] == [0, 1, 2]

    async def test_a_partial_reorder_is_refused(self, session, org) -> None:
        sequence = await with_steps(session, org, 3)
        with pytest.raises(ValidationError):
            await svc.reorder_steps(
                session, org.id, sequence.id, [sequence.steps[0].id]
            )

    async def test_a_duplicated_id_is_refused(self, session, org) -> None:
        sequence = await with_steps(session, org, 2)
        first = sequence.steps[0].id
        with pytest.raises(ValidationError):
            await svc.reorder_steps(session, org.id, sequence.id, [first, first])

    async def test_a_foreign_step_id_is_refused(self, session, org) -> None:
        sequence = await with_steps(session, org, 2)
        ids = [step.id for step in sequence.steps]
        with pytest.raises(ValidationError):
            await svc.reorder_steps(
                session, org.id, sequence.id, [ids[0], uuid.uuid4()]
            )


class TestStepAt:
    async def test_the_matching_position_is_returned(self, session, org) -> None:
        sequence = await with_steps(session, org, 2)
        step = svc.step_at(sequence, 1)
        assert step is not None and step.body_override == "Message 1"

    async def test_past_the_end_is_none(self, session, org) -> None:
        sequence = await with_steps(session, org, 2)
        assert svc.step_at(sequence, 2) is None


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
@pytest.fixture
async def campaign(session, org) -> dict:
    """An active two-step sequence and one fully consenting candidate."""
    sequence = await with_steps(session, org, 2)
    await svc.activate(session, org.id, sequence.id, now=NOW)
    candidate = await make_candidate(session, org)
    await grant(session, org, candidate)
    return {"sequence": sequence, "candidate": candidate}


class TestEnroll:
    async def test_a_consenting_candidate_is_enrolled(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert report.enrolled_count == 1
        assert report.skipped == []
        assert report.enrolled[0].status == EnrollmentStatus.ACTIVE
        assert report.enrolled[0].current_step == 0

    async def test_the_first_step_is_scheduled_from_enrolment(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert report.enrolled[0].next_send_at == NOW

    async def test_enrolment_outside_the_window_waits_for_the_morning(
        self, session, org, campaign
    ) -> None:
        night = datetime(2027, 3, 8, 23, 0, tzinfo=UTC)
        report = await svc.enroll(
            session,
            org.id,
            campaign["sequence"].id,
            [campaign["candidate"].id],
            now=night,
        )
        assert report.enrolled[0].next_send_at == datetime(
            2027, 3, 9, 9, 0, tzinfo=UTC
        )

    async def test_a_missing_candidate_is_skipped_not_raised(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [uuid.uuid4()], now=NOW
        )
        assert report.enrolled == []
        assert report.skipped[0]["reason"] == "not_found"

    async def test_a_candidate_from_another_org_is_skipped(
        self, session, org, other_org, campaign
    ) -> None:
        outsider = await make_candidate(session, other_org)
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [outsider.id], now=NOW
        )
        assert report.skipped[0]["reason"] == "not_found"

    async def test_a_blacklisted_candidate_is_skipped(
        self, session, org, campaign
    ) -> None:
        blocked = await make_candidate(session, org, is_blacklisted=True)
        await grant(session, org, blocked)
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [blocked.id], now=NOW
        )
        assert report.skipped[0]["reason"] == "blacklisted"

    async def test_a_candidate_without_consent_is_skipped(
        self, session, org, campaign
    ) -> None:
        silent = await make_candidate(session, org)
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [silent.id], now=NOW
        )
        assert report.skipped[0]["reason"] == "no_consent"

    async def test_withdrawn_consent_counts_as_no_consent(
        self, session, org, campaign
    ) -> None:
        gone = await make_candidate(session, org)
        await grant(session, org, gone)
        await grant(
            session,
            org,
            gone,
            status=ConsentStatus.WITHDRAWN,
            granted_at=NOW + timedelta(days=1),
        )
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [gone.id], now=NOW
        )
        assert report.skipped[0]["reason"] == "no_consent"

    async def test_a_whatsapp_step_demands_whatsapp_consent(
        self, session, org
    ) -> None:
        """Each channel in the sequence adds its own consent requirement."""
        sequence = await make_sequence(session, org)
        await svc.add_step(session, org.id, sequence.id, body_override="Email")
        await svc.add_step(
            session,
            org.id,
            sequence.id,
            body_override="WhatsApp",
            channel=OutreachChannel.WHATSAPP,
        )
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate, ConsentType.EMAIL_COMMUNICATION)

        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.skipped[0]["reason"] == "no_consent"
        assert "whatsapp" in report.skipped[0]["detail"]

        await grant(session, org, candidate, ConsentType.WHATSAPP_COMMUNICATION)
        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.enrolled_count == 1

    async def test_a_linkedin_step_needs_no_consent_of_ours(
        self, session, org
    ) -> None:
        """InMail travels under LinkedIn's consent regime, not this one."""
        sequence = await make_sequence(session, org)
        await svc.add_step(
            session,
            org.id,
            sequence.id,
            body_override="InMail",
            channel=OutreachChannel.LINKEDIN,
        )
        candidate = await make_candidate(session, org)
        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.enrolled_count == 1

    async def test_enrolling_the_same_candidate_twice_is_skipped(
        self, session, org, campaign
    ) -> None:
        ids = [campaign["candidate"].id]
        await svc.enroll(session, org.id, campaign["sequence"].id, ids, now=NOW)
        report = await svc.enroll(session, org.id, campaign["sequence"].id, ids, now=NOW)
        assert report.enrolled == []
        assert report.skipped[0]["reason"] == "already_enrolled"

    async def test_a_repeated_id_in_one_call_enrolls_once(
        self, session, org, campaign
    ) -> None:
        """Deduplicating in-call avoids tripping the uniqueness index."""
        candidate_id = campaign["candidate"].id
        report = await svc.enroll(
            session,
            org.id,
            campaign["sequence"].id,
            [candidate_id, candidate_id],
            now=NOW,
        )
        assert report.enrolled_count == 1
        assert report.skipped == []

    async def test_one_refusal_does_not_sink_the_batch(
        self, session, org, campaign
    ) -> None:
        blocked = await make_candidate(session, org, is_blacklisted=True)
        report = await svc.enroll(
            session,
            org.id,
            campaign["sequence"].id,
            [campaign["candidate"].id, blocked.id, uuid.uuid4()],
            now=NOW,
        )
        assert report.enrolled_count == 1
        assert report.skipped_count == 2

    async def test_enrolment_bumps_the_rolling_counter(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        sequence = await svc.get_sequence(session, org.id, campaign["sequence"].id)
        assert sequence.stats_json["enrolled"] == 1

    async def test_a_sequence_with_no_steps_cannot_enrol(
        self, session, org
    ) -> None:
        sequence = await make_sequence(session, org)
        candidate = await make_candidate(session, org)
        with pytest.raises(ValidationError):
            await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)

    async def test_an_archived_sequence_cannot_take_enrollments(
        self, session, org, campaign
    ) -> None:
        await svc.delete_sequence(session, org.id, campaign["sequence"].id, now=NOW)
        with pytest.raises(NotFoundError):
            await svc.enroll(
                session,
                org.id,
                campaign["sequence"].id,
                [campaign["candidate"].id],
                now=NOW,
            )

    async def test_a_draft_sequence_can_be_pre_loaded(
        self, session, org
    ) -> None:
        """Enrolling before launch is normal; due_enrollments is the gate."""
        sequence = await with_steps(session, org, 1)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.enrolled_count == 1


class TestEnrollmentApplicationLink:
    async def _job(self, session, org) -> Job:
        job = Job(
            organization_id=org.id,
            title="Backend Engineer",
            slug=f"be-{uuid.uuid4().hex[:6]}",
        )
        session.add(job)
        await session.commit()
        return job

    async def _application(
        self, session, org, job, candidate, status=ApplicationStatus.ACTIVE
    ) -> Application:
        application = Application(
            organization_id=org.id,
            job_id=job.id,
            candidate_id=candidate.id,
            stage=PipelineStage.APPLIED,
            status=status,
            applied_at=NOW,
        )
        session.add(application)
        await session.commit()
        return application

    async def test_an_open_application_is_linked(self, session, org) -> None:
        job = await self._job(session, org)
        sequence = await with_steps(session, org, 1, job_id=job.id)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        application = await self._application(session, org, job, candidate)

        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.enrolled[0].application_id == application.id

    async def test_a_rejected_application_is_not_linked(self, session, org) -> None:
        """Outreach should not attach itself to a closed application."""
        job = await self._job(session, org)
        sequence = await with_steps(session, org, 1, job_id=job.id)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        await self._application(
            session, org, job, candidate, status=ApplicationStatus.REJECTED
        )

        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert report.enrolled[0].application_id is None

    async def test_a_sequence_without_a_job_links_nothing(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert report.enrolled[0].application_id is None


class TestListEnrollments:
    async def test_filters_compose(self, session, org, campaign) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        found = await svc.list_enrollments(
            session,
            org.id,
            sequence_id=campaign["sequence"].id,
            candidate_id=campaign["candidate"].id,
            status=EnrollmentStatus.ACTIVE,
        )
        assert len(found) == 1

    async def test_a_status_filter_that_matches_nothing_is_empty(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        found = await svc.list_enrollments(
            session, org.id, status=EnrollmentStatus.REPLIED
        )
        assert found == []

    async def test_another_orgs_enrollments_are_invisible(
        self, session, org, other_org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert await svc.list_enrollments(session, other_org.id) == []


# --------------------------------------------------------------------------- #
# The due queue
# --------------------------------------------------------------------------- #
class TestDueEnrollments:
    async def test_an_enrollment_due_now_is_returned(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        due = await svc.due_enrollments(session, now=NOW)
        assert len(due) == 1

    async def test_an_enrollment_due_later_is_not(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert await svc.due_enrollments(session, now=NOW - timedelta(hours=1)) == []

    async def test_a_missed_tick_is_caught_up_rather_than_skipped(
        self, session, org, campaign
    ) -> None:
        """A worker down for an hour must not skip everyone in the gap."""
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        due = await svc.due_enrollments(session, now=NOW + timedelta(hours=3))
        assert len(due) == 1

    async def test_pausing_the_sequence_empties_the_queue(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await svc.pause(session, org.id, campaign["sequence"].id)
        assert await svc.due_enrollments(session, now=NOW) == []

    async def test_a_draft_sequence_sends_nothing(self, session, org) -> None:
        sequence = await with_steps(session, org, 1)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)
        assert await svc.due_enrollments(session, now=NOW) == []

    async def test_a_deleted_sequence_sends_nothing(
        self, session, org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await svc.delete_sequence(session, org.id, campaign["sequence"].id, now=NOW)
        assert await svc.due_enrollments(session, now=NOW) == []

    async def test_a_paused_enrollment_is_not_due(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await svc.pause_enrollment(session, org.id, report.enrolled[0].id)
        assert await svc.due_enrollments(session, now=NOW) == []

    async def test_the_queue_is_oldest_first(self, session, org, campaign) -> None:
        first = campaign["candidate"]
        second = await make_candidate(session, org)
        await grant(session, org, second)
        sequence_id = campaign["sequence"].id
        await svc.enroll(session, org.id, sequence_id, [second.id], now=NOW)
        await svc.enroll(
            session, org.id, sequence_id, [first.id], now=NOW - timedelta(hours=1)
        )

        due = await svc.due_enrollments(session, now=NOW)
        assert [e.candidate_id for e in due] == [first.id, second.id]

    async def test_the_limit_is_honoured(self, session, org, campaign) -> None:
        extra = await make_candidate(session, org)
        await grant(session, org, extra)
        await svc.enroll(
            session,
            org.id,
            campaign["sequence"].id,
            [campaign["candidate"].id, extra.id],
            now=NOW,
        )
        assert len(await svc.due_enrollments(session, now=NOW, limit=1)) == 1

    async def test_the_queue_can_be_narrowed_to_one_org(
        self, session, org, other_org, campaign
    ) -> None:
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        assert await svc.due_enrollments(session, now=NOW, organization_id=org.id)
        assert (
            await svc.due_enrollments(session, now=NOW, organization_id=other_org.id)
            == []
        )

    async def test_the_queue_spans_orgs_by_default(
        self, session, org, other_org, campaign
    ) -> None:
        """The worker runs once for the whole install, not once per tenant."""
        await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        rival_sequence = await with_steps(session, other_org, 1)
        await svc.activate(session, other_org.id, rival_sequence.id, now=NOW)
        rival = await make_candidate(session, other_org)
        await grant(session, other_org, rival)
        await svc.enroll(session, other_org.id, rival_sequence.id, [rival.id], now=NOW)

        assert len(await svc.due_enrollments(session, now=NOW)) == 2


# --------------------------------------------------------------------------- #
# Progression
# --------------------------------------------------------------------------- #
class TestAdvance:
    async def test_advancing_schedules_the_next_step(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        enrollment = report.enrolled[0]
        svc.advance(enrollment, campaign["sequence"], now=NOW)

        assert enrollment.current_step == 1
        # Step 1 carries a two-day delay.
        assert enrollment.next_send_at == NOW + timedelta(days=2)
        assert enrollment.status == EnrollmentStatus.ACTIVE

    async def test_running_off_the_end_completes_the_enrollment(
        self, session, org
    ) -> None:
        """ACTIVE with nothing to do is indistinguishable from a stuck row."""
        sequence = await with_steps(session, org, 1)
        await svc.activate(session, org.id, sequence.id, now=NOW)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)

        enrollment = report.enrolled[0]
        svc.advance(enrollment, sequence, now=NOW)
        assert enrollment.status == EnrollmentStatus.COMPLETED
        assert enrollment.completed_at == NOW
        assert enrollment.next_send_at is None

    async def test_the_next_step_respects_the_window(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        enrollment = report.enrolled[0]
        # Sending at 22:00 Wednesday, +2 days lands Friday 22:00 → Monday 09:00.
        late = datetime(2027, 3, 10, 22, 0, tzinfo=UTC)
        svc.advance(enrollment, campaign["sequence"], now=late)
        assert enrollment.next_send_at == datetime(2027, 3, 15, 9, 0, tzinfo=UTC)


class TestHalt:
    def test_halting_clears_the_clock(self) -> None:
        enrollment = SequenceEnrollment(
            organization_id=uuid.uuid4(),
            sequence_id=uuid.uuid4(),
            candidate_id=uuid.uuid4(),
            status=EnrollmentStatus.ACTIVE,
            enrolled_at=NOW,
            next_send_at=NOW,
        )
        svc.halt(enrollment, EnrollmentStatus.BOUNCED, reason="hard bounce", now=NOW)
        assert enrollment.status == EnrollmentStatus.BOUNCED
        assert enrollment.next_send_at is None
        assert enrollment.paused_reason == "hard bounce"
        assert enrollment.completed_at == NOW

    def test_a_non_terminal_halt_leaves_the_completion_stamp_alone(self) -> None:
        enrollment = SequenceEnrollment(
            organization_id=uuid.uuid4(),
            sequence_id=uuid.uuid4(),
            candidate_id=uuid.uuid4(),
            status=EnrollmentStatus.ACTIVE,
            enrolled_at=NOW,
            next_send_at=NOW,
        )
        svc.halt(enrollment, EnrollmentStatus.PAUSED, now=NOW)
        assert enrollment.completed_at is None


class TestPauseAndResume:
    async def test_pausing_keeps_the_position(self, session, org, campaign) -> None:
        """Resuming should pick the thread up, not restart the candidate."""
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        paused = await svc.pause_enrollment(
            session, org.id, report.enrolled[0].id, reason="recruiter asked"
        )
        assert paused.status == EnrollmentStatus.PAUSED
        assert paused.paused_reason == "recruiter asked"
        assert paused.next_send_at == NOW

    async def test_a_finished_enrollment_cannot_be_paused(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await svc.unsubscribe(session, org.id, report.enrolled[0].id, now=NOW)
        with pytest.raises(ConflictError):
            await svc.pause_enrollment(session, org.id, report.enrolled[0].id)

    async def test_resuming_a_step_whose_time_passed_makes_it_due_now(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        enrollment_id = report.enrolled[0].id
        await svc.pause_enrollment(session, org.id, enrollment_id)

        later = NOW + timedelta(days=3)  # Thursday midday, inside the window
        resumed = await svc.resume_enrollment(session, org.id, enrollment_id, now=later)
        assert resumed.status == EnrollmentStatus.ACTIVE
        assert resumed.paused_reason is None
        assert resumed.next_send_at == later

    async def test_resuming_before_the_step_was_due_keeps_the_schedule(
        self, session, org, campaign
    ) -> None:
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        night = datetime(2027, 3, 8, 23, 0, tzinfo=UTC)
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [candidate.id], now=night
        )
        enrollment_id = report.enrolled[0].id
        await svc.pause_enrollment(session, org.id, enrollment_id)

        resumed = await svc.resume_enrollment(session, org.id, enrollment_id, now=night)
        assert resumed.next_send_at == datetime(2027, 3, 9, 9, 0, tzinfo=UTC)

    async def test_only_a_paused_enrollment_can_be_resumed(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        with pytest.raises(ConflictError):
            await svc.resume_enrollment(session, org.id, report.enrolled[0].id, now=NOW)

    async def test_resuming_past_the_last_step_completes_instead(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        enrollment = report.enrolled[0]
        enrollment.current_step = 99
        await svc.pause_enrollment(session, org.id, enrollment.id)

        resumed = await svc.resume_enrollment(session, org.id, enrollment.id, now=NOW)
        assert resumed.status == EnrollmentStatus.COMPLETED
        assert resumed.next_send_at is None


class TestRecordReply:
    async def test_a_reply_stops_the_remaining_steps(
        self, session, org, campaign
    ) -> None:
        """Nothing reads as more automated than a follow-up after a real reply."""
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        replied = await svc.record_reply(
            session, org.id, report.enrolled[0].id, now=NOW
        )
        assert replied.status == EnrollmentStatus.REPLIED
        assert replied.replied_at == NOW
        assert replied.next_send_at is None

    async def test_a_sequence_that_keeps_going_records_but_does_not_stop(
        self, session, org
    ) -> None:
        sequence = await with_steps(session, org, 2, stop_on_reply=False)
        await svc.activate(session, org.id, sequence.id, now=NOW)
        candidate = await make_candidate(session, org)
        await grant(session, org, candidate)
        report = await svc.enroll(session, org.id, sequence.id, [candidate.id], now=NOW)

        replied = await svc.record_reply(
            session, org.id, report.enrolled[0].id, now=NOW
        )
        assert replied.status == EnrollmentStatus.ACTIVE
        assert replied.replied_at == NOW
        assert replied.next_send_at == NOW

    async def test_a_reply_bumps_the_counter(self, session, org, campaign) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await svc.record_reply(session, org.id, report.enrolled[0].id, now=NOW)
        sequence = await svc.get_sequence(session, org.id, campaign["sequence"].id)
        assert sequence.stats_json["replied"] == 1


class TestUnsubscribe:
    async def test_unsubscribing_ends_the_enrollment(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        gone = await svc.unsubscribe(session, org.id, report.enrolled[0].id, now=NOW)
        assert gone.status == EnrollmentStatus.UNSUBSCRIBED
        assert gone.next_send_at is None
        assert gone.completed_at == NOW

    async def test_an_enrollment_from_another_org_is_not_found(
        self, session, org, other_org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        with pytest.raises(NotFoundError):
            await svc.get_enrollment(session, other_org.id, report.enrolled[0].id)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
class TestBumpStats:
    def test_the_dict_is_replaced_so_sqlalchemy_notices(self) -> None:
        """An in-place edit of a JSON column does not mark the row dirty."""
        sequence = sequence_stub(stats_json={"sent": 1})
        before = sequence.stats_json
        svc.bump_stats(sequence, "sent")
        assert sequence.stats_json is not before
        assert sequence.stats_json["sent"] == 2

    def test_an_unseen_key_starts_at_zero(self) -> None:
        sequence = sequence_stub(stats_json={})
        assert svc.bump_stats(sequence, "opened")["opened"] == 1

    def test_a_missing_dict_is_tolerated(self) -> None:
        sequence = sequence_stub(stats_json=None)
        assert svc.bump_stats(sequence, "sent", 3)["sent"] == 3


class TestSequenceStats:
    async def _message(self, session, org, enrollment, status: MessageStatus):
        message = OutreachMessage(
            organization_id=org.id,
            enrollment_id=enrollment.id,
            candidate_id=enrollment.candidate_id,
            status=status,
            to_address="someone@example.test",
        )
        session.add(message)
        await session.commit()
        return message

    async def test_an_untouched_sequence_reports_zeroes(
        self, session, org, campaign
    ) -> None:
        stats = await svc.sequence_stats(session, org.id, campaign["sequence"].id)
        assert stats["enrollments_total"] == 0
        assert stats["sent"] == 0
        assert stats["messages"] == {}

    async def test_enrollments_are_counted_by_status(
        self, session, org, campaign
    ) -> None:
        extra = await make_candidate(session, org)
        await grant(session, org, extra)
        report = await svc.enroll(
            session,
            org.id,
            campaign["sequence"].id,
            [campaign["candidate"].id, extra.id],
            now=NOW,
        )
        await svc.unsubscribe(session, org.id, report.enrolled[1].id, now=NOW)

        stats = await svc.sequence_stats(session, org.id, campaign["sequence"].id)
        assert stats["enrollments"] == {"active": 1, "unsubscribed": 1}
        assert stats["enrollments_total"] == 2

    async def test_anything_that_left_the_box_counts_as_sent(
        self, session, org, campaign
    ) -> None:
        """Opened and replied are sent messages too; they must not be lost."""
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        enrollment = report.enrolled[0]
        for status in (
            MessageStatus.SENT,
            MessageStatus.DELIVERED,
            MessageStatus.OPENED,
            MessageStatus.CLICKED,
            MessageStatus.REPLIED,
            MessageStatus.BOUNCED,
            MessageStatus.FAILED,
            MessageStatus.QUEUED,
        ):
            await self._message(session, org, enrollment, status)

        stats = await svc.sequence_stats(session, org.id, campaign["sequence"].id)
        assert stats["sent"] == 5
        assert stats["bounced"] == 1
        assert stats["failed"] == 1
        assert stats["queued"] == 1

    async def test_another_sequences_messages_are_not_counted(
        self, session, org, campaign
    ) -> None:
        report = await svc.enroll(
            session, org.id, campaign["sequence"].id, [campaign["candidate"].id], now=NOW
        )
        await self._message(session, org, report.enrolled[0], MessageStatus.SENT)

        rival = await with_steps(session, org, 1, name="Other campaign")
        await svc.activate(session, org.id, rival.id, now=NOW)
        other_candidate = await make_candidate(session, org)
        await grant(session, org, other_candidate)
        other = await svc.enroll(
            session, org.id, rival.id, [other_candidate.id], now=NOW
        )
        await self._message(session, org, other.enrolled[0], MessageStatus.SENT)

        stats = await svc.sequence_stats(session, org.id, campaign["sequence"].id)
        assert stats["sent"] == 1

    async def test_stats_for_another_orgs_sequence_are_not_found(
        self, session, other_org, campaign
    ) -> None:
        with pytest.raises(NotFoundError):
            await svc.sequence_stats(session, other_org.id, campaign["sequence"].id)


def test_now_defaults_to_the_current_moment() -> None:
    assert svc._now(NOW) == NOW
    assert (datetime.now(UTC) - svc._now()).total_seconds() < 5
