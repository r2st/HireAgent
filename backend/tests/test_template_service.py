"""Message templates and ``{{variable}}`` rendering (design §4.2).

The behaviour worth defending: an unresolved placeholder is reported, never
quietly blanked. "Hi {{first_name}}" reaching a real candidate is the single
worst outcome of a personalization feature, so every test below is ultimately
about making that impossible to do by accident.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.models.candidate import Candidate
from app.models.enums import EmploymentType, OutreachChannel
from app.models.job import Job
from app.models.organization import Organization
from app.models.outreach import EmailAccount, MessageTemplate
from app.services import template_service as svc


async def make_org(session, name: str = "Acme") -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    session.add(org)
    await session.commit()
    return org


# --------------------------------------------------------------------------- #
# Variable discovery
# --------------------------------------------------------------------------- #
class TestFindVariables:
    def test_it_lists_variables_in_first_appearance_order(self) -> None:
        text = "Hi {{first_name}}, about {{job_title}} — {{first_name}} again"
        assert svc.find_variables(text) == ["first_name", "job_title"]

    def test_whitespace_inside_the_braces_is_tolerated(self) -> None:
        assert svc.find_variables("{{  first_name  }}") == ["first_name"]

    def test_a_fallback_does_not_change_the_variable_name(self) -> None:
        assert svc.find_variables("{{first_name|there}}") == ["first_name"]

    def test_prose_braces_are_not_variables(self) -> None:
        # A single brace is ordinary text; only the doubled form is a token.
        assert svc.find_variables("use { this } literally") == []

    def test_a_name_starting_with_a_digit_is_not_a_variable(self) -> None:
        assert svc.find_variables("{{2days}}") == []

    def test_empty_text_has_no_variables(self) -> None:
        assert svc.find_variables(None) == []
        assert svc.find_variables("") == []


class TestMalformedPlaceholders:
    @pytest.mark.parametrize(
        "text",
        [
            "Hi {{first_name}",
            "Hi {first_name}}",
            "Hi {{2days}} left",
            "Hi {{ }}",
            "{{unclosed",
        ],
    )
    def test_broken_braces_are_flagged(self, text: str) -> None:
        assert svc.has_malformed_placeholder(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Hi {{first_name}}",
            "Hi {{first_name|there}}, we use { braces } in prose",
            "No placeholders at all",
            "",
        ],
    )
    def test_well_formed_text_is_not_flagged(self, text: str) -> None:
        assert svc.has_malformed_placeholder(text) is False


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class TestRender:
    def test_it_substitutes_a_known_variable(self) -> None:
        result = svc.render("Hi {{first_name}}", {"first_name": "Ada"})
        assert result.text == "Hi Ada"
        assert result.ok is True
        assert result.used == ["first_name"]

    def test_a_missing_variable_is_reported_not_blanked(self) -> None:
        result = svc.render("Hi {{first_name}}", {})
        assert result.missing == ["first_name"]
        assert result.ok is False

    def test_a_missing_variable_keeps_its_placeholder_for_the_preview(self) -> None:
        # Leaving the token in place shows the author which one failed, and
        # where — a blank would just look like a spacing bug.
        assert svc.render("Hi {{first_name}}!", {}).text == "Hi {{first_name}}!"

    def test_a_fallback_covers_an_absent_value(self) -> None:
        result = svc.render("Hi {{first_name|there}}", {})
        assert result.text == "Hi there"
        assert result.ok is True

    def test_a_present_value_wins_over_its_fallback(self) -> None:
        assert svc.render("Hi {{n|there}}", {"n": "Ada"}).text == "Hi Ada"

    def test_an_empty_fallback_is_an_explicit_blank(self) -> None:
        result = svc.render("Hi{{suffix|}}", {})
        assert result.text == "Hi"
        assert result.ok is True

    def test_an_empty_value_takes_the_fallback(self) -> None:
        # A candidate row with current_company = "" should read as absent, not
        # leave a hole in the middle of a sentence.
        result = svc.render("at {{company|your current role}}", {"company": "   "})
        assert result.text == "at your current role"

    def test_an_empty_value_without_a_fallback_is_missing(self) -> None:
        assert svc.render("at {{company}}", {"company": ""}).missing == ["company"]

    def test_a_none_value_is_missing(self) -> None:
        assert svc.render("at {{company}}", {"company": None}).missing == ["company"]

    def test_a_numeric_value_renders(self) -> None:
        assert svc.render("{{years}} years", {"years": 7}).text == "7 years"

    def test_a_zero_renders_rather_than_reading_as_absent(self) -> None:
        assert svc.render("{{n}} left", {"n": 0}).text == "0 left"

    def test_a_boolean_is_treated_as_absent(self) -> None:
        # "True" is never the word a template wanted in a sentence.
        assert svc.render("{{flag}}", {"flag": True}).missing == ["flag"]

    def test_each_missing_variable_is_reported_once(self) -> None:
        result = svc.render("{{a}} {{a}} {{b}}", {})
        assert result.missing == ["a", "b"]

    def test_a_repeated_variable_is_substituted_everywhere(self) -> None:
        assert svc.render("{{n}}/{{n}}", {"n": "x"}).text == "x/x"

    def test_unknown_context_keys_are_ignored(self) -> None:
        assert svc.render("Hi {{a}}", {"a": "A", "unused": "B"}).text == "Hi A"

    def test_a_value_containing_braces_is_not_re_rendered(self) -> None:
        # Otherwise a candidate could inject a placeholder through their own
        # profile and have it filled from the org's context.
        result = svc.render("{{a}}", {"a": "{{secret}}", "secret": "leaked"})
        assert result.text == "{{secret}}"

    def test_the_template_is_returned_unchanged_when_it_has_no_variables(self) -> None:
        assert svc.render("Plain text", {"a": "1"}).text == "Plain text"

    def test_none_renders_as_empty(self) -> None:
        assert svc.render(None, {}).text == ""


class TestHtmlEscaping:
    def test_substituted_values_are_escaped_for_html(self) -> None:
        result = svc.render("<p>{{n}}</p>", {"n": "O'Brien & Sons"}, escape=True)
        assert "&amp;" in result.text
        assert "O'Brien & Sons" not in result.text

    def test_markup_in_a_value_cannot_become_markup(self) -> None:
        result = svc.render("<p>{{n}}</p>", {"n": "<script>x()</script>"}, escape=True)
        assert "<script>" not in result.text
        assert "&lt;script&gt;" in result.text

    def test_the_templates_own_markup_is_left_alone(self) -> None:
        result = svc.render("<b>{{n}}</b>", {"n": "Ada"}, escape=True)
        assert result.text == "<b>Ada</b>"

    def test_a_fallback_is_escaped_too(self) -> None:
        result = svc.render("{{n|A & B}}", {}, escape=True)
        assert result.text == "A &amp; B"

    def test_the_text_part_is_not_escaped(self) -> None:
        assert svc.render("{{n}}", {"n": "A & B"}).text == "A & B"


class TestRenderMessage:
    def test_it_renders_every_part(self) -> None:
        message = svc.render_message(
            subject="Role at {{company_name}}",
            body="Hi {{first_name}}",
            body_html="<p>Hi {{first_name}}</p>",
            context={"company_name": "Acme", "first_name": "Ada"},
        )
        assert message.subject == "Role at Acme"
        assert message.body == "Hi Ada"
        assert message.body_html == "<p>Hi Ada</p>"
        assert message.ok is True

    def test_missing_variables_are_pooled_across_parts(self) -> None:
        # An author fixing a template should see every broken token at once,
        # not discover them one send at a time.
        message = svc.render_message(
            subject="{{a}}", body="{{b}}", body_html="{{c}}", context={}
        )
        assert message.missing == ["a", "b", "c"]
        assert message.ok is False

    def test_a_variable_missing_from_two_parts_is_reported_once(self) -> None:
        message = svc.render_message(subject="{{a}}", body="{{a}}", context={})
        assert message.missing == ["a"]

    def test_an_absent_html_part_stays_absent(self) -> None:
        message = svc.render_message(subject="s", body="b", context={})
        assert message.body_html is None

    def test_the_html_part_escapes_while_the_text_part_does_not(self) -> None:
        message = svc.render_message(
            subject="s",
            body="{{n}}",
            body_html="<p>{{n}}</p>",
            context={"n": "A & B"},
        )
        assert message.body == "A & B"
        assert message.body_html == "<p>A &amp; B</p>"


# --------------------------------------------------------------------------- #
# Context building
# --------------------------------------------------------------------------- #
class TestBuildContext:
    def candidate(self, **fields) -> Candidate:
        defaults = dict(
            organization_id=uuid.uuid4(),
            full_name="Ada Lovelace",
            email="ada@example.com",
            email_index="idx",
            current_company="Analytical Engines",
            current_role="Mathematician",
            location="London",
            experience_years=12,
        )
        defaults.update(fields)
        return Candidate(**defaults)

    def test_it_derives_a_first_name(self) -> None:
        context = svc.build_context(candidate=self.candidate())
        assert context["first_name"] == "Ada"
        assert context["candidate_name"] == "Ada Lovelace"

    def test_a_single_word_name_is_its_own_first_name(self) -> None:
        context = svc.build_context(candidate=self.candidate(full_name="Prince"))
        assert context["first_name"] == "Prince"

    def test_a_blank_name_yields_no_first_name(self) -> None:
        # None rather than "", so a {{first_name|there}} fallback fires.
        context = svc.build_context(candidate=self.candidate(full_name="  "))
        assert context["first_name"] is None
        assert svc.render("Hi {{first_name|there}}", context).text == "Hi there"

    def test_job_fields_are_exposed(self) -> None:
        job = Job(
            organization_id=uuid.uuid4(),
            title="Staff Engineer",
            slug="staff-engineer",
            location="Remote",
            department="Platform",
            employment_type=EmploymentType.FULL_TIME,
        )
        context = svc.build_context(job=job)
        assert context["job_title"] == "Staff Engineer"
        assert context["job_location"] == "Remote"
        assert context["job_department"] == "Platform"

    def test_organization_and_sender_fields_are_exposed(self) -> None:
        org = Organization(name="Acme Talent", slug="acme")
        sender = EmailAccount(
            organization_id=uuid.uuid4(),
            email="talent@acme.test",
            display_name="Acme Talent Team",
        )
        context = svc.build_context(organization=org, sender=sender)
        assert context["company_name"] == "Acme Talent"
        assert context["sender_name"] == "Acme Talent Team"
        assert context["sender_email"] == "talent@acme.test"

    def test_a_sender_without_a_display_name_falls_back_to_its_address(self) -> None:
        sender = EmailAccount(
            organization_id=uuid.uuid4(), email="talent@acme.test", display_name=None
        )
        assert svc.build_context(sender=sender)["sender_name"] == "talent@acme.test"

    def test_extra_values_override_derived_ones(self) -> None:
        # The dispatcher injects per-message values (unsubscribe links, booking
        # URLs) that nothing else can know.
        context = svc.build_context(
            candidate=self.candidate(), extra={"first_name": "Countess"}
        )
        assert context["first_name"] == "Countess"

    def test_everything_is_optional(self) -> None:
        assert svc.build_context() == {}


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
class TestCreateTemplate:
    async def test_it_records_the_variables_it_finds(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session,
            org.id,
            name="Intro",
            subject="A role at {{company_name}}",
            body="Hi {{first_name}}",
            body_html="<p>Hi {{first_name}} — {{job_title}}</p>",
        )
        assert template.variables == ["company_name", "first_name", "job_title"]

    async def test_a_nameless_template_is_rejected(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError, match="name"):
            await svc.create_template(session, org.id, name="  ", body="Hi")

    async def test_an_empty_body_is_rejected(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError, match="body"):
            await svc.create_template(session, org.id, name="Intro", body="   ")

    async def test_an_email_template_needs_a_subject(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError, match="subject"):
            await svc.create_template(session, org.id, name="Intro", body="Hi")

    async def test_a_whatsapp_template_needs_no_subject(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session,
            org.id,
            name="Confirm",
            body="Your interview is confirmed",
            channel=OutreachChannel.WHATSAPP,
        )
        assert template.subject is None

    async def test_a_malformed_placeholder_is_rejected(self, session) -> None:
        # Better a 422 at authoring time than raw braces in a candidate's inbox.
        org = await make_org(session)
        with pytest.raises(ValidationError, match="placeholder"):
            await svc.create_template(
                session, org.id, name="Intro", subject="Hi", body="Hi {{first_name}"
            )

    async def test_an_overlong_subject_is_rejected(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(ValidationError, match="characters"):
            await svc.create_template(
                session,
                org.id,
                name="Intro",
                subject="x" * (svc.MAX_SUBJECT_LENGTH + 1),
                body="Hi",
            )

    async def test_a_duplicate_name_on_the_same_channel_conflicts(
        self, session
    ) -> None:
        org = await make_org(session)
        await svc.create_template(session, org.id, name="Intro", subject="s", body="b")
        with pytest.raises(ConflictError):
            await svc.create_template(
                session, org.id, name="Intro", subject="s", body="b"
            )

    async def test_the_same_name_may_exist_on_another_channel(self, session) -> None:
        org = await make_org(session)
        await svc.create_template(session, org.id, name="Intro", subject="s", body="b")
        other = await svc.create_template(
            session,
            org.id,
            name="Intro",
            body="b",
            channel=OutreachChannel.WHATSAPP,
        )
        assert other.channel == OutreachChannel.WHATSAPP

    async def test_the_same_name_may_exist_in_another_tenant(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        for org in (mine, theirs):
            await svc.create_template(
                session, org.id, name="Intro", subject="s", body="b"
            )
        assert len(await svc.list_templates(session, mine.id)) == 1


@pytest.mark.anyio
class TestTemplateLookup:
    async def test_a_missing_template_is_not_found(self, session) -> None:
        org = await make_org(session)
        with pytest.raises(NotFoundError):
            await svc.get_template(session, org.id, uuid.uuid4())

    async def test_another_tenants_template_is_not_found(self, session) -> None:
        mine = await make_org(session, "Mine")
        theirs = await make_org(session, "Theirs")
        template = await svc.create_template(
            session, theirs.id, name="Intro", subject="s", body="b"
        )
        with pytest.raises(NotFoundError):
            await svc.get_template(session, mine.id, template.id)

    async def test_listing_is_alphabetical(self, session) -> None:
        org = await make_org(session)
        for name in ("Zeta", "Alpha", "Mu"):
            await svc.create_template(session, org.id, name=name, subject="s", body="b")
        listed = await svc.list_templates(session, org.id)
        assert [t.name for t in listed] == ["Alpha", "Mu", "Zeta"]

    async def test_listing_can_filter_by_channel(self, session) -> None:
        org = await make_org(session)
        await svc.create_template(session, org.id, name="Mail", subject="s", body="b")
        await svc.create_template(
            session, org.id, name="Chat", body="b", channel=OutreachChannel.WHATSAPP
        )
        listed = await svc.list_templates(
            session, org.id, channel=OutreachChannel.WHATSAPP
        )
        assert [t.name for t in listed] == ["Chat"]

    async def test_listing_can_hide_inactive_templates(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Old", subject="s", body="b"
        )
        await svc.update_template(
            session, org.id, template.id, changes={"is_active": False}
        )
        assert await svc.list_templates(session, org.id, active_only=True) == []
        assert len(await svc.list_templates(session, org.id)) == 1


@pytest.mark.anyio
class TestUpdateTemplate:
    async def test_editing_the_body_re_derives_the_variables(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="s", body="Hi {{first_name}}"
        )
        updated = await svc.update_template(
            session, org.id, template.id, changes={"body": "Hi {{candidate_name}}"}
        )
        assert updated.variables == ["candidate_name"]

    async def test_none_leaves_a_field_alone(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="Keep", body="b"
        )
        updated = await svc.update_template(
            session, org.id, template.id, changes={"subject": None}
        )
        assert updated.subject == "Keep"

    async def test_an_edit_that_breaks_a_placeholder_is_rejected(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="s", body="b"
        )
        with pytest.raises(ValidationError, match="placeholder"):
            await svc.update_template(
                session, org.id, template.id, changes={"body": "Hi {{name}"}
            )

    async def test_an_edit_that_empties_the_body_is_rejected(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="s", body="b"
        )
        with pytest.raises(ValidationError, match="body"):
            await svc.update_template(
                session, org.id, template.id, changes={"body": "   "}
            )


@pytest.mark.anyio
class TestDeleteTemplate:
    async def test_deletion_is_soft_and_deactivates(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="s", body="b"
        )
        await svc.delete_template(session, org.id, template.id)
        assert template.deleted_at is not None
        assert template.is_active is False
        assert await svc.list_templates(session, org.id) == []

    async def test_a_deleted_template_cannot_be_fetched(self, session) -> None:
        org = await make_org(session)
        template = await svc.create_template(
            session, org.id, name="Intro", subject="s", body="b"
        )
        await svc.delete_template(session, org.id, template.id)
        with pytest.raises(NotFoundError):
            await svc.get_template(session, org.id, template.id)


def test_preview_reports_what_a_sample_context_cannot_fill() -> None:
    template = MessageTemplate(
        organization_id=uuid.uuid4(),
        name="Intro",
        subject="Role at {{company_name}}",
        body="Hi {{first_name}}, about {{job_title}}",
        created_at=datetime.now(UTC),
    )
    rendered = svc.preview(template, {"company_name": "Acme", "first_name": "Ada"})
    assert rendered.subject == "Role at Acme"
    assert rendered.missing == ["job_title"]
