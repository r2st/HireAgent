"""Resume parsing: the heuristic engine, the LLM engine, and their handoff.

The heuristic engine is not a stub — it is what runs whenever the free-tier
model is rate-limited — so it is tested to the same standard as the LLM path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.resume_parser import (
    ParsedResume,
    _confidence,
    _split_role_company,
    _years_from_experience,
    parse_heuristic,
    parse_resume,
)
from tests.factories import (
    ANONYMOUS_RESUME,
    LLM_RESUME_PAYLOAD,
    MINIMAL_RESUME,
    SAMPLE_RESUME,
    FakeLLMClient,
)


@pytest.fixture
def parsed() -> ParsedResume:
    return parse_heuristic(SAMPLE_RESUME)


class TestHeuristicContactExtraction:
    def test_extracts_name_from_the_header(self, parsed):
        assert parsed.full_name == "Ada Lovelace"

    def test_extracts_email(self, parsed):
        assert parsed.email == "ada.lovelace@example.com"

    def test_extracts_phone(self, parsed):
        assert parsed.phone is not None
        assert "98765" in parsed.phone

    def test_extracts_profile_urls(self, parsed):
        assert parsed.linkedin_url == "https://linkedin.com/in/adalovelace"
        assert parsed.github_url == "https://github.com/adalovelace"
        # The portfolio is the first URL that is neither of the above.
        assert parsed.portfolio_url == "https://ada.dev"

    def test_falls_back_to_the_email_local_part_for_a_name(self):
        # No plausible name line, but "ada.lovelace@" still identifies them.
        text = "Contact: ada.lovelace@example.com\n" + MINIMAL_RESUME.split("\n", 1)[1]
        assert parse_heuristic(text).full_name == "Ada Lovelace"

    def test_does_not_mistake_a_year_range_for_a_phone_number(self):
        text = "Ada Lovelace\nada@example.com\n\nEducation\nB.Tech, IIT, 2013 - 2017\n"
        assert parse_heuristic(text).phone is None

    def test_ignores_section_headings_when_guessing_the_name(self):
        text = "Professional Summary\nSenior engineer.\nada@example.com\n"
        assert parse_heuristic(text).full_name != "Professional Summary"


class TestHeuristicSections:
    def test_extracts_skills_from_the_skills_section(self, parsed):
        names = parsed.skill_names
        assert "Python" in names
        assert "PostgreSQL" in names
        assert "Kubernetes" in names
        # "ReactJS" in the source text is normalised by the taxonomy.
        assert "React" in names
        assert "ReactJS" not in names

    def test_scans_the_whole_document_when_there_is_no_skills_section(self):
        text = (
            "Grace Hopper\ngrace@example.com\n\n"
            "I have shipped production Python services backed by PostgreSQL "
            "and deployed them with Docker for the last several years.\n"
        )
        names = parse_heuristic(text).skill_names
        assert {"Python", "PostgreSQL", "Docker"} <= set(names)

    def test_extracts_experience_entries(self, parsed):
        assert len(parsed.experience) == 2
        latest = parsed.experience[0]
        assert latest["role"] == "Staff Engineer"
        assert latest["company"] == "Analytical Engines"
        assert latest["start_date"] == "2021-01"
        assert latest["is_current"] is True
        assert latest["end_date"] is None

    def test_experience_is_ordered_most_recent_first(self, parsed):
        starts = [e["start_date"] for e in parsed.experience]
        assert starts == sorted(starts, reverse=True)

    def test_derives_the_current_role_from_the_timeline(self, parsed):
        assert parsed.current_role == "Staff Engineer"
        assert parsed.current_company == "Analytical Engines"

    def test_extracts_education(self, parsed):
        assert parsed.education
        degree = parsed.education[0]
        assert degree["degree"] == "B.TECH"
        assert degree["start_year"] == 2013
        assert degree["end_year"] == 2017

    def test_extracts_certifications(self, parsed):
        assert [c["name"] for c in parsed.certifications] == [
            "AWS Certified Solutions Architect"
        ]

    def test_prefers_the_stated_years_of_experience(self, parsed):
        # The summary says "8 years of professional experience"; that beats
        # anything inferred from the date ranges.
        assert parsed.total_experience_years == 8.0

    def test_reports_the_engine_used(self, parsed):
        assert parsed.engine == "heuristic"
        assert parsed.model is None


class TestSplitRoleCompany:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("Staff Engineer at Analytical Engines", ("Staff Engineer", "Analytical Engines")),
            ("Staff Engineer | Analytical Engines", ("Staff Engineer", "Analytical Engines")),
            ("Staff Engineer — Analytical Engines", ("Staff Engineer", "Analytical Engines")),
            ("Staff Engineer, Analytical Engines", ("Staff Engineer", "Analytical Engines")),
            ("Staff Engineer - Analytical Engines", ("Staff Engineer", "Analytical Engines")),
        ],
    )
    def test_splits_on_common_separators(self, header, expected):
        assert _split_role_company(header) == expected

    def test_unsplittable_header_becomes_the_role(self):
        assert _split_role_company("Consultant") == ("Consultant", None)

    def test_empty_header(self):
        assert _split_role_company("") == (None, None)


class TestYearsFromExperience:
    def test_sums_sequential_roles(self):
        years = _years_from_experience(
            [
                {"start_date": "2015-01", "end_date": "2018-01", "is_current": False},
                {"start_date": "2019-01", "end_date": "2021-01", "is_current": False},
            ]
        )
        # 3 years + 2 years; the 2018-2019 gap is excluded, not penalised.
        assert years == pytest.approx(5.0, abs=0.1)

    def test_merges_overlapping_roles_instead_of_double_counting(self):
        years = _years_from_experience(
            [
                {"start_date": "2015-01", "end_date": "2020-01", "is_current": False},
                # A concurrent contract inside the same window.
                {"start_date": "2017-01", "end_date": "2018-01", "is_current": False},
            ]
        )
        assert years == pytest.approx(5.0, abs=0.1)

    def test_current_roles_run_to_today(self):
        start_year = datetime.now(UTC).year - 3
        years = _years_from_experience(
            [{"start_date": f"{start_year}-01", "end_date": None, "is_current": True}]
        )
        # January of (this year - 3) to today is between 3 and 4 years,
        # depending on how far into the current year "today" falls.
        assert years is not None
        assert 3.0 <= years < 4.05

    def test_ignores_entries_without_a_start_date(self):
        assert _years_from_experience([{"company": "Acme", "role": "Engineer"}]) is None

    def test_ignores_reversed_date_ranges(self):
        assert (
            _years_from_experience(
                [{"start_date": "2020-01", "end_date": "2015-01", "is_current": False}]
            )
            is None
        )

    def test_no_experience(self):
        assert _years_from_experience([]) is None


class TestConfidence:
    def test_a_complete_parse_scores_above_the_review_threshold(self, parsed):
        assert parsed.confidence >= 0.70

    def test_a_sparse_parse_is_flagged_for_review(self):
        sparse = parse_heuristic(MINIMAL_RESUME)
        assert sparse.confidence < 0.70

    def test_an_empty_parse_scores_zero(self):
        assert _confidence(ParsedResume()) == 0.0

    def test_confidence_never_exceeds_one(self):
        full = ParsedResume(
            full_name="A",
            email="a@b.com",
            phone="1",
            skills=[{"name": "Python"}],
            experience=[{"company": "X"}],
            education=[{"institution": "Y"}],
            total_experience_years=1.0,
            current_role="Engineer",
        )
        assert _confidence(full) == 1.0


class TestParseResumeDispatch:
    async def test_uses_the_heuristic_engine_when_the_llm_is_unavailable(self):
        client = FakeLLMClient(available=False)
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "heuristic"
        assert client.calls == []

    async def test_uses_the_llm_when_available(self):
        client = FakeLLMClient([LLM_RESUME_PAYLOAD])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "llm"
        assert result.model == "fake/model:free"
        assert result.current_role == "Staff Engineer"
        assert result.total_experience_years == 8.0

    async def test_llm_skills_are_run_through_the_taxonomy(self):
        client = FakeLLMClient([LLM_RESUME_PAYLOAD])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        names = result.skill_names
        # "python" -> "Python", "postgres" -> "PostgreSQL", "ReactJS" -> "React".
        assert {"Python", "PostgreSQL", "React", "Django"} <= set(names)

    async def test_falls_back_to_heuristics_when_the_model_returns_nothing(self):
        client = FakeLLMClient([None])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "heuristic"
        assert result.email == "ada.lovelace@example.com"

    async def test_falls_back_when_the_model_returns_unparseable_prose(self):
        client = FakeLLMClient(["I'm sorry, I can't help with that."])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "heuristic"

    async def test_tolerates_a_fenced_json_response(self):
        # Free-tier models routinely wrap JSON in a markdown fence.
        import json

        fenced = "Here you go:\n```json\n" + json.dumps(LLM_RESUME_PAYLOAD) + "\n```"
        client = FakeLLMClient([fenced])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "llm"
        assert result.full_name == "Ada Lovelace"

    async def test_backfills_contact_fields_the_model_dropped(self):
        # Free models often omit contact details a regex finds reliably.
        partial = {**LLM_RESUME_PAYLOAD, "email": None, "phone": None, "github_url": None}
        client = FakeLLMClient([partial])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.engine == "llm"
        assert result.email == "ada.lovelace@example.com"
        assert result.github_url == "https://github.com/adalovelace"

    async def test_does_not_overwrite_fields_the_model_did_supply(self):
        payload = {**LLM_RESUME_PAYLOAD, "location": "Remote (EU)"}
        client = FakeLLMClient([payload])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        assert result.location == "Remote (EU)"

    async def test_rejects_impossible_experience_totals(self):
        payload = {**LLM_RESUME_PAYLOAD, "total_experience_years": 400}
        client = FakeLLMClient([payload])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        # 400 years is discarded and the value is re-derived from the timeline.
        assert result.total_experience_years is not None
        assert result.total_experience_years < 60

    async def test_drops_hallucinated_experience_entries_with_no_company_or_role(self):
        payload = {
            **LLM_RESUME_PAYLOAD,
            "experience": [{"start_date": "2020-01", "is_current": True}],
        }
        client = FakeLLMClient([payload])
        result = await parse_resume(SAMPLE_RESUME, client=client)
        # The empty entry is dropped, so the heuristic timeline backfills.
        assert all(e.get("company") or e.get("role") for e in result.experience)

    async def test_empty_input_returns_an_empty_parse(self):
        result = await parse_resume("   ")
        assert result.confidence == 0.0
        assert result.full_name is None

    async def test_anonymous_resume_yields_no_email(self):
        result = await parse_resume(ANONYMOUS_RESUME)
        assert result.email is None

    async def test_the_prompt_forbids_inferring_protected_characteristics(self):
        client = FakeLLMClient([LLM_RESUME_PAYLOAD])
        await parse_resume(SAMPLE_RESUME, client=client)
        system = client.calls[0]["system"]
        assert "protected characteristic" in system.lower()

    async def test_long_resumes_are_truncated_before_the_model_call(self):
        client = FakeLLMClient([LLM_RESUME_PAYLOAD])
        await parse_resume("A" * 60_000, client=client)
        # The free-tier context window must not be blown by a 40-page CV.
        assert len(client.calls[0]["prompt"]) < 30_000
