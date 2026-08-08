"""Candidate scoring: component scorers, weighting, confidence, and the LLM path."""

from __future__ import annotations

import pytest

from app.models.job import DEFAULT_SCORING_WEIGHTS, Job
from app.services.scoring import (
    CONFIDENCE_THRESHOLD,
    UNSCORABLE,
    CandidateProfile,
    score_candidate,
    score_cultural,
    score_education,
    score_experience,
    score_skills,
)
from tests.factories import FakeLLMClient

REQUIREMENTS = {
    "required_skills": [
        {"name": "Python", "weight": 3, "min_years": 4},
        {"name": "PostgreSQL", "weight": 2},
    ],
    "preferred_skills": [{"name": "Kubernetes", "weight": 1}],
    "education": {"degree": "B.Tech", "field_of_study": "Computer Science"},
}


def make_job(**overrides) -> Job:
    """An unsaved Job carrying just what the scorer reads."""
    job = Job(
        title="Senior Backend Engineer",
        slug="senior-backend-engineer",
        seniority="senior",
        min_experience_years=4,
        max_experience_years=9,
        requirements_json=dict(REQUIREMENTS),
        scoring_weights=dict(DEFAULT_SCORING_WEIGHTS),
    )
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


def make_profile(**overrides) -> CandidateProfile:
    profile = CandidateProfile(
        skills=["Python", "PostgreSQL", "Kubernetes"],
        experience_years=6.0,
        education=[
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
            }
        ],
        experience=[
            {"company": "Acme", "role": "Staff Engineer", "start_date": "2021-01", "is_current": True},
            {
                "company": "Babbage",
                "role": "Senior Engineer",
                "start_date": "2017-03",
                "end_date": "2020-12",
            },
        ],
        current_role="Staff Engineer",
        current_company="Acme",
        has_parsed_resume=True,
    )
    for key, value in overrides.items():
        setattr(profile, key, value)
    return profile


class TestScoreSkills:
    def test_all_required_skills_present_scores_full(self):
        score, matched, missing = score_skills(
            ["Python", "PostgreSQL"],
            {"required_skills": REQUIREMENTS["required_skills"]},
        )
        assert score == 100.0
        assert set(matched) == {"Python", "PostgreSQL"}
        assert missing == []

    def test_matching_is_taxonomy_aware(self):
        # The candidate wrote "Postgres"; the job asked for "PostgreSQL".
        score, _, missing = score_skills(
            ["python", "Postgres"],
            {"required_skills": REQUIREMENTS["required_skills"]},
        )
        assert score == 100.0
        assert missing == []

    def test_weights_make_some_skills_matter_more(self):
        required = REQUIREMENTS["required_skills"]  # Python 3, PostgreSQL 2
        with_python, _, _ = score_skills(["Python"], {"required_skills": required})
        with_postgres, _, _ = score_skills(["PostgreSQL"], {"required_skills": required})
        assert with_python == pytest.approx(60.0)
        assert with_postgres == pytest.approx(40.0)

    def test_missing_skills_are_named(self):
        _, matched, missing = score_skills(
            ["Python"], {"required_skills": REQUIREMENTS["required_skills"]}
        )
        assert matched == ["Python"]
        assert missing == ["PostgreSQL"]

    def test_insufficient_depth_earns_partial_credit(self):
        # Has Python, but 2 years against a 4-year requirement.
        shallow, _, missing = score_skills(
            [{"name": "Python", "years": 2}, {"name": "PostgreSQL"}],
            {"required_skills": REQUIREMENTS["required_skills"]},
        )
        deep, _, _ = score_skills(
            [{"name": "Python", "years": 6}, {"name": "PostgreSQL"}],
            {"required_skills": REQUIREMENTS["required_skills"]},
        )
        assert shallow < deep == 100.0
        # Partial credit, not a miss: they still have the skill.
        assert missing == []
        assert shallow > 60.0

    def test_preferred_skills_add_a_capped_bonus(self):
        without, _, _ = score_skills(
            ["Python"],
            {
                "required_skills": REQUIREMENTS["required_skills"],
                "preferred_skills": REQUIREMENTS["preferred_skills"],
            },
        )
        with_preferred, _, _ = score_skills(
            ["Python", "Kubernetes"],
            {
                "required_skills": REQUIREMENTS["required_skills"],
                "preferred_skills": REQUIREMENTS["preferred_skills"],
            },
        )
        assert with_preferred > without
        # A preferred skill cannot rescue a weak core match.
        assert with_preferred < 100.0

    def test_no_required_skills_is_unscorable_rather_than_zero(self):
        score, matched, missing = score_skills(["Python"], {})
        assert score == UNSCORABLE
        assert matched == [] and missing == []

    def test_candidate_with_no_skills_scores_zero_against_a_specified_job(self):
        score, matched, missing = score_skills(
            [], {"required_skills": REQUIREMENTS["required_skills"]}
        )
        assert score == 0.0
        assert matched == []
        assert missing == ["Python", "PostgreSQL"]

    def test_accepts_plain_string_requirements(self):
        score, _, _ = score_skills(["Python", "Go"], {"required_skills": ["python", "go"]})
        assert score == 100.0

    def test_zero_weight_requirements_are_ignored(self):
        score, matched, _ = score_skills(
            ["Python"],
            {"required_skills": [{"name": "Python"}, {"name": "Cobol", "weight": 0}]},
        )
        assert score == 100.0
        assert matched == ["Python"]


class TestScoreExperience:
    @pytest.mark.parametrize("years", [4.0, 6.0, 9.0])
    def test_inside_the_band_scores_full(self, years):
        assert score_experience(years, 4, 9) == 100.0

    def test_below_the_band_falls_off_proportionally(self):
        assert score_experience(2, 4, 9) == pytest.approx(70.0)
        assert score_experience(0, 4, 9) == pytest.approx(40.0)

    def test_a_promising_junior_is_not_scored_to_zero(self):
        # Zeroing out career changers is exactly what design §4.1 warns against.
        assert score_experience(0.5, 8, None) > 40.0

    def test_above_the_band_is_penalised_gently(self):
        overqualified = score_experience(14, 4, 9)
        underqualified = score_experience(1, 4, 9)
        assert overqualified > underqualified
        # Seniority is a weaker misfit signal than inexperience.
        assert overqualified >= 70.0

    def test_over_the_band_never_drops_below_the_floor(self):
        assert score_experience(60, 4, 9) == 70.0

    def test_unknown_experience_is_unscorable(self):
        assert score_experience(None, 4, 9) == UNSCORABLE

    def test_job_with_no_band_is_unscorable(self):
        assert score_experience(6, None, None) == UNSCORABLE

    def test_zero_minimum_accepts_anyone(self):
        assert score_experience(0, 0, 3) == 100.0


class TestScoreEducation:
    def test_exact_degree_and_field_match(self):
        assert (
            score_education(
                [{"degree": "B.Tech", "field_of_study": "Computer Science"}],
                REQUIREMENTS["education"],
            )
            == 100.0
        )

    def test_degree_only_match_scores_partially(self):
        score = score_education(
            [{"degree": "B.Tech", "field_of_study": "Mechanical Engineering"}],
            REQUIREMENTS["education"],
        )
        assert score == 75.0

    def test_field_folded_into_the_institution_line_still_matches(self):
        # Heuristic parses often produce "Computer Science, IIT" as institution.
        score = score_education(
            [{"institution": "Computer Science, IIT", "degree": "B.Tech"}],
            REQUIREMENTS["education"],
        )
        assert score == 100.0

    def test_missing_mandatory_degree_costs_more_than_a_preference(self):
        mandatory = score_education([], {"degree": "PhD", "is_mandatory": True})
        preferred = score_education([], {"degree": "PhD", "is_mandatory": False})
        assert mandatory < preferred

    def test_a_missing_preference_is_survivable(self):
        # Design §4.1: skill-based matching over credential matching.
        assert score_education([], {"degree": "PhD", "is_mandatory": False}) >= 50.0

    def test_no_requirement_is_unscorable(self):
        assert score_education([{"degree": "B.Tech"}], None) == UNSCORABLE
        assert score_education([{"degree": "B.Tech"}], {}) == UNSCORABLE


class TestScoreCultural:
    def test_long_tenure_scores_above_short_tenure(self):
        stable = score_cultural(
            CandidateProfile(
                experience=[
                    {"start_date": "2015-01", "end_date": "2020-01"},
                    {"start_date": "2020-02", "end_date": "2024-02"},
                ]
            )
        )
        churn = score_cultural(
            CandidateProfile(
                experience=[
                    {"start_date": "2022-01", "end_date": "2022-08"},
                    {"start_date": "2022-09", "end_date": "2023-03"},
                ]
            )
        )
        assert stable > churn

    def test_short_tenure_is_a_dent_not_a_disqualification(self):
        churn = score_cultural(
            CandidateProfile(experience=[{"start_date": "2023-01", "end_date": "2023-06"}])
        )
        assert churn > 50.0

    def test_no_work_history_is_unscorable(self):
        assert score_cultural(CandidateProfile()) == UNSCORABLE

    def test_undated_history_is_unscorable(self):
        assert score_cultural(CandidateProfile(experience=[{"company": "Acme"}])) == UNSCORABLE


class TestScoreCandidate:
    async def test_a_strong_match_scores_high(self):
        result = await score_candidate(make_job(), make_profile())
        assert result.overall_score >= 90.0
        assert result.engine == "heuristic"
        assert result.missing_skills == []

    async def test_a_weak_match_scores_low(self):
        weak = make_profile(
            skills=["Excel"],
            experience_years=0.5,
            education=[{"degree": "Diploma", "field_of_study": "Hospitality"}],
        )
        result = await score_candidate(make_job(), weak)
        assert result.overall_score < 60.0
        assert set(result.missing_skills) == {"Python", "PostgreSQL"}

    async def test_the_overall_score_is_the_weighted_blend(self):
        result = await score_candidate(make_job(), make_profile())
        expected = (
            result.skill_match * 0.40
            + result.experience_score * 0.30
            + result.education_score * 0.15
            + result.cultural_score * 0.15
        )
        assert result.overall_score == pytest.approx(expected, abs=0.02)

    async def test_custom_job_weights_are_applied_and_snapshotted(self):
        skills_only = make_job(
            scoring_weights={"skills": 1.0, "experience": 0, "education": 0, "cultural": 0}
        )
        profile = make_profile(skills=["Python", "PostgreSQL"], experience_years=0.1)
        result = await score_candidate(skills_only, profile)
        # Experience is weighted out entirely, so the skills match carries it.
        assert result.overall_score == pytest.approx(result.skill_match, abs=0.02)
        assert result.weights["skills"] == 1.0

    async def test_weights_are_normalised_before_use(self):
        doubled = make_job(
            scoring_weights={"skills": 0.8, "experience": 0.6, "education": 0.3, "cultural": 0.3}
        )
        result = await score_candidate(doubled, make_profile())
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert 0 <= result.overall_score <= 100

    async def test_scoring_is_deterministic_without_a_model(self):
        job, profile = make_job(), make_profile()
        first = await score_candidate(job, profile)
        second = await score_candidate(job, profile)
        assert first.overall_score == second.overall_score

    async def test_heuristic_results_still_explain_themselves(self):
        result = await score_candidate(make_job(), make_profile(skills=["Python"]))
        assert result.reasoning is not None
        assert "PostgreSQL" in result.reasoning

    async def test_a_thin_profile_is_flagged_for_human_review(self):
        thin = CandidateProfile()
        result = await score_candidate(make_job(), thin)
        assert result.confidence < CONFIDENCE_THRESHOLD
        assert result.requires_human_review is True

    async def test_a_complete_profile_clears_the_review_threshold(self):
        result = await score_candidate(make_job(), make_profile())
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.requires_human_review is False

    async def test_an_unspecified_job_lowers_confidence(self):
        vague = make_job(requirements_json={})
        specific = await score_candidate(make_job(), make_profile())
        result = await score_candidate(vague, make_profile())
        assert result.confidence < specific.confidence


class TestScoreCandidateWithLLM:
    LLM_JUDGEMENT = {
        "cultural_score": 88,
        "reasoning": "Consistent ownership of payment systems across two employers.",
        "strengths": ["Deep payments domain experience", "Led a platform team"],
        "concerns": ["No exposure to the job's Kubernetes stack"],
    }

    async def test_the_model_supplies_the_cultural_score_and_narrative(self):
        client = FakeLLMClient([self.LLM_JUDGEMENT])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.engine == "llm"
        assert result.cultural_score == 88.0
        assert result.reasoning == self.LLM_JUDGEMENT["reasoning"]
        assert result.strengths == self.LLM_JUDGEMENT["strengths"]
        assert result.concerns == self.LLM_JUDGEMENT["concerns"]

    async def test_the_model_does_not_get_to_move_the_structured_components(self):
        # Auditability: skills/experience/education stay reproducible.
        tampering = {**self.LLM_JUDGEMENT, "skill_match": 5, "overall_score": 3}
        client = FakeLLMClient([tampering])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.skill_match == 100.0
        assert result.overall_score > 50.0

    async def test_an_out_of_range_cultural_score_is_clamped(self):
        client = FakeLLMClient([{**self.LLM_JUDGEMENT, "cultural_score": 900}])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.cultural_score == 100.0

    async def test_a_model_failure_falls_back_to_the_heuristic_read(self):
        client = FakeLLMClient([None])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.engine == "heuristic"
        assert result.overall_score > 0
        assert result.reasoning is not None

    async def test_unparseable_output_falls_back(self):
        client = FakeLLMClient(["The candidate seems fine to me."])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.engine == "heuristic"

    async def test_a_model_answer_raises_confidence(self):
        with_llm = await score_candidate(
            make_job(), make_profile(), client=FakeLLMClient([self.LLM_JUDGEMENT])
        )
        without = await score_candidate(make_job(), make_profile())
        assert with_llm.confidence > without.confidence

    async def test_the_model_records_which_model_ran(self):
        client = FakeLLMClient([self.LLM_JUDGEMENT])
        result = await score_candidate(make_job(), make_profile(), client=client)
        assert result.model_used == "fake/model:free"
        assert result.latency_ms is not None

    async def test_the_prompt_forbids_protected_characteristics(self):
        client = FakeLLMClient([self.LLM_JUDGEMENT])
        await score_candidate(make_job(), make_profile(), client=client)
        system = client.calls[0]["system"].lower()
        assert "protected characteristic" in system
        assert "do not penalise career gaps" in system

    async def test_the_prompt_carries_no_candidate_identity(self):
        client = FakeLLMClient([self.LLM_JUDGEMENT])
        profile = make_profile()
        await score_candidate(make_job(), profile, client=client)
        prompt = client.calls[0]["prompt"]
        # Only work history and skills reach the model — no name, email, or phone.
        assert "@" not in prompt
        assert "Ada" not in prompt
        assert "Staff Engineer" in prompt
