"""Skill taxonomy: canonicalisation, merging, and job↔candidate matching."""

from __future__ import annotations

import pytest

from app.services.skill_taxonomy import (
    SKILL_ALIASES,
    canonicalize,
    match_skills,
    normalize_skills,
    skill_names,
)


class TestCanonicalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The design doc's own example: surface variants collapse to one tag.
            ("React.js", "React"),
            ("ReactJS", "React"),
            ("react", "React"),
            ("REACT", "React"),
            ("  react.js  ", "React"),
        ],
    )
    def test_variants_of_one_skill_collapse(self, raw, expected):
        assert canonicalize(raw) == expected

    def test_aliases_resolve_to_their_canonical_name(self):
        # Every alias in the table must round-trip, or matching silently misses.
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in aliases:
                assert canonicalize(alias) == canonical, f"{alias} -> {canonical}"

    def test_strips_trailing_qualifiers(self):
        assert canonicalize("Python (advanced)") == "Python"
        assert canonicalize("Python - 5 years") == "Python"
        assert canonicalize("Python - 5 yrs") == "Python"

    def test_unknown_skills_pass_through_title_cased(self):
        # The taxonomy normalises; it is not a whitelist, so niche skills survive.
        assert canonicalize("quantum annealing") == "Quantum Annealing"

    def test_preserves_short_all_caps_tokens(self):
        assert canonicalize("SAP") == "SAP"

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_input_yields_empty_string(self, raw):
        assert canonicalize(raw or "") == ""


class TestNormalizeSkills:
    def test_accepts_strings_and_dicts_together(self):
        result = normalize_skills(
            ["python", {"name": "React.js", "proficiency": "advanced", "years": 3}]
        )
        assert result == [
            {"name": "Python", "proficiency": None, "years": None},
            {"name": "React", "proficiency": "advanced", "years": 3.0},
        ]

    def test_deduplicates_across_surface_forms(self):
        result = normalize_skills(["React.js", "ReactJS", "react"])
        assert [s["name"] for s in result] == ["React"]

    def test_merge_keeps_the_richest_detail(self):
        result = normalize_skills(
            [
                {"name": "python", "proficiency": None, "years": 2},
                {"name": "Python", "proficiency": "expert", "years": 7},
            ]
        )
        assert result == [{"name": "Python", "proficiency": "expert", "years": 7.0}]

    def test_accepts_alternate_key_spellings(self):
        result = normalize_skills(
            [{"skill": "postgres", "level": "expert", "years_of_experience": "4"}]
        )
        assert result == [{"name": "PostgreSQL", "proficiency": "expert", "years": 4.0}]

    def test_discards_invalid_proficiency_and_years(self):
        result = normalize_skills(
            [{"name": "Python", "proficiency": "wizard", "years": "a while"}]
        )
        assert result == [{"name": "Python", "proficiency": None, "years": None}]

    def test_ignores_unusable_entries(self):
        assert normalize_skills(["", None, 42, {"name": ""}, {}]) == []

    def test_none_input_is_safe(self):
        assert normalize_skills(None) == []

    def test_skill_names_are_canonical_and_ordered(self):
        assert skill_names(["react.js", "python", "ReactJS"]) == ["React", "Python"]


class TestMatchSkills:
    def test_splits_required_skills_into_matched_and_missing(self):
        matched, missing = match_skills(
            ["Python", "React.js", "Docker"],
            ["python", "reactjs", "Kubernetes"],
        )
        assert matched == ["Python", "React"]
        assert missing == ["Kubernetes"]

    def test_matching_is_alias_aware(self):
        # A candidate writing "Postgres" must satisfy a job asking for
        # "PostgreSQL" — this is the whole point of the taxonomy.
        matched, missing = match_skills(["Postgres"], ["PostgreSQL"])
        assert matched == ["PostgreSQL"]
        assert missing == []

    def test_no_requirements_means_nothing_missing(self):
        assert match_skills(["Python"], []) == ([], [])

    def test_candidate_with_no_skills_misses_everything(self):
        matched, missing = match_skills([], ["Python", "Go"])
        assert matched == []
        assert missing == ["Python", "Go"]

    def test_duplicate_requirements_are_collapsed(self):
        matched, missing = match_skills(["Python"], ["python", "Python", "PYTHON"])
        assert matched == ["Python"]
        assert missing == []
