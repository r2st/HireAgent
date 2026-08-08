"""Candidate↔job scoring (design §2.2 step 6, §4.1, §5).

The overall score is a weighted blend of four components, defaulting to the
design's 40/30/15/15 split and overridable per job:

* **skills** — taxonomy-aware match against the job's required and preferred
  skill lists.
* **experience** — how the candidate's years line up with the job's band.
* **education** — degree/field fit, with a non-mandatory requirement costing
  far less than a mandatory one.
* **cultural** — tenure stability and career progression, refined by the LLM
  when one is reachable.

The first three are deterministic and reproducible: the same inputs always
produce the same number, which is what makes a score defensible to a candidate
who asks why they were ranked where they were. The LLM contributes the soft
dimension and the human-readable reasoning, and its absence degrades quality
rather than blocking the pipeline.

Nothing here reads name, gender, age, nationality, or any other protected
characteristic — only skills, dates, and qualifications (design §4.9).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.tenancy import scoped_select
from app.integrations.openrouter import OpenRouterClient, get_llm_client
from app.models.application import Application, Screening
from app.models.candidate import Candidate, Resume
from app.models.job import Job
from app.services.skill_taxonomy import canonicalize, skill_names

logger = logging.getLogger(__name__)

# What a component scores when the inputs simply do not support a judgement —
# a job that lists no skills, or a candidate with no parsed resume. Scoring
# these as 0 would rank an under-specified job's applicants below everyone
# else's for reasons that have nothing to do with the candidates.
UNSCORABLE = 60.0

# Below this, the screening is surfaced for human review (design §5).
CONFIDENCE_THRESHOLD = 0.70

# A required skill the candidate has, but with less depth than the job asks
# for, still counts for most of its weight.
PARTIAL_SKILL_CREDIT = 0.6

# Preferred skills can lift a skill score by at most this much over the
# required-skill baseline, so a candidate cannot pad a weak core match.
PREFERRED_SKILL_BONUS = 15.0

SCORING_SYSTEM_PROMPT = """\
You are a recruitment screening assistant. Given a job and a candidate \
profile, judge only the qualitative fit that structured data cannot capture, \
and return ONLY a JSON object — no prose, no markdown fences.

Schema:
{
  "cultural_score": number,        // 0-100: tenure stability, career progression, scope of ownership
  "reasoning": string,             // 2-3 sentences explaining the assessment
  "strengths": [string],           // up to 4 concrete, evidence-backed strengths
  "concerns": [string]             // up to 4 concrete, evidence-backed concerns
}

Rules:
- Judge only on work history, skills, and qualifications.
- Never infer or comment on gender, age, race, nationality, religion, marital \
status, parental status, disability, or any other protected characteristic, \
even if the profile mentions one.
- Do not penalise career gaps, non-linear paths, or non-traditional \
backgrounds on their own; assess demonstrated capability instead.
- Cite only what the profile actually states. Do not invent employers, dates, \
or achievements.
"""


@dataclass
class ScoreBreakdown:
    """A full screening result before it is persisted."""

    skill_match: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    cultural_score: float = 0.0
    overall_score: float = 0.0

    weights: dict[str, float] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    reasoning: str | None = None

    confidence: float = 0.0
    engine: str = "heuristic"
    model_used: str | None = None
    latency_ms: int | None = None

    @property
    def requires_human_review(self) -> bool:
        return self.confidence < CONFIDENCE_THRESHOLD


@dataclass
class CandidateProfile:
    """Everything the scorer is allowed to look at.

    Assembling this explicitly keeps protected characteristics out of scoring
    by construction: the name and contact fields are simply never copied in.
    """

    skills: list[str] = field(default_factory=list)
    experience_years: float | None = None
    education: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    current_role: str | None = None
    current_company: str | None = None
    has_parsed_resume: bool = False


# --------------------------------------------------------------------------- #
# Component scorers
# --------------------------------------------------------------------------- #
def score_skills(
    candidate_skills: list, requirements: dict
) -> tuple[float, list[str], list[str]]:
    """Weighted match against required skills, with a preferred-skill bonus.

    Returns ``(score, matched, missing)`` where the lists name the *required*
    skills only — those are what a recruiter acts on.
    """
    required = _skill_requirements(requirements.get("required_skills"))
    preferred = _skill_requirements(requirements.get("preferred_skills"))

    have: dict[str, dict] = {}
    for entry in candidate_skills or []:
        if isinstance(entry, dict):
            name = canonicalize(str(entry.get("name") or ""))
            years = entry.get("years")
        else:
            name, years = canonicalize(str(entry)), None
        if name:
            have[name.lower()] = {"name": name, "years": _as_float(years)}

    matched: list[str] = []
    missing: list[str] = []

    if not required:
        # Nothing to match against — every applicant is equally (un)scorable.
        base = UNSCORABLE
    else:
        earned = 0.0
        total_weight = 0.0
        for requirement in required:
            weight = requirement["weight"]
            total_weight += weight
            held = have.get(requirement["name"].lower())
            if held is None:
                missing.append(requirement["name"])
                continue
            matched.append(requirement["name"])
            min_years = requirement["min_years"]
            if (
                min_years is not None
                and held["years"] is not None
                and held["years"] < min_years
            ):
                # They have the skill but less depth than asked for.
                earned += weight * PARTIAL_SKILL_CREDIT
            else:
                earned += weight
        base = (earned / total_weight * 100.0) if total_weight > 0 else UNSCORABLE

    if preferred:
        hits = sum(1 for p in preferred if p["name"].lower() in have)
        base += PREFERRED_SKILL_BONUS * (hits / len(preferred))

    return _clamp(base), matched, missing


def score_experience(
    candidate_years: float | None,
    min_years: float | None,
    max_years: float | None,
) -> float:
    """How the candidate's years sit against the job's band.

    Being over the band is treated far more gently than being under it: an
    experienced applicant is a weaker signal of misfit than an inexperienced
    one, and over-penalising seniority pushes out career changers.
    """
    if candidate_years is None:
        return UNSCORABLE
    if min_years is None and max_years is None:
        return UNSCORABLE

    if min_years is not None and candidate_years < min_years:
        if min_years <= 0:
            return 100.0
        ratio = max(0.0, candidate_years / min_years)
        # Half the score is retained at the boundary of "no experience at all",
        # so a promising junior is not scored to zero.
        return _clamp(40.0 + 60.0 * ratio)

    if max_years is not None and candidate_years > max_years:
        over = candidate_years - max_years
        return _clamp(100.0 - 4.0 * over, floor=70.0)

    return 100.0


def score_education(education: list[dict], requirement: dict | None) -> float:
    """Degree and field fit.

    A non-mandatory requirement is a preference, not a gate: missing it costs a
    little, while missing a mandatory one costs most of the component. This is
    the design's "skill-based matching over credential matching" (§4.1).
    """
    if not requirement or not (requirement.get("degree") or requirement.get("field_of_study")):
        return UNSCORABLE
    if not education:
        return 30.0 if requirement.get("is_mandatory") else 55.0

    wanted_degree = _normalise(requirement.get("degree"))
    wanted_field = _normalise(requirement.get("field_of_study"))

    degree_hit = not wanted_degree
    field_hit = not wanted_field
    for entry in education:
        degree = _normalise(entry.get("degree"))
        study = _normalise(entry.get("field_of_study"))
        institution = _normalise(entry.get("institution"))
        if wanted_degree and degree and (
            wanted_degree in degree or degree in wanted_degree
        ):
            degree_hit = True
        if wanted_field and (
            (study and (wanted_field in study or study in wanted_field))
            # Many resumes fold the field into the institution line.
            or (institution and wanted_field in institution)
        ):
            field_hit = True

    if degree_hit and field_hit:
        return 100.0
    if degree_hit or field_hit:
        return 75.0
    return 40.0 if requirement.get("is_mandatory") else 60.0


def score_cultural(profile: CandidateProfile) -> float:
    """Deterministic stand-in for the LLM's qualitative read.

    Uses two signals that are visible in any parsed resume: how long the
    candidate tends to stay in a role, and whether they are currently employed
    in a role at all. Short tenure is a mild negative, not a disqualifier.
    """
    if not profile.experience:
        return UNSCORABLE

    tenures = [t for t in (_tenure_years(e) for e in profile.experience) if t is not None]
    if not tenures:
        return UNSCORABLE

    average = sum(tenures) / len(tenures)
    # ~2.5 years per role is the point where tenure stops being a concern.
    stability = min(1.0, average / 2.5)
    score = 55.0 + 35.0 * stability
    if profile.current_role:
        score += 5.0
    return _clamp(score)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def score_candidate(
    job: Job,
    profile: CandidateProfile,
    *,
    client: OpenRouterClient | None = None,
) -> ScoreBreakdown:
    """Score one candidate against one job, using the LLM when available."""
    requirements = dict(job.requirements_json or {})
    weights = job.effective_weights

    skill_match, matched, missing = score_skills(profile.skills, requirements)
    experience = score_experience(
        profile.experience_years,
        _as_float(job.min_experience_years),
        _as_float(job.max_experience_years),
    )
    education = score_education(profile.education, requirements.get("education"))
    cultural = score_cultural(profile)

    breakdown = ScoreBreakdown(
        skill_match=skill_match,
        experience_score=experience,
        education_score=education,
        cultural_score=cultural,
        weights=weights,
        matched_skills=matched,
        missing_skills=missing,
    )

    llm = client or get_llm_client()
    if llm.is_available:
        await _apply_llm_judgement(job, profile, breakdown, llm)

    if breakdown.engine == "heuristic":
        breakdown.reasoning = _heuristic_reasoning(breakdown, profile)

    breakdown.overall_score = _weighted_total(breakdown, weights)
    breakdown.confidence = _confidence(profile, breakdown, requirements)
    return breakdown


async def _apply_llm_judgement(
    job: Job,
    profile: CandidateProfile,
    breakdown: ScoreBreakdown,
    client: OpenRouterClient,
) -> None:
    data, result = await client.complete_json(
        prompt=_build_prompt(job, profile, breakdown),
        system=SCORING_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1200,
    )
    if not isinstance(data, dict):
        logger.info("Scoring LLM unavailable or unusable; keeping heuristic result")
        return

    cultural = _as_float(data.get("cultural_score"))
    if cultural is not None:
        breakdown.cultural_score = _clamp(cultural)
    breakdown.reasoning = _clean_text(data.get("reasoning"), 2000)
    breakdown.strengths = _string_list(data.get("strengths"))
    breakdown.concerns = _string_list(data.get("concerns"))
    breakdown.engine = "llm"
    breakdown.model_used = result.model if result else None
    breakdown.latency_ms = result.latency_ms if result else None


def _build_prompt(job: Job, profile: CandidateProfile, breakdown: ScoreBreakdown) -> str:
    requirements = dict(job.requirements_json or {})
    lines = [
        "JOB",
        f"Title: {job.title}",
        f"Seniority: {job.seniority or 'unspecified'}",
        f"Experience band: {job.min_experience_years or 0}-{job.max_experience_years or 'any'} years",
        f"Required skills: {', '.join(_requirement_names(requirements.get('required_skills'))) or 'none listed'}",
        f"Preferred skills: {', '.join(_requirement_names(requirements.get('preferred_skills'))) or 'none listed'}",
    ]
    responsibilities = requirements.get("responsibilities") or []
    if responsibilities:
        lines.append("Responsibilities: " + "; ".join(str(r) for r in responsibilities[:8]))

    lines += [
        "",
        "CANDIDATE",
        f"Current role: {profile.current_role or 'unknown'}"
        f" at {profile.current_company or 'unknown'}",
        f"Total experience: {profile.experience_years if profile.experience_years is not None else 'unknown'} years",
        f"Skills: {', '.join(profile.skills[:60]) or 'none extracted'}",
    ]
    if profile.experience:
        lines.append("Work history:")
        for entry in profile.experience[:8]:
            lines.append(
                f"  - {entry.get('role') or 'unknown role'}"
                f" at {entry.get('company') or 'unknown company'}"
                f" ({entry.get('start_date') or '?'} to "
                f"{'present' if entry.get('is_current') else entry.get('end_date') or '?'})"
            )
    if profile.education:
        lines.append("Education:")
        for entry in profile.education[:5]:
            lines.append(
                f"  - {entry.get('degree') or 'degree'}"
                f" {entry.get('field_of_study') or ''}"
                f" at {entry.get('institution') or 'unknown'}".rstrip()
            )
    if profile.certifications:
        lines.append("Certifications: " + ", ".join(profile.certifications[:10]))

    lines += [
        "",
        "STRUCTURED SCORES ALREADY COMPUTED (for context only — do not restate)",
        f"skills={breakdown.skill_match:.0f} experience={breakdown.experience_score:.0f}"
        f" education={breakdown.education_score:.0f}",
        f"Missing required skills: {', '.join(breakdown.missing_skills) or 'none'}",
    ]
    return "\n".join(lines)


def _weighted_total(breakdown: ScoreBreakdown, weights: dict[str, float]) -> float:
    total = (
        breakdown.skill_match * weights.get("skills", 0.0)
        + breakdown.experience_score * weights.get("experience", 0.0)
        + breakdown.education_score * weights.get("education", 0.0)
        + breakdown.cultural_score * weights.get("cultural", 0.0)
    )
    return round(_clamp(total), 2)


def _confidence(
    profile: CandidateProfile, breakdown: ScoreBreakdown, requirements: dict
) -> float:
    """How much the inputs actually supported the judgement.

    A score computed from an empty profile against an empty job description is
    arithmetically valid and practically meaningless; confidence is what stops
    it being trusted (design §5).
    """
    checks = [
        (profile.has_parsed_resume, 0.15),
        (bool(profile.skills), 0.20),
        (profile.experience_years is not None, 0.15),
        (bool(profile.experience), 0.10),
        (bool(requirements.get("required_skills")), 0.20),
        (bool(profile.education), 0.05),
        (breakdown.engine == "llm", 0.15),
    ]
    return round(min(1.0, sum(w for ok, w in checks if ok)), 4)


def _heuristic_reasoning(breakdown: ScoreBreakdown, profile: CandidateProfile) -> str:
    parts = []
    if breakdown.matched_skills:
        parts.append(
            f"Matches {len(breakdown.matched_skills)} of "
            f"{len(breakdown.matched_skills) + len(breakdown.missing_skills)} "
            f"required skills ({', '.join(breakdown.matched_skills[:6])})."
        )
    if breakdown.missing_skills:
        parts.append(f"Missing: {', '.join(breakdown.missing_skills[:6])}.")
    if profile.experience_years is not None:
        parts.append(f"{profile.experience_years:g} years of experience.")
    parts.append("Scored without a language model; components are rule-based.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
async def build_profile(
    session: AsyncSession, organization_id: uuid.UUID, candidate: Candidate
) -> CandidateProfile:
    """Assemble a scoring profile from the candidate and their primary resume."""
    resume = await session.scalar(
        scoped_select(Resume, organization_id)
        .where(Resume.candidate_id == candidate.id, Resume.is_primary.is_(True))
        .order_by(Resume.created_at.desc())
        .limit(1)
    )
    parsed: dict[str, Any] = dict(resume.parsed_json or {}) if resume else {}

    return CandidateProfile(
        skills=skill_names([*(candidate.skills_json or []), *(parsed.get("skills") or [])]),
        experience_years=_as_float(candidate.experience_years)
        or _as_float(parsed.get("total_experience_years")),
        education=[e for e in (parsed.get("education") or []) if isinstance(e, dict)],
        certifications=[
            str(c.get("name")) if isinstance(c, dict) else str(c)
            for c in (parsed.get("certifications") or [])
        ][:20],
        experience=[e for e in (parsed.get("experience") or []) if isinstance(e, dict)],
        current_role=candidate.current_role or parsed.get("current_role"),
        current_company=candidate.current_company or parsed.get("current_company"),
        has_parsed_resume=bool(parsed),
    )


async def screen_application(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application: Application,
    job: Job,
    candidate: Candidate,
    *,
    client: OpenRouterClient | None = None,
) -> Screening:
    """Score an application and persist the result.

    Screenings are append-only, so re-screening after a resume update leaves
    the previous score in place for comparison and audit.
    """
    profile = await build_profile(session, organization_id, candidate)
    breakdown = await score_candidate(job, profile, client=client)

    screening = Screening(
        organization_id=organization_id,
        application_id=application.id,
        skill_match=breakdown.skill_match,
        experience_score=breakdown.experience_score,
        education_score=breakdown.education_score,
        cultural_score=breakdown.cultural_score,
        overall_score=breakdown.overall_score,
        weights_json=breakdown.weights,
        reasoning=breakdown.reasoning,
        matched_skills=breakdown.matched_skills,
        missing_skills=breakdown.missing_skills,
        strengths=breakdown.strengths,
        concerns=breakdown.concerns,
        confidence=breakdown.confidence,
        requires_human_review=breakdown.requires_human_review,
        model_used=breakdown.model_used,
        engine=breakdown.engine,
        latency_ms=breakdown.latency_ms,
    )
    session.add(screening)

    # Denormalised onto the application so the pipeline board can sort without
    # joining screenings.
    application.score = breakdown.overall_score
    return screening


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _skill_requirements(raw: Any) -> list[dict]:
    """Normalise a requirement list that may hold strings or dicts."""
    out: list[dict] = []
    for entry in raw or []:
        if isinstance(entry, str):
            name, weight, min_years = entry, 1.0, None
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "")
            weight = _as_float(entry.get("weight"))
            weight = 1.0 if weight is None else max(0.0, weight)
            min_years = _as_float(entry.get("min_years"))
        else:
            continue
        canonical = canonicalize(name)
        if canonical and weight > 0:
            out.append({"name": canonical, "weight": weight, "min_years": min_years})
    return out


def _requirement_names(raw: Any) -> list[str]:
    return [r["name"] for r in _skill_requirements(raw)]


def _tenure_years(entry: dict) -> float | None:
    start = _parse_month(entry.get("start_date"))
    if start is None:
        return None
    end = (
        datetime.now(UTC).date()
        if entry.get("is_current")
        else _parse_month(entry.get("end_date")) or datetime.now(UTC).date()
    )
    if end < start:
        return None
    return (end - start).days / 365.25


def _parse_month(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    parts = text.split("-")
    try:
        if len(parts) >= 2:
            return date(int(parts[0]), max(1, min(12, int(parts[1]))), 1)
        if len(parts[0]) == 4:
            return date(int(parts[0]), 1, 1)
    except ValueError:
        return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, *, floor: float = 0.0, ceiling: float = 100.0) -> float:
    return round(max(floor, min(ceiling, value)), 2)


def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace(".", "")


def _clean_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _string_list(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = _clean_text(item, 300)
        if text:
            out.append(text)
    return out[:limit]
