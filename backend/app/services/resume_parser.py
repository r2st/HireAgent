"""Structured resume parsing (design §2.2 steps 3-4).

Two engines share one output shape:

* **LLM** (``llm``) — an OpenRouter free-tier model returns structured JSON.
* **Heuristic** (``heuristic``) — regex/section extraction, used when the LLM
  is disabled, unreachable, or returns nothing parseable.

The heuristic path is not just a test stub: a resume must still land in the
pipeline when the free-tier model is rate-limited, so every field the scorer
depends on has a deterministic derivation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from app.integrations.openrouter import OpenRouterClient, get_llm_client
from app.services.skill_taxonomy import canonicalize, normalize_skills

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """\
You are a resume parsing engine. Extract structured data from the resume text \
and return ONLY a JSON object — no prose, no markdown fences.

Schema:
{
  "full_name": string|null,
  "email": string|null,
  "phone": string|null,
  "location": string|null,
  "linkedin_url": string|null,
  "github_url": string|null,
  "portfolio_url": string|null,
  "current_company": string|null,
  "current_role": string|null,
  "total_experience_years": number|null,
  "summary": string|null,
  "skills": [{"name": string, "proficiency": "beginner"|"intermediate"|"advanced"|"expert"|null, "years": number|null}],
  "experience": [{"company": string, "role": string, "start_date": "YYYY-MM"|null, "end_date": "YYYY-MM"|null, "is_current": boolean, "location": string|null, "description": string|null}],
  "education": [{"institution": string, "degree": string|null, "field_of_study": string|null, "start_year": number|null, "end_year": number|null, "grade": string|null}],
  "certifications": [{"name": string, "issuer": string|null, "year": number|null}],
  "projects": [{"name": string, "description": string|null, "technologies": [string]}],
  "languages": [string]
}

Rules:
- Use null for anything not stated. Never invent employers, dates, or degrees.
- total_experience_years counts professional work only, excluding internships \
and education.
- Do not infer gender, age, nationality, marital status, or any protected \
characteristic, even if the resume mentions it.
"""

# --- Heuristic extraction patterns ---
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{3,5}[\s.-]?\d{3,4}[\s.-]?\d{0,4}"
)
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.I)
_URL_RE = re.compile(r"https?://[^\s,;)\]]+", re.I)
_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:professional\s+|work\s+|industry\s+)?experience",
    re.I,
)
_DATE_RANGE_RE = re.compile(
    r"(?P<start>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})"
    r"\s*(?:-|–|—|to|until)\s*"
    r"(?P<end>present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})",
    re.I,
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

SECTION_HEADINGS = {
    "experience": [
        "work experience", "professional experience", "employment history",
        "experience", "career history", "work history",
    ],
    "education": ["education", "academic background", "qualifications", "academics"],
    "skills": [
        "skills", "technical skills", "core competencies", "technologies",
        "skill set", "expertise",
    ],
    "projects": ["projects", "personal projects", "key projects", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses", "courses"],
    "summary": ["summary", "profile", "objective", "about me", "professional summary"],
}

DEGREE_KEYWORDS = [
    "phd", "doctorate", "m.tech", "mtech", "m.s", "ms", "msc", "m.sc", "mba",
    "master", "b.tech", "btech", "b.e", "be", "b.s", "bs", "bsc", "b.sc",
    "bachelor", "diploma", "associate", "b.com", "m.com", "bca", "mca",
]


@dataclass
class ParsedResume:
    """Normalised parse output, independent of which engine produced it."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None
    current_company: str | None = None
    current_role: str | None = None
    total_experience_years: float | None = None
    summary: str | None = None
    skills: list[dict] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    certifications: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    engine: str = "heuristic"
    model: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "linkedin_url": self.linkedin_url,
            "github_url": self.github_url,
            "portfolio_url": self.portfolio_url,
            "current_company": self.current_company,
            "current_role": self.current_role,
            "total_experience_years": self.total_experience_years,
            "summary": self.summary,
            "skills": self.skills,
            "experience": self.experience,
            "education": self.education,
            "certifications": self.certifications,
            "projects": self.projects,
            "languages": self.languages,
            "engine": self.engine,
            "model": self.model,
            "confidence": self.confidence,
        }

    @property
    def skill_names(self) -> list[str]:
        return [s["name"] for s in self.skills]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def parse_resume(
    text: str, *, client: OpenRouterClient | None = None
) -> ParsedResume:
    """Parse resume text, preferring the LLM and falling back to heuristics."""
    if not text or not text.strip():
        return ParsedResume(engine="heuristic", confidence=0.0)

    llm = client or get_llm_client()
    if llm.is_available:
        parsed = await _parse_with_llm(text, llm)
        if parsed is not None:
            return parsed
        logger.info("LLM parse unavailable or unusable; using heuristic parser")

    return parse_heuristic(text)


async def _parse_with_llm(
    text: str, client: OpenRouterClient
) -> ParsedResume | None:
    # Cap input so a 40-page CV cannot blow the free-tier context window.
    excerpt = text[:24_000]
    data, result = await client.complete_json(
        prompt=f"Resume text:\n\n{excerpt}",
        system=PARSER_SYSTEM_PROMPT,
        model=None,
        temperature=0.0,
        max_tokens=4096,
    )
    if not isinstance(data, dict):
        return None

    parsed = _from_llm_payload(data)
    parsed.engine = "llm"
    parsed.model = result.model if result else None

    # Backfill anything the model missed with heuristics: free models routinely
    # drop contact fields that a regex finds reliably.
    fallback = parse_heuristic(text)
    _merge_missing(parsed, fallback)

    parsed.confidence = _confidence(parsed)
    return parsed


def _from_llm_payload(data: dict) -> ParsedResume:
    parsed = ParsedResume()
    parsed.full_name = _clean_str(data.get("full_name"))
    parsed.email = _clean_email(data.get("email"))
    parsed.phone = _clean_str(data.get("phone"))
    parsed.location = _clean_str(data.get("location"))
    parsed.linkedin_url = _clean_str(data.get("linkedin_url"))
    parsed.github_url = _clean_str(data.get("github_url"))
    parsed.portfolio_url = _clean_str(data.get("portfolio_url"))
    parsed.current_company = _clean_str(data.get("current_company"))
    parsed.current_role = _clean_str(data.get("current_role"))
    parsed.summary = _clean_str(data.get("summary"))
    parsed.total_experience_years = _clean_float(data.get("total_experience_years"))

    parsed.skills = normalize_skills(data.get("skills") or [])
    parsed.experience = _clean_experience(data.get("experience") or [])
    parsed.education = _clean_education(data.get("education") or [])
    parsed.certifications = _clean_certifications(data.get("certifications") or [])
    parsed.projects = _clean_projects(data.get("projects") or [])
    parsed.languages = [
        s for s in (_clean_str(x) for x in (data.get("languages") or [])) if s
    ]

    # Derive experience from the timeline when the model omitted the total.
    if parsed.total_experience_years is None and parsed.experience:
        parsed.total_experience_years = _years_from_experience(parsed.experience)
    if not parsed.current_company or not parsed.current_role:
        current = next(
            (e for e in parsed.experience if e.get("is_current")), None
        ) or (parsed.experience[0] if parsed.experience else None)
        if current:
            parsed.current_company = parsed.current_company or current.get("company")
            parsed.current_role = parsed.current_role or current.get("role")
    return parsed


def _merge_missing(target: ParsedResume, source: ParsedResume) -> None:
    """Fill only the fields the primary engine left empty."""
    for attr in (
        "full_name", "email", "phone", "location", "linkedin_url",
        "github_url", "portfolio_url", "current_company", "current_role",
        "total_experience_years", "summary",
    ):
        if getattr(target, attr) in (None, ""):
            setattr(target, attr, getattr(source, attr))
    if not target.skills:
        target.skills = source.skills
    if not target.experience:
        target.experience = source.experience
    if not target.education:
        target.education = source.education


# --------------------------------------------------------------------------- #
# Heuristic parser
# --------------------------------------------------------------------------- #
def parse_heuristic(text: str) -> ParsedResume:
    """Deterministic extraction with no model call."""
    parsed = ParsedResume(engine="heuristic")
    lines = [line.strip() for line in text.split("\n")]
    non_empty = [line for line in lines if line]

    parsed.email = _clean_email(_first(_EMAIL_RE.findall(text)))
    parsed.phone = _extract_phone(text)
    parsed.linkedin_url = _first(_LINKEDIN_RE.findall(text))
    parsed.github_url = _first(_GITHUB_RE.findall(text))
    parsed.portfolio_url = _extract_portfolio(text)
    parsed.full_name = _extract_name(non_empty, parsed.email)

    sections = _split_sections(lines)
    parsed.summary = _truncate(sections.get("summary"), 800)
    parsed.skills = _extract_skills(sections.get("skills"), text)
    parsed.experience = _extract_experience(sections.get("experience"))
    parsed.education = _extract_education(sections.get("education"))
    parsed.certifications = [
        {"name": line.lstrip("-•* ").strip(), "issuer": None, "year": None}
        for line in (sections.get("certifications") or "").split("\n")
        if line.strip()
    ][:20]

    stated = _YEARS_RE.search(text)
    if stated:
        parsed.total_experience_years = _clean_float(stated.group(1))
    elif parsed.experience:
        parsed.total_experience_years = _years_from_experience(parsed.experience)

    current = next((e for e in parsed.experience if e.get("is_current")), None)
    if current is None and parsed.experience:
        current = parsed.experience[0]
    if current:
        parsed.current_company = current.get("company")
        parsed.current_role = current.get("role")

    parsed.confidence = _confidence(parsed)
    return parsed


def _split_sections(lines: list[str]) -> dict[str, str]:
    """Bucket lines under the resume heading they fall beneath."""
    heading_lookup: dict[str, str] = {}
    for section, headings in SECTION_HEADINGS.items():
        for heading in headings:
            heading_lookup[heading] = section

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        key = re.sub(r"[^a-z ]", "", line.lower()).strip()
        # A heading is a short standalone line matching a known label.
        if key in heading_lookup and len(line) <= 60:
            current = heading_lookup[key]
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _extract_skills(skills_section: str | None, full_text: str) -> list[dict]:
    """Prefer an explicit skills section; otherwise scan for known skills."""
    if skills_section:
        # Skills are typically comma/pipe/bullet separated.
        tokens = re.split(r"[,;|•·\n\t]+", skills_section)
        candidates = [t.strip(" -–—:") for t in tokens if t.strip(" -–—:")]
        # Drop prose lines: a skills entry is short.
        candidates = [c for c in candidates if 1 < len(c) <= 60 and len(c.split()) <= 6]
        normalized = normalize_skills(candidates)
        if normalized:
            return normalized[:80]

    # No usable section — look for taxonomy terms anywhere in the document.
    from app.services.skill_taxonomy import SKILL_ALIASES

    found: list[str] = []
    lowered = full_text.lower()
    for canonical, aliases in SKILL_ALIASES.items():
        for term in (canonical, *aliases):
            # Word-boundary match so "R" does not match every capital R.
            if re.search(rf"(?<![\w+#]){re.escape(term.lower())}(?![\w+#])", lowered):
                found.append(canonical)
                break
    return normalize_skills(found)[:80]


def _extract_experience(section: str | None) -> list[dict]:
    """Pull role/company/date-range entries out of the experience section."""
    if not section:
        return []

    entries: list[dict] = []
    lines = [line for line in section.split("\n") if line.strip()]

    for index, line in enumerate(lines):
        match = _DATE_RANGE_RE.search(line)
        if not match:
            continue

        start = _parse_month_year(match.group("start"))
        end_raw = match.group("end")
        is_current = end_raw.lower() in {"present", "current", "now"}
        end = None if is_current else _parse_month_year(end_raw)

        # The role/company usually sit on this line before the dates, or on the
        # line immediately above.
        header = _DATE_RANGE_RE.sub("", line).strip(" |,-–—\t")
        if not header and index > 0:
            header = lines[index - 1].strip()

        role, company = _split_role_company(header)
        if not role and not company:
            continue

        entries.append(
            {
                "company": company,
                "role": role,
                "start_date": start.strftime("%Y-%m") if start else None,
                "end_date": end.strftime("%Y-%m") if end else None,
                "is_current": is_current,
                "location": None,
                "description": None,
            }
        )

    # Most recent first.
    entries.sort(key=lambda e: e.get("start_date") or "", reverse=True)
    return entries[:20]


def _split_role_company(header: str) -> tuple[str | None, str | None]:
    """Split "Senior Engineer at Acme" / "Senior Engineer | Acme" into parts."""
    if not header:
        return None, None
    header = header.strip(" |,-–—\t")
    for separator in (r"\s+at\s+", r"\s*\|\s*", r"\s*[–—]\s*", r"\s*,\s*", r"\s+-\s+"):
        parts = re.split(separator, header, maxsplit=1, flags=re.I)
        if len(parts) == 2 and all(p.strip() for p in parts):
            return parts[0].strip(), parts[1].strip()
    return header, None


def _extract_education(section: str | None) -> list[dict]:
    if not section:
        return []

    entries: list[dict] = []
    for line in section.split("\n"):
        stripped = line.strip(" -•*")
        if not stripped:
            continue
        lowered = stripped.lower()
        degree = next((d for d in DEGREE_KEYWORDS if d in lowered), None)
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", stripped)]
        if not degree and not years:
            continue

        # The institution is whatever is left after removing degree and years.
        institution = re.sub(r"\b(19\d{2}|20\d{2})\b", "", stripped)
        institution = re.sub(r"[,\-–—|]{1,}", " ", institution).strip()

        entries.append(
            {
                "institution": institution or stripped,
                "degree": degree.upper() if degree else None,
                "field_of_study": None,
                "start_year": min(years) if len(years) > 1 else None,
                "end_year": max(years) if years else None,
                "grade": None,
            }
        )
    return entries[:10]


def _extract_name(lines: list[str], email: str | None) -> str | None:
    """Guess the candidate's name from the document header.

    Resumes put the name first; the main risk is picking up a heading or the
    contact line instead, so obviously non-name lines are skipped.
    """
    for line in lines[:6]:
        if _EMAIL_RE.search(line) or _URL_RE.search(line):
            continue
        if any(ch.isdigit() for ch in line):
            continue
        if len(line) > 60 or "@" in line:
            continue
        words = line.split()
        if not 1 < len(words) <= 5:
            continue
        lowered = line.lower().strip(" :")
        if lowered in {h for hs in SECTION_HEADINGS.values() for h in hs}:
            continue
        if any(word[:1].isupper() for word in words):
            return line.strip(" ,|")

    # Fall back to the local part of the email: "ada.lovelace@x.com" -> "Ada Lovelace".
    if email:
        local = email.split("@")[0]
        parts = [p for p in re.split(r"[._-]+", local) if p.isalpha() and len(p) > 1]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3])
    return None


def _extract_phone(text: str) -> str | None:
    """Find a phone number, rejecting matches that are really dates or IDs."""
    for candidate in _PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if not 7 <= len(digits) <= 15:
            continue
        # A bare 4-digit year or a year range is not a phone number.
        if len(digits) <= 8 and re.fullmatch(r"(19|20)\d{2}.*", digits):
            continue
        return candidate.strip()
    return None


def _extract_portfolio(text: str) -> str | None:
    """First URL that is neither the LinkedIn nor the GitHub profile."""
    for url in _URL_RE.findall(text):
        lowered = url.lower()
        if "linkedin.com" in lowered or "github.com" in lowered:
            continue
        if any(lowered.endswith(ext) for ext in (".png", ".jpg", ".gif")):
            continue
        return url.rstrip(".,;")
    return None


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-", "unknown"}:
        return None
    return text


def _clean_email(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    match = _EMAIL_RE.search(text)
    return match.group(0).lower() if match else None


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group(0))
    # Reject impossible values rather than letting them skew scoring.
    if not 0 <= number <= 60:
        return None
    return round(number, 1)


def _truncate(value: str | None, limit: int) -> str | None:
    text = _clean_str(value)
    if text and len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def _clean_experience(items: list) -> list[dict]:
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        company = _clean_str(item.get("company"))
        role = _clean_str(item.get("role") or item.get("title"))
        if not company and not role:
            continue
        out.append(
            {
                "company": company,
                "role": role,
                "start_date": _clean_month(item.get("start_date")),
                "end_date": _clean_month(item.get("end_date")),
                "is_current": bool(item.get("is_current")),
                "location": _clean_str(item.get("location")),
                "description": _truncate(item.get("description"), 1000),
            }
        )
    return out[:25]


def _clean_education(items: list) -> list[dict]:
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        institution = _clean_str(item.get("institution") or item.get("school"))
        degree = _clean_str(item.get("degree"))
        if not institution and not degree:
            continue
        out.append(
            {
                "institution": institution,
                "degree": degree,
                "field_of_study": _clean_str(item.get("field_of_study")),
                "start_year": _clean_year(item.get("start_year")),
                "end_year": _clean_year(item.get("end_year")),
                "grade": _clean_str(item.get("grade")),
            }
        )
    return out[:10]


def _clean_certifications(items: list) -> list[dict]:
    out: list[dict] = []
    for item in items:
        if isinstance(item, str):
            name = _clean_str(item)
            if name:
                out.append({"name": name, "issuer": None, "year": None})
        elif isinstance(item, dict):
            name = _clean_str(item.get("name"))
            if name:
                out.append(
                    {
                        "name": name,
                        "issuer": _clean_str(item.get("issuer")),
                        "year": _clean_year(item.get("year")),
                    }
                )
    return out[:20]


def _clean_projects(items: list) -> list[dict]:
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name"))
        if not name:
            continue
        technologies = [
            canonicalize(str(t))
            for t in (item.get("technologies") or [])
            if _clean_str(t)
        ]
        out.append(
            {
                "name": name,
                "description": _truncate(item.get("description"), 500),
                "technologies": [t for t in technologies if t][:20],
            }
        )
    return out[:15]


def _clean_month(value: Any) -> str | None:
    """Normalise a date to ``YYYY-MM``."""
    text = _clean_str(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-01"
    parsed = _parse_month_year(text)
    return parsed.strftime("%Y-%m") if parsed else None


def _clean_year(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    year = int(match.group(0))
    return year if 1950 <= year <= date.today().year + 10 else None


def _parse_month_year(value: str) -> date | None:
    """Parse 'Jan 2020', '01/2020', or '2020' into a date."""
    text = value.strip().lower()
    month_match = re.match(r"([a-z]{3,})\.?\s*(\d{4})", text)
    if month_match:
        month = _MONTHS.get(month_match.group(1)[:3])
        if month:
            return date(int(month_match.group(2)), month, 1)
    slash = re.match(r"(\d{1,2})/(\d{4})", text)
    if slash:
        month = max(1, min(12, int(slash.group(1))))
        return date(int(slash.group(2)), month, 1)
    year_match = re.fullmatch(r"(19|20)\d{2}", text)
    if year_match:
        return date(int(text), 1, 1)
    return None


def _years_from_experience(experience: list[dict]) -> float | None:
    """Total professional experience from the role timeline.

    Overlapping roles are merged so concurrent positions are not double
    counted, and gaps between roles are excluded rather than penalised
    (design §4.1: career gaps get contextual analysis, not a penalty).
    """
    intervals: list[tuple[date, date]] = []
    today = datetime.now(UTC).date()

    for entry in experience:
        start = _parse_iso_month(entry.get("start_date"))
        if start is None:
            continue
        end = (
            today
            if entry.get("is_current")
            else _parse_iso_month(entry.get("end_date")) or today
        )
        if end < start:
            continue
        intervals.append((start, end))

    if not intervals:
        return None

    intervals.sort()
    merged: list[list[date]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total_days = sum((end - start).days for start, end in merged)
    years = round(total_days / 365.25, 1)
    return years if 0 <= years <= 60 else None


def _parse_iso_month(value: Any) -> date | None:
    text = _clean_str(value)
    if not text:
        return None
    match = re.match(r"(\d{4})-(\d{1,2})", text)
    if match:
        year, month = int(match.group(1)), max(1, min(12, int(match.group(2))))
        return date(year, month, 1)
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    return None


def _confidence(parsed: ParsedResume) -> float:
    """How complete the parse is, 0-1.

    Below 0.70 the result is flagged for human review (design §5).
    """
    checks = [
        parsed.full_name is not None,
        parsed.email is not None,
        parsed.phone is not None,
        bool(parsed.skills),
        bool(parsed.experience),
        bool(parsed.education),
        parsed.total_experience_years is not None,
        parsed.current_role is not None,
    ]
    # Contact details and skills matter most for downstream matching.
    weights = [0.18, 0.18, 0.08, 0.20, 0.16, 0.08, 0.06, 0.06]
    score = sum(w for ok, w in zip(checks, weights, strict=True) if ok)
    return round(min(1.0, score), 4)
