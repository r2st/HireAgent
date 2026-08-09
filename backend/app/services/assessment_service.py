"""Skill assessments (design §4.4, and the ``assessed`` pipeline stage of §4.5).

A recruiter builds a reusable template, issues it against one application, and
the candidate sits the paper from a link. Three decisions shape the module.

**The invite token is the candidate's credential.** Candidates have no account,
so the token in the link authorises viewing, starting, and submitting. Anything
reached by token is looked up by token alone and takes its tenant from the row
it finds, never from caller-supplied input — the same rule the booking links
follow.

**The paper is snapshotted at issue time.** ``questions_json``, the pass mark,
and the duration are copied onto the assessment, so editing a template does not
rewrite a test somebody is part-way through, nor change the mark of one already
sat. A score stays reproducible from the row that produced it.

**A partial grade never becomes a verdict.** Multiple-choice answers mark
themselves; essays, code, and video do not. When something is left awaiting
review, the assessment still records a score for what *could* be marked, but
``passed`` stays ``None`` until a human — or the model, when one is configured
— has ruled on the rest. Inferring a fail from the half of the paper a
regex can read is exactly the failure mode §5 asks us to avoid.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import count_scoped, get_scoped, scoped_select
from app.integrations.openrouter import OpenRouterClient, get_llm_client
from app.models.application import Application
from app.models.assessment import Assessment, AssessmentTemplate
from app.models.enums import (
    STAGE_INDEX,
    ApplicationStatus,
    AssessmentStatus,
    AssessmentType,
    ConsentType,
    PipelineStage,
)
from app.schemas.common import PaginationParams
from app.services import application_service, candidate_service

logger = logging.getLogger(__name__)

# Question kinds a paper may contain.
CHOICE_TYPES = frozenset({"single_choice", "multi_choice"})
# Free-form answers. ``short_text`` marks itself when the author supplied a list
# of accepted answers, and otherwise joins the rest here.
FREE_TYPES = frozenset({"short_text", "long_text", "code", "video"})
QUESTION_TYPES = CHOICE_TYPES | FREE_TYPES

MAX_QUESTIONS = 100
MAX_PROMPT_CHARS = 4000
MAX_ANSWER_CHARS = 20000

# Assessments the candidate may still act on from their link.
OPEN_STATUSES = frozenset(
    {
        AssessmentStatus.PENDING,
        AssessmentStatus.SENT,
        AssessmentStatus.IN_PROGRESS,
    }
)

# End of the line for a row.
TERMINAL_STATUSES = frozenset(
    {
        AssessmentStatus.COMPLETED,
        AssessmentStatus.EXPIRED,
        AssessmentStatus.SKIPPED,
    }
)

# Applications in these states should not be gaining new assessments.
CLOSED_APPLICATION_STATUSES = frozenset(
    {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}
)

_JUDGE_SYSTEM = (
    "You are a technical assessor marking one candidate's answers. Mark only "
    "what the answer demonstrates. Award partial credit. Return strict JSON."
)


# --------------------------------------------------------------------------- #
# Question definitions
# --------------------------------------------------------------------------- #
def normalise_questions(raw: object) -> list[dict]:
    """Validate and canonicalise a paper.

    Run at authoring time rather than at grading time so a malformed question
    is a 422 on the recruiter's screen, not a crash in front of a candidate who
    has already typed six answers.
    """
    if not isinstance(raw, list) or not raw:
        raise ValidationError("An assessment needs at least one question")
    if len(raw) > MAX_QUESTIONS:
        raise ValidationError(
            f"An assessment may have at most {MAX_QUESTIONS} questions",
            details={"count": len(raw)},
        )

    seen: set[str] = set()
    return [_normalise_question(entry, index, seen) for index, entry in enumerate(raw)]


def _normalise_question(entry: object, index: int, seen: set[str]) -> dict:
    position = index + 1

    def fail(message: str, **extra: Any) -> NoReturn:
        raise ValidationError(
            message, details={"index": index, "question": position, **extra}
        )

    if not isinstance(entry, dict):
        fail(f"Question {position} is not an object")

    qtype = str(entry.get("type") or "single_choice").strip().lower()
    if qtype not in QUESTION_TYPES:
        fail(
            f"Question {position} has unknown type '{qtype}'",
            allowed=sorted(QUESTION_TYPES),
        )

    prompt = str(entry.get("prompt") or "").strip()
    if not prompt:
        fail(f"Question {position} has no prompt")

    qid = str(entry.get("id") or f"q{position}").strip()
    if not qid:
        fail(f"Question {position} has a blank id")
    if qid in seen:
        fail(f"Duplicate question id '{qid}'", id=qid)
    seen.add(qid)

    weight = _as_float(entry.get("weight", 1.0))
    if weight is None or weight <= 0:
        fail(f"Question {position} needs a weight above zero")

    question: dict[str, Any] = {
        "id": qid,
        "type": qtype,
        "prompt": prompt[:MAX_PROMPT_CHARS],
        "weight": float(weight or 1.0),
        "required": bool(entry.get("required", True)),
    }

    if qtype in CHOICE_TYPES:
        options = [
            str(option).strip()
            for option in (entry.get("options") or [])
            if str(option).strip()
        ]
        if len(options) < 2:
            fail(f"Question {position} needs at least two options")
        if len({o.casefold() for o in options}) != len(options):
            fail(f"Question {position} has duplicate options")

        expected_raw = entry.get("expected")
        expected_list = (
            list(expected_raw) if isinstance(expected_raw, list) else [expected_raw]
        )
        lookup = {o.casefold(): o for o in options}
        expected: list[str] = []
        for value in expected_list:
            key = str(value or "").strip().casefold()
            if key not in lookup:
                fail(
                    f"Question {position} expects '{value}', which is not one of "
                    "its options",
                    options=options,
                )
            if lookup[key] not in expected:
                expected.append(lookup[key])
        if not expected:
            fail(f"Question {position} has no correct option")
        if qtype == "single_choice" and len(expected) != 1:
            fail(f"Question {position} is single-choice but has {len(expected)} answers")

        question["options"] = options
        question["expected"] = expected
    else:
        if qtype == "short_text":
            accepted = [
                str(value).strip()
                for value in (entry.get("accepted") or [])
                if str(value).strip()
            ]
            if accepted:
                question["accepted"] = accepted
        rubric = str(entry.get("rubric") or "").strip()
        if rubric:
            question["rubric"] = rubric[:MAX_PROMPT_CHARS]

    return question


def is_self_marking(question: dict) -> bool:
    """Whether this question can be marked without a judgement call."""
    return question.get("type") in CHOICE_TYPES or bool(question.get("accepted"))


def candidate_view(questions: list[dict]) -> list[dict]:
    """The paper with the answer key removed.

    The candidate-facing routes return this and never the stored question, so
    ``expected``/``accepted``/``rubric`` cannot leak through the link.
    """
    visible: list[dict] = []
    for question in questions:
        entry = {
            "id": question.get("id"),
            "type": question.get("type"),
            "prompt": question.get("prompt"),
            "weight": question.get("weight", 1.0),
            "required": question.get("required", True),
        }
        if question.get("options"):
            entry["options"] = list(question["options"])
        visible.append(entry)
    return visible


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
@dataclass
class QuestionGrade:
    """One question's mark.

    ``fraction`` is the share of the question's weight earned, or ``None`` while
    the answer is still waiting on a judgement.
    """

    question_id: str
    type: str
    weight: float
    fraction: float | None
    method: str  # auto | llm | manual | unanswered | pending
    comment: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.fraction is None

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "type": self.type,
            "weight": self.weight,
            "fraction": self.fraction,
            "method": self.method,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> QuestionGrade:
        return cls(
            question_id=str(raw.get("question_id")),
            type=str(raw.get("type") or "short_text"),
            weight=float(_as_float(raw.get("weight")) or 1.0),
            fraction=_as_float(raw.get("fraction")),
            method=str(raw.get("method") or "manual"),
            comment=raw.get("comment"),
        )


@dataclass
class GradeOutcome:
    """The result of marking a whole paper."""

    score: float | None
    passed: bool | None
    grades: list[QuestionGrade] = field(default_factory=list)
    feedback: str | None = None

    @property
    def pending(self) -> list[str]:
        return [g.question_id for g in self.grades if g.is_pending]

    @property
    def engine(self) -> str:
        methods = {g.method for g in self.grades}
        if methods == {"llm"}:
            return "llm"
        if "llm" in methods:
            return "mixed"
        if methods & {"manual", "pending"}:
            return "manual" if methods <= {"manual", "pending"} else "mixed"
        return "auto"

    def breakdown(self) -> dict:
        graded = [g for g in self.grades if not g.is_pending]
        return {
            "questions": [g.as_dict() for g in self.grades],
            "pending_review": self.pending,
            "engine": self.engine,
            "graded_weight": round(sum(g.weight for g in graded), 4),
            "total_weight": round(sum(g.weight for g in self.grades), 4),
        }


def _grade_choice(question: dict, answer: object) -> QuestionGrade:
    expected = {str(v).strip().casefold() for v in question.get("expected") or []}
    if question["type"] == "single_choice":
        chosen = (
            {str(answer).strip().casefold()}
            if answer is not None and str(answer).strip()
            else set()
        )
    else:
        raw = answer if isinstance(answer, list) else ([answer] if answer else [])
        chosen = {str(v).strip().casefold() for v in raw if str(v).strip()}

    if not chosen:
        return QuestionGrade(
            question_id=question["id"],
            type=question["type"],
            weight=question["weight"],
            fraction=0.0,
            method="unanswered",
            comment="No answer given",
        )

    hits = len(chosen & expected)
    misses = len(chosen - expected)
    # Wrong picks cancel right ones, so selecting everything scores zero rather
    # than full marks. Floored at zero: a question cannot cost more than itself.
    fraction = max(0.0, (hits - misses) / len(expected)) if expected else 0.0
    return QuestionGrade(
        question_id=question["id"],
        type=question["type"],
        weight=question["weight"],
        fraction=min(1.0, fraction),
        method="auto",
        comment=None if fraction >= 1.0 else f"{hits} of {len(expected)} correct",
    )


def _grade_accepted(question: dict, text: str) -> QuestionGrade:
    haystack = " ".join(text.split()).casefold()
    matched = any(
        " ".join(str(candidate).split()).casefold() in haystack
        for candidate in question.get("accepted") or []
    )
    return QuestionGrade(
        question_id=question["id"],
        type=question["type"],
        weight=question["weight"],
        fraction=1.0 if matched else 0.0,
        method="auto",
        comment=None if matched else "No accepted answer found in the response",
    )


def _was_answered(answer: object) -> bool:
    """Whether the candidate put anything at all against this question.

    Distinct from having text a grader can read: a video with no transcript is
    an answer we cannot mark, not an answer nobody gave, and the two must not
    collapse into the same zero.
    """
    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    if isinstance(answer, list | dict):
        return bool(answer)
    return True


def _answer_text(answer: object) -> str:
    """Flatten one answer to the text a grader can read.

    Video answers arrive as ``{"video_url": ..., "transcript": ...}``; only the
    transcript is markable, which is why a video with none goes to review.
    """
    if answer is None:
        return ""
    if isinstance(answer, dict):
        for key in ("transcript", "text", "answer", "content"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:MAX_ANSWER_CHARS]
        return ""
    if isinstance(answer, list):
        return ", ".join(str(v) for v in answer)[:MAX_ANSWER_CHARS]
    return str(answer).strip()[:MAX_ANSWER_CHARS]


async def grade(
    questions: list[dict],
    responses: dict,
    *,
    passing_score: float,
    llm: OpenRouterClient | None = None,
) -> GradeOutcome:
    """Mark a paper, using the model only for what cannot mark itself."""
    grades: list[QuestionGrade] = []
    judged: list[tuple[dict, str]] = []

    for question in questions:
        answer = responses.get(question["id"])
        if question["type"] in CHOICE_TYPES:
            grades.append(_grade_choice(question, answer))
            continue

        text = _answer_text(answer)
        if not text:
            answered = _was_answered(answer)
            grades.append(
                QuestionGrade(
                    question_id=question["id"],
                    type=question["type"],
                    weight=question["weight"],
                    # Something was submitted that no grader here can read — a
                    # video with no transcript, say. That is a review job, not
                    # a zero.
                    fraction=None if answered else 0.0,
                    method="pending" if answered else "unanswered",
                    comment=(
                        "Answered, but nothing readable to mark"
                        if answered
                        else "No answer given"
                    ),
                )
            )
            continue

        if question.get("accepted"):
            grades.append(_grade_accepted(question, text))
            continue

        grades.append(
            QuestionGrade(
                question_id=question["id"],
                type=question["type"],
                weight=question["weight"],
                fraction=None,
                method="pending",
                comment="Awaiting review",
            )
        )
        judged.append((question, text))

    feedback: str | None = None
    if judged:
        feedback = await _judge(judged, grades, llm=llm)

    return _finalise(grades, passing_score=passing_score, feedback=feedback)


def _finalise(
    grades: list[QuestionGrade], *, passing_score: float, feedback: str | None = None
) -> GradeOutcome:
    """Turn per-question marks into a score, and a score into a verdict."""
    graded = [g for g in grades if not g.is_pending]
    graded_weight = sum(g.weight for g in graded)

    score: float | None = None
    if graded_weight > 0:
        earned = sum((g.fraction or 0.0) * g.weight for g in graded)
        score = round(100.0 * earned / graded_weight, 2)

    # A verdict needs the whole paper. Anything still awaiting review leaves
    # ``passed`` open rather than guessing from the part that marked itself.
    passed: bool | None = None
    if score is not None and not any(g.is_pending for g in grades):
        passed = score >= passing_score

    return GradeOutcome(score=score, passed=passed, grades=grades, feedback=feedback)


async def _judge(
    judged: list[tuple[dict, str]],
    grades: list[QuestionGrade],
    *,
    llm: OpenRouterClient | None = None,
) -> str | None:
    """Ask the model to mark the free-form answers, in one call.

    A model that is unavailable, slow, or incoherent leaves every judged
    question pending — the recruiter marks them by hand. Failing open to a
    score would be inventing one.
    """
    client = llm or get_llm_client()
    if not client.is_available:
        return None

    by_id = {g.question_id: g for g in grades}
    parsed, result = await client.complete_json(
        prompt=_judge_prompt(judged),
        system=_JUDGE_SYSTEM,
        model=settings.llm_model_scoring,
        temperature=0.0,
    )
    if not isinstance(parsed, dict):
        logger.warning("Assessment grader returned nothing usable")
        return None

    marked = 0
    for entry in parsed.get("grades") or []:
        if not isinstance(entry, dict):
            continue
        grade_row = by_id.get(str(entry.get("question_id") or ""))
        if grade_row is None or not grade_row.is_pending:
            continue
        value = _as_float(entry.get("score"))
        if value is None:
            continue
        grade_row.fraction = min(1.0, max(0.0, value / 100.0))
        grade_row.method = "llm"
        comment = entry.get("comment")
        grade_row.comment = str(comment)[:1000] if comment else None
        marked += 1

    if marked:
        logger.info(
            "Assessment grader marked %d/%d free-form answers via %s",
            marked,
            len(judged),
            result.model if result else "unknown model",
        )
    feedback = parsed.get("feedback")
    return str(feedback)[:4000] if feedback else None


def _judge_prompt(judged: list[tuple[dict, str]]) -> str:
    lines = [
        "Mark each answer from 0 to 100 against its question and rubric.",
        "",
    ]
    for question, text in judged:
        lines.append(f"### Question id: {question['id']} ({question['type']})")
        lines.append(f"Prompt: {question['prompt']}")
        if question.get("rubric"):
            lines.append(f"Rubric: {question['rubric']}")
        lines.append(f"Answer: {text}")
        lines.append("")
    lines.append(
        'Reply with JSON: {"grades": [{"question_id": "...", "score": 0-100, '
        '"comment": "one sentence"}], "feedback": "two sentences overall"}'
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
async def create_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    name: str,
    questions: list,
    type: AssessmentType = AssessmentType.MCQ,
    description: str | None = None,
    duration_minutes: int = 60,
    passing_score: float = 60,
) -> AssessmentTemplate:
    template = AssessmentTemplate(
        organization_id=organization_id,
        name=name,
        type=type,
        description=description,
        questions_json=normalise_questions(questions),
        duration_minutes=duration_minutes,
        passing_score=passing_score,
    )
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def get_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> AssessmentTemplate:
    template = await get_scoped(
        session, AssessmentTemplate, template_id, organization_id
    )
    if template is None:
        raise NotFoundError("Assessment template not found")
    return template


async def list_templates(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    type: AssessmentType | None = None,
    is_active: bool | None = None,
    params: PaginationParams | None = None,
) -> tuple[list[AssessmentTemplate], int]:
    params = params or PaginationParams()
    filters: list[ColumnElement[bool]] = []
    if type is not None:
        filters.append(AssessmentTemplate.type == type)
    if is_active is not None:
        filters.append(AssessmentTemplate.is_active.is_(is_active))

    stmt = scoped_select(AssessmentTemplate, organization_id)
    for condition in filters:
        stmt = stmt.where(condition)

    total = await count_scoped(session, AssessmentTemplate, organization_id, *filters)
    rows = await session.execute(
        stmt.order_by(AssessmentTemplate.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    return list(rows.scalars().all()), total


async def update_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    template_id: uuid.UUID,
    **changes: Any,
) -> AssessmentTemplate:
    """Edit a template. In-flight assessments keep the paper they were issued."""
    template = await get_template(session, organization_id, template_id)
    if "questions" in changes:
        questions = changes.pop("questions")
        if questions is not None:
            template.questions_json = normalise_questions(questions)
    for attribute, value in changes.items():
        if value is not None:
            setattr(template, attribute, value)
    await session.commit()
    await session.refresh(template)
    return template


async def delete_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> None:
    template = await get_template(session, organization_id, template_id)
    template.soft_delete()
    await session.commit()


# --------------------------------------------------------------------------- #
# Issuing
# --------------------------------------------------------------------------- #
async def issue(
    session: AsyncSession,
    organization_id: uuid.UUID,
    application_id: uuid.UUID,
    *,
    template_id: uuid.UUID | None = None,
    questions: list | None = None,
    type: AssessmentType | None = None,
    duration_minutes: int | None = None,
    passing_score: float | None = None,
    expires_in_hours: int | None = None,
) -> Assessment:
    """Issue an assessment against one application.

    Issuing makes the link live and marks the row ``sent``; putting the link in
    front of the candidate is the outreach engine's job (§4.2), which is why
    nothing here touches a mailbox.
    """
    application = await application_service.get_application(
        session, organization_id, application_id
    )
    if application.status in CLOSED_APPLICATION_STATUSES:
        raise ConflictError(
            f"Application is {application.status} and cannot be assessed",
            details={"status": application.status},
        )
    if await candidate_service.consent_refused(
        session, organization_id, application.candidate_id, ConsentType.ASSESSMENT
    ):
        raise ConflictError(
            "This candidate has withdrawn consent to be assessed",
            details={"candidate_id": str(application.candidate_id)},
        )

    template: AssessmentTemplate | None = None
    if template_id is not None:
        template = await get_template(session, organization_id, template_id)
        if not template.is_active:
            raise ConflictError(
                f"Assessment template '{template.name}' is not active"
            )

    if questions is not None:
        paper = normalise_questions(questions)
    elif template is not None:
        # Re-validated on the way out: a template written before a rule was
        # tightened must not produce a paper the grader cannot mark.
        paper = normalise_questions(template.questions_json)
    else:
        raise ValidationError("An assessment needs either a template or questions")

    open_existing = await session.scalar(
        scoped_select(Assessment, organization_id).where(
            Assessment.application_id == application_id,
            Assessment.status.in_(sorted(OPEN_STATUSES)),
        )
    )
    if open_existing is not None:
        raise ConflictError(
            "This application already has an assessment outstanding",
            details={
                "assessment_id": str(open_existing.id),
                "status": open_existing.status,
            },
        )

    now = datetime.now(UTC)
    ttl = expires_in_hours or settings.assessment_token_ttl_hours
    assessment = Assessment(
        organization_id=organization_id,
        application_id=application_id,
        template_id=template.id if template is not None else None,
        type=type
        or (template.type if template is not None else AssessmentType.MCQ),
        status=AssessmentStatus.SENT,
        questions_json=paper,
        passing_score=(
            passing_score
            if passing_score is not None
            else float(template.passing_score) if template is not None else 60.0
        ),
        duration_minutes=(
            duration_minutes
            if duration_minutes is not None
            else template.duration_minutes if template is not None else None
        ),
        invite_token=_new_invite_token(),
        sent_at=now,
        expires_at=now + timedelta(hours=ttl),
    )
    session.add(assessment)
    await session.commit()
    await session.refresh(assessment)
    return assessment


def _new_invite_token() -> str:
    return secrets.token_urlsafe(32)


def invite_url(assessment: Assessment) -> str | None:
    """The candidate-facing link, while the assessment still has a token."""
    if not assessment.invite_token:
        return None
    base = settings.public_base_url.rstrip("/")
    return f"{base}/assessment/{assessment.invite_token}"


# --------------------------------------------------------------------------- #
# Recruiter-side reads and actions
# --------------------------------------------------------------------------- #
async def get_assessment(
    session: AsyncSession, organization_id: uuid.UUID, assessment_id: uuid.UUID
) -> Assessment:
    assessment = await get_scoped(session, Assessment, assessment_id, organization_id)
    if assessment is None:
        raise NotFoundError("Assessment not found")
    return assessment


async def list_assessments(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    application_id: uuid.UUID | None = None,
    status: AssessmentStatus | None = None,
    params: PaginationParams | None = None,
) -> tuple[list[Assessment], int]:
    params = params or PaginationParams()
    filters: list[ColumnElement[bool]] = []
    if application_id is not None:
        filters.append(Assessment.application_id == application_id)
    if status is not None:
        filters.append(Assessment.status == status)

    stmt = scoped_select(Assessment, organization_id)
    for condition in filters:
        stmt = stmt.where(condition)

    total = await count_scoped(session, Assessment, organization_id, *filters)
    rows = await session.execute(
        stmt.order_by(Assessment.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    return list(rows.scalars().all()), total


async def cancel(
    session: AsyncSession,
    organization_id: uuid.UUID,
    assessment_id: uuid.UUID,
    *,
    reason: str | None = None,
) -> Assessment:
    """Withdraw an outstanding assessment and kill its link."""
    assessment = await get_assessment(session, organization_id, assessment_id)
    if assessment.status == AssessmentStatus.COMPLETED:
        raise ConflictError("A completed assessment cannot be cancelled")

    assessment.status = AssessmentStatus.SKIPPED
    assessment.invite_token = None
    if reason:
        assessment.ai_feedback = reason
    await session.commit()
    await session.refresh(assessment)
    return assessment


async def grade_manually(
    session: AsyncSession,
    organization_id: uuid.UUID,
    assessment_id: uuid.UUID,
    *,
    grades: dict[str, float],
    feedback: str | None = None,
    changed_by_id: uuid.UUID | None = None,
    advance_application: bool = True,
) -> Assessment:
    """Record a reviewer's marks and re-derive the verdict.

    Takes 0-100 per question id, and may override a mark the machine already
    made — the reviewer is the authority, not the tie-breaker.
    """
    assessment = await get_assessment(session, organization_id, assessment_id)
    if assessment.status != AssessmentStatus.COMPLETED:
        raise ConflictError(
            f"An assessment that is {assessment.status} has nothing to grade",
            details={"status": assessment.status},
        )

    breakdown = dict(assessment.breakdown_json or {})
    rows = [QuestionGrade.from_dict(r) for r in breakdown.get("questions") or []]
    by_id = {row.question_id: row for row in rows}

    unknown = sorted(set(grades) - set(by_id))
    if unknown:
        raise ValidationError(
            "Marks given for questions that are not on this paper",
            details={"unknown": unknown},
        )

    for question_id, value in grades.items():
        score = _as_float(value)
        if score is None or not 0 <= score <= 100:
            raise ValidationError(
                f"Mark for '{question_id}' must be between 0 and 100",
                details={"question_id": question_id, "score": value},
            )
        row = by_id[question_id]
        row.fraction = score / 100.0
        row.method = "manual"

    outcome = _finalise(
        rows,
        passing_score=float(assessment.passing_score),
        feedback=feedback or breakdown.get("feedback"),
    )
    _apply_outcome(assessment, outcome, feedback=feedback)
    await session.commit()
    await session.refresh(assessment)

    if advance_application and outcome.passed:
        await _advance(session, assessment, changed_by_id=changed_by_id)
        await session.refresh(assessment)
    return assessment


def _apply_outcome(
    assessment: Assessment, outcome: GradeOutcome, *, feedback: str | None = None
) -> None:
    assessment.score = outcome.score
    assessment.passed = outcome.passed
    assessment.breakdown_json = outcome.breakdown()
    if feedback or outcome.feedback:
        assessment.ai_feedback = feedback or outcome.feedback


async def _advance(
    session: AsyncSession,
    assessment: Assessment,
    *,
    changed_by_id: uuid.UUID | None = None,
) -> None:
    """Move a passing candidate's card to ``assessed``.

    Forward only, and only on a pass. A fail never auto-rejects: §4.9's line
    that these features assist rather than decide applies most sharply where
    the machine marked the paper itself.
    """
    application = await session.get(Application, assessment.application_id)
    if application is None or application.status in CLOSED_APPLICATION_STATUSES:
        return
    if STAGE_INDEX[PipelineStage(application.stage)] >= STAGE_INDEX[
        PipelineStage.ASSESSED
    ]:
        return

    await application_service.move_stage(
        session,
        assessment.organization_id,
        application.id,
        PipelineStage.ASSESSED,
        changed_by_id=changed_by_id,
        trigger="assessment_passed",
        note=f"Passed assessment with {assessment.score}",
        # The pass is evidence in its own right; a card that reached an
        # assessment without a screening row should not be stuck behind one.
        force=True,
    )


# --------------------------------------------------------------------------- #
# Candidate self-service (token-authenticated)
# --------------------------------------------------------------------------- #
async def get_by_invite_token(session: AsyncSession, token: str) -> Assessment:
    """Resolve an invite token to its assessment.

    A wrong token and an expired one look identical from outside, so a guess
    cannot be distinguished from a near miss.
    """
    if not token:
        raise NotFoundError("This assessment link is not valid")

    assessment = await session.scalar(
        select(Assessment).where(
            Assessment.invite_token == token,
            Assessment.deleted_at.is_(None),
        )
    )
    if assessment is None:
        raise NotFoundError("This assessment link is not valid")
    if _is_expired(assessment):
        raise NotFoundError("This assessment link has expired")
    return assessment


def _is_expired(assessment: Assessment, *, now: datetime | None = None) -> bool:
    if assessment.status == AssessmentStatus.EXPIRED:
        return True
    if assessment.status not in OPEN_STATUSES:
        return False
    if assessment.expires_at is None:
        return False
    expires = assessment.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return expires < (now or datetime.now(UTC))


async def start(session: AsyncSession, token: str) -> Assessment:
    """Start the clock. Idempotent, so a reload does not reset the timer."""
    assessment = await get_by_invite_token(session, token)
    if assessment.status == AssessmentStatus.IN_PROGRESS:
        return assessment
    if assessment.status not in OPEN_STATUSES:
        raise ConflictError(
            f"This assessment is {assessment.status} and cannot be started",
            details={"status": assessment.status},
        )

    assessment.status = AssessmentStatus.IN_PROGRESS
    assessment.started_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(assessment)
    return assessment


async def submit(
    session: AsyncSession,
    token: str,
    responses: dict | None,
    *,
    llm: OpenRouterClient | None = None,
    advance_application: bool = True,
) -> Assessment:
    """Accept the candidate's answers, mark them, and close the assessment.

    Answers for questions that are not on the paper are dropped rather than
    rejected — a stale tab should not cost somebody their submission — and an
    overrun clock is recorded, never punished. Whether a late paper counts is a
    hiring decision, and it needs the human who set the time limit, not a
    timeout in a request handler.
    """
    assessment = await get_by_invite_token(session, token)
    if assessment.status == AssessmentStatus.COMPLETED:
        raise ConflictError("This assessment has already been submitted")
    if assessment.status not in OPEN_STATUSES:
        raise ConflictError(
            f"This assessment is {assessment.status} and cannot be submitted",
            details={"status": assessment.status},
        )
    if not isinstance(responses, dict):
        raise ValidationError("Responses must be an object keyed by question id")

    questions = list(assessment.questions_json or [])
    known = {q["id"] for q in questions}
    answers = {k: v for k, v in responses.items() if k in known}

    now = datetime.now(UTC)
    started = assessment.started_at
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    elapsed = int((now - started).total_seconds()) if started else None

    outcome = await grade(
        questions,
        answers,
        passing_score=float(assessment.passing_score),
        llm=llm,
    )

    assessment.responses_json = answers
    assessment.status = AssessmentStatus.COMPLETED
    assessment.completed_at = now
    assessment.time_spent_seconds = elapsed
    _apply_outcome(assessment, outcome)

    breakdown = dict(assessment.breakdown_json or {})
    breakdown["over_time"] = _over_time(assessment, elapsed)
    breakdown["unanswered"] = sorted(known - set(answers))
    assessment.breakdown_json = breakdown

    await session.commit()
    await session.refresh(assessment)

    if advance_application and outcome.passed:
        await _advance(session, assessment)
        await session.refresh(assessment)
    return assessment


def _over_time(assessment: Assessment, elapsed: int | None) -> bool:
    if elapsed is None or not assessment.duration_minutes:
        return False
    allowed = (
        assessment.duration_minutes + settings.assessment_time_grace_minutes
    ) * 60
    return elapsed > allowed


# --------------------------------------------------------------------------- #
# Expiry sweep
# --------------------------------------------------------------------------- #
async def expire_due(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    organization_id: uuid.UUID | None = None,
    limit: int = 500,
) -> list[Assessment]:
    """Close out assessments whose window has passed.

    ``get_by_invite_token`` already refuses an expired link, so this is about
    the recruiter's view rather than access control: an assessment sitting at
    ``sent`` three weeks after its deadline reads as waiting on the candidate
    when it is in fact over.
    """
    moment = now or datetime.now(UTC)
    stmt = select(Assessment).where(
        Assessment.deleted_at.is_(None),
        Assessment.status.in_(sorted(OPEN_STATUSES)),
        Assessment.expires_at.is_not(None),
        Assessment.expires_at < moment,
    )
    if organization_id is not None:
        stmt = stmt.where(Assessment.organization_id == organization_id)

    rows = list(
        (await session.execute(stmt.order_by(Assessment.expires_at).limit(limit)))
        .scalars()
        .all()
    )
    for assessment in rows:
        assessment.status = AssessmentStatus.EXPIRED
        assessment.invite_token = None
    if rows:
        await session.commit()
        logger.info("Expired %d assessment(s)", len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
