"""Question validation and the grader itself (design §4.4).

These are the parts with arithmetic in them, exercised directly rather than
through HTTP so a wrong mark shows up as a wrong number instead of a 422.
"""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.services import assessment_service as svc
from tests.factories import FakeLLMClient


def question(**overrides) -> dict:
    return {
        "id": "q1",
        "type": "single_choice",
        "prompt": "Pick one",
        "options": ["A", "B", "C"],
        "expected": "B",
        **overrides,
    }


# --------------------------------------------------------------------------- #
# Question validation
# --------------------------------------------------------------------------- #
class TestNormaliseQuestions:
    def test_fills_in_the_defaults(self) -> None:
        [q] = svc.normalise_questions([{"prompt": "Pick one", "options": ["A", "B"], "expected": "A"}])
        assert q["id"] == "q1"
        assert q["type"] == "single_choice"
        assert q["weight"] == 1.0
        assert q["required"] is True

    def test_matches_the_expected_answer_case_insensitively(self) -> None:
        [q] = svc.normalise_questions([question(expected="b")])
        # Stored in the option's own casing so the answer key reads correctly.
        assert q["expected"] == ["B"]

    def test_keeps_the_rubric_for_free_text(self) -> None:
        [q] = svc.normalise_questions(
            [{"type": "long_text", "prompt": "Explain", "rubric": "Mentions indexes"}]
        )
        assert q["rubric"] == "Mentions indexes"

    def test_rejects_an_empty_paper(self) -> None:
        with pytest.raises(ValidationError):
            svc.normalise_questions([])

    def test_rejects_too_many_questions(self) -> None:
        paper = [question(id=f"q{i}") for i in range(svc.MAX_QUESTIONS + 1)]
        with pytest.raises(ValidationError):
            svc.normalise_questions(paper)

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param({"prompt": "x", "type": "telepathy"}, id="unknown type"),
            pytest.param(question(prompt="   "), id="blank prompt"),
            pytest.param(question(options=["A"]), id="one option"),
            pytest.param(question(options=["A", "a"]), id="duplicate options"),
            pytest.param(question(expected="Z"), id="answer not an option"),
            pytest.param(question(expected=[]), id="no answer"),
            pytest.param(
                question(expected=["A", "B"]), id="two answers for single choice"
            ),
            pytest.param(question(weight=0), id="zero weight"),
            pytest.param(question(weight=-3), id="negative weight"),
            pytest.param("not a question", id="not an object"),
        ],
    )
    def test_rejects_malformed_questions(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            svc.normalise_questions([bad])

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValidationError) as exc:
            svc.normalise_questions([question(id="same"), question(id="same")])
        assert exc.value.details["question"] == 2

    def test_reports_which_question_failed(self) -> None:
        with pytest.raises(ValidationError) as exc:
            svc.normalise_questions([question(), question(id="q2", options=["A"])])
        assert exc.value.details["index"] == 1


class TestCandidateView:
    def test_strips_the_answer_key(self) -> None:
        paper = svc.normalise_questions(
            [
                question(),
                {"type": "short_text", "prompt": "Complexity?", "accepted": ["O(n)"]},
                {"type": "long_text", "prompt": "Explain", "rubric": "secret rubric"},
            ]
        )
        visible = svc.candidate_view(paper)
        serialised = str(visible)
        assert "expected" not in serialised
        assert "accepted" not in serialised
        assert "secret rubric" not in serialised
        # What the candidate does need is still there.
        assert visible[0]["options"] == ["A", "B", "C"]
        assert visible[1]["prompt"] == "Complexity?"


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
MIXED_PAPER = [
    question(),
    {
        "id": "q2",
        "type": "multi_choice",
        "prompt": "Pick the two",
        "options": ["A", "B", "C", "D"],
        "expected": ["A", "C"],
    },
    {
        "id": "q3",
        "type": "short_text",
        "prompt": "Complexity of a good sort?",
        "accepted": ["O(n log n)"],
    },
]


class TestAutoGrading:
    async def test_marks_a_perfect_paper(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(MIXED_PAPER),
            {"q1": "B", "q2": ["A", "C"], "q3": "It is O(n log n)."},
            passing_score=60,
        )
        assert outcome.score == 100.0
        assert outcome.passed is True
        assert outcome.engine == "auto"
        assert outcome.pending == []

    async def test_choice_answers_are_case_insensitive(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions([question()]), {"q1": "  b "}, passing_score=60
        )
        assert outcome.score == 100.0

    async def test_multi_choice_gives_partial_credit(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions([MIXED_PAPER[1]]), {"q2": ["A"]}, passing_score=60
        )
        assert outcome.score == 50.0
        assert outcome.passed is False

    async def test_a_wrong_pick_cancels_a_right_one(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions([MIXED_PAPER[1]]),
            {"q2": ["A", "B", "C"]},
            passing_score=60,
        )
        assert outcome.score == 50.0

    async def test_selecting_everything_scores_nothing(self) -> None:
        """Otherwise ticking every box would be the optimal strategy."""
        outcome = await svc.grade(
            svc.normalise_questions([MIXED_PAPER[1]]),
            {"q2": ["A", "B", "C", "D"]},
            passing_score=60,
        )
        assert outcome.score == 0.0

    async def test_an_unanswered_question_scores_zero_rather_than_pending(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(MIXED_PAPER), {"q1": "B"}, passing_score=60
        )
        assert outcome.pending == []
        assert outcome.passed is False
        methods = {g.question_id: g.method for g in outcome.grades}
        assert methods["q2"] == "unanswered"

    async def test_weights_shift_the_total(self) -> None:
        paper = svc.normalise_questions(
            [question(weight=3), {**MIXED_PAPER[1], "weight": 1}]
        )
        # Right on the heavy question, wrong on the light one: 3 of 4.
        outcome = await svc.grade(paper, {"q1": "B", "q2": ["B"]}, passing_score=60)
        assert outcome.score == 75.0

    async def test_accepted_answers_match_inside_a_sentence(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions([MIXED_PAPER[2]]),
            {"q3": "Probably   o(N LOG N) for comparison sorts"},
            passing_score=60,
        )
        assert outcome.score == 100.0


class TestJudgedQuestions:
    PAPER = [
        question(),
        {"id": "q2", "type": "long_text", "prompt": "Explain indexes", "rubric": "B-trees"},
    ]

    async def test_free_text_waits_for_review_when_no_model_is_configured(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Indexes make lookups fast."},
            passing_score=60,
        )
        # The half that marks itself is scored; the verdict is not guessed.
        assert outcome.score == 100.0
        assert outcome.passed is None
        assert outcome.pending == ["q2"]
        assert outcome.engine == "mixed"

    async def test_the_model_marks_what_it_can(self) -> None:
        llm = FakeLLMClient(
            [
                {
                    "grades": [
                        {"question_id": "q2", "score": 40, "comment": "Thin"}
                    ],
                    "feedback": "Solid on the basics.",
                }
            ]
        )
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Indexes make lookups fast."},
            passing_score=60,
            llm=llm,
        )
        assert outcome.pending == []
        assert outcome.score == 70.0
        assert outcome.passed is True
        assert outcome.feedback == "Solid on the basics."
        assert outcome.engine == "mixed"

    async def test_the_prompt_carries_the_rubric_not_the_other_answers(self) -> None:
        llm = FakeLLMClient([{"grades": []}])
        await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Indexes make lookups fast."},
            passing_score=60,
            llm=llm,
        )
        prompt = llm.calls[0]["prompt"]
        assert "B-trees" in prompt
        assert "Explain indexes" in prompt
        # The self-marking question is not the model's business.
        assert "Pick one" not in prompt

    async def test_a_silent_model_leaves_the_question_pending(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Something"},
            passing_score=60,
            llm=FakeLLMClient([None]),
        )
        assert outcome.pending == ["q2"]
        assert outcome.passed is None

    async def test_prose_instead_of_json_leaves_the_question_pending(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Something"},
            passing_score=60,
            llm=FakeLLMClient(["Looks good to me!"]),
        )
        assert outcome.pending == ["q2"]

    async def test_an_out_of_range_mark_is_clamped(self) -> None:
        llm = FakeLLMClient(
            [{"grades": [{"question_id": "q2", "score": 900}]}]
        )
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Something"},
            passing_score=60,
            llm=llm,
        )
        assert outcome.score == 100.0

    async def test_a_mark_for_an_unknown_question_is_ignored(self) -> None:
        llm = FakeLLMClient(
            [{"grades": [{"question_id": "does-not-exist", "score": 100}]}]
        )
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "B", "q2": "Something"},
            passing_score=60,
            llm=llm,
        )
        assert outcome.pending == ["q2"]

    async def test_the_model_cannot_overwrite_a_self_marked_answer(self) -> None:
        """A tampering response must not turn a wrong tick into a right one."""
        llm = FakeLLMClient(
            [{"grades": [{"question_id": "q1", "score": 100}, {"question_id": "q2", "score": 0}]}]
        )
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER),
            {"q1": "C", "q2": "Something"},
            passing_score=60,
            llm=llm,
        )
        assert outcome.score == 0.0
        assert outcome.passed is False

    async def test_an_unanswered_free_text_never_reaches_the_model(self) -> None:
        llm = FakeLLMClient([{"grades": []}])
        outcome = await svc.grade(
            svc.normalise_questions(self.PAPER), {"q1": "B"}, passing_score=60
        )
        assert llm.calls == []
        assert outcome.pending == []
        assert outcome.score == 50.0

    async def test_a_video_answer_is_marked_from_its_transcript(self) -> None:
        paper = svc.normalise_questions(
            [{"id": "v1", "type": "video", "prompt": "Introduce yourself"}]
        )
        llm = FakeLLMClient([{"grades": [{"question_id": "v1", "score": 80}]}])
        outcome = await svc.grade(
            paper,
            {"v1": {"video_url": "https://cdn.test/v.mp4", "transcript": "Hello, I am Ada."}},
            passing_score=60,
            llm=llm,
        )
        assert outcome.score == 80.0
        assert "Hello, I am Ada." in llm.calls[0]["prompt"]

    async def test_a_video_with_no_transcript_is_left_for_a_human(self) -> None:
        paper = svc.normalise_questions(
            [{"id": "v1", "type": "video", "prompt": "Introduce yourself"}]
        )
        outcome = await svc.grade(
            paper,
            {"v1": {"video_url": "https://cdn.test/v.mp4"}},
            passing_score=60,
            llm=FakeLLMClient([{"grades": []}]),
        )
        assert outcome.score is None
        assert outcome.passed is None


class TestScoreShape:
    async def test_a_wholly_unmarkable_paper_has_no_score(self) -> None:
        paper = svc.normalise_questions(
            [{"id": "q1", "type": "code", "prompt": "Write fizzbuzz"}]
        )
        outcome = await svc.grade(paper, {"q1": "print(1)"}, passing_score=60)
        assert outcome.score is None
        assert outcome.passed is None
        assert outcome.breakdown()["graded_weight"] == 0

    async def test_the_breakdown_lists_every_question(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions(MIXED_PAPER),
            {"q1": "B", "q2": ["A"], "q3": "O(n log n)"},
            passing_score=60,
        )
        breakdown = outcome.breakdown()
        assert [q["question_id"] for q in breakdown["questions"]] == ["q1", "q2", "q3"]
        assert breakdown["total_weight"] == 3
        assert breakdown["engine"] == "auto"

    async def test_the_pass_mark_is_inclusive(self) -> None:
        outcome = await svc.grade(
            svc.normalise_questions([question(), {**MIXED_PAPER[1]}]),
            {"q1": "B", "q2": ["B"]},
            passing_score=50,
        )
        assert outcome.score == 50.0
        assert outcome.passed is True
