"""Assessments end to end (design §4.4).

The arc the design describes: author a paper, issue it against an application,
let the candidate sit it from a link with no account, mark it, and move the
pipeline card — plus the tenant, consent, and permission boundaries around it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.integrations import openrouter as openrouter_module
from app.models.assessment import Assessment
from app.models.enums import UserRole
from app.services import assessment_service
from tests.conftest import make_user
from tests.factories import FakeLLMClient
from tests.test_applications import apply_candidate, create_candidate, create_job

MCQ_QUESTIONS = [
    {
        "id": "q1",
        "type": "single_choice",
        "prompt": "Which index type does PostgreSQL use by default?",
        "options": ["B-tree", "Hash", "GiST"],
        "expected": "B-tree",
    },
    {
        "id": "q2",
        "type": "multi_choice",
        "prompt": "Which of these are ACID properties?",
        "options": ["Atomicity", "Availability", "Isolation", "Partitioning"],
        "expected": ["Atomicity", "Isolation"],
    },
]

MIXED_QUESTIONS = [
    MCQ_QUESTIONS[0],
    {
        "id": "q2",
        "type": "long_text",
        "prompt": "Explain how you would debug a slow query.",
        "rubric": "Mentions EXPLAIN and indexes",
    },
]

TEMPLATE = {
    "name": "Backend basics",
    "type": "mcq",
    "description": "Screening paper for backend roles",
    "questions": MCQ_QUESTIONS,
    "duration_minutes": 30,
    "passing_score": 60,
}


async def create_template(client: AsyncClient, headers: dict, **overrides) -> dict:
    resp = await client.post(
        "/api/v1/assessments/templates", headers=headers, json={**TEMPLATE, **overrides}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
async def scene(client: AsyncClient, auth_headers: dict) -> dict:
    """A job, a candidate, their application, and a stored paper."""
    job = await create_job(client, auth_headers)
    candidate = await create_candidate(client, auth_headers)
    application = await apply_candidate(
        client, auth_headers, job["id"], candidate["id"]
    )
    template = await create_template(client, auth_headers)
    return {
        "job": job,
        "candidate": candidate,
        "application": application,
        "template": template,
    }


async def issue(client: AsyncClient, headers: dict, scene: dict, **overrides) -> dict:
    payload = {
        "application_id": scene["application"]["id"],
        "template_id": scene["template"]["id"],
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/assessments", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def token_of(issued: dict) -> str:
    return issued["invite_url"].rsplit("/", 1)[-1]


async def fetch_assessment(session, assessment_id: str) -> Assessment:
    return await session.scalar(
        select(Assessment).where(Assessment.id == uuid.UUID(assessment_id))
    )


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
class TestTemplates:
    async def test_creating_a_template_normalises_the_paper(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        assert template["name"] == "Backend basics"
        assert template["duration_minutes"] == 30
        assert [q["id"] for q in template["questions_json"]] == ["q1", "q2"]
        assert template["questions_json"][0]["weight"] == 1.0
        assert template["is_active"] is True

    async def test_a_malformed_question_is_refused_with_its_position(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/assessments/templates",
            headers=auth_headers,
            json={
                **TEMPLATE,
                "questions": [
                    MCQ_QUESTIONS[0],
                    {"type": "single_choice", "prompt": "Pick", "options": ["only one"]},
                ],
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["code"] == "validation_error"
        assert detail["details"]["question"] == 2

    async def test_an_empty_paper_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/assessments/templates",
            headers=auth_headers,
            json={**TEMPLATE, "questions": []},
        )
        assert resp.status_code == 422

    async def test_templates_can_be_listed_and_filtered(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        await create_template(client, auth_headers)
        retired = await create_template(client, auth_headers, name="Retired paper")
        await client.patch(
            f"/api/v1/assessments/templates/{retired['id']}",
            headers=auth_headers,
            json={"is_active": False},
        )

        resp = await client.get(
            "/api/v1/assessments/templates", headers=auth_headers, params={"is_active": True}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["name"] == "Backend basics"

    async def test_editing_the_questions_revalidates_them(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.patch(
            f"/api/v1/assessments/templates/{template['id']}",
            headers=auth_headers,
            json={"questions": [{"type": "single_choice", "prompt": "Pick", "options": ["A"]}]},
        )
        assert resp.status_code == 422

    async def test_a_deleted_template_is_gone(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.delete(
            f"/api/v1/assessments/templates/{template['id']}", headers=auth_headers
        )
        assert resp.status_code == 200

        resp = await client.get(
            f"/api/v1/assessments/templates/{template['id']}", headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_another_tenant_cannot_read_the_template(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.get(
            f"/api/v1/assessments/templates/{template['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Issuing
# --------------------------------------------------------------------------- #
class TestIssuing:
    async def test_issuing_mints_a_live_link(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        assessment = issued["assessment"]
        assert assessment["status"] == "sent"
        assert assessment["sent_at"] is not None
        assert assessment["expires_at"] is not None
        assert issued["invite_url"].endswith(token_of(issued))
        assert "/assessment/" in issued["invite_url"]

    async def test_the_paper_is_snapshotted_from_the_template(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        assessment = issued["assessment"]
        assert [q["id"] for q in assessment["questions_json"]] == ["q1", "q2"]
        assert float(assessment["passing_score"]) == 60.0
        assert assessment["duration_minutes"] == 30

    async def test_editing_the_template_leaves_an_issued_paper_alone(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """The whole reason the questions are copied rather than referenced."""
        issued = await issue(client, auth_headers, scene)
        await client.patch(
            f"/api/v1/assessments/templates/{scene['template']['id']}",
            headers=auth_headers,
            json={
                "questions": [
                    {"type": "single_choice", "prompt": "Totally new", "options": ["A", "B"], "expected": "A"}
                ],
                "passing_score": 95,
            },
        )

        resp = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        assert resp.status_code == 200
        assessment = resp.json()
        assert [q["id"] for q in assessment["questions_json"]] == ["q1", "q2"]
        assert float(assessment["passing_score"]) == 60.0

    async def test_a_one_off_paper_needs_no_template(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "questions": MIXED_QUESTIONS,
                "type": "take_home",
                "passing_score": 70,
            },
        )
        assert resp.status_code == 201, resp.text
        assessment = resp.json()["assessment"]
        assert assessment["template_id"] is None
        assert assessment["type"] == "take_home"
        assert float(assessment["passing_score"]) == 70.0

    async def test_neither_a_template_nor_questions_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={"application_id": scene["application"]["id"]},
        )
        assert resp.status_code == 422

    async def test_only_one_assessment_may_be_outstanding(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        first = await issue(client, auth_headers, scene)
        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["details"]["assessment_id"] == first["assessment"]["id"]

    async def test_a_cancelled_assessment_frees_the_slot(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        first = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessments/{first['assessment']['id']}/cancel",
            headers=auth_headers,
            json={"reason": "Wrong paper"},
        )
        second = await issue(client, auth_headers, scene)
        assert second["assessment"]["id"] != first["assessment"]["id"]

    async def test_a_rejected_application_is_not_assessed(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await client.post(
            f"/api/v1/applications/{scene['application']['id']}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
            },
        )
        assert resp.status_code == 409

    async def test_a_withdrawn_consent_blocks_the_assessment(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            f"/api/v1/candidates/{scene['candidate']['id']}/consents",
            headers=auth_headers,
            json={"consent_type": "assessment", "granted": False},
        )
        assert resp.status_code == 201, resp.text

        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
            },
        )
        assert resp.status_code == 409

    async def test_silence_on_consent_is_not_a_refusal(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """Applying for the job is the opt-in; only a withdrawal stops this."""
        issued = await issue(client, auth_headers, scene)
        assert issued["assessment"]["status"] == "sent"

    async def test_an_inactive_template_cannot_be_issued(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        await client.patch(
            f"/api/v1/assessments/templates/{scene['template']['id']}",
            headers=auth_headers,
            json={"is_active": False},
        )
        resp = await client.post(
            "/api/v1/assessments",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
            },
        )
        assert resp.status_code == 409

    async def test_another_tenants_application_cannot_be_assessed(
        self, client: AsyncClient, second_org: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/assessments",
            headers=second_org["headers"],
            json={
                "application_id": scene["application"]["id"],
                "questions": MCQ_QUESTIONS,
            },
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# The candidate's side of the link
# --------------------------------------------------------------------------- #
class TestCandidateFlow:
    async def test_the_link_shows_the_paper_without_the_answers(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(f"/api/v1/assessment/{token_of(issued)}")
        assert resp.status_code == 200
        view = resp.json()
        assert view["question_count"] == 2
        assert view["job_title"] == scene["job"]["title"]
        assert view["organization_name"] == "Acme Talent"
        assert view["duration_minutes"] == 30
        assert "expected" not in str(view["questions"])
        assert view["questions"][0]["options"] == ["B-tree", "Hash", "GiST"]

    async def test_the_link_needs_no_authentication(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(f"/api/v1/assessment/{token_of(issued)}")
        assert resp.status_code == 200

    async def test_a_guessed_token_is_a_flat_not_found(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/assessment/definitely-not-a-token")
        assert resp.status_code == 404

    async def test_an_expired_link_is_indistinguishable_from_a_wrong_one(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        row = await fetch_assessment(session, issued["assessment"]["id"])
        row.expires_at = datetime.now(UTC) - timedelta(hours=1)
        await session.commit()

        resp = await client.get(f"/api/v1/assessment/{token_of(issued)}")
        assert resp.status_code == 404

    async def test_starting_sets_the_clock_and_is_idempotent(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)

        first = await client.post(f"/api/v1/assessment/{token}/start")
        assert first.status_code == 200
        assert first.json()["status"] == "in_progress"
        started = first.json()["started_at"]

        # A reload must not hand the candidate a fresh half hour.
        second = await client.post(f"/api/v1/assessment/{token}/start")
        assert second.status_code == 200
        assert second.json()["started_at"] == started

    async def test_submitting_marks_the_paper_and_advances_the_card(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)
        await client.post(f"/api/v1/assessment/{token}/start")

        resp = await client.post(
            f"/api/v1/assessment/{token}/submit",
            json={"responses": {"q1": "B-tree", "q2": ["Atomicity", "Isolation"]}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = detail.json()
        assert float(body["score"]) == 100.0
        assert body["passed"] is True
        assert body["responses_json"]["q1"] == "B-tree"
        assert body["time_spent_seconds"] is not None

        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "assessed"

    async def test_the_candidate_is_never_told_their_score(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)
        resp = await client.post(
            f"/api/v1/assessment/{token}/submit",
            json={"responses": {"q1": "B-tree", "q2": ["Atomicity", "Isolation"]}},
        )
        body = resp.json()
        assert "score" not in body
        assert "passed" not in body
        # And the questions are withheld once there is nothing left to answer.
        assert body["questions"] == []

    async def test_a_failing_paper_does_not_move_the_card(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """A machine-marked fail is evidence, not a rejection (§4.9)."""
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "Hash", "q2": ["Availability"]}},
        )

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        assert detail.json()["passed"] is False

        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "applied"
        assert application.json()["status"] == "active"

    async def test_a_paper_cannot_be_handed_in_twice(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)
        await client.post(
            f"/api/v1/assessment/{token}/submit", json={"responses": {"q1": "B-tree"}}
        )
        resp = await client.post(
            f"/api/v1/assessment/{token}/submit", json={"responses": {"q1": "Hash"}}
        )
        assert resp.status_code == 409

    async def test_a_submitted_paper_cannot_be_restarted(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)
        await client.post(
            f"/api/v1/assessment/{token}/submit", json={"responses": {"q1": "B-tree"}}
        )
        resp = await client.post(f"/api/v1/assessment/{token}/start")
        assert resp.status_code == 409

    async def test_answers_to_questions_not_on_the_paper_are_dropped(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """A stale tab should not cost somebody their submission."""
        issued = await issue(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree", "q99": "from an older version"}},
        )
        assert resp.status_code == 200

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = detail.json()
        assert set(body["responses_json"]) == {"q1"}
        assert body["breakdown_json"]["unanswered"] == ["q2"]

    async def test_an_overrun_clock_is_recorded_not_punished(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)
        await client.post(f"/api/v1/assessment/{token}/start")

        row = await fetch_assessment(session, issued["assessment"]["id"])
        row.started_at = datetime.now(UTC) - timedelta(hours=3)
        await session.commit()

        resp = await client.post(
            f"/api/v1/assessment/{token}/submit",
            json={"responses": {"q1": "B-tree", "q2": ["Atomicity", "Isolation"]}},
        )
        assert resp.status_code == 200

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = detail.json()
        assert body["breakdown_json"]["over_time"] is True
        assert body["passed"] is True
        assert body["time_spent_seconds"] > 3600

    async def test_a_never_started_paper_can_still_be_submitted(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree", "q2": ["Atomicity", "Isolation"]}},
        )
        assert resp.status_code == 200

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        assert detail.json()["time_spent_seconds"] is None


# --------------------------------------------------------------------------- #
# Free-form answers and review
# --------------------------------------------------------------------------- #
class TestReview:
    async def test_free_text_leaves_the_verdict_open(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(
            client, auth_headers, scene, template_id=None, questions=MIXED_QUESTIONS
        )
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree", "q2": "I would run EXPLAIN first."}},
        )

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = detail.json()
        assert body["passed"] is None
        assert body["breakdown_json"]["pending_review"] == ["q2"]

        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "applied"

    async def test_a_reviewers_marks_settle_it_and_move_the_card(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(
            client, auth_headers, scene, template_id=None, questions=MIXED_QUESTIONS
        )
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree", "q2": "I would run EXPLAIN first."}},
        )

        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=auth_headers,
            json={"grades": {"q2": 80}, "feedback": "Good instincts."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert float(body["score"]) == 90.0
        assert body["passed"] is True
        assert body["ai_feedback"] == "Good instincts."

        application = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert application.json()["stage"] == "assessed"

    async def test_a_reviewer_may_overrule_an_automatic_mark(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "Hash", "q2": ["Availability"]}},
        )
        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=auth_headers,
            json={"grades": {"q1": 100, "q2": 100}},
        )
        assert resp.status_code == 200
        assert float(resp.json()["score"]) == 100.0
        assert resp.json()["passed"] is True

    async def test_marks_for_a_question_that_is_not_on_the_paper_are_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree"}},
        )
        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=auth_headers,
            json={"grades": {"q7": 100}},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["details"]["unknown"] == ["q7"]

    async def test_a_mark_outside_the_scale_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree"}},
        )
        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=auth_headers,
            json={"grades": {"q1": 150}},
        )
        assert resp.status_code == 422

    async def test_an_unsat_paper_has_nothing_to_grade(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=auth_headers,
            json={"grades": {"q1": 100}},
        )
        assert resp.status_code == 409

    async def test_the_model_marks_the_free_text_when_one_is_configured(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        openrouter_module.set_llm_client(
            FakeLLMClient(
                [
                    {
                        "grades": [{"question_id": "q2", "score": 90, "comment": "Thorough"}],
                        "feedback": "Strong debugging instincts.",
                    }
                ]
            )
        )
        issued = await issue(
            client, auth_headers, scene, template_id=None, questions=MIXED_QUESTIONS
        )
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree", "q2": "EXPLAIN, then check the indexes."}},
        )

        detail = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = detail.json()
        assert body["breakdown_json"]["pending_review"] == []
        assert float(body["score"]) == 95.0
        assert body["passed"] is True
        assert body["ai_feedback"] == "Strong debugging instincts."


# --------------------------------------------------------------------------- #
# Cancelling and expiry
# --------------------------------------------------------------------------- #
class TestCancellationAndExpiry:
    async def test_cancelling_kills_the_link(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        token = token_of(issued)

        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/cancel",
            headers=auth_headers,
            json={"reason": "Sent by mistake"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

        assert (await client.get(f"/api/v1/assessment/{token}")).status_code == 404

    async def test_a_sat_paper_cannot_be_cancelled(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree"}},
        )
        resp = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/cancel",
            headers=auth_headers,
            json={},
        )
        assert resp.status_code == 409

    async def test_the_sweep_closes_out_an_overdue_assessment(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        row = await fetch_assessment(session, issued["assessment"]["id"])
        row.expires_at = datetime.now(UTC) - timedelta(days=1)
        await session.commit()

        expired = await assessment_service.expire_due(session)
        assert [a.id for a in expired] == [row.id]

        resp = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}", headers=auth_headers
        )
        body = resp.json()
        assert body["status"] == "expired"
        # And the link goes with it.
        assert body["invite_url"] is None

    async def test_the_sweep_leaves_a_live_assessment_alone(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        await issue(client, auth_headers, scene)
        assert await assessment_service.expire_due(session) == []

    async def test_the_sweep_does_not_reopen_a_completed_paper(
        self, client: AsyncClient, auth_headers: dict, scene: dict, session
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree"}},
        )
        row = await fetch_assessment(session, issued["assessment"]["id"])
        row.expires_at = datetime.now(UTC) - timedelta(days=1)
        await session.commit()

        assert await assessment_service.expire_due(session) == []


# --------------------------------------------------------------------------- #
# Listing, permissions, and tenancy
# --------------------------------------------------------------------------- #
class TestAccess:
    async def test_assessments_can_be_listed_per_application(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(
            "/api/v1/assessments",
            headers=auth_headers,
            params={"application_id": scene["application"]["id"], "status": "sent"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["id"] == issued["assessment"]["id"]

    async def test_a_recruiter_can_issue_an_assessment(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
        resp = await client.post(
            "/api/v1/assessments",
            headers=recruiter["headers"],
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
            },
        )
        assert resp.status_code == 201, resp.text

    async def test_a_hiring_manager_marks_but_does_not_issue(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        manager = await make_user(client, auth_headers, UserRole.HIRING_MANAGER)
        issued = await issue(client, auth_headers, scene)
        await client.post(
            f"/api/v1/assessment/{token_of(issued)}/submit",
            json={"responses": {"q1": "B-tree"}},
        )

        blocked = await client.post(
            "/api/v1/assessments",
            headers=manager["headers"],
            json={
                "application_id": scene["application"]["id"],
                "questions": MCQ_QUESTIONS,
            },
        )
        assert blocked.status_code == 403

        allowed = await client.post(
            f"/api/v1/assessments/{issued['assessment']['id']}/grade",
            headers=manager["headers"],
            json={"grades": {"q2": 50}},
        )
        assert allowed.status_code == 200

    async def test_an_interviewer_sees_no_assessments(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}",
            headers=interviewer["headers"],
        )
        assert resp.status_code == 403

    async def test_another_tenant_cannot_read_the_result(
        self, client: AsyncClient, auth_headers: dict, second_org: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(
            f"/api/v1/assessments/{issued['assessment']['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404

    async def test_an_unauthenticated_recruiter_route_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        issued = await issue(client, auth_headers, scene)
        resp = await client.get(f"/api/v1/assessments/{issued['assessment']['id']}")
        assert resp.status_code == 401
