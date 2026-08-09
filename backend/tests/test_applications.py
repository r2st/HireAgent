"""Applications: pipeline stages, screening automation, the board, and tenancy."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.models.enums import STAGE_ORDER, UserRole
from tests.conftest import make_user
from tests.factories import SAMPLE_RESUME, make_pdf
from tests.test_jobs import PUBLISHABLE


async def create_job(client: AsyncClient, headers: dict, **overrides) -> dict:
    resp = await client.post(
        "/api/v1/jobs", headers=headers, json={**PUBLISHABLE, **overrides}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def create_candidate(client: AsyncClient, headers: dict, **overrides) -> dict:
    payload = {
        "full_name": "Grace Hopper",
        "email": f"grace-{uuid.uuid4().hex[:8]}@example.com",
        "location": "Bengaluru",
        "experience_years": 6,
        "skills": ["Python", "PostgreSQL", "Kubernetes"],
        "source": "direct",
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/candidates", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def apply_candidate(
    client: AsyncClient, headers: dict, job_id: str, candidate_id: str, **overrides
) -> dict:
    resp = await client.post(
        "/api/v1/applications",
        headers=headers,
        json={"job_id": job_id, "candidate_id": candidate_id, **overrides},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
async def pipeline(client: AsyncClient, auth_headers: dict) -> dict:
    """One job, one candidate, and the application joining them."""
    job = await create_job(client, auth_headers)
    candidate = await create_candidate(client, auth_headers)
    application = await apply_candidate(
        client, auth_headers, job["id"], candidate["id"]
    )
    return {"job": job, "candidate": candidate, "application": application}


async def upload_resume(client: AsyncClient, headers: dict, candidate_id: str) -> dict:
    """Attach a parsed resume, which is what lifts screening confidence.

    Without a parsed resume the rule-based scorer stays below the confidence
    threshold and deliberately declines to automate anything.
    """
    resp = await client.post(
        "/api/v1/candidates/upload-resume",
        headers=headers,
        files={"file": ("ada.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")},
        data={"candidate_id": candidate_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
class TestCreateApplication:
    async def test_application_starts_in_applied(self, pipeline: dict) -> None:
        application = pipeline["application"]
        assert application["stage"] == "applied"
        assert application["status"] == "active"
        assert application["score"] is None
        assert application["applied_at"] is not None

    async def test_creation_records_an_opening_stage_event(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/applications/{pipeline['application']['id']}/events",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        events = resp.json()
        assert len(events) == 1
        assert events[0]["from_stage"] is None
        assert events[0]["to_stage"] == "applied"
        assert events[0]["trigger"] == "manual"

    async def test_import_source_is_recorded_as_an_import_trigger(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(client, auth_headers)
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"], source="import"
        )

        resp = await client.get(
            f"/api/v1/applications/{application['id']}/events", headers=auth_headers
        )
        assert resp.json()[0]["trigger"] == "import"

    async def test_can_start_at_a_non_default_stage(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(client, auth_headers)
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"], stage="sourced"
        )
        assert application["stage"] == "sourced"

    async def test_duplicate_application_is_rejected(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/applications",
            headers=auth_headers,
            json={
                "job_id": pipeline["job"]["id"],
                "candidate_id": pipeline["candidate"]["id"],
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "conflict"
        # The existing application is named so the UI can navigate to it.
        assert detail["details"]["application_id"] == pipeline["application"]["id"]

    async def test_same_candidate_can_apply_to_a_different_job(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        other_job = await create_job(client, auth_headers, title="Platform Engineer")
        application = await apply_candidate(
            client, auth_headers, other_job["id"], pipeline["candidate"]["id"]
        )
        assert application["job_id"] == other_job["id"]

    async def test_closed_job_stops_accepting_applications(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(client, auth_headers)
        closed = await client.post(
            f"/api/v1/jobs/{job['id']}/status",
            headers=auth_headers,
            json={"status": "closed"},
        )
        assert closed.status_code == 200, closed.text

        resp = await client.post(
            "/api/v1/applications",
            headers=auth_headers,
            json={"job_id": job["id"], "candidate_id": candidate["id"]},
        )
        assert resp.status_code == 409
        assert "closed" in resp.json()["detail"]["message"]

    async def test_blacklisted_candidate_cannot_be_applied(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(client, auth_headers)
        blacklisted = await client.patch(
            f"/api/v1/candidates/{candidate['id']}",
            headers=auth_headers,
            json={"is_blacklisted": True},
        )
        assert blacklisted.status_code == 200, blacklisted.text

        resp = await client.post(
            "/api/v1/applications",
            headers=auth_headers,
            json={"job_id": job["id"], "candidate_id": candidate["id"]},
        )
        assert resp.status_code == 409
        assert "blacklisted" in resp.json()["detail"]["message"]

    async def test_unknown_job_is_not_found(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        candidate = await create_candidate(client, auth_headers)
        resp = await client.post(
            "/api/v1/applications",
            headers=auth_headers,
            json={"job_id": str(uuid.uuid4()), "candidate_id": candidate["id"]},
        )
        assert resp.status_code == 404

    async def test_cannot_graft_another_tenants_candidate_onto_a_local_job(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        """The ids are re-read through the tenant scope, not trusted."""
        job = await create_job(client, auth_headers)
        foreign = await create_candidate(client, second_org["headers"])

        resp = await client.post(
            "/api/v1/applications",
            headers=auth_headers,
            json={"job_id": job["id"], "candidate_id": foreign["id"]},
        )
        assert resp.status_code == 404

    async def test_cards_entering_a_column_get_increasing_positions(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        positions = []
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            application = await apply_candidate(
                client, auth_headers, job["id"], candidate["id"]
            )
            positions.append(float(application["board_position"]))
        assert positions == sorted(positions)
        assert len(set(positions)) == 3


# --------------------------------------------------------------------------- #
# Reading and filtering
# --------------------------------------------------------------------------- #
class TestListApplications:
    async def test_lists_only_the_callers_tenant(
        self, client: AsyncClient, auth_headers: dict, second_org: dict, pipeline: dict
    ) -> None:
        foreign_job = await create_job(client, second_org["headers"])
        foreign_candidate = await create_candidate(client, second_org["headers"])
        await apply_candidate(
            client, second_org["headers"], foreign_job["id"], foreign_candidate["id"]
        )

        resp = await client.get("/api/v1/applications", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["id"] == pipeline["application"]["id"]

    async def test_filters_by_job(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        other_job = await create_job(client, auth_headers, title="Data Engineer")
        other_candidate = await create_candidate(client, auth_headers)
        await apply_candidate(
            client, auth_headers, other_job["id"], other_candidate["id"]
        )

        resp = await client.get(
            "/api/v1/applications",
            headers=auth_headers,
            params={"job_id": other_job["id"]},
        )
        body = resp.json()
        # The fixture's application belongs to a different job and is excluded.
        assert body["total"] == 1
        assert body["items"][0]["id"] != pipeline["application"]["id"]

    async def test_filters_by_candidate(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            "/api/v1/applications",
            headers=auth_headers,
            params={"candidate_id": pipeline["candidate"]["id"]},
        )
        assert resp.json()["total"] == 1

    async def test_filters_by_stage_and_status(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "screened"},
        )

        screened = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"stage": "screened"}
        )
        assert screened.json()["total"] == 1

        applied = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"stage": "applied"}
        )
        assert applied.json()["total"] == 0

        active = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"status": "active"}
        )
        assert active.json()["total"] == 1

    async def test_filters_by_source(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for source in ("referral", "linkedin"):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(
                client, auth_headers, job["id"], candidate["id"], source=source
            )

        resp = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"source": "referral"}
        )
        assert resp.json()["total"] == 1

    async def test_filters_by_assignee(
        self, client: AsyncClient, auth_headers: dict, registered: dict, pipeline: dict
    ) -> None:
        me = registered["user"]["id"]
        await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/assignee",
            headers=auth_headers,
            json={"user_id": me},
        )

        mine = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"assigned_to_id": me}
        )
        assert mine.json()["total"] == 1

        theirs = await client.get(
            "/api/v1/applications",
            headers=auth_headers,
            params={"assigned_to_id": str(uuid.uuid4())},
        )
        assert theirs.json()["total"] == 0

    async def test_unscored_applications_sort_last_not_as_zero(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        """A fresh application must not be presented as a bad one."""
        job = await create_job(client, auth_headers)
        scored_candidate = await create_candidate(client, auth_headers)
        scored = await apply_candidate(
            client, auth_headers, job["id"], scored_candidate["id"]
        )
        unscored_candidate = await create_candidate(client, auth_headers)
        unscored = await apply_candidate(
            client, auth_headers, job["id"], unscored_candidate["id"]
        )

        await client.post(
            f"/api/v1/applications/{scored['id']}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )

        resp = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"order_by": "score"}
        )
        ids = [item["id"] for item in resp.json()["items"]]
        assert ids == [scored["id"], unscored["id"]]

    async def test_min_score_filter_excludes_unscored(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        assert pipeline["application"]["score"] is None

        resp = await client.get(
            "/api/v1/applications", headers=auth_headers, params={"min_score": 1}
        )
        assert resp.json()["total"] == 0

    async def test_pagination_reports_the_full_total(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(client, auth_headers, job["id"], candidate["id"])

        resp = await client.get(
            "/api/v1/applications",
            headers=auth_headers,
            params={"page": 2, "page_size": 2},
        )
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 1
        assert body["page"] == 2


class TestGetApplication:
    async def test_detail_embeds_the_candidate_and_job(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/applications/{pipeline['application']['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["candidate"]["id"] == pipeline["candidate"]["id"]
        assert body["job"]["id"] == pipeline["job"]["id"]
        assert body["latest_screening"] is None

    async def test_detail_carries_the_latest_screening(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )

        resp = await client.get(
            f"/api/v1/applications/{application_id}", headers=auth_headers
        )
        assert resp.json()["latest_screening"] is not None

    async def test_another_tenant_cannot_read_it(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/applications/{pipeline['application']['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404

    async def test_unknown_id_is_not_found(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/applications/{uuid.uuid4()}", headers=auth_headers
        )
        assert resp.status_code == 404


class TestUpdateApplication:
    async def test_updates_the_source_detail(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.patch(
            f"/api/v1/applications/{pipeline['application']['id']}",
            headers=auth_headers,
            json={"source_detail": "Referred by Ada"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["source_detail"] == "Referred by Ada"

    async def test_another_tenant_cannot_update(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.patch(
            f"/api/v1/applications/{pipeline['application']['id']}",
            headers=second_org["headers"],
            json={"source_detail": "hijacked"},
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Stage transitions
# --------------------------------------------------------------------------- #
class TestStageMoves:
    async def test_moves_forward_and_records_the_transition(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "screened", "note": "Looks strong"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["stage"] == "screened"

        events = (
            await client.get(
                f"/api/v1/applications/{application_id}/events", headers=auth_headers
            )
        ).json()
        assert [e["to_stage"] for e in events] == ["applied", "screened"]
        assert events[-1]["from_stage"] == "applied"
        assert events[-1]["note"] == "Looks strong"
        # Time-in-stage is what makes funnel analytics possible after the fact.
        assert events[-1]["seconds_in_previous_stage"] is not None

    async def test_moving_backwards_is_allowed(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        """A candidate genuinely does get sent back for another round."""
        application_id = pipeline["application"]["id"]
        await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "interviewed"},
        )
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "screened"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "screened"

    async def test_moving_to_the_same_stage_is_a_no_op(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "applied"},
        )
        assert resp.status_code == 200

        events = (
            await client.get(
                f"/api/v1/applications/{application_id}/events", headers=auth_headers
            )
        ).json()
        assert len(events) == 1

    async def test_explicit_board_position_is_honoured(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "screened", "board_position": 250.5},
        )
        assert float(resp.json()["board_position"]) == 250.5

    @pytest.mark.parametrize("stage", ["offered", "hired"])
    async def test_cannot_reach_an_offer_without_screening(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict, stage: str
    ) -> None:
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": stage},
        )
        assert resp.status_code == 422
        assert "screened" in resp.json()["detail"]["message"]

    async def test_force_overrides_the_screening_gate(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "offered", "force": True},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "offered"

    async def test_screening_opens_the_gate(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "offered"},
        )
        assert resp.status_code == 200

    async def test_reaching_hired_sets_the_hired_status(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "hired", "force": True},
        )
        body = resp.json()
        assert body["status"] == "hired"
        assert body["hired_at"] is not None

    async def test_moving_back_out_of_hired_clears_the_hire(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )
        hired = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "hired"},
        )
        assert hired.json()["status"] == "hired"

        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "offered"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "active"
        assert body["hired_at"] is None

    async def test_a_rejected_application_cannot_be_moved(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "screened"},
        )
        assert resp.status_code == 409

    async def test_another_tenant_cannot_move_a_card(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=second_org["headers"],
            json={"stage": "hired", "force": True},
        )
        assert resp.status_code == 404


class TestBulkStageMove:
    async def test_moves_every_valid_application(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        ids = []
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            application = await apply_candidate(
                client, auth_headers, job["id"], candidate["id"]
            )
            ids.append(application["id"])

        resp = await client.post(
            "/api/v1/applications/bulk/stage",
            headers=auth_headers,
            json={"application_ids": ids, "stage": "screened"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["total"], body["moved"], body["failed"]) == (3, 3, 0)
        assert {a["stage"] for a in body["applications"]} == {"screened"}

    async def test_one_bad_card_does_not_abort_the_batch(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        missing = str(uuid.uuid4())
        resp = await client.post(
            "/api/v1/applications/bulk/stage",
            headers=auth_headers,
            json={
                "application_ids": [pipeline["application"]["id"], missing],
                "stage": "screened",
            },
        )
        body = resp.json()
        assert (body["total"], body["moved"], body["failed"]) == (2, 1, 1)
        assert body["errors"][0]["application_id"] == missing

    async def test_bulk_moves_are_tagged_as_bulk(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            "/api/v1/applications/bulk/stage",
            headers=auth_headers,
            json={"application_ids": [application_id], "stage": "screened"},
        )
        events = (
            await client.get(
                f"/api/v1/applications/{application_id}/events", headers=auth_headers
            )
        ).json()
        assert events[-1]["trigger"] == "bulk"

    async def test_screening_gate_applies_in_bulk_too(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/applications/bulk/stage",
            headers=auth_headers,
            json={
                "application_ids": [pipeline["application"]["id"]],
                "stage": "offered",
            },
        )
        body = resp.json()
        assert body["moved"] == 0
        assert body["failed"] == 1

    async def test_empty_batch_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/applications/bulk/stage",
            headers=auth_headers,
            json={"application_ids": [], "stage": "screened"},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Rejection, withdrawal, assignment, deletion
# --------------------------------------------------------------------------- #
class TestOutcomes:
    async def test_reject_records_the_reason_and_keeps_the_stage(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        """The stage is preserved so analytics can see where candidates are lost."""
        application_id = pipeline["application"]["id"]
        await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "interviewed"},
        )
        resp = await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Wants a different role"},
        )
        body = resp.json()
        assert body["status"] == "rejected"
        assert body["stage"] == "interviewed"
        assert body["rejection_reason"] == "Wants a different role"
        assert body["rejected_at"] is not None

    async def test_rejecting_twice_is_idempotent(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        first = await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        second = await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Changed my mind"},
        )
        assert second.status_code == 200
        assert second.json()["rejection_reason"] == first.json()["rejection_reason"]

    async def test_withdrawal_is_distinct_from_rejection(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.post(
            f"/api/v1/applications/{pipeline['application']['id']}/withdraw",
            headers=auth_headers,
            json={"reason": "Accepted another offer"},
        )
        body = resp.json()
        assert body["status"] == "withdrawn"
        assert body["rejected_at"] is None

    async def test_reopen_restores_an_active_application(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Too early"},
        )
        resp = await client.post(
            f"/api/v1/applications/{application_id}/reopen", headers=auth_headers
        )
        body = resp.json()
        assert body["status"] == "active"
        assert body["rejected_at"] is None
        assert body["rejection_reason"] is None

    async def test_reopened_applications_can_move_again(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        await client.post(
            f"/api/v1/applications/{application_id}/reject",
            headers=auth_headers,
            json={"reason": "Too early"},
        )
        await client.post(
            f"/api/v1/applications/{application_id}/reopen", headers=auth_headers
        )
        resp = await client.put(
            f"/api/v1/applications/{application_id}/stage",
            headers=auth_headers,
            json={"stage": "screened"},
        )
        assert resp.status_code == 200

    async def test_assignment_can_be_set_and_cleared(
        self, client: AsyncClient, auth_headers: dict, registered: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        assigned = await client.put(
            f"/api/v1/applications/{application_id}/assignee",
            headers=auth_headers,
            json={"user_id": registered["user"]["id"]},
        )
        assert assigned.json()["assigned_to_id"] == registered["user"]["id"]

        cleared = await client.put(
            f"/api/v1/applications/{application_id}/assignee",
            headers=auth_headers,
            json={"user_id": None},
        )
        assert cleared.json()["assigned_to_id"] is None


class TestDeleteApplication:
    async def test_delete_hides_it_from_reads(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        resp = await client.delete(
            f"/api/v1/applications/{application_id}", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text

        assert (
            await client.get(
                f"/api/v1/applications/{application_id}", headers=auth_headers
            )
        ).status_code == 404
        listed = await client.get("/api/v1/applications", headers=auth_headers)
        assert listed.json()["total"] == 0

    async def test_delete_is_soft(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict, session
    ) -> None:
        """The row survives for audit; only ``deleted_at`` changes."""
        from sqlalchemy import text

        application_id = pipeline["application"]["id"]
        await client.delete(
            f"/api/v1/applications/{application_id}", headers=auth_headers
        )

        row = (
            await session.execute(
                text("SELECT deleted_at FROM applications WHERE id = :id"),
                # SQLite stores UUIDs undashed.
                {"id": uuid.UUID(application_id).hex},
            )
        ).first()
        assert row is not None
        assert row[0] is not None

    async def test_another_tenant_cannot_delete(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.delete(
            f"/api/v1/applications/{pipeline['application']['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
class TestScreening:
    async def test_screening_scores_the_application(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.post(
            f"/api/v1/applications/{pipeline['application']['id']}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        screening = body["screening"]
        assert 0 <= float(screening["overall_score"]) <= 100
        # No LLM is configured in tests, so this must degrade, not fail.
        assert screening["engine"] == "heuristic"
        assert body["application"]["score"] == screening["overall_score"]

    async def test_screening_history_is_append_only(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        application_id = pipeline["application"]["id"]
        for _ in range(2):
            await client.post(
                f"/api/v1/applications/{application_id}/screen",
                headers=auth_headers,
                json={"auto_advance": False},
            )

        resp = await client.get(
            f"/api/v1/applications/{application_id}/screenings", headers=auth_headers
        )
        assert len(resp.json()) == 2

    async def test_a_thin_profile_is_flagged_for_human_review(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        """Confidence, not the score, is what stops a guess being trusted."""
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(
            client, auth_headers, skills=[], experience_years=None
        )
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"]
        )

        resp = await client.post(
            f"/api/v1/applications/{application['id']}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )
        assert resp.json()["screening"]["requires_human_review"] is True

    async def test_low_confidence_screening_never_triggers_automation(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        """Auto-rejecting on a parse the system does not trust erodes trust."""
        job = await create_job(client, auth_headers, auto_reject_threshold=99)
        candidate = await create_candidate(
            client, auth_headers, skills=[], experience_years=None
        )
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"]
        )

        resp = await client.post(
            f"/api/v1/applications/{application['id']}/screen", headers=auth_headers
        )
        body = resp.json()
        assert body["screening"]["requires_human_review"] is True
        assert body["application"]["status"] == "active"

    async def test_a_confident_high_score_auto_advances(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers, auto_advance_threshold=1)
        candidate = await create_candidate(client, auth_headers)
        await upload_resume(client, auth_headers, candidate["id"])
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"]
        )

        resp = await client.post(
            f"/api/v1/applications/{application['id']}/screen", headers=auth_headers
        )
        body = resp.json()
        assert body["screening"]["requires_human_review"] is False
        assert body["application"]["stage"] == "screened"

        events = (
            await client.get(
                f"/api/v1/applications/{application['id']}/events", headers=auth_headers
            )
        ).json()
        assert events[-1]["trigger"] == "auto_advance"

    async def test_a_confident_low_score_is_auto_rejected(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        # A resume that is genuinely wrong for the job: none of the required
        # skills, far too little experience, the wrong degree.
        job = await create_job(
            client,
            auth_headers,
            title="Principal SAP Consultant",
            min_experience_years=25,
            max_experience_years=30,
            auto_reject_threshold=50,
            auto_advance_threshold=90,
            requirements={
                "required_skills": [
                    {"name": "SAP", "weight": 3},
                    {"name": "Salesforce", "weight": 3},
                    {"name": "COBOL", "weight": 2},
                ],
                "education": {
                    "degree": "PhD",
                    "field_of_study": "Physics",
                    "is_mandatory": True,
                },
            },
        )
        candidate = await create_candidate(client, auth_headers, skills=[])
        await upload_resume(client, auth_headers, candidate["id"])
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"]
        )

        resp = await client.post(
            f"/api/v1/applications/{application['id']}/screen", headers=auth_headers
        )
        body = resp.json()
        assert body["screening"]["requires_human_review"] is False
        assert body["application"]["status"] == "rejected"
        assert "threshold" in body["application"]["rejection_reason"]

    async def test_auto_advance_can_be_declined_per_request(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers, auto_advance_threshold=1)
        candidate = await create_candidate(client, auth_headers)
        await upload_resume(client, auth_headers, candidate["id"])
        application = await apply_candidate(
            client, auth_headers, job["id"], candidate["id"]
        )

        resp = await client.post(
            f"/api/v1/applications/{application['id']}/screen",
            headers=auth_headers,
            json={"auto_advance": False},
        )
        assert resp.json()["application"]["stage"] == "applied"

    async def test_screening_another_tenants_application_is_not_found(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.post(
            f"/api/v1/applications/{pipeline['application']['id']}/screen",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


class TestBulkScreen:
    async def test_screens_a_jobs_unscored_applications(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(client, auth_headers, job["id"], candidate["id"])

        resp = await client.post(
            f"/api/v1/jobs/{job['id']}/screen", headers=auth_headers, json={}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["total"], body["succeeded"], body["failed"]) == (3, 3, 0)

    async def test_already_scored_applications_are_skipped(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        """Re-running after new applicants must not burn credits on old ones."""
        job_id = pipeline["job"]["id"]
        await client.post(f"/api/v1/jobs/{job_id}/screen", headers=auth_headers, json={})

        again = await client.post(
            f"/api/v1/jobs/{job_id}/screen", headers=auth_headers, json={}
        )
        assert again.json()["total"] == 0

    async def test_rescore_revisits_scored_applications(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        job_id = pipeline["job"]["id"]
        await client.post(f"/api/v1/jobs/{job_id}/screen", headers=auth_headers, json={})

        again = await client.post(
            f"/api/v1/jobs/{job_id}/screen", headers=auth_headers, json={"rescore": True}
        )
        assert again.json()["succeeded"] == 1

    async def test_can_be_narrowed_to_one_stage(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for stage in ("applied", "sourced"):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(
                client, auth_headers, job["id"], candidate["id"], stage=stage
            )

        resp = await client.post(
            f"/api/v1/jobs/{job['id']}/screen",
            headers=auth_headers,
            json={"stage": "sourced"},
        )
        assert resp.json()["succeeded"] == 1

    async def test_limit_caps_the_batch(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(client, auth_headers, job["id"], candidate["id"])

        resp = await client.post(
            f"/api/v1/jobs/{job['id']}/screen", headers=auth_headers, json={"limit": 2}
        )
        assert resp.json()["succeeded"] == 2

    async def test_rejected_applications_are_left_alone(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        await client.post(
            f"/api/v1/applications/{pipeline['application']['id']}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        resp = await client.post(
            f"/api/v1/jobs/{pipeline['job']['id']}/screen",
            headers=auth_headers,
            json={},
        )
        assert resp.json()["total"] == 0

    async def test_unknown_job_is_not_found(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            f"/api/v1/jobs/{uuid.uuid4()}/screen", headers=auth_headers, json={}
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Board and ranked shortlist
# --------------------------------------------------------------------------- #
class TestBoard:
    async def test_board_returns_every_stage_including_empty_ones(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        """A board that hides its empty columns cannot be dropped into."""
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/board", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [c["stage"] for c in body["columns"]] == [s.value for s in STAGE_ORDER]
        assert body["total"] == 1

    async def test_cards_carry_the_candidate(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/board", headers=auth_headers
        )
        applied = next(
            c for c in resp.json()["columns"] if c["stage"] == "applied"
        )
        assert applied["total"] == 1
        card = applied["applications"][0]
        assert card["candidate"]["id"] == pipeline["candidate"]["id"]
        assert card["application"]["id"] == pipeline["application"]["id"]

    async def test_cards_land_in_the_column_they_were_moved_to(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "interviewed"},
        )
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/board", headers=auth_headers
        )
        by_stage = {c["stage"]: c["total"] for c in resp.json()["columns"]}
        assert by_stage["applied"] == 0
        assert by_stage["interviewed"] == 1

    async def test_rejected_cards_are_hidden_by_default(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        await client.post(
            f"/api/v1/applications/{pipeline['application']['id']}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        job_id = pipeline["job"]["id"]

        default = await client.get(f"/api/v1/jobs/{job_id}/board", headers=auth_headers)
        assert default.json()["total"] == 0

        inclusive = await client.get(
            f"/api/v1/jobs/{job_id}/board",
            headers=auth_headers,
            params={"include_inactive": True},
        )
        assert inclusive.json()["total"] == 1

    async def test_hired_cards_stay_on_the_board(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=auth_headers,
            json={"stage": "hired", "force": True},
        )
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/board", headers=auth_headers
        )
        by_stage = {c["stage"]: c["total"] for c in resp.json()["columns"]}
        assert by_stage["hired"] == 1

    async def test_per_stage_limit_truncates_cards_but_not_the_count(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        for _ in range(3):
            candidate = await create_candidate(client, auth_headers)
            await apply_candidate(client, auth_headers, job["id"], candidate["id"])

        resp = await client.get(
            f"/api/v1/jobs/{job['id']}/board",
            headers=auth_headers,
            params={"per_stage_limit": 2},
        )
        applied = next(c for c in resp.json()["columns"] if c["stage"] == "applied")
        assert applied["total"] == 3
        assert len(applied["applications"]) == 2

    async def test_another_tenants_board_is_not_found(
        self, client: AsyncClient, second_org: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/board",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


class TestRankedCandidates:
    async def test_ranks_best_first_by_default(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        strong = await create_candidate(
            client, auth_headers, skills=["Python", "PostgreSQL", "Kubernetes"]
        )
        weak = await create_candidate(client, auth_headers, skills=["Excel"])
        for candidate in (weak, strong):
            await apply_candidate(client, auth_headers, job["id"], candidate["id"])
        await client.post(f"/api/v1/jobs/{job['id']}/screen", headers=auth_headers, json={})

        resp = await client.get(
            f"/api/v1/jobs/{job['id']}/candidates", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text
        items = resp.json()["items"]
        assert [i["candidate"]["id"] for i in items] == [strong["id"], weak["id"]]
        assert items[0]["screening"] is not None

    async def test_rows_without_a_screening_still_render(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/candidates", headers=auth_headers
        )
        item = resp.json()["items"][0]
        assert item["screening"] is None
        assert item["candidate"]["id"] == pipeline["candidate"]["id"]

    async def test_filters_by_stage(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/jobs/{pipeline['job']['id']}/candidates",
            headers=auth_headers,
            params={"stage": "hired"},
        )
        assert resp.json()["total"] == 0

    async def test_unknown_job_is_not_found(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/jobs/{uuid.uuid4()}/candidates", headers=auth_headers
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Permissions (design §8.3)
# --------------------------------------------------------------------------- #
class TestPermissions:
    async def test_unauthenticated_requests_are_refused(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/applications")
        assert resp.status_code == 401

    async def test_interviewers_can_read_but_not_create(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        headers = interviewer["headers"]

        assert (
            await client.get("/api/v1/applications", headers=headers)
        ).status_code == 200
        created = await client.post(
            "/api/v1/applications",
            headers=headers,
            json={
                "job_id": pipeline["job"]["id"],
                "candidate_id": pipeline["candidate"]["id"],
            },
        )
        assert created.status_code == 403

    async def test_interviewers_cannot_move_cards(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=interviewer["headers"],
            json={"stage": "screened"},
        )
        assert resp.status_code == 403

    async def test_hiring_managers_can_move_but_not_create(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        manager = await make_user(client, auth_headers, UserRole.HIRING_MANAGER)
        headers = manager["headers"]

        moved = await client.put(
            f"/api/v1/applications/{pipeline['application']['id']}/stage",
            headers=headers,
            json={"stage": "screened"},
        )
        assert moved.status_code == 200

        created = await client.post(
            "/api/v1/applications",
            headers=headers,
            json={
                "job_id": pipeline["job"]["id"],
                "candidate_id": pipeline["candidate"]["id"],
            },
        )
        assert created.status_code == 403

    async def test_recruiters_can_run_the_pipeline(
        self, client: AsyncClient, auth_headers: dict, pipeline: dict
    ) -> None:
        recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
        headers = recruiter["headers"]

        other = await create_candidate(client, headers)
        created = await client.post(
            "/api/v1/applications",
            headers=headers,
            json={"job_id": pipeline["job"]["id"], "candidate_id": other["id"]},
        )
        assert created.status_code == 201

        screened = await client.post(
            f"/api/v1/applications/{created.json()['id']}/screen",
            headers=headers,
            json={"auto_advance": False},
        )
        assert screened.status_code == 200
