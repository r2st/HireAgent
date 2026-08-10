"""Outreach sequence management API (design §4.2, §6.1).

The service layer (``test_outreach_service.py``) already exercises the send
window, step ordering, and enrollment filtering in depth. This file is about
the surface recruiters actually touch: creating a sequence, building its
steps, turning it on, enrolling candidates, and reading its stats back —
plus the permission and tenancy boundaries around all of it.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.models.enums import UserRole
from tests.conftest import make_user
from tests.test_applications import create_candidate


async def create_sequence(client: AsyncClient, headers: dict, **overrides) -> dict:
    payload = {"name": "Senior Backend Outreach", "timezone": "UTC"}
    payload.update(overrides)
    resp = await client.post(
        "/api/v1/outreach/sequences", headers=headers, json=payload
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def add_step(client: AsyncClient, headers: dict, sequence_id: str, **overrides) -> dict:
    payload = {"channel": "email", "body_override": "Hi {{first_name}}, are you open to chat?"}
    payload.update(overrides)
    resp = await client.post(
        f"/api/v1/outreach/sequences/{sequence_id}/steps", headers=headers, json=payload
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def create_consented_candidate(client: AsyncClient, headers: dict, **overrides) -> dict:
    consents = [{"consent_type": "email_communication", "granted": True}]
    return await create_candidate(client, headers, consents=consents, **overrides)


async def create_template(client: AsyncClient, headers: dict, **overrides) -> dict:
    payload = {
        "name": "Cold outreach",
        "channel": "email",
        "subject": "Hi {{first_name}}, quick question",
        "body": "Hi {{first_name}}, are you open to a chat about {{job_title}}?",
    }
    payload.update(overrides)
    resp = await client.post(
        "/api/v1/outreach/templates", headers=headers, json=payload
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Sequence CRUD
# --------------------------------------------------------------------------- #
class TestSequenceCRUD:
    async def test_create_defaults_to_draft(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        assert sequence["status"] == "draft"
        assert sequence["steps"] == []
        assert sequence["stats_json"]["enrolled"] == 0

    async def test_a_blank_name_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/outreach/sequences", headers=auth_headers, json={"name": "  "}
        )
        assert resp.status_code == 422

    async def test_list_and_get(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        created = await create_sequence(client, auth_headers)

        listed = await client.get("/api/v1/outreach/sequences", headers=auth_headers)
        assert listed.status_code == 200
        assert any(s["id"] == created["id"] for s in listed.json())

        fetched = await client.get(
            f"/api/v1/outreach/sequences/{created['id']}", headers=auth_headers
        )
        assert fetched.status_code == 200
        assert fetched.json()["name"] == created["name"]

    async def test_update_changes_the_window(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.patch(
            f"/api/v1/outreach/sequences/{sequence['id']}",
            headers=auth_headers,
            json={"send_window_start_hour": 10, "send_window_end_hour": 16},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["send_window_start_hour"] == 10
        assert body["send_window_end_hour"] == 16

    async def test_an_invalid_window_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.patch(
            f"/api/v1/outreach/sequences/{sequence['id']}",
            headers=auth_headers,
            json={"send_window_start_hour": 18, "send_window_end_hour": 9},
        )
        assert resp.status_code == 422

    async def test_delete_archives_it(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.delete(
            f"/api/v1/outreach/sequences/{sequence['id']}", headers=auth_headers
        )
        assert resp.status_code == 200

        fetched = await client.get(
            f"/api/v1/outreach/sequences/{sequence['id']}", headers=auth_headers
        )
        assert fetched.status_code == 404

    async def test_another_tenant_cannot_read_the_sequence(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.get(
            f"/api/v1/outreach/sequences/{sequence['id']}", headers=second_org["headers"]
        )
        assert resp.status_code == 404

    async def test_a_nonexistent_sequence_is_404(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.get(
            f"/api/v1/outreach/sequences/{uuid.uuid4()}", headers=auth_headers
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
class TestSteps:
    async def test_a_step_needs_a_template_or_body(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps",
            headers=auth_headers,
            json={"channel": "email"},
        )
        assert resp.status_code == 422

    async def test_steps_append_in_order(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        first = await add_step(client, auth_headers, sequence["id"], body_override="Step one")
        second = await add_step(
            client, auth_headers, sequence["id"], body_override="Step two", delay_days=2
        )
        assert first["step_order"] == 0
        assert second["step_order"] == 1
        assert second["delay_days"] == 2

        fetched = await client.get(
            f"/api/v1/outreach/sequences/{sequence['id']}", headers=auth_headers
        )
        assert [s["step_order"] for s in fetched.json()["steps"]] == [0, 1]

    async def test_deleting_a_step_closes_the_gap(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        first = await add_step(client, auth_headers, sequence["id"], body_override="A")
        second = await add_step(client, auth_headers, sequence["id"], body_override="B")
        third = await add_step(client, auth_headers, sequence["id"], body_override="C")

        resp = await client.delete(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps/{second['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200

        fetched = await client.get(
            f"/api/v1/outreach/sequences/{sequence['id']}", headers=auth_headers
        )
        remaining = fetched.json()["steps"]
        assert [s["id"] for s in remaining] == [first["id"], third["id"]]
        assert [s["step_order"] for s in remaining] == [0, 1]

    async def test_reorder_requires_every_step(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        first = await add_step(client, auth_headers, sequence["id"], body_override="A")
        await add_step(client, auth_headers, sequence["id"], body_override="B")

        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps/reorder",
            headers=auth_headers,
            json={"step_ids": [first["id"]]},
        )
        assert resp.status_code == 422

    async def test_reorder_rewrites_the_order(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        first = await add_step(client, auth_headers, sequence["id"], body_override="A")
        second = await add_step(client, auth_headers, sequence["id"], body_override="B")

        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps/reorder",
            headers=auth_headers,
            json={"step_ids": [second["id"], first["id"]]},
        )
        assert resp.status_code == 200
        ordered = resp.json()
        assert [s["id"] for s in ordered] == [second["id"], first["id"]]
        assert [s["step_order"] for s in ordered] == [0, 1]

    async def test_a_step_from_another_sequence_is_404(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence_a = await create_sequence(client, auth_headers, name="A")
        sequence_b = await create_sequence(client, auth_headers, name="B")
        step = await add_step(client, auth_headers, sequence_a["id"], body_override="A step")

        resp = await client.patch(
            f"/api/v1/outreach/sequences/{sequence_b['id']}/steps/{step['id']}",
            headers=auth_headers,
            json={"delay_days": 3},
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
class TestTemplates:
    async def test_create_derives_variables_from_the_text(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        assert template["is_active"] is True
        assert set(template["variables"]) == {"first_name", "job_title"}

    async def test_an_email_template_needs_a_subject(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/outreach/templates",
            headers=auth_headers,
            json={"name": "No subject", "channel": "email", "body": "Hi there"},
        )
        assert resp.status_code == 422

    async def test_duplicate_name_and_channel_is_a_conflict(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        await create_template(client, auth_headers, name="Repeat")
        resp = await client.post(
            "/api/v1/outreach/templates",
            headers=auth_headers,
            json={
                "name": "Repeat",
                "channel": "email",
                "subject": "Hi",
                "body": "Hi there",
            },
        )
        assert resp.status_code == 409

    async def test_list_get_update_delete(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)

        listed = await client.get("/api/v1/outreach/templates", headers=auth_headers)
        assert listed.status_code == 200
        assert any(t["id"] == template["id"] for t in listed.json())

        fetched = await client.get(
            f"/api/v1/outreach/templates/{template['id']}", headers=auth_headers
        )
        assert fetched.status_code == 200

        updated = await client.patch(
            f"/api/v1/outreach/templates/{template['id']}",
            headers=auth_headers,
            json={"body": "Hi {{first_name}}, new pitch entirely"},
        )
        assert updated.status_code == 200
        assert updated.json()["variables"] == ["first_name"]

        deleted = await client.delete(
            f"/api/v1/outreach/templates/{template['id']}", headers=auth_headers
        )
        assert deleted.status_code == 200

        after_delete = await client.get(
            f"/api/v1/outreach/templates/{template['id']}", headers=auth_headers
        )
        assert after_delete.status_code == 404

    async def test_another_tenant_cannot_read_the_template(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.get(
            f"/api/v1/outreach/templates/{template['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404

    async def test_a_step_can_reference_a_template(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        sequence = await create_sequence(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps",
            headers=auth_headers,
            json={"channel": "email", "template_id": template["id"]},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["template_id"] == template["id"]

    async def test_a_step_referencing_the_wrong_channel_template_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers, channel="email")
        sequence = await create_sequence(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/steps",
            headers=auth_headers,
            json={"channel": "whatsapp", "template_id": template["id"]},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Activation
# --------------------------------------------------------------------------- #
class TestActivation:
    async def test_activating_an_empty_sequence_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/activate", headers=auth_headers
        )
        assert resp.status_code == 422

    async def test_activate_then_pause_then_complete(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        sequence = await create_sequence(client, auth_headers)
        await add_step(client, auth_headers, sequence["id"], body_override="Hello")

        active = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/activate", headers=auth_headers
        )
        assert active.status_code == 200
        assert active.json()["status"] == "active"
        assert active.json()["started_at"] is not None

        paused = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/pause", headers=auth_headers
        )
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        completed = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/complete", headers=auth_headers
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
@pytest.fixture
async def active_sequence(client: AsyncClient, auth_headers: dict) -> dict:
    sequence = await create_sequence(client, auth_headers)
    await add_step(client, auth_headers, sequence["id"], body_override="Hello there")
    resp = await client.post(
        f"/api/v1/outreach/sequences/{sequence['id']}/activate", headers=auth_headers
    )
    return resp.json()


class TestEnrollment:
    async def test_a_consented_candidate_is_enrolled(
        self, client: AsyncClient, auth_headers: dict, active_sequence: dict
    ) -> None:
        candidate = await create_consented_candidate(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enroll",
            headers=auth_headers,
            json={"candidate_ids": [candidate["id"]]},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["enrolled_count"] == 1
        assert body["skipped_count"] == 0
        assert body["enrolled"][0]["candidate_id"] == candidate["id"]
        assert body["enrolled"][0]["status"] == "active"

    async def test_a_candidate_without_consent_is_skipped_not_rejected(
        self, client: AsyncClient, auth_headers: dict, active_sequence: dict
    ) -> None:
        candidate = await create_candidate(client, auth_headers)
        resp = await client.post(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enroll",
            headers=auth_headers,
            json={"candidate_ids": [candidate["id"]]},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["enrolled_count"] == 0
        assert body["skipped_count"] == 1
        assert body["skipped"][0]["reason"] == "no_consent"

    async def test_stats_reflect_the_enrollment(
        self, client: AsyncClient, auth_headers: dict, active_sequence: dict
    ) -> None:
        candidate = await create_consented_candidate(client, auth_headers)
        await client.post(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enroll",
            headers=auth_headers,
            json={"candidate_ids": [candidate["id"]]},
        )
        resp = await client.get(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/stats",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["enrollments_total"] == 1
        assert stats["enrollments"]["active"] == 1

    async def test_pause_and_resume_an_enrollment(
        self, client: AsyncClient, auth_headers: dict, active_sequence: dict
    ) -> None:
        candidate = await create_consented_candidate(client, auth_headers)
        enroll = await client.post(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enroll",
            headers=auth_headers,
            json={"candidate_ids": [candidate["id"]]},
        )
        enrollment_id = enroll.json()["enrolled"][0]["id"]

        paused = await client.post(
            f"/api/v1/outreach/enrollments/{enrollment_id}/pause",
            headers=auth_headers,
            json={"reason": "Recruiter asked to hold"},
        )
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        resumed = await client.post(
            f"/api/v1/outreach/enrollments/{enrollment_id}/resume", headers=auth_headers
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"

    async def test_list_enrollments_for_a_sequence(
        self, client: AsyncClient, auth_headers: dict, active_sequence: dict
    ) -> None:
        candidate = await create_consented_candidate(client, auth_headers)
        await client.post(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enroll",
            headers=auth_headers,
            json={"candidate_ids": [candidate["id"]]},
        )
        resp = await client.get(
            f"/api/v1/outreach/sequences/{active_sequence['id']}/enrollments",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #
class TestPermissions:
    async def test_an_interviewer_cannot_read_sequences(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await client.get(
            "/api/v1/outreach/sequences", headers=interviewer["headers"]
        )
        assert resp.status_code == 403

    async def test_a_hiring_manager_can_read_but_not_create(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        manager = await make_user(client, auth_headers, UserRole.HIRING_MANAGER)

        listed = await client.get(
            "/api/v1/outreach/sequences", headers=manager["headers"]
        )
        assert listed.status_code == 200

        resp = await client.post(
            "/api/v1/outreach/sequences",
            headers=manager["headers"],
            json={"name": "Should be refused"},
        )
        assert resp.status_code == 403

    async def test_a_recruiter_can_create_and_manage(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
        sequence = await create_sequence(client, recruiter["headers"])
        resp = await client.post(
            f"/api/v1/outreach/sequences/{sequence['id']}/pause",
            headers=recruiter["headers"],
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"
