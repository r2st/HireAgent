"""Offer letters end to end (design §4.7).

The arc: draft a letter against an application, send it through approval,
mint the candidate's link once it is approved, let the candidate accept or
decline with no account of their own, and check the pipeline card and the
tenant/permission boundaries around every step.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.enums import UserRole
from app.models.offer import OfferLetter
from app.services import offer_service
from tests.conftest import make_user
from tests.test_applications import apply_candidate, create_candidate, create_job

TEMPLATE = {
    "name": "Standard offer",
    "subject": "Your offer from {{company_name}}",
    "body": (
        "Dear {{first_name}},\n\n"
        "We are pleased to offer you the {{job_title}} role at {{company_name}}, "
        "reporting to {{reporting_manager}}, starting {{start_date}}.\n\n"
        "Compensation: {{salary}} per {{salary_period}}."
    ),
}

GENERATE_FIELDS = {
    "salary_amount": 150000,
    "salary_currency": "USD",
    "salary_period": "annual",
    "reporting_manager": "Priya Patel",
    "start_date": "2026-09-01",
}


async def create_template(client: AsyncClient, headers: dict, **overrides) -> dict:
    resp = await client.post(
        "/api/v1/offers/templates", headers=headers, json={**TEMPLATE, **overrides}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
async def scene(client: AsyncClient, auth_headers: dict) -> dict:
    """A job, a candidate, their application, and a stored letter template."""
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


async def draft(client: AsyncClient, headers: dict, scene: dict, **overrides) -> dict:
    payload = {
        "application_id": scene["application"]["id"],
        "template_id": scene["template"]["id"],
        **GENERATE_FIELDS,
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/offers", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def approved_offer(
    client: AsyncClient, admin_headers: dict, scene: dict, **overrides
) -> dict:
    """A drafted, submitted, and approved offer — one step from sending."""
    offer = await draft(client, admin_headers, scene, **overrides)
    resp = await client.post(
        f"/api/v1/offers/{offer['id']}/submit", headers=admin_headers
    )
    assert resp.status_code == 200, resp.text
    resp = await client.post(
        f"/api/v1/offers/{offer['id']}/approve", headers=admin_headers
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def sent_offer(client: AsyncClient, admin_headers: dict, scene: dict, **overrides) -> dict:
    offer = await approved_offer(client, admin_headers, scene, **overrides)
    resp = await client.post(f"/api/v1/offers/{offer['id']}/send", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def token_of(offer: dict) -> str:
    return offer["invite_url"].rsplit("/", 1)[-1]


async def fetch_offer(session, offer_id: str) -> OfferLetter:
    return await session.scalar(
        select(OfferLetter).where(OfferLetter.id == uuid.UUID(offer_id))
    )


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
class TestTemplates:
    async def test_creating_a_template_derives_its_variables(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        assert template["name"] == "Standard offer"
        assert set(template["variables"]) >= {
            "company_name",
            "first_name",
            "job_title",
            "reporting_manager",
            "start_date",
            "salary",
            "salary_period",
        }
        assert template["is_active"] is True

    async def test_a_malformed_placeholder_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/offers/templates",
            headers=auth_headers,
            json={**TEMPLATE, "body": "Hi {{first_name}, welcome aboard"},
        )
        assert resp.status_code == 422

    async def test_an_empty_body_is_refused(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/offers/templates",
            headers=auth_headers,
            json={**TEMPLATE, "body": "   "},
        )
        assert resp.status_code == 422

    async def test_templates_can_be_listed_and_filtered(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        await create_template(client, auth_headers)
        retired = await create_template(client, auth_headers, name="Retired letter")
        await client.patch(
            f"/api/v1/offers/templates/{retired['id']}",
            headers=auth_headers,
            json={"is_active": False},
        )

        resp = await client.get(
            "/api/v1/offers/templates", headers=auth_headers, params={"is_active": True}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["name"] == "Standard offer"

    async def test_a_deleted_template_is_gone(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.delete(
            f"/api/v1/offers/templates/{template['id']}", headers=auth_headers
        )
        assert resp.status_code == 200

        resp = await client.get(
            f"/api/v1/offers/templates/{template['id']}", headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_another_tenant_cannot_read_the_template(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        template = await create_template(client, auth_headers)
        resp = await client.get(
            f"/api/v1/offers/templates/{template['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #
class TestDrafting:
    async def test_drafting_renders_the_letter_and_sends_nothing(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await draft(client, auth_headers, scene)
        assert offer["status"] == "draft"
        assert offer["invite_url"] is None
        assert "Priya Patel" in offer["rendered_body"]
        assert "USD 150,000.00" in offer["rendered_body"]

    async def test_a_missing_template_variable_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        """``reporting_manager`` is referenced by the template but not supplied."""
        fields = {**GENERATE_FIELDS}
        fields.pop("reporting_manager")
        resp = await client.post(
            "/api/v1/offers",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
                **fields,
            },
        )
        assert resp.status_code == 422
        assert "reporting_manager" in resp.json()["detail"]["details"]["missing"]

    async def test_a_one_off_letter_needs_no_template(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/offers",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "body": "Hi {{first_name}}, you're hired at {{salary}}.",
                "job_title": "Staff Engineer",
                **{k: v for k, v in GENERATE_FIELDS.items() if k != "reporting_manager"},
            },
        )
        assert resp.status_code == 201, resp.text
        offer = resp.json()
        assert offer["template_id"] is None
        assert offer["job_title"] == "Staff Engineer"
        assert "hired at USD 150,000.00" in offer["rendered_body"]

    async def test_neither_a_template_nor_a_body_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        resp = await client.post(
            "/api/v1/offers",
            headers=auth_headers,
            json={"application_id": scene["application"]["id"], **GENERATE_FIELDS},
        )
        assert resp.status_code == 422

    async def test_only_one_offer_may_be_outstanding(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        first = await draft(client, auth_headers, scene)
        resp = await client.post(
            "/api/v1/offers",
            headers=auth_headers,
            json={
                "application_id": scene["application"]["id"],
                "template_id": scene["template"]["id"],
                **GENERATE_FIELDS,
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["details"]["offer_id"] == first["id"]

    async def test_a_withdrawn_offer_frees_the_slot(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        first = await draft(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/offers/{first['id']}/withdraw",
            headers=auth_headers,
            json={"reason": "Budget changed"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "withdrawn"

        second = await draft(client, auth_headers, scene)
        assert second["id"] != first["id"]

    async def test_editing_the_template_leaves_a_drafted_letter_alone(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await draft(client, auth_headers, scene)
        await client.patch(
            f"/api/v1/offers/templates/{scene['template']['id']}",
            headers=auth_headers,
            json={"body": "Completely different letter, {{first_name}}."},
        )
        resp = await client.get(f"/api/v1/offers/{offer['id']}", headers=auth_headers)
        assert resp.status_code == 200
        assert "Priya Patel" in resp.json()["rendered_body"]


# --------------------------------------------------------------------------- #
# Approval and sending
# --------------------------------------------------------------------------- #
class TestApprovalWorkflow:
    async def test_sending_before_approval_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await draft(client, auth_headers, scene)
        resp = await client.post(f"/api/v1/offers/{offer['id']}/send", headers=auth_headers)
        assert resp.status_code == 409

    async def test_approving_before_submission_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await draft(client, auth_headers, scene)
        resp = await client.post(f"/api/v1/offers/{offer['id']}/approve", headers=auth_headers)
        assert resp.status_code == 409

    async def test_a_recruiter_cannot_approve_their_own_draft(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
        offer = await draft(client, recruiter["headers"], scene)
        await client.post(f"/api/v1/offers/{offer['id']}/submit", headers=recruiter["headers"])

        resp = await client.post(
            f"/api/v1/offers/{offer['id']}/approve", headers=recruiter["headers"]
        )
        assert resp.status_code == 403

    async def test_a_hiring_manager_can_approve(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        manager = await make_user(client, auth_headers, UserRole.HIRING_MANAGER)
        offer = await draft(client, auth_headers, scene)
        await client.post(f"/api/v1/offers/{offer['id']}/submit", headers=auth_headers)

        resp = await client.post(
            f"/api/v1/offers/{offer['id']}/approve", headers=manager["headers"]
        )
        assert resp.status_code == 200
        approved = resp.json()
        assert approved["status"] == "approved"
        assert approved["approved_by_id"] == manager["user"]["id"]

    async def test_sending_mints_a_link_and_moves_the_pipeline_card(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        assert offer["status"] == "sent"
        assert offer["sent_at"] is not None
        assert "/offer/" in offer["invite_url"]

        resp = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert resp.json()["stage"] == "offered"


# --------------------------------------------------------------------------- #
# Candidate self-service
# --------------------------------------------------------------------------- #
class TestCandidateResponse:
    async def test_viewing_marks_the_offer_viewed(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        resp = await client.get(f"/api/v1/offer/{token_of(offer)}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "viewed"
        assert body["salary_amount"] == 150000
        assert "Priya Patel" in body["rendered_body"]

        recruiter_view = await client.get(
            f"/api/v1/offers/{offer['id']}", headers=auth_headers
        )
        assert recruiter_view.json()["status"] == "viewed"

    async def test_accepting_hires_the_candidate(
        self, client: AsyncClient, auth_headers: dict, session, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/offer/{token_of(offer)}/accept",
            json={"signature_name": "Grace Hopper"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

        row = await fetch_offer(session, offer["id"])
        assert row.signed_by_name == "Grace Hopper"
        assert row.esign_provider == "self_serve"
        assert row.signed_at is not None

        resp = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        body = resp.json()
        assert body["stage"] == "hired"
        assert body["status"] == "hired"

    async def test_accepting_without_a_name_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/offer/{token_of(offer)}/accept", json={"signature_name": "   "}
        )
        assert resp.status_code == 422

    async def test_declining_does_not_move_the_pipeline(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        resp = await client.post(
            f"/api/v1/offer/{token_of(offer)}/decline",
            json={"reason": "Accepted elsewhere"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "declined"

        recruiter_view = await client.get(
            f"/api/v1/offers/{offer['id']}", headers=auth_headers
        )
        assert recruiter_view.json()["decline_reason"] == "Accepted elsewhere"

        app_resp = await client.get(
            f"/api/v1/applications/{scene['application']['id']}", headers=auth_headers
        )
        assert app_resp.json()["stage"] == "offered"

    async def test_a_second_response_is_refused(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        token = token_of(offer)
        resp = await client.post(
            f"/api/v1/offer/{token}/accept", json={"signature_name": "Grace Hopper"}
        )
        assert resp.status_code == 200

        resp = await client.post(
            f"/api/v1/offer/{token}/decline", json={"reason": "Changed my mind"}
        )
        assert resp.status_code == 409

    async def test_a_wrong_token_looks_like_an_expired_one(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/offer/not-a-real-token")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "not_found"


# --------------------------------------------------------------------------- #
# Expiry sweep
# --------------------------------------------------------------------------- #
class TestExpiry:
    async def test_expire_due_closes_out_stale_offers(
        self, client: AsyncClient, auth_headers: dict, session, scene: dict
    ) -> None:
        offer = await sent_offer(client, auth_headers, scene)
        row = await fetch_offer(session, offer["id"])
        row.expiry_date = date.today() - timedelta(days=1)
        await session.commit()

        expired = await offer_service.expire_due(session)
        assert [str(o.id) for o in expired] == [offer["id"]]

        resp = await client.get(f"/api/v1/offer/{token_of(offer)}")
        assert resp.status_code == 404

    async def test_a_default_expiry_is_set_when_none_is_given(
        self, client: AsyncClient, auth_headers: dict, scene: dict
    ) -> None:
        offer = await draft(client, auth_headers, scene)
        assert offer["expiry_date"] is not None
        expiry = date.fromisoformat(offer["expiry_date"])
        assert expiry > date.today()
