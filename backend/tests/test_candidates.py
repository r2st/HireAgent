"""Candidate CRUD, resume ingestion, consent, tenancy, and encryption at rest."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.integrations import openrouter as openrouter_module
from app.models.enums import ConsentType
from tests.conftest import make_user
from tests.factories import (
    ANONYMOUS_RESUME,
    LLM_RESUME_PAYLOAD,
    MINIMAL_RESUME,
    SAMPLE_RESUME,
    FakeLLMClient,
    make_docx,
    make_pdf,
)
from app.models.enums import UserRole

DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


def candidate_payload(**overrides) -> dict:
    payload = {
        "full_name": "Grace Hopper",
        "email": f"grace-{uuid.uuid4().hex[:8]}@example.com",
        "phone": "+1 202 555 0143",
        "location": "Arlington, VA",
        "current_company": "Navy",
        "current_role": "Rear Admiral",
        "experience_years": 12,
        "skills": ["COBOL", "python", "React.js"],
        "source": "direct",
        "tags": ["senior"],
    }
    payload.update(overrides)
    return payload


def upload_file(text_body: str = SAMPLE_RESUME, *, name: str = "ada.pdf") -> dict:
    return {"file": (name, make_pdf(text_body), "application/pdf")}


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
class TestCandidateCrud:
    async def test_create_returns_the_candidate(self, client: AsyncClient, auth_headers):
        resp = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["full_name"] == "Grace Hopper"
        # Skills are canonicalised on the way in.
        assert [s["name"] for s in body["skills_json"]] == ["COBOL", "Python", "React"]

    async def test_create_records_consent_grants(
        self, client: AsyncClient, auth_headers
    ):
        payload = candidate_payload(
            consents=[
                {"consent_type": "resume_processing", "granted": True},
                {"consent_type": "email_communication", "granted": True},
            ]
        )
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=payload
        )
        assert created.status_code == 201

        resp = await client.get(
            f"/api/v1/candidates/{created.json()['id']}/consents", headers=auth_headers
        )
        assert resp.status_code == 200
        assert {c["consent_type"] for c in resp.json()} == {
            "resume_processing",
            "email_communication",
        }

    async def test_duplicate_email_is_rejected(self, client: AsyncClient, auth_headers):
        payload = candidate_payload()
        first = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=payload
        )
        assert first.status_code == 201

        second = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=payload
        )
        assert second.status_code == 409

    async def test_email_uniqueness_is_per_organization(
        self, client: AsyncClient, auth_headers, second_org
    ):
        payload = candidate_payload()
        assert (
            await client.post("/api/v1/candidates", headers=auth_headers, json=payload)
        ).status_code == 201
        # The same person can exist in two unrelated tenants.
        assert (
            await client.post(
                "/api/v1/candidates", headers=second_org["headers"], json=payload
            )
        ).status_code == 201

    async def test_get_returns_the_full_record(self, client: AsyncClient, auth_headers):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.get(
            f"/api/v1/candidates/{created.json()['id']}", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["current_role"] == "Rear Admiral"

    async def test_get_unknown_candidate_is_404(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.get(
            f"/api/v1/candidates/{uuid.uuid4()}", headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_update_patches_only_supplied_fields(
        self, client: AsyncClient, auth_headers
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        candidate_id = created.json()["id"]

        resp = await client.patch(
            f"/api/v1/candidates/{candidate_id}",
            headers=auth_headers,
            json={"current_role": "Consultant", "skills": ["golang"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_role"] == "Consultant"
        assert [s["name"] for s in body["skills_json"]] == ["Go"]
        # Untouched fields survive.
        assert body["current_company"] == "Navy"

    async def test_update_to_a_taken_email_is_rejected(
        self, client: AsyncClient, auth_headers
    ):
        first = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        second = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.patch(
            f"/api/v1/candidates/{second.json()['id']}",
            headers=auth_headers,
            json={"email": first.json()["email"]},
        )
        assert resp.status_code == 409

    async def test_delete_is_a_soft_delete(self, client: AsyncClient, auth_headers):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        candidate_id = created.json()["id"]

        assert (
            await client.delete(
                f"/api/v1/candidates/{candidate_id}", headers=auth_headers
            )
        ).status_code == 200
        # Gone from the API...
        assert (
            await client.get(
                f"/api/v1/candidates/{candidate_id}", headers=auth_headers
            )
        ).status_code == 404
        assert (
            await client.get("/api/v1/candidates", headers=auth_headers)
        ).json()["total"] == 0

    async def test_deleted_candidate_row_is_retained(
        self, client: AsyncClient, auth_headers, session
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        await client.delete(
            f"/api/v1/candidates/{created.json()['id']}", headers=auth_headers
        )
        # Soft delete: the row stays for audit and retention policy.
        remaining = await session.scalar(
            text("SELECT COUNT(*) FROM candidates WHERE deleted_at IS NOT NULL")
        )
        assert remaining == 1


# --------------------------------------------------------------------------- #
# Listing and filters
# --------------------------------------------------------------------------- #
class TestCandidateListing:
    async def test_lists_and_paginates(self, client: AsyncClient, auth_headers):
        for _ in range(3):
            await client.post(
                "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
            )
        resp = await client.get(
            "/api/v1/candidates?page=1&page_size=2", headers=auth_headers
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 2

    async def test_filters_by_canonical_skill(self, client: AsyncClient, auth_headers):
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(skills=["ReactJS"]),
        )
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(skills=["Go"]),
        )
        # Querying the canonical name finds the candidate who wrote "ReactJS".
        resp = await client.get("/api/v1/candidates?skill=React", headers=auth_headers)
        assert resp.json()["total"] == 1

    async def test_filters_by_experience_range(self, client: AsyncClient, auth_headers):
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(experience_years=2),
        )
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(experience_years=11),
        )
        resp = await client.get(
            "/api/v1/candidates?min_experience=5&max_experience=15", headers=auth_headers
        )
        assert resp.json()["total"] == 1

    async def test_filters_by_source(self, client: AsyncClient, auth_headers):
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(source="referral"),
        )
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(source="linkedin"),
        )
        resp = await client.get(
            "/api/v1/candidates?source=referral", headers=auth_headers
        )
        assert resp.json()["total"] == 1

    async def test_searches_unencrypted_columns(
        self, client: AsyncClient, auth_headers
    ):
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(current_company="Analytical Engines"),
        )
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(current_company="Babbage Systems"),
        )
        resp = await client.get(
            "/api/v1/candidates?search=Analytical", headers=auth_headers
        )
        assert resp.json()["total"] == 1


# --------------------------------------------------------------------------- #
# Tenancy (design §3.1: org_id on every query)
# --------------------------------------------------------------------------- #
class TestCandidateTenancy:
    async def test_another_org_cannot_read_a_candidate(
        self, client: AsyncClient, auth_headers, second_org
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.get(
            f"/api/v1/candidates/{created.json()['id']}",
            headers=second_org["headers"],
        )
        # 404 rather than 403: the API must not confirm the record exists.
        assert resp.status_code == 404

    async def test_another_org_cannot_update_a_candidate(
        self, client: AsyncClient, auth_headers, second_org
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.patch(
            f"/api/v1/candidates/{created.json()['id']}",
            headers=second_org["headers"],
            json={"notes": "poached"},
        )
        assert resp.status_code == 404

    async def test_another_org_cannot_delete_a_candidate(
        self, client: AsyncClient, auth_headers, second_org
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.delete(
            f"/api/v1/candidates/{created.json()['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404

    async def test_listing_never_crosses_tenants(
        self, client: AsyncClient, auth_headers, second_org
    ):
        await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        resp = await client.get("/api/v1/candidates", headers=second_org["headers"])
        assert resp.json()["total"] == 0

    async def test_requires_authentication(self, client: AsyncClient):
        assert (await client.get("/api/v1/candidates")).status_code == 401

    async def test_interviewers_cannot_create_candidates(
        self, client: AsyncClient, auth_headers
    ):
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await client.post(
            "/api/v1/candidates",
            headers=interviewer["headers"],
            json=candidate_payload(),
        )
        assert resp.status_code == 403

    async def test_interviewers_may_read_candidates(
        self, client: AsyncClient, auth_headers
    ):
        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await client.get(
            "/api/v1/candidates", headers=interviewer["headers"]
        )
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Encryption at rest (design §8.1)
# --------------------------------------------------------------------------- #
class TestCandidateEncryption:
    async def test_pii_columns_are_ciphertext_in_the_database(
        self, client: AsyncClient, auth_headers, session
    ):
        payload = candidate_payload(full_name="Grace Hopper")
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=payload
        )
        assert created.status_code == 201

        row = (
            await session.execute(
                text("SELECT full_name, email FROM candidates LIMIT 1")
            )
        ).one()
        stored_name, stored_email = row
        assert "Grace Hopper" not in stored_name
        assert payload["email"] not in stored_email
        # ...but the API still returns plaintext.
        assert created.json()["full_name"] == "Grace Hopper"

    async def test_email_lookup_still_works_through_the_blind_index(
        self, client: AsyncClient, auth_headers
    ):
        payload = candidate_payload()
        await client.post("/api/v1/candidates", headers=auth_headers, json=payload)
        # A second create with the same address must still collide even though
        # the column is encrypted and not directly searchable.
        again = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=payload
        )
        assert again.status_code == 409


# --------------------------------------------------------------------------- #
# Consent (design §8.2)
# --------------------------------------------------------------------------- #
class TestConsent:
    @pytest.fixture
    async def candidate_id(self, client: AsyncClient, auth_headers) -> str:
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        return created.json()["id"]

    async def test_records_a_grant(self, client: AsyncClient, auth_headers, candidate_id):
        resp = await client.post(
            f"/api/v1/candidates/{candidate_id}/consents",
            headers=auth_headers,
            json={
                "consent_type": "whatsapp_communication",
                "granted": True,
                "policy_version": "2026-01",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "granted"
        assert body["policy_version"] == "2026-01"

    async def test_withdrawal_appends_rather_than_editing(
        self, client: AsyncClient, auth_headers, candidate_id
    ):
        for granted in (True, False):
            await client.post(
                f"/api/v1/candidates/{candidate_id}/consents",
                headers=auth_headers,
                json={"consent_type": "email_communication", "granted": granted},
            )

        resp = await client.get(
            f"/api/v1/candidates/{candidate_id}/consents", headers=auth_headers
        )
        records = resp.json()
        # Both records survive: consent history is immutable and auditable.
        assert len(records) == 2
        assert {r["status"] for r in records} == {"granted", "withdrawn"}

    async def test_has_consent_reflects_the_latest_record(
        self, client: AsyncClient, auth_headers, candidate_id, session
    ):
        from app.services import candidate_service

        org_id = uuid.UUID(
            (
                await client.get(
                    f"/api/v1/candidates/{candidate_id}", headers=auth_headers
                )
            ).json()["organization_id"]
        )
        for granted in (True, False):
            await client.post(
                f"/api/v1/candidates/{candidate_id}/consents",
                headers=auth_headers,
                json={"consent_type": "email_communication", "granted": granted},
            )

        assert not await candidate_service.has_consent(
            session, org_id, uuid.UUID(candidate_id), ConsentType.EMAIL_COMMUNICATION
        )

    async def test_consent_for_an_unknown_candidate_is_404(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            f"/api/v1/candidates/{uuid.uuid4()}/consents",
            headers=auth_headers,
            json={"consent_type": "assessment", "granted": True},
        )
        assert resp.status_code == 404

    async def test_another_org_cannot_read_consents(
        self, client: AsyncClient, candidate_id, second_org
    ):
        resp = await client.get(
            f"/api/v1/candidates/{candidate_id}/consents",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Resume ingestion (design §2.2, §6.1)
# --------------------------------------------------------------------------- #
class TestResumeUpload:
    async def test_creates_a_candidate_from_a_parsed_pdf(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["is_existing_candidate"] is False
        assert body["candidate"]["email"] == "ada.lovelace@example.com"
        assert body["candidate"]["full_name"] == "Ada Lovelace"
        assert body["resume"]["parse_status"] == "parsed"
        assert body["resume"]["is_primary"] is True

    async def test_extracted_skills_are_canonical(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        names = {s["name"] for s in resp.json()["resume"]["skills_extracted"]}
        assert {"Python", "PostgreSQL", "Docker", "React"} <= names

    async def test_accepts_docx(self, client: AsyncClient, auth_headers):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files={"file": ("ada.docx", make_docx(SAMPLE_RESUME), DOCX_CONTENT_TYPE)},
        )
        assert resp.status_code == 201
        assert resp.json()["candidate"]["email"] == "ada.lovelace@example.com"

    async def test_records_resume_processing_consent_by_default(
        self, client: AsyncClient, auth_headers
    ):
        upload = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        candidate_id = upload.json()["candidate"]["id"]
        consents = (
            await client.get(
                f"/api/v1/candidates/{candidate_id}/consents", headers=auth_headers
            )
        ).json()
        assert [c["consent_type"] for c in consents] == ["resume_processing"]

    async def test_warns_when_consent_was_not_captured(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
            data={"consent_granted": "false"},
        )
        assert resp.status_code == 201
        assert any("consent" in w.lower() for w in resp.json()["warnings"])

    async def test_second_upload_matches_the_existing_candidate(
        self, client: AsyncClient, auth_headers
    ):
        first = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        second = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(name="ada-v2.pdf"),
        )
        assert second.status_code == 201
        assert second.json()["is_existing_candidate"] is True
        # Deduplicated by email rather than creating a second profile.
        assert second.json()["candidate"]["id"] == first.json()["candidate"]["id"]

    async def test_newest_resume_becomes_primary(
        self, client: AsyncClient, auth_headers
    ):
        first = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        candidate_id = first.json()["candidate"]["id"]
        second = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(name="ada-v2.pdf"),
        )

        resumes = (
            await client.get(
                f"/api/v1/candidates/{candidate_id}/resumes", headers=auth_headers
            )
        ).json()
        assert len(resumes) == 2
        primary = [r for r in resumes if r["is_primary"]]
        assert len(primary) == 1
        assert primary[0]["id"] == second.json()["resume"]["id"]

    async def test_upload_enriches_a_manually_created_candidate(
        self, client: AsyncClient, auth_headers
    ):
        created = await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(
                full_name="Ada Lovelace",
                email="ada.lovelace@example.com",
                current_company=None,
                current_role=None,
                location=None,
                skills=[],
            ),
        )
        assert created.status_code == 201

        upload = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        candidate = upload.json()["candidate"]
        assert upload.json()["is_existing_candidate"] is True
        assert candidate["current_company"] == "Analytical Engines"
        assert {s["name"] for s in candidate["skills_json"]} >= {"Python", "Docker"}

    async def test_upload_never_overwrites_a_recruiters_correction(
        self, client: AsyncClient, auth_headers
    ):
        await client.post(
            "/api/v1/candidates",
            headers=auth_headers,
            json=candidate_payload(
                email="ada.lovelace@example.com",
                current_role="Head of Payments",
            ),
        )
        upload = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        # A machine parse must not outrank a human edit.
        assert upload.json()["candidate"]["current_role"] == "Head of Payments"

    async def test_upload_against_an_explicit_candidate_id(
        self, client: AsyncClient, auth_headers
    ):
        created = await client.post(
            "/api/v1/candidates", headers=auth_headers, json=candidate_payload()
        )
        candidate_id = created.json()["id"]

        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
            data={"candidate_id": candidate_id},
        )
        assert resp.status_code == 201
        # The explicit target wins over the email found in the document.
        assert resp.json()["candidate"]["id"] == candidate_id

    async def test_resume_without_an_email_is_rejected(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(ANONYMOUS_RESUME, name="anon.pdf"),
        )
        assert resp.status_code == 422
        assert "email" in resp.json()["detail"]["message"].lower()

    async def test_low_confidence_parse_is_flagged_for_review(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(MINIMAL_RESUME, name="grace.pdf"),
        )
        assert resp.status_code == 201
        assert any("confidence" in w.lower() for w in resp.json()["warnings"])

    async def test_unsupported_format_is_rejected(
        self, client: AsyncClient, auth_headers
    ):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n" + b"0" * 200, "image/png")},
        )
        assert resp.status_code == 422

    async def test_empty_file_is_rejected(self, client: AsyncClient, auth_headers):
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 422

    async def test_hiring_managers_cannot_upload_resumes(
        self, client: AsyncClient, auth_headers
    ):
        manager = await make_user(client, auth_headers, UserRole.HIRING_MANAGER)
        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=manager["headers"],
            files=upload_file(),
        )
        assert resp.status_code == 403

    async def test_uses_the_llm_parser_when_one_is_configured(
        self, client: AsyncClient, auth_headers
    ):
        client_stub = FakeLLMClient([LLM_RESUME_PAYLOAD])
        openrouter_module.set_llm_client(client_stub)

        resp = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        assert resp.status_code == 201
        assert resp.json()["resume"]["parser_model"] == "fake/model:free"
        assert client_stub.calls


class TestResumeDetail:
    async def test_detail_returns_the_decrypted_parse(
        self, client: AsyncClient, auth_headers
    ):
        upload = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        resume_id = upload.json()["resume"]["id"]

        resp = await client.get(
            f"/api/v1/candidates/resumes/{resume_id}", headers=auth_headers
        )
        assert resp.status_code == 200
        parsed = resp.json()["parsed_json"]
        assert parsed["email"] == "ada.lovelace@example.com"
        assert parsed["engine"] in {"llm", "heuristic"}

    async def test_parsed_payload_is_encrypted_at_rest(
        self, client: AsyncClient, auth_headers, session
    ):
        await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        stored = await session.scalar(
            text("SELECT parsed_json FROM resumes LIMIT 1")
        )
        # A database dump must not reveal candidate PII (design §8.1).
        assert "ada.lovelace@example.com" not in stored
        assert not stored.lstrip().startswith("{")

    async def test_another_org_cannot_read_a_resume(
        self, client: AsyncClient, auth_headers, second_org
    ):
        upload = await client.post(
            "/api/v1/candidates/upload-resume",
            headers=auth_headers,
            files=upload_file(),
        )
        resp = await client.get(
            f"/api/v1/candidates/resumes/{upload.json()['resume']['id']}",
            headers=second_org["headers"],
        )
        assert resp.status_code == 404


class TestBulkUpload:
    async def test_ingests_many_resumes(self, client: AsyncClient, auth_headers):
        files = [
            ("files", ("ada.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")),
            ("files", ("grace.pdf", make_pdf(MINIMAL_RESUME), "application/pdf")),
        ]
        resp = await client.post(
            "/api/v1/candidates/bulk-upload", headers=auth_headers, files=files
        )
        assert resp.status_code == 207
        body = resp.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2
        assert body["failed"] == 0

    async def test_one_bad_file_does_not_abort_the_batch(
        self, client: AsyncClient, auth_headers
    ):
        files = [
            ("files", ("ada.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")),
            ("files", ("broken.pdf", b"not really a pdf", "application/pdf")),
            ("files", ("anon.pdf", make_pdf(ANONYMOUS_RESUME), "application/pdf")),
        ]
        resp = await client.post(
            "/api/v1/candidates/bulk-upload", headers=auth_headers, files=files
        )
        body = resp.json()
        assert body["succeeded"] == 1
        assert body["failed"] == 2
        assert {e["filename"] for e in body["errors"]} == {"broken.pdf", "anon.pdf"}

    async def test_rejects_oversized_batches(self, client: AsyncClient, auth_headers):
        from app.api.v1.candidates import MAX_BULK_FILES

        files = [
            ("files", (f"cv{i}.txt", SAMPLE_RESUME.encode(), "text/plain"))
            for i in range(MAX_BULK_FILES + 1)
        ]
        resp = await client.post(
            "/api/v1/candidates/bulk-upload", headers=auth_headers, files=files
        )
        assert resp.status_code == 422

    async def test_duplicate_resumes_in_one_batch_dedupe_to_one_candidate(
        self, client: AsyncClient, auth_headers
    ):
        files = [
            ("files", ("ada.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")),
            ("files", ("ada-copy.pdf", make_pdf(SAMPLE_RESUME), "application/pdf")),
        ]
        resp = await client.post(
            "/api/v1/candidates/bulk-upload", headers=auth_headers, files=files
        )
        assert resp.json()["succeeded"] == 2
        listing = await client.get("/api/v1/candidates", headers=auth_headers)
        assert listing.json()["total"] == 1
