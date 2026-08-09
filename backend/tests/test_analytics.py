"""Live pipeline analytics (design §4.5, §6.1).

Computed straight off ``applications``/``stage_events`` rather than a
pre-aggregated rollup — see ``analytics_service``. These tests build a small
pipeline by hand (move stages, reject, withdraw) and check the funnel,
conversion rates, and hire/rejection counts come out right, plus that a
second tenant's applications never leak into the count.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.test_applications import apply_candidate, create_candidate, create_job


async def move(client: AsyncClient, headers: dict, application_id: str, stage: str, **kw) -> dict:
    resp = await client.put(
        f"/api/v1/applications/{application_id}/stage",
        headers=headers,
        json={"stage": stage, "force": True, **kw},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestPipelineAnalytics:
    async def test_an_empty_org_reports_zero(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        resp = await client.get("/api/v1/analytics/pipeline", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_applications"] == 0
        assert body["funnel"] == []

    async def test_funnel_counts_and_conversion(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job = await create_job(client, auth_headers)

        # Three candidates: one hired, one rejected at screening, one sits
        # untouched at applied.
        hired = await create_candidate(client, auth_headers, email="hired@example.com")
        rejected = await create_candidate(client, auth_headers, email="rejected@example.com")
        pending = await create_candidate(client, auth_headers, email="pending@example.com")

        hired_app = await apply_candidate(client, auth_headers, job["id"], hired["id"])
        rejected_app = await apply_candidate(client, auth_headers, job["id"], rejected["id"])
        await apply_candidate(client, auth_headers, job["id"], pending["id"])

        for stage in ("screened", "interviewed", "assessed", "offered", "hired"):
            await move(client, auth_headers, hired_app["id"], stage)

        await move(client, auth_headers, rejected_app["id"], "screened")
        resp = await client.post(
            f"/api/v1/applications/{rejected_app['id']}/reject",
            headers=auth_headers,
            json={"reason": "Not a fit"},
        )
        assert resp.status_code == 200

        resp = await client.get("/api/v1/analytics/pipeline", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()

        assert body["total_applications"] == 3
        assert body["hires"] == 1
        assert body["rejections"] == 1

        funnel = {step["stage"]: step for step in body["funnel"]}
        assert funnel["applied"]["reached"] == 3
        assert funnel["screened"]["reached"] == 2
        assert funnel["hired"]["reached"] == 1
        # 2 of 3 applied made it to screened.
        assert funnel["screened"]["conversion_from_previous"] == pytest.approx(66.7, abs=0.1)

        assert body["stage_counts"]["hired"] == 1
        assert body["stage_counts"]["screened"] == 1
        assert body["stage_counts"]["applied"] == 1
        assert body["avg_time_to_hire_days"] is not None

    async def test_filters_by_job(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        job_a = await create_job(client, auth_headers, title="Backend Engineer")
        job_b = await create_job(client, auth_headers, title="Frontend Engineer")
        candidate_a = await create_candidate(client, auth_headers, email="a@example.com")
        candidate_b = await create_candidate(client, auth_headers, email="b@example.com")
        await apply_candidate(client, auth_headers, job_a["id"], candidate_a["id"])
        await apply_candidate(client, auth_headers, job_b["id"], candidate_b["id"])

        resp = await client.get(
            "/api/v1/analytics/pipeline",
            headers=auth_headers,
            params={"job_id": job_a["id"]},
        )
        assert resp.status_code == 200
        assert resp.json()["total_applications"] == 1

    async def test_another_tenant_is_isolated(
        self, client: AsyncClient, auth_headers: dict, second_org: dict
    ) -> None:
        job = await create_job(client, auth_headers)
        candidate = await create_candidate(client, auth_headers)
        await apply_candidate(client, auth_headers, job["id"], candidate["id"])

        resp = await client.get(
            "/api/v1/analytics/pipeline", headers=second_org["headers"]
        )
        assert resp.status_code == 200
        assert resp.json()["total_applications"] == 0

    async def test_an_interviewer_cannot_read_analytics(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        from app.models.enums import UserRole
        from tests.conftest import make_user

        interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
        resp = await client.get(
            "/api/v1/analytics/pipeline", headers=interviewer["headers"]
        )
        assert resp.status_code == 403
