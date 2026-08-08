"""Job CRUD, publishing rules, plan limits, and tenant isolation."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.models.enums import UserRole
from app.services.job_service import slugify
from tests.conftest import make_user

PUBLISHABLE = {
    "title": "Senior Backend Engineer",
    "department": "Engineering",
    "location": "Bengaluru",
    "work_mode": "hybrid",
    "employment_type": "full_time",
    "seniority": "senior",
    "description": "Build and operate the services behind our hiring platform.",
    "min_experience_years": 4,
    "max_experience_years": 9,
    "salary_min": "2500000",
    "salary_max": "4000000",
    "salary_currency": "INR",
    "requirements": {
        "required_skills": [
            {"name": "Python", "weight": 3, "min_years": 4},
            {"name": "PostgreSQL", "weight": 2},
        ],
        "preferred_skills": [{"name": "Kubernetes", "weight": 1}],
        "education": {"degree": "B.Tech", "is_mandatory": False},
        "responsibilities": ["Own the API layer"],
    },
}


async def create_job(client: AsyncClient, headers: dict, **overrides) -> dict:
    resp = await client.post(
        "/api/v1/jobs", headers=headers, json={**PUBLISHABLE, **overrides}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Slug generation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Senior Backend Engineer", "senior-backend-engineer"),
        ("  C++ Developer!! ", "c-developer"),
        ("Data/ML Scientist", "data-ml-scientist"),
        ("###", "job"),
    ],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
async def test_create_job_starts_in_draft(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    assert job["status"] == "draft"
    assert job["slug"] == "senior-backend-engineer"
    assert job["published_at"] is None
    assert job["salary_currency"] == "INR"


async def test_create_job_applies_default_scoring_weights(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    assert job["scoring_weights"] == {
        "skills": 0.40,
        "experience": 0.30,
        "education": 0.15,
        "cultural": 0.15,
    }


async def test_duplicate_title_gets_distinct_slug(
    client: AsyncClient, auth_headers: dict
) -> None:
    first = await create_job(client, auth_headers)
    second = await create_job(client, auth_headers)
    assert first["slug"] != second["slug"]


async def test_create_rejects_inverted_experience_range(
    client: AsyncClient, auth_headers: dict
) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        headers=auth_headers,
        json={**PUBLISHABLE, "min_experience_years": 10, "max_experience_years": 2},
    )
    assert resp.status_code == 422


async def test_create_rejects_inverted_salary_range(
    client: AsyncClient, auth_headers: dict
) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        headers=auth_headers,
        json={**PUBLISHABLE, "salary_min": "900000", "salary_max": "100000"},
    )
    assert resp.status_code == 422


async def test_create_rejects_reject_threshold_above_advance(
    client: AsyncClient, auth_headers: dict
) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        headers=auth_headers,
        json={**PUBLISHABLE, "auto_advance_threshold": 60, "auto_reject_threshold": 80},
    )
    assert resp.status_code == 422


async def test_create_requires_authentication(client: AsyncClient) -> None:
    assert (await client.post("/api/v1/jobs", json=PUBLISHABLE)).status_code == 401


async def test_interviewer_cannot_create_jobs(
    client: AsyncClient, auth_headers: dict
) -> None:
    interviewer = await make_user(client, auth_headers, UserRole.INTERVIEWER)
    resp = await client.post(
        "/api/v1/jobs", headers=interviewer["headers"], json=PUBLISHABLE
    )
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Reading and listing
# --------------------------------------------------------------------------- #
async def test_get_job(client: AsyncClient, auth_headers: dict) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.get(f"/api/v1/jobs/{job['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == job["id"]


async def test_get_missing_job_returns_404(
    client: AsyncClient, auth_headers: dict
) -> None:
    resp = await client.get(
        "/api/v1/jobs/00000000-0000-0000-0000-000000000000", headers=auth_headers
    )
    assert resp.status_code == 404


async def test_list_jobs_paginates(client: AsyncClient, auth_headers: dict) -> None:
    for i in range(5):
        await create_job(client, auth_headers, title=f"Engineer {i}")
    resp = await client.get(
        "/api/v1/jobs?page=1&page_size=2", headers=auth_headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5


async def test_list_jobs_filters_by_status(
    client: AsyncClient, auth_headers: dict
) -> None:
    draft = await create_job(client, auth_headers, title="Stays Draft")
    published = await create_job(client, auth_headers, title="Gets Published")
    await client.post(
        f"/api/v1/jobs/{published['id']}/publish", headers=auth_headers
    )

    resp = await client.get("/api/v1/jobs?status=draft", headers=auth_headers)
    ids = {j["id"] for j in resp.json()["items"]}
    assert draft["id"] in ids
    assert published["id"] not in ids


async def test_list_jobs_searches_title(
    client: AsyncClient, auth_headers: dict
) -> None:
    await create_job(client, auth_headers, title="Rust Systems Engineer")
    await create_job(client, auth_headers, title="Marketing Lead")
    resp = await client.get("/api/v1/jobs?search=Rust", headers=auth_headers)
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["title"] == "Rust Systems Engineer"


# --------------------------------------------------------------------------- #
# Updating
# --------------------------------------------------------------------------- #
async def test_update_job_partial(client: AsyncClient, auth_headers: dict) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/jobs/{job['id']}",
        headers=auth_headers,
        json={"department": "Platform", "openings": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["department"] == "Platform"
    assert body["openings"] == 3
    # Untouched fields survive.
    assert body["title"] == PUBLISHABLE["title"]


async def test_update_rejects_range_violation_across_patches(
    client: AsyncClient, auth_headers: dict
) -> None:
    """A PATCH moving one side of a pair must still be validated."""
    job = await create_job(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/jobs/{job['id']}",
        headers=auth_headers,
        json={"min_experience_years": 20},
    )
    assert resp.status_code == 422


async def test_update_scoring_weights(client: AsyncClient, auth_headers: dict) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/jobs/{job['id']}",
        headers=auth_headers,
        json={
            "scoring_weights": {
                "skills": 0.6,
                "experience": 0.2,
                "education": 0.1,
                "cultural": 0.1,
            }
        },
    )
    assert resp.status_code == 200
    assert resp.json()["scoring_weights"]["skills"] == 0.6


async def test_update_rejects_all_zero_weights(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/jobs/{job['id']}",
        headers=auth_headers,
        json={
            "scoring_weights": {
                "skills": 0,
                "experience": 0,
                "education": 0,
                "cultural": 0,
            }
        },
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Publishing lifecycle
# --------------------------------------------------------------------------- #
async def test_publish_sets_published_at(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "published"
    assert body["published_at"] is not None


async def test_cannot_publish_without_description(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers, description="   ")
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers
    )
    assert resp.status_code == 422
    assert "description" in resp.json()["detail"]["details"]["missing"]


async def test_cannot_publish_without_required_skills(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(
        client, auth_headers, requirements={"required_skills": []}
    )
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers
    )
    assert resp.status_code == 422
    assert (
        "requirements.required_skills"
        in resp.json()["detail"]["details"]["missing"]
    )


async def test_status_transitions(client: AsyncClient, auth_headers: dict) -> None:
    job = await create_job(client, auth_headers)
    job_id = job["id"]

    async def move(to: str) -> int:
        resp = await client.post(
            f"/api/v1/jobs/{job_id}/status",
            headers=auth_headers,
            json={"status": to},
        )
        return resp.status_code

    assert await move("published") == 200
    assert await move("paused") == 200
    assert await move("published") == 200
    assert await move("closed") == 200
    # closed -> paused is not a legal transition.
    assert await move("paused") == 422
    assert await move("archived") == 200


async def test_reopening_clears_publication_timestamps(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    await client.post(f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers)
    await client.post(
        f"/api/v1/jobs/{job['id']}/status",
        headers=auth_headers,
        json={"status": "closed"},
    )
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/status",
        headers=auth_headers,
        json={"status": "draft"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["published_at"] is None
    assert body["closed_at"] is None


async def test_same_status_is_a_noop(client: AsyncClient, auth_headers: dict) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/status",
        headers=auth_headers,
        json={"status": "draft"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "draft"


# --------------------------------------------------------------------------- #
# Plan limits (design §7.1)
# --------------------------------------------------------------------------- #
async def test_startup_plan_caps_active_jobs_at_five(
    client: AsyncClient, auth_headers: dict
) -> None:
    for i in range(5):
        job = await create_job(client, auth_headers, title=f"Role {i}")
        resp = await client.post(
            f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text

    sixth = await create_job(client, auth_headers, title="Role Six")
    resp = await client.post(
        f"/api/v1/jobs/{sixth['id']}/publish", headers=auth_headers
    )
    assert resp.status_code == 402
    assert resp.json()["detail"]["details"]["limit"] == 5


async def test_closing_a_job_frees_a_plan_slot(
    client: AsyncClient, auth_headers: dict
) -> None:
    published = []
    for i in range(5):
        job = await create_job(client, auth_headers, title=f"Slot {i}")
        await client.post(f"/api/v1/jobs/{job['id']}/publish", headers=auth_headers)
        published.append(job)

    await client.post(
        f"/api/v1/jobs/{published[0]['id']}/status",
        headers=auth_headers,
        json={"status": "closed"},
    )
    extra = await create_job(client, auth_headers, title="Now Fits")
    resp = await client.post(
        f"/api/v1/jobs/{extra['id']}/publish", headers=auth_headers
    )
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
async def test_delete_is_soft_and_hides_the_job(
    client: AsyncClient, auth_headers: dict
) -> None:
    job = await create_job(client, auth_headers)
    assert (
        await client.delete(f"/api/v1/jobs/{job['id']}", headers=auth_headers)
    ).status_code == 200
    assert (
        await client.get(f"/api/v1/jobs/{job['id']}", headers=auth_headers)
    ).status_code == 404


async def test_recruiter_cannot_delete_jobs(
    client: AsyncClient, auth_headers: dict
) -> None:
    recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
    job = await create_job(client, auth_headers)
    resp = await client.delete(
        f"/api/v1/jobs/{job['id']}", headers=recruiter["headers"]
    )
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Tenant isolation
# --------------------------------------------------------------------------- #
async def test_jobs_are_invisible_across_organizations(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    job = await create_job(client, auth_headers)

    assert (
        await client.get(
            f"/api/v1/jobs/{job['id']}", headers=second_org["headers"]
        )
    ).status_code == 404

    listing = await client.get("/api/v1/jobs", headers=second_org["headers"])
    assert listing.json()["total"] == 0


async def test_cannot_update_another_orgs_job(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/jobs/{job['id']}",
        headers=second_org["headers"],
        json={"title": "Hijacked"},
    )
    assert resp.status_code == 404


async def test_cannot_publish_another_orgs_job(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    job = await create_job(client, auth_headers)
    resp = await client.post(
        f"/api/v1/jobs/{job['id']}/publish", headers=second_org["headers"]
    )
    assert resp.status_code == 404


async def test_slug_uniqueness_is_per_organization(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    ours = await create_job(client, auth_headers)
    theirs = await create_job(client, second_org["headers"])
    # Same title in different tenants may share a slug.
    assert ours["slug"] == theirs["slug"] == "senior-backend-engineer"
