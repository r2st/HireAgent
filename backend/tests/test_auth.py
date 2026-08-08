"""Auth: registration, login, tokens, RBAC, and tenant isolation."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.permissions import has_permission
from app.core.security import (
    TokenError,
    blind_index,
    create_token,
    decode_token,
    decrypt_text,
    encrypt_text,
    hash_password,
    verify_password,
)
from app.models.enums import UserRole
from tests.conftest import make_user, unique_email


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_password_hash_roundtrip() -> None:
    hashed = hash_password("Str0ngPassword1")
    assert hashed != "Str0ngPassword1"
    assert verify_password("Str0ngPassword1", hashed)
    assert not verify_password("wrong", hashed)


def test_password_over_72_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="72 bytes"):
        hash_password("a" * 73)


def test_verify_password_handles_garbage_hash() -> None:
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_token_roundtrip_carries_tenant_context() -> None:
    token = create_token(
        subject="11111111-1111-1111-1111-111111111111",
        organization_id="22222222-2222-2222-2222-222222222222",
        role="admin",
    )
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == "11111111-1111-1111-1111-111111111111"
    assert payload["org"] == "22222222-2222-2222-2222-222222222222"
    assert payload["role"] == "admin"


def test_token_type_is_enforced() -> None:
    refresh = create_token(
        subject="a", organization_id="b", role="admin", token_type="refresh"
    )
    with pytest.raises(TokenError, match="Expected access token"):
        decode_token(refresh, expected_type="access")


def test_tampered_token_rejected() -> None:
    token = create_token(subject="a", organization_id="b", role="admin")
    with pytest.raises(TokenError):
        decode_token(token[:-4] + "AAAA", expected_type="access")


def test_encryption_roundtrip() -> None:
    secret = "candidate@example.com"
    ciphertext = encrypt_text(secret)
    assert ciphertext != secret
    assert decrypt_text(ciphertext) == secret


def test_encryption_is_non_deterministic() -> None:
    # Fernet includes a random IV, so equal plaintexts must not produce equal
    # ciphertexts — that is why blind indexes exist for lookups.
    assert encrypt_text("same") != encrypt_text("same")


def test_blind_index_is_deterministic_and_normalised() -> None:
    assert blind_index("Foo@Bar.com") == blind_index("  foo@bar.com ")
    assert blind_index("a@b.com") != blind_index("c@d.com")


# --------------------------------------------------------------------------- #
# Permission matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("role", "permission", "expected"),
    [
        (UserRole.ADMIN, "anything:at-all", True),
        (UserRole.RECRUITER, "candidate:create", True),
        (UserRole.RECRUITER, "offer:approve", False),
        (UserRole.HIRING_MANAGER, "offer:approve", True),
        (UserRole.HIRING_MANAGER, "candidate:delete", False),
        (UserRole.INTERVIEWER, "interview:read", True),
        (UserRole.INTERVIEWER, "candidate:create", False),
    ],
)
def test_role_permission_matrix(
    role: UserRole, permission: str, expected: bool
) -> None:
    assert has_permission(role, permission) is expected


def test_permission_overrides_grant_extra_access() -> None:
    assert not has_permission(UserRole.INTERVIEWER, "offer:create")
    assert has_permission(
        UserRole.INTERVIEWER, "offer:create", {"grant": ["offer:create"]}
    )


def test_resource_wildcard_permission() -> None:
    assert has_permission(UserRole.INTERVIEWER, "offer:create", {"grant": ["offer:*"]})


# --------------------------------------------------------------------------- #
# Registration and login
# --------------------------------------------------------------------------- #
async def test_register_creates_org_and_admin(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "organization_name": "New Co",
            "full_name": "Ada",
            "email": unique_email(),
            "password": "Str0ngPassword1",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["user"]["role"] == "admin"
    assert body["organization"]["slug"] == "new-co"
    assert body["organization"]["plan"] == "startup"
    assert body["tokens"]["access_token"]


async def test_register_allocates_unique_slug_for_duplicate_name(
    client: AsyncClient,
) -> None:
    payload = {
        "organization_name": "Same Name",
        "full_name": "One",
        "email": unique_email(),
        "password": "Str0ngPassword1",
    }
    first = await client.post("/api/v1/auth/register", json=payload)
    second = await client.post(
        "/api/v1/auth/register", json={**payload, "email": unique_email()}
    )
    assert first.status_code == second.status_code == 201
    assert first.json()["organization"]["slug"] != second.json()["organization"]["slug"]


@pytest.mark.parametrize(
    "password", ["short1", "nodigitshereatall", "1234567890", "aB1"]
)
async def test_weak_passwords_rejected(client: AsyncClient, password: str) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "organization_name": "Weak Co",
            "full_name": "Ada",
            "email": unique_email(),
            "password": password,
        },
    )
    assert resp.status_code == 422


async def test_login_succeeds_and_returns_tokens(
    client: AsyncClient, registered: dict
) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": registered["email"], "password": registered["password"]},
    )
    assert resp.status_code == 200
    assert resp.json()["tokens"]["access_token"]


async def test_login_with_wrong_password_rejected(
    client: AsyncClient, registered: dict
) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": registered["email"], "password": "WrongPassword1"},
    )
    assert resp.status_code == 401


async def test_login_unknown_email_rejected(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "Str0ngPassword1"},
    )
    assert resp.status_code == 401


async def test_same_email_in_two_orgs_requires_slug(client: AsyncClient) -> None:
    email = unique_email("shared")
    for name in ("Org One", "Org Two"):
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": name,
                "full_name": "Shared",
                "email": email,
                "password": "Str0ngPassword1",
            },
        )
        assert resp.status_code == 201

    ambiguous = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "Str0ngPassword1"}
    )
    assert ambiguous.status_code == 422

    resolved = await client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "Str0ngPassword1",
            "organization_slug": "org-two",
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["organization"]["name"] == "Org Two"


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
async def test_refresh_returns_new_pair(client: AsyncClient, registered: dict) -> None:
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": registered["tokens"]["refresh_token"]},
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]


async def test_access_token_rejected_at_refresh_endpoint(
    client: AsyncClient, registered: dict
) -> None:
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": registered["tokens"]["access_token"]},
    )
    assert resp.status_code == 401


async def test_me_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_me_rejects_invalid_token(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer garbage"}
    )
    assert resp.status_code == 401


async def test_me_returns_current_user(client: AsyncClient, registered: dict) -> None:
    resp = await client.get("/api/v1/auth/me", headers=registered["headers"])
    assert resp.status_code == 200
    assert resp.json()["email"] == registered["email"]


# --------------------------------------------------------------------------- #
# User administration
# --------------------------------------------------------------------------- #
async def test_admin_can_invite_user(client: AsyncClient, auth_headers: dict) -> None:
    created = await make_user(client, auth_headers, UserRole.RECRUITER)
    assert created["user"]["role"] == "recruiter"


async def test_duplicate_email_within_org_conflicts(
    client: AsyncClient, auth_headers: dict, registered: dict
) -> None:
    resp = await client.post(
        "/api/v1/auth/users",
        headers=auth_headers,
        json={
            "email": registered["email"],
            "full_name": "Dupe",
            "role": "recruiter",
            "password": "Str0ngPassword1",
        },
    )
    assert resp.status_code == 409


async def test_non_admin_cannot_invite_users(
    client: AsyncClient, auth_headers: dict
) -> None:
    recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
    resp = await client.post(
        "/api/v1/auth/users",
        headers=recruiter["headers"],
        json={
            "email": unique_email(),
            "full_name": "Nope",
            "role": "recruiter",
            "password": "Str0ngPassword1",
        },
    )
    assert resp.status_code == 403


async def test_cannot_remove_last_admin(
    client: AsyncClient, auth_headers: dict, registered: dict
) -> None:
    """Demoting the only admin would lock the organization out."""
    admin_id = registered["user"]["id"]
    resp = await client.patch(
        f"/api/v1/auth/users/{admin_id}",
        headers=auth_headers,
        json={"role": "recruiter"},
    )
    assert resp.status_code == 403


async def test_admin_can_be_demoted_when_another_admin_exists(
    client: AsyncClient, auth_headers: dict, registered: dict
) -> None:
    await make_user(client, auth_headers, UserRole.ADMIN)
    resp = await client.patch(
        f"/api/v1/auth/users/{registered['user']['id']}",
        headers=auth_headers,
        json={"role": "recruiter"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "recruiter"


async def test_cannot_delete_own_account(
    client: AsyncClient, auth_headers: dict, registered: dict
) -> None:
    resp = await client.delete(
        f"/api/v1/auth/users/{registered['user']['id']}", headers=auth_headers
    )
    assert resp.status_code == 422


async def test_deactivated_user_cannot_authenticate(
    client: AsyncClient, auth_headers: dict
) -> None:
    recruiter = await make_user(client, auth_headers, UserRole.RECRUITER)
    resp = await client.patch(
        f"/api/v1/auth/users/{recruiter['user']['id']}",
        headers=auth_headers,
        json={"is_active": False},
    )
    assert resp.status_code == 200

    # The token was valid at issue time; deactivation must take effect
    # immediately rather than at token expiry.
    check = await client.get("/api/v1/auth/me", headers=recruiter["headers"])
    assert check.status_code == 401


async def test_change_password(client: AsyncClient, registered: dict) -> None:
    resp = await client.post(
        "/api/v1/auth/change-password",
        headers=registered["headers"],
        json={
            "current_password": registered["password"],
            "new_password": "BrandNewP4ss",
        },
    )
    assert resp.status_code == 200

    old = await client.post(
        "/api/v1/auth/login",
        json={"email": registered["email"], "password": registered["password"]},
    )
    assert old.status_code == 401

    new = await client.post(
        "/api/v1/auth/login",
        json={"email": registered["email"], "password": "BrandNewP4ss"},
    )
    assert new.status_code == 200


async def test_change_password_requires_correct_current(
    client: AsyncClient, registered: dict
) -> None:
    resp = await client.post(
        "/api/v1/auth/change-password",
        headers=registered["headers"],
        json={"current_password": "WrongPassword1", "new_password": "BrandNewP4ss"},
    )
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Tenant isolation
# --------------------------------------------------------------------------- #
async def test_user_list_is_scoped_to_own_organization(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    ours = await client.get("/api/v1/auth/users", headers=auth_headers)
    theirs = await client.get("/api/v1/auth/users", headers=second_org["headers"])
    assert ours.status_code == theirs.status_code == 200

    our_emails = {u["email"] for u in ours.json()}
    their_emails = {u["email"] for u in theirs.json()}
    assert our_emails.isdisjoint(their_emails)


async def test_cannot_modify_user_in_another_organization(
    client: AsyncClient, auth_headers: dict, second_org: dict
) -> None:
    victim_id = second_org["user"]["id"]
    resp = await client.patch(
        f"/api/v1/auth/users/{victim_id}",
        headers=auth_headers,
        json={"is_active": False},
    )
    # 404, not 403: the API must not confirm that the id exists elsewhere.
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# System endpoints
# --------------------------------------------------------------------------- #
async def test_health_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
