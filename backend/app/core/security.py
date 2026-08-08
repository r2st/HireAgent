"""Password hashing, JWT issuance/verification, and column encryption helpers."""

from __future__ import annotations

import base64
import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TokenType = Literal["access", "refresh"]


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    # bcrypt silently truncates at 72 bytes; reject rather than create a
    # password where trailing characters are meaningless.
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def create_token(
    *,
    subject: str | uuid.UUID,
    organization_id: str | uuid.UUID,
    role: str,
    token_type: TokenType = "access",
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Mint a signed JWT carrying the tenant context.

    ``organization_id`` is embedded in every token so that request handlers can
    scope queries without a second lookup.
    """
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = (
            timedelta(minutes=settings.access_token_expire_minutes)
            if token_type == "access"
            else timedelta(days=settings.refresh_token_expire_days)
        )
    claims: dict[str, Any] = {
        "sub": str(subject),
        "org": str(organization_id),
        "role": role,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": secrets.token_urlsafe(16),
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, settings.secret_key, algorithm=settings.jwt_algorithm)


class TokenError(Exception):
    """Raised when a token is malformed, expired, or of the wrong type."""


def decode_token(token: str, *, expected_type: TokenType | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise TokenError(str(exc)) from exc

    if expected_type is not None and payload.get("type") != expected_type:
        raise TokenError(
            f"Expected {expected_type} token, got {payload.get('type')!r}"
        )
    if not payload.get("sub") or not payload.get("org"):
        raise TokenError("Token missing subject or organization claim")
    return payload


# --------------------------------------------------------------------------- #
# Column-level encryption (design §8.1: AES-256 for candidate PII at rest)
# --------------------------------------------------------------------------- #
def _build_fernet(key: str) -> Fernet:
    """Accept either a real Fernet key or any passphrase.

    A passphrase is stretched to a valid 32-byte urlsafe-base64 key so local
    setup does not require key generation, while production can supply a
    properly generated key.
    """
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


_fernet = _build_fernet(settings.encryption_key)


def encrypt_text(plaintext: str) -> str:
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_text(ciphertext: str) -> str:
    try:
        return _fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise ValueError("Could not decrypt value") from exc


def blind_index(value: str) -> str:
    """Deterministic HMAC of a value, for equality lookups on encrypted columns.

    Encrypted columns cannot be indexed or searched. Storing this alongside
    lets us find e.g. a candidate by email without decrypting every row.
    """
    normalized = value.strip().lower().encode("utf-8")
    return hashlib.blake2b(
        normalized, key=settings.secret_key.encode("utf-8")[:64], digest_size=32
    ).hexdigest()
