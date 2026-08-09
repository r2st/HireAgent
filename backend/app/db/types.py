"""Custom SQLAlchemy column types.

Kept dialect-agnostic: JSONB/native UUID on PostgreSQL, portable equivalents on
SQLite so the test suite can run without a database server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB

from app.core.security import decrypt_text, encrypt_text

# JSON that becomes JSONB on PostgreSQL (indexable, binary) and plain JSON
# elsewhere.
JSONColumn = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator):
    """A timestamp that is always tz-aware UTC in Python, on every dialect.

    PostgreSQL's ``timestamptz`` round-trips tzinfo, but SQLite has no time
    zone type and hands back naive datetimes. That difference is invisible
    until a value is read back — an interview scheduled at an aware datetime
    serialises as ``...Z`` before ``session.refresh()`` and without the ``Z``
    after — so it is normalised here rather than at each call site.

    Stored values are UTC; a naive input is taken to already be UTC, which
    matches the convention used throughout the scheduling code.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_bind_param(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return self._as_utc(value)

    def process_result_value(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return self._as_utc(value)


class EncryptedText(TypeDecorator):
    """Text transparently encrypted at rest with Fernet (AES-128-CBC + HMAC).

    Values are encrypted on the way in and decrypted on the way out, so callers
    work with plaintext and the database only ever holds ciphertext.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return encrypt_text(str(value))

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        try:
            return decrypt_text(value)
        except ValueError:
            # Tolerate pre-encryption rows rather than failing the whole query;
            # a backfill migration can re-encrypt them.
            return value


class EncryptedJSON(TypeDecorator):
    """A JSON document stored as an encrypted blob.

    Used for parsed resume payloads, which are candidate PII and must not be
    readable from a database dump.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return encrypt_text(json.dumps(value, default=str))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        try:
            return json.loads(decrypt_text(value))
        except (ValueError, json.JSONDecodeError):
            return None
