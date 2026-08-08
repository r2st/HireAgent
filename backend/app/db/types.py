"""Custom SQLAlchemy column types.

Kept dialect-agnostic: JSONB/native UUID on PostgreSQL, portable equivalents on
SQLite so the test suite can run without a database server.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import JSON, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB

from app.core.security import decrypt_text, encrypt_text

# JSON that becomes JSONB on PostgreSQL (indexable, binary) and plain JSON
# elsewhere.
JSONColumn = JSON().with_variant(JSONB(), "postgresql")


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
