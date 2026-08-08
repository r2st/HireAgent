"""Resume file storage.

Files land on local disk under a per-organization directory. The interface is
deliberately narrow (``save``/``read``/``delete``) so swapping in the
S3-compatible backend the design specifies means implementing one class.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.errors import ValidationError

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Strip path separators and unsafe characters from an uploaded name.

    Without this, a filename like ``../../etc/passwd`` would escape the storage
    root when joined to a path.
    """
    name = Path(filename or "resume").name
    name = _SAFE_NAME.sub("_", name).strip("._") or "resume"
    return name[:180]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStorage:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.storage_dir).resolve()

    def _org_dir(self, organization_id: uuid.UUID) -> Path:
        path = self.root / str(organization_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self, organization_id: uuid.UUID, filename: str, data: bytes
    ) -> str:
        """Persist a file and return its storage-relative path."""
        if not data:
            raise ValidationError("Cannot store an empty file")
        if len(data) > settings.max_upload_bytes:
            raise ValidationError(
                f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)}MB limit",
                details={"size": len(data), "limit": settings.max_upload_bytes},
            )

        safe = sanitize_filename(filename)
        # Prefix with a UUID so two candidates uploading "resume.pdf" coexist.
        stored_name = f"{uuid.uuid4().hex}_{safe}"
        target = self._org_dir(organization_id) / stored_name
        target.write_bytes(data)
        return f"{organization_id}/{stored_name}"

    def read(self, path: str) -> bytes:
        target = self._resolve(path)
        if not target.is_file():
            raise ValidationError(f"Stored file not found: {path}")
        return target.read_bytes()

    def delete(self, path: str) -> bool:
        try:
            target = self._resolve(path)
        except ValidationError:
            return False
        if target.is_file():
            target.unlink()
            return True
        return False

    def _resolve(self, path: str) -> Path:
        """Resolve a stored path, refusing anything outside the storage root."""
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValidationError("Invalid storage path")
        return candidate


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage()
    return _storage


def set_storage(storage: LocalStorage | None) -> None:
    """Override the process-wide storage backend (used by tests)."""
    global _storage
    _storage = storage
