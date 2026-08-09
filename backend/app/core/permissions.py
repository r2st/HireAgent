"""Role-based permission matrix (design §8.3).

Four built-in roles, each granting a set of ``resource:action`` permissions.
Organizations can widen (never narrow) a specific user's access through
``User.permissions_json``.
"""

from __future__ import annotations

from app.models.enums import UserRole

# Wildcards are supported at the resource level ("job:*") and globally ("*").
ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.ADMIN: {"*"},
    UserRole.HIRING_MANAGER: {
        "job:read",
        "job:create",
        "job:update",
        "job:publish",
        "candidate:read",
        "application:read",
        "application:update",
        "application:move",
        "screening:read",
        "screening:create",
        "interview:read",
        "interview:create",
        "interview:update",
        # Hiring managers sit on panels, so they file scorecards too.
        "interview:feedback",
        "assessment:read",
        # A hiring manager marks the free-form answers on their own req, but
        # does not author the papers or issue them.
        "assessment:update",
        "offer:read",
        "offer:create",
        "offer:approve",
        "analytics:read",
        "outreach:read",
    },
    UserRole.RECRUITER: {
        "job:read",
        "job:create",
        "job:update",
        "candidate:read",
        "candidate:create",
        "candidate:update",
        "candidate:delete",
        "resume:read",
        "resume:create",
        "application:read",
        "application:create",
        "application:update",
        "application:move",
        "screening:read",
        "screening:create",
        "interview:read",
        "interview:create",
        "interview:update",
        "interview:feedback",
        "assessment:read",
        "assessment:create",
        "assessment:update",
        "assessment:delete",
        "outreach:read",
        "outreach:create",
        "outreach:update",
        "offer:read",
        "offer:create",
        "analytics:read",
    },
    # Interviewers see only what they need to run their assigned interviews.
    UserRole.INTERVIEWER: {
        "interview:read",
        "interview:feedback",
        "candidate:read",
        "application:read",
        "job:read",
    },
}


def permissions_for(role: UserRole, overrides: dict | None = None) -> set[str]:
    """Effective permission set for a role plus any per-user grants."""
    perms = set(ROLE_PERMISSIONS.get(role, set()))
    if overrides:
        granted = overrides.get("grant") or []
        if isinstance(granted, list):
            perms |= {str(p) for p in granted}
    return perms


def has_permission(
    role: UserRole, permission: str, overrides: dict | None = None
) -> bool:
    """Whether ``role`` (plus overrides) satisfies ``permission``.

    Matches exact grants, resource wildcards (``job:*``), and the global ``*``.
    """
    perms = permissions_for(role, overrides)
    if "*" in perms or permission in perms:
        return True
    resource = permission.split(":", 1)[0]
    return f"{resource}:*" in perms
