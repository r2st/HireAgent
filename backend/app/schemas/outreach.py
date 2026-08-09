"""Outreach request/response schemas.

Only the candidate-facing opt-out lives here so far. It is deliberately thin:
the caller is a mail client or an unauthenticated browser page, and every field
it can see is a field an attacker holding a guessed token can see too.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def mask_email(address: str | None) -> str | None:
    """``grace.hopper@example.test`` -> ``g***r@example.test``.

    Enough for the reader to recognise which of their addresses this is, and
    not enough to be worth harvesting. The token travelled by email and could
    be sitting in a forwarded thread or a proxy log; the confirmation page has
    no reason to reprint the full address back out of it.
    """
    if not address or "@" not in address:
        return None
    local, _, domain = address.partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"


class UnsubscribeView(BaseModel):
    """What the landing page renders before the candidate confirms."""

    organization_name: str | None = None
    recipient: str | None = Field(
        default=None, description="Masked address this link was mailed to"
    )
    sequence_name: str | None = None
    already_unsubscribed: bool = False


class UnsubscribeResponse(BaseModel):
    """The confirmation, after the opt-out has been recorded."""

    organization_name: str | None = None
    already_unsubscribed: bool = False
    sequences_stopped: int = 0
    message: str = "You will not receive further emails."
    ok: bool = True
