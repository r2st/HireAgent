"""Message templates and ``{{variable}}`` rendering (design §4.2).

One rule drives every decision here: **an unresolved placeholder must never
reach a candidate.** "Hi {{first_name}}, we loved your work at {{company}}"
sent verbatim is worse than not sending at all — it identifies the sender as a
bulk mailer to the one person they were trying to impress. So rendering is
strict by default: a missing variable is reported, not silently blanked, and
the dispatcher refuses the message rather than sending a broken one.

The escape hatch is an inline fallback — ``{{first_name|there}}`` — which is
the honest way to handle a field that is legitimately optional. A template
author who wants a blank writes ``{{middle_name|}}`` and says so.

Rendering is deliberately not a general template language. There are no
conditionals, no loops, and no attribute traversal: every one of those turns a
recruiter's typo into a server-side error at send time, and none of them earn
their keep for a five-paragraph email.
"""

from __future__ import annotations

import html
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.tenancy import get_scoped, scoped_select
from app.models.enums import OutreachChannel
from app.models.outreach import MessageTemplate

# ``{{ name }}`` or ``{{ name | fallback text }}``. The name is restricted to
# identifier characters so a stray brace in prose cannot become a variable, and
# the fallback runs to the closing brace so it may contain spaces.
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\|([^}]*))?\}\}")

MAX_SUBJECT_LENGTH = 500


@dataclass
class RenderResult:
    """Rendered text plus what the renderer could not fill in.

    ``missing`` lists variables with neither a value nor a fallback. A caller
    that intends to send must treat a non-empty ``missing`` as a hard stop; a
    caller previewing a template shows it to the author instead.
    """

    text: str
    missing: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def find_variables(text: str | None) -> list[str]:
    """Every variable a template references, in first-appearance order."""
    seen: dict[str, None] = {}
    for match in PLACEHOLDER.finditer(text or ""):
        seen.setdefault(match.group(1), None)
    return list(seen)


def has_malformed_placeholder(text: str | None) -> bool:
    """Whether the text contains braces that look like a broken placeholder.

    Found by deleting every well-formed placeholder and looking at what is
    left: any surviving double brace is a typo — ``{{name}``, ``{{2days}}``,
    an unclosed tag — that would otherwise ship to a candidate as raw braces.
    Single braces are left alone, since prose and code samples use them.
    """
    residue = PLACEHOLDER.sub("", text or "")
    return "{{" in residue or "}}" in residue


def _lookup(context: dict, name: str) -> str | None:
    """The context value for ``name``, or ``None`` when it cannot be used.

    Empty and whitespace-only values count as absent: a candidate row with
    ``current_company = ""`` should take the fallback, not render a gap in the
    middle of a sentence.
    """
    value = context.get(name)
    if value is None or isinstance(value, bool):
        # A bool would render as "True", which is never what a template wants.
        return None
    text = str(value).strip()
    return text or None


def render(text: str | None, context: dict, *, escape: bool = False) -> RenderResult:
    """Substitute ``{{variables}}`` in ``text`` from ``context``.

    ``escape`` HTML-escapes each substituted value, for rendering into an HTML
    body: a candidate named ``O'Brien & Sons`` must not break the markup, and a
    value carrying a ``<script>`` must not become one.
    """
    missing: list[str] = []
    used: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        value = _lookup(context, name)
        if value is None:
            if fallback is None:
                if name not in missing:
                    missing.append(name)
                # Leave the placeholder intact so a preview shows the author
                # exactly which token failed, at the position it failed in.
                return match.group(0)
            value = fallback.strip()
        if name not in used:
            used.append(name)
        return html.escape(value) if escape else value

    return RenderResult(
        text=PLACEHOLDER.sub(substitute, text or ""), missing=missing, used=used
    )


@dataclass
class RenderedMessage:
    """A fully rendered subject and body, ready for the gateway."""

    subject: str
    body: str
    body_html: str | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def render_message(
    *,
    subject: str | None,
    body: str,
    body_html: str | None = None,
    context: dict,
) -> RenderedMessage:
    """Render a whole message, pooling the missing variables across its parts.

    All three parts are rendered even when the first one fails, so an author
    fixing a template sees every broken token at once instead of discovering
    them one send at a time.
    """
    rendered_subject = render(subject or "", context)
    rendered_body = render(body, context)
    rendered_html = render(body_html, context, escape=True) if body_html else None

    missing: list[str] = []
    for part in (rendered_subject, rendered_body, rendered_html):
        if part is None:
            continue
        for name in part.missing:
            if name not in missing:
                missing.append(name)

    return RenderedMessage(
        subject=rendered_subject.text,
        body=rendered_body.text,
        body_html=rendered_html.text if rendered_html else None,
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# Variable context
# --------------------------------------------------------------------------- #
def _first_name(full_name: str | None) -> str | None:
    return (full_name or "").strip().split(" ")[0] or None


def build_context(
    *,
    candidate=None,
    job=None,
    organization=None,
    sender=None,
    extra: dict | None = None,
) -> dict:
    """The variable map a candidate-facing message is rendered against.

    Everything is optional: a sourcing sequence has no job, and a template that
    references ``{{job_title}}`` anyway should fail loudly at render time
    rather than be prevented from existing.
    """
    context: dict = {}

    if candidate is not None:
        context.update(
            {
                "candidate_name": candidate.full_name,
                "first_name": _first_name(candidate.full_name),
                "candidate_email": candidate.email,
                "current_company": candidate.current_company,
                "current_role": candidate.current_role,
                "location": candidate.location,
                "years_experience": candidate.experience_years,
            }
        )
    if job is not None:
        context.update(
            {
                "job_title": job.title,
                "job_location": job.location,
                "job_department": job.department,
                "employment_type": job.employment_type,
            }
        )
    if organization is not None:
        context.update(
            {
                "company_name": organization.name,
                "organization_name": organization.name,
            }
        )
    if sender is not None:
        context.update(
            {
                "sender_name": sender.display_name or sender.email,
                "sender_email": sender.email,
            }
        )
    if extra:
        context.update(extra)
    return context


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def _validate(subject: str | None, body: str, channel: OutreachChannel) -> None:
    if not (body or "").strip():
        raise ValidationError("A template needs a body")
    if channel == OutreachChannel.EMAIL and not (subject or "").strip():
        raise ValidationError("An email template needs a subject")
    if subject and len(subject) > MAX_SUBJECT_LENGTH:
        raise ValidationError(
            f"Subject must be {MAX_SUBJECT_LENGTH} characters or fewer"
        )
    for part, label in ((subject, "subject"), (body, "body")):
        if has_malformed_placeholder(part):
            raise ValidationError(
                f"The {label} has an unclosed or malformed {{{{placeholder}}}}",
                details={"part": label},
            )


async def create_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    name: str,
    body: str,
    subject: str | None = None,
    body_html: str | None = None,
    channel: OutreachChannel = OutreachChannel.EMAIL,
    category: str | None = None,
    provider_template_name: str | None = None,
) -> MessageTemplate:
    label = (name or "").strip()
    if not label:
        raise ValidationError("A template needs a name")
    _validate(subject, body, channel)

    existing = await session.scalar(
        scoped_select(MessageTemplate, organization_id).where(
            MessageTemplate.name == label, MessageTemplate.channel == channel
        )
    )
    if existing is not None:
        raise ConflictError(f"A {channel} template named {label!r} already exists")

    template = MessageTemplate(
        organization_id=organization_id,
        name=label,
        channel=channel,
        subject=subject,
        body=body,
        body_html=body_html,
        category=category,
        provider_template_name=provider_template_name,
        # Recorded rather than required: the declared list drives editor
        # autocomplete, and deriving it means it cannot drift from the text.
        variables=sorted(
            set(find_variables(subject))
            | set(find_variables(body))
            | set(find_variables(body_html))
        ),
    )
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def get_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> MessageTemplate:
    template = await get_scoped(session, MessageTemplate, template_id, organization_id)
    if template is None:
        raise NotFoundError("Template not found")
    return template


async def list_templates(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    channel: OutreachChannel | None = None,
    active_only: bool = False,
) -> list[MessageTemplate]:
    stmt = scoped_select(MessageTemplate, organization_id)
    if channel is not None:
        stmt = stmt.where(MessageTemplate.channel == channel)
    if active_only:
        stmt = stmt.where(MessageTemplate.is_active.is_(True))
    stmt = stmt.order_by(MessageTemplate.name.asc())
    return list((await session.execute(stmt)).scalars().all())


async def update_template(
    session: AsyncSession,
    organization_id: uuid.UUID,
    template_id: uuid.UUID,
    *,
    changes: dict,
) -> MessageTemplate:
    template = await get_template(session, organization_id, template_id)

    for field_name in (
        "name",
        "subject",
        "body",
        "body_html",
        "category",
        "provider_template_name",
        "is_active",
    ):
        if field_name in changes and changes[field_name] is not None:
            setattr(template, field_name, changes[field_name])

    _validate(template.subject, template.body, OutreachChannel(template.channel))
    template.variables = sorted(
        set(find_variables(template.subject))
        | set(find_variables(template.body))
        | set(find_variables(template.body_html))
    )
    await session.commit()
    await session.refresh(template)
    return template


async def delete_template(
    session: AsyncSession, organization_id: uuid.UUID, template_id: uuid.UUID
) -> None:
    template = await get_template(session, organization_id, template_id)
    template.deleted_at = datetime.now(UTC)
    template.is_active = False
    await session.commit()


def preview(template: MessageTemplate, context: dict) -> RenderedMessage:
    """Render a template against sample values, missing variables and all."""
    return render_message(
        subject=template.subject,
        body=template.body,
        body_html=template.body_html,
        context=context,
    )
