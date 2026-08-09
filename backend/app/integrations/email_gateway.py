"""Outbound email gateway (design §4.2).

Three transports cover the sender accounts the product supports: SMTP (which
also serves Amazon SES, since SES exposes an SMTP endpoint), the Gmail API, and
Microsoft Graph. They share one contract:

**A send either happened or it did not, and the caller must be able to tell the
difference between "try again" and "never try again".** That distinction is the
whole point of ``SendResult.retryable``. A refused recipient or a rejected
sender is permanent — retrying it burns sender reputation, which is the one
resource outreach cannot buy back — while a timeout or a 4xx greylist is worth
another pass. Everything is reported, never raised, so one bad address cannot
abort a batch.

Unlike the calendar gateway, an unreachable transport is *not* a soft failure:
there is no degraded mode for sending mail. The message stays queued and the
worker retries it.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import smtplib
import ssl
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

import httpx

from app.core.config import settings
from app.integrations import oauth
from app.integrations.oauth import TokenRefresh

logger = logging.getLogger(__name__)

SMTP = "smtp"
SES = "ses"
GMAIL = "gmail"
OUTLOOK = "outlook"

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"
MICROSOFT_API = "https://graph.microsoft.com/v1.0"

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GRAPH_SCOPE = "https://graph.microsoft.com/.default offline_access"

_TIMEOUT = 30.0
_SMTP_TIMEOUT = 30.0


@dataclass
class EmailCredentials:
    """The parts of an ``EmailAccount`` a transport needs, without the ORM row."""

    provider: str
    email: str
    display_name: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        return oauth.is_expired(self.expires_at)

    @property
    def sender(self) -> str:
        """The ``From`` header value, with a display name when there is one."""
        return (
            formataddr((self.display_name, self.email))
            if self.display_name
            else self.email
        )


@dataclass
class OutboundEmail:
    """One rendered message, ready to hand to a transport."""

    to_email: str
    subject: str
    text_body: str
    to_name: str | None = None
    html_body: str | None = None
    reply_to: str | None = None
    # Extra headers, used for List-Unsubscribe and threading.
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def recipient(self) -> str:
        return (
            formataddr((self.to_name, self.to_email)) if self.to_name else self.to_email
        )


@dataclass
class SendResult:
    """The outcome of one send attempt.

    ``retryable=False`` means the address or the account is the problem and the
    message must not be tried again. ``bounced`` narrows that to "the recipient
    rejected it", which additionally penalises the sender's reputation.
    """

    ok: bool
    provider_message_id: str | None = None
    error: str | None = None
    retryable: bool = True
    bounced: bool = False
    refreshed: TokenRefresh | None = None


def build_mime(credentials: EmailCredentials, message: OutboundEmail) -> EmailMessage:
    """Assemble the RFC 5322 message every transport ultimately sends.

    A plain-text part is always present even when HTML is supplied: a
    text/html-only message is a strong spam signal, and outreach lives or dies
    by deliverability.
    """
    mime = EmailMessage()
    mime["From"] = credentials.sender
    mime["To"] = message.recipient
    mime["Subject"] = message.subject
    mime["Message-ID"] = make_msgid()
    if message.reply_to:
        mime["Reply-To"] = message.reply_to
    for key, value in message.headers.items():
        if value is None:
            continue
        # Assigning to an existing key appends a duplicate header rather than
        # replacing it, so drop any prior value first.
        if key in mime:
            del mime[key]
        mime[key] = value

    mime.set_content(message.text_body or "")
    if message.html_body:
        mime.add_alternative(message.html_body, subtype="html")
    return mime


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class EmailTransport:
    """Base transport. Subclasses talk to one vendor."""

    name = "unknown"

    @property
    def is_configured(self) -> bool:
        return False

    async def send(
        self, credentials: EmailCredentials, message: OutboundEmail
    ) -> SendResult:
        raise NotImplementedError


class UnavailableTransport(EmailTransport):
    """Stand-in for a transport that is not configured or not recognised.

    Reports a permanent failure: with no way to send, retrying the same message
    against the same account will fail identically forever.
    """

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    async def send(
        self, credentials: EmailCredentials, message: OutboundEmail
    ) -> SendResult:
        return SendResult(ok=False, error=self.reason, retryable=False)


class SmtpTransport(EmailTransport):
    """Generic SMTP, including Amazon SES's SMTP endpoint.

    ``smtplib`` is synchronous, so the whole exchange runs in a worker thread.
    That is deliberate: an async SMTP client is another dependency, and sending
    happens on a background worker where a blocked thread costs nothing.
    """

    name = SMTP

    @property
    def is_configured(self) -> bool:
        # Configuration is per account, not per process — an org supplies its
        # own host — so the transport itself is always usable.
        return True

    async def send(
        self, credentials: EmailCredentials, message: OutboundEmail
    ) -> SendResult:
        if not credentials.smtp_host:
            return SendResult(
                ok=False,
                error="No SMTP host configured for this account",
                retryable=False,
            )
        mime = build_mime(credentials, message)
        try:
            await asyncio.to_thread(self._deliver, credentials, mime, message.to_email)
        except smtplib.SMTPRecipientsRefused as exc:
            return SendResult(
                ok=False,
                error=f"Recipient refused: {exc}",
                retryable=False,
                bounced=True,
            )
        except smtplib.SMTPSenderRefused as exc:
            return SendResult(
                ok=False, error=f"Sender refused: {exc}", retryable=False
            )
        except smtplib.SMTPResponseException as exc:
            # 5xx is a permanent refusal; 4xx is a greylist or a rate limit.
            permanent = 500 <= int(exc.smtp_code) < 600
            return SendResult(
                ok=False,
                error=f"SMTP {exc.smtp_code}: {exc.smtp_error!r}",
                retryable=not permanent,
                bounced=permanent,
            )
        except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
            return SendResult(ok=False, error=f"SMTP transport error: {exc}")

        return SendResult(ok=True, provider_message_id=mime["Message-ID"])

    def _deliver(
        self, credentials: EmailCredentials, mime: EmailMessage, to_email: str
    ) -> None:
        port = credentials.smtp_port or 587
        host = credentials.smtp_host or ""
        if port == 465:
            client: smtplib.SMTP = smtplib.SMTP_SSL(
                host, port, timeout=_SMTP_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            client = smtplib.SMTP(host, port, timeout=_SMTP_TIMEOUT)
        try:
            client.ehlo()
            if port != 465 and client.has_extn("starttls"):
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
            if credentials.smtp_username and credentials.smtp_password:
                client.login(credentials.smtp_username, credentials.smtp_password)
            client.send_message(mime, from_addr=credentials.email, to_addrs=[to_email])
        finally:
            try:
                client.quit()
            except (smtplib.SMTPException, OSError):
                # The message is already handed over; a rude disconnect on the
                # way out must not turn a successful send into a failure.
                pass


class _OAuthTransport(EmailTransport):
    """Shared token handling for the API-based transports."""

    token_url = ""
    refresh_scope: str | None = None

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self.client_id = client_id
        self.client_secret = client_secret

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def _authorize(
        self, credentials: EmailCredentials
    ) -> tuple[str | None, TokenRefresh | None]:
        """Return a usable access token, refreshing first if it has expired."""
        if credentials.access_token and not credentials.is_expired:
            return credentials.access_token, None
        refreshed = None
        if credentials.refresh_token and self.is_configured:
            refreshed = await oauth.refresh_access_token(
                self.token_url,
                client_id=self.client_id,
                client_secret=self.client_secret,
                refresh_token=credentials.refresh_token,
                scope=self.refresh_scope,
                provider=self.name,
                timeout=_TIMEOUT,
            )
        if refreshed is not None:
            return refreshed.access_token, refreshed
        # No refresh available: try the existing token anyway. It may still
        # work if our expiry bookkeeping is wrong.
        return credentials.access_token, None

    @staticmethod
    def _classify(status_code: int) -> tuple[bool, bool]:
        """Map an HTTP status onto ``(retryable, bounced)``.

        401 is retryable because the next attempt refreshes the token first,
        and 429 because the rate limit lifts. The rest of the 4xx range means
        the request itself is wrong and will stay wrong.
        """
        if status_code in (401, 408, 429):
            return True, False
        if 400 <= status_code < 500:
            return False, status_code in (400, 422)
        return True, False


class GmailTransport(_OAuthTransport):
    """Gmail API ``users.messages.send``."""

    name = GMAIL
    token_url = oauth.GOOGLE_TOKEN_URL
    refresh_scope = GMAIL_SCOPE

    async def send(
        self, credentials: EmailCredentials, message: OutboundEmail
    ) -> SendResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return SendResult(
                ok=False, error="No usable Gmail access token", retryable=False
            )

        mime = build_mime(credentials, message)
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{GMAIL_API}/users/me/messages/send",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"raw": raw},
                )
        except httpx.HTTPError as exc:
            return SendResult(
                ok=False, error=f"Gmail transport error: {exc}", refreshed=refreshed
            )

        if response.status_code >= 400:
            retryable, bounced = self._classify(response.status_code)
            return SendResult(
                ok=False,
                error=f"Gmail HTTP {response.status_code}: {response.text[:200]}",
                retryable=retryable,
                bounced=bounced,
                refreshed=refreshed,
            )

        try:
            body = response.json()
        except ValueError:
            body = {}
        return SendResult(
            ok=True,
            provider_message_id=(body or {}).get("id") or mime["Message-ID"],
            refreshed=refreshed,
        )


class OutlookTransport(_OAuthTransport):
    """Microsoft Graph ``/me/sendMail``."""

    name = OUTLOOK
    refresh_scope = GRAPH_SCOPE

    def __init__(
        self, client_id: str = "", client_secret: str = "", tenant: str = "common"
    ) -> None:
        super().__init__(client_id, client_secret)
        self.tenant = tenant or "common"

    @property
    def token_url(self) -> str:  # type: ignore[override]
        return oauth.microsoft_token_url(self.tenant)

    async def send(
        self, credentials: EmailCredentials, message: OutboundEmail
    ) -> SendResult:
        token, refreshed = await self._authorize(credentials)
        if not token:
            return SendResult(
                ok=False, error="No usable Outlook access token", retryable=False
            )

        # Graph takes structured JSON rather than raw MIME, so the body is
        # described here instead of going through build_mime.
        content_type = "HTML" if message.html_body else "Text"
        payload: dict = {
            "message": {
                "subject": message.subject,
                "body": {
                    "contentType": content_type,
                    "content": message.html_body or message.text_body or "",
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": message.to_email,
                            "name": message.to_name or message.to_email,
                        }
                    }
                ],
            },
            "saveToSentItems": True,
        }
        if message.reply_to:
            payload["message"]["replyTo"] = [
                {"emailAddress": {"address": message.reply_to}}
            ]
        if message.headers:
            # Graph only accepts custom headers prefixed with "x-".
            payload["message"]["internetMessageHeaders"] = [
                {"name": name, "value": value}
                for name, value in message.headers.items()
                if name.lower().startswith("x-")
            ]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{MICROSOFT_API}/me/sendMail",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            return SendResult(
                ok=False, error=f"Outlook transport error: {exc}", refreshed=refreshed
            )

        if response.status_code >= 400:
            retryable, bounced = self._classify(response.status_code)
            return SendResult(
                ok=False,
                error=f"Outlook HTTP {response.status_code}: {response.text[:200]}",
                retryable=retryable,
                bounced=bounced,
                refreshed=refreshed,
            )

        # sendMail returns 202 with an empty body and no id of its own.
        return SendResult(
            ok=True,
            provider_message_id=response.headers.get("request-id")
            or f"graph-{uuid.uuid4().hex}",
            refreshed=refreshed,
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_registry: dict[str, EmailTransport] | None = None


def _build_registry() -> dict[str, EmailTransport]:
    smtp = SmtpTransport()
    gmail = GmailTransport(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
    )
    outlook = OutlookTransport(
        client_id=settings.microsoft_client_id,
        client_secret=settings.microsoft_client_secret,
        tenant=settings.microsoft_tenant_id,
    )

    def usable(transport: EmailTransport) -> EmailTransport:
        if not settings.outreach_sending_enabled:
            return UnavailableTransport(
                transport.name, "Outreach sending is disabled for this deployment"
            )
        if not transport.is_configured:
            return UnavailableTransport(
                transport.name, f"{transport.name} sending is not configured"
            )
        return transport

    # SES speaks SMTP, so it shares the SMTP transport rather than duplicating
    # a signed-API client for the same job.
    return {
        SMTP: usable(smtp),
        SES: usable(smtp),
        GMAIL: usable(gmail),
        OUTLOOK: usable(outlook),
    }


def get_transport(name: str) -> EmailTransport:
    """The transport for ``name``; never raises for an unknown vendor."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    transport = _registry.get((name or "").lower())
    if transport is None:
        return UnavailableTransport(
            name or "unknown", f"Unsupported email provider {name!r}"
        )
    return transport


def set_transports(registry: dict[str, EmailTransport] | None) -> None:
    """Swap the process-wide transport registry (used by tests)."""
    global _registry
    _registry = registry
