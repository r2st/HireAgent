"""The outbound email gateway (design §4.2).

The contract under test is the retry classification. A transport that reports
every failure as retryable turns one dead address into an infinite loop that
burns the sender's reputation; one that reports every failure as permanent
throws away mail over a transient timeout. Each transport is checked on both
sides of that line, plus the MIME it hands over.
"""

from __future__ import annotations

import smtplib
import ssl
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.integrations import email_gateway as email_api
from app.integrations import oauth as oauth_api
from app.integrations.email_gateway import (
    EmailCredentials,
    GmailTransport,
    OutboundEmail,
    OutlookTransport,
    SmtpTransport,
    UnavailableTransport,
    build_mime,
)
from tests.factories import FakeHTTP, FakeResponse, install_http


def creds(**overrides) -> EmailCredentials:
    base = {
        "provider": "smtp",
        "email": "talent@acme.test",
        "display_name": "Acme Talent",
        "smtp_host": "smtp.acme.test",
        "smtp_port": 587,
        "smtp_username": "talent@acme.test",
        "smtp_password": "hunter2",
        "access_token": "at-live",
        "refresh_token": "rt",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
    }
    base.update(overrides)
    return EmailCredentials(**base)  # type: ignore[arg-type]


def outbound(**overrides) -> OutboundEmail:
    base = {
        "to_email": "ada@example.test",
        "to_name": "Ada Lovelace",
        "subject": "A role at Acme",
        "text_body": "Hello Ada,\n\nWe are hiring.",
    }
    base.update(overrides)
    return OutboundEmail(**base)  # type: ignore[arg-type]


def install(monkeypatch: pytest.MonkeyPatch, http: FakeHTTP) -> FakeHTTP:
    return install_http(monkeypatch, http, email_api, oauth_api)


# --------------------------------------------------------------------------- #
# MIME construction
# --------------------------------------------------------------------------- #
class TestBuildMime:
    def test_uses_the_display_name_in_the_from_header(self) -> None:
        mime = build_mime(creds(), outbound())
        assert mime["From"] == "Acme Talent <talent@acme.test>"
        assert mime["To"] == "Ada Lovelace <ada@example.test>"

    def test_falls_back_to_a_bare_address_without_a_display_name(self) -> None:
        mime = build_mime(creds(display_name=None), outbound(to_name=None))
        assert mime["From"] == "talent@acme.test"
        assert mime["To"] == "ada@example.test"

    def test_html_is_sent_alongside_text_never_alone(self) -> None:
        # A text/html-only message is a strong spam signal.
        mime = build_mime(creds(), outbound(html_body="<p>We are hiring.</p>"))
        types = [part.get_content_type() for part in mime.walk()]
        assert "text/plain" in types
        assert "text/html" in types

    def test_a_text_only_message_stays_single_part(self) -> None:
        mime = build_mime(creds(), outbound())
        assert mime.get_content_type() == "text/plain"

    def test_extra_headers_are_carried(self) -> None:
        mime = build_mime(
            creds(),
            outbound(headers={"List-Unsubscribe": "<https://acme.test/u/abc>"}),
        )
        assert mime["List-Unsubscribe"] == "<https://acme.test/u/abc>"

    def test_a_repeated_header_replaces_rather_than_duplicates(self) -> None:
        # Two Reply-To headers is a malformed message, not a stronger hint.
        mime = build_mime(
            creds(),
            outbound(
                reply_to="recruiter@acme.test",
                headers={"Reply-To": "override@acme.test"},
            ),
        )
        assert mime.get_all("Reply-To") == ["override@acme.test"]

    def test_every_message_gets_a_message_id(self) -> None:
        assert build_mime(creds(), outbound())["Message-ID"]


# --------------------------------------------------------------------------- #
# SMTP
# --------------------------------------------------------------------------- #
class FakeSMTP:
    """Records the exchange instead of opening a socket."""

    instances: list[FakeSMTP] = []

    def __init__(self, host: str, port: int, timeout: float = 0, **kwargs) -> None:
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.starttls_called = False
        self.login_args: tuple | None = None
        self.sent: list[dict] = []
        self.quit_called = False
        self.extensions = {"starttls"}
        self.raise_on_send: Exception | None = None
        self.raise_on_quit: Exception | None = None
        FakeSMTP.instances.append(self)

    def ehlo(self) -> None: ...

    def has_extn(self, name: str) -> bool:
        return name in self.extensions

    def starttls(self, context=None) -> None:
        self.starttls_called = True

    def login(self, username: str, password: str) -> None:
        self.login_args = (username, password)

    def send_message(self, mime, from_addr=None, to_addrs=None) -> None:
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append({"from": from_addr, "to": to_addrs, "mime": mime})

    def quit(self) -> None:
        self.quit_called = True
        if self.raise_on_quit is not None:
            raise self.raise_on_quit


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch):
    FakeSMTP.instances = []
    made: dict = {}

    def factory(host, port, timeout=0, **kwargs):
        client = FakeSMTP(host, port, timeout, **kwargs)
        for key, value in made.get("prime", {}).items():
            setattr(client, key, value)
        return client

    monkeypatch.setattr(smtplib, "SMTP", factory)
    monkeypatch.setattr(smtplib, "SMTP_SSL", factory)
    return made


class TestSmtpTransport:
    async def test_sends_and_reports_the_message_id(self, smtp) -> None:
        result = await SmtpTransport().send(creds(), outbound())
        assert result.ok
        assert result.provider_message_id
        client = FakeSMTP.instances[0]
        assert client.sent[0]["to"] == ["ada@example.test"]
        assert client.sent[0]["from"] == "talent@acme.test"

    async def test_upgrades_to_tls_and_authenticates(self, smtp) -> None:
        await SmtpTransport().send(creds(), outbound())
        client = FakeSMTP.instances[0]
        assert client.starttls_called
        assert client.login_args == ("talent@acme.test", "hunter2")

    async def test_port_465_uses_implicit_tls_without_starttls(self, smtp) -> None:
        await SmtpTransport().send(creds(smtp_port=465), outbound())
        client = FakeSMTP.instances[0]
        assert client.port == 465
        assert not client.starttls_called

    async def test_skips_starttls_when_the_server_does_not_offer_it(self, smtp) -> None:
        smtp["prime"] = {"extensions": set()}
        result = await SmtpTransport().send(creds(), outbound())
        assert result.ok
        assert not FakeSMTP.instances[0].starttls_called

    async def test_an_account_without_a_host_fails_permanently(self, smtp) -> None:
        result = await SmtpTransport().send(creds(smtp_host=None), outbound())
        assert not result.ok
        assert result.retryable is False

    async def test_a_refused_recipient_is_a_permanent_bounce(self, smtp) -> None:
        smtp["prime"] = {
            "raise_on_send": smtplib.SMTPRecipientsRefused(
                {"ada@example.test": (550, b"No such user")}
            )
        }
        result = await SmtpTransport().send(creds(), outbound())
        assert not result.ok
        assert result.retryable is False
        assert result.bounced is True

    async def test_a_refused_sender_is_permanent_but_not_a_bounce(self, smtp) -> None:
        # The mailbox is the problem, not the candidate's address.
        smtp["prime"] = {
            "raise_on_send": smtplib.SMTPSenderRefused(
                530, b"Not authenticated", "talent@acme.test"
            )
        }
        result = await SmtpTransport().send(creds(), outbound())
        assert not result.ok
        assert result.retryable is False
        assert result.bounced is False

    async def test_a_4xx_response_is_retryable(self, smtp) -> None:
        # Greylisting and rate limits both answer 4xx and both clear.
        smtp["prime"] = {
            "raise_on_send": smtplib.SMTPResponseException(451, b"Try again later")
        }
        result = await SmtpTransport().send(creds(), outbound())
        assert not result.ok
        assert result.retryable is True
        assert result.bounced is False

    async def test_a_5xx_response_is_a_permanent_bounce(self, smtp) -> None:
        smtp["prime"] = {
            "raise_on_send": smtplib.SMTPResponseException(552, b"Mailbox full")
        }
        result = await SmtpTransport().send(creds(), outbound())
        assert result.retryable is False
        assert result.bounced is True

    async def test_a_dead_socket_is_retryable(self, smtp) -> None:
        smtp["prime"] = {"raise_on_send": OSError("connection reset")}
        result = await SmtpTransport().send(creds(), outbound())
        assert not result.ok
        assert result.retryable is True

    async def test_a_tls_failure_is_reported_not_raised(self, smtp) -> None:
        smtp["prime"] = {"raise_on_send": ssl.SSLError("handshake failed")}
        result = await SmtpTransport().send(creds(), outbound())
        assert not result.ok

    async def test_a_rude_disconnect_after_delivery_still_counts_as_sent(
        self, smtp
    ) -> None:
        # The message is already handed over; QUIT failing changes nothing.
        smtp["prime"] = {"raise_on_quit": smtplib.SMTPServerDisconnected("bye")}
        result = await SmtpTransport().send(creds(), outbound())
        assert result.ok

    async def test_an_account_without_credentials_skips_login(self, smtp) -> None:
        result = await SmtpTransport().send(
            creds(smtp_username=None, smtp_password=None), outbound()
        )
        assert result.ok
        assert FakeSMTP.instances[0].login_args is None


# --------------------------------------------------------------------------- #
# Gmail
# --------------------------------------------------------------------------- #
@pytest.fixture
def gmail() -> GmailTransport:
    return GmailTransport(client_id="cid", client_secret="secret")


class TestGmailTransport:
    async def test_posts_base64url_raw_mime(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(monkeypatch, FakeHTTP(FakeResponse(200, {"id": "gmail-1"})))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert result.ok
        assert result.provider_message_id == "gmail-1"
        body = http.calls[0]["json"]
        # URL-safe alphabet only: the standard one breaks the API's parser.
        assert "+" not in body["raw"] and "/" not in body["raw"]

    async def test_a_401_is_retryable_so_the_next_pass_refreshes(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(401, {"error": "unauthorized"})))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert not result.ok
        assert result.retryable is True

    async def test_a_429_is_retryable(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(429, {"error": "rate"})))
        assert (await gmail.send(creds(provider="gmail"), outbound())).retryable

    async def test_a_403_is_permanent(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(403, {"error": "forbidden"})))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert result.retryable is False

    async def test_a_400_is_treated_as_a_bounce(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gmail answers a malformed recipient with 400 rather than a bounce.
        install(monkeypatch, FakeHTTP(FakeResponse(400, {"error": "bad address"})))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert result.bounced is True
        assert result.retryable is False

    async def test_a_5xx_is_retryable(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(503, {"error": "down"})))
        assert (await gmail.send(creds(provider="gmail"), outbound())).retryable

    async def test_a_transport_fault_is_reported_not_raised(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(httpx.ConnectError("no route")))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert not result.ok
        assert result.retryable is True

    async def test_an_expired_token_is_refreshed_first(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(200, {"access_token": "at-new", "expires_in": 3600}),
                FakeResponse(200, {"id": "gmail-2"}),
            ),
        )
        result = await gmail.send(
            creds(provider="gmail", expires_at=datetime.now(UTC) - timedelta(hours=1)),
            outbound(),
        )
        assert result.ok
        assert result.refreshed is not None
        assert result.refreshed.access_token == "at-new"
        assert http.calls[1]["headers"]["Authorization"] == "Bearer at-new"

    async def test_no_token_at_all_fails_permanently(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP())
        result = await gmail.send(
            creds(provider="gmail", access_token=None, refresh_token=None), outbound()
        )
        assert not result.ok
        assert result.retryable is False

    async def test_a_success_without_json_still_reports_an_id(
        self, gmail: GmailTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(FakeResponse(200, None)))
        result = await gmail.send(creds(provider="gmail"), outbound())
        assert result.ok
        assert result.provider_message_id


# --------------------------------------------------------------------------- #
# Outlook
# --------------------------------------------------------------------------- #
@pytest.fixture
def outlook() -> OutlookTransport:
    return OutlookTransport(client_id="cid", client_secret="secret")


class GraphResponse(FakeResponse):
    def __init__(self, status_code: int, payload=None, headers=None) -> None:
        super().__init__(status_code, payload)
        self.headers = headers or {}


class TestOutlookTransport:
    async def test_sends_structured_json_not_raw_mime(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(monkeypatch, FakeHTTP(GraphResponse(202)))
        result = await outlook.send(creds(provider="outlook"), outbound())
        assert result.ok
        payload = http.calls[0]["json"]["message"]
        assert payload["subject"] == "A role at Acme"
        assert payload["toRecipients"][0]["emailAddress"]["address"] == (
            "ada@example.test"
        )
        assert payload["body"]["contentType"] == "Text"

    async def test_html_switches_the_content_type(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(monkeypatch, FakeHTTP(GraphResponse(202)))
        await outlook.send(
            creds(provider="outlook"), outbound(html_body="<p>Hi</p>")
        )
        body = http.calls[0]["json"]["message"]["body"]
        assert body["contentType"] == "HTML"
        assert body["content"] == "<p>Hi</p>"

    async def test_only_x_prefixed_headers_reach_graph(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Graph rejects the whole request over a non-"x-" custom header.
        http = install(monkeypatch, FakeHTTP(GraphResponse(202)))
        await outlook.send(
            creds(provider="outlook"),
            outbound(
                headers={
                    "X-Campaign": "abc",
                    "List-Unsubscribe": "<https://acme.test/u>",
                }
            ),
        )
        names = [
            h["name"]
            for h in http.calls[0]["json"]["message"]["internetMessageHeaders"]
        ]
        assert names == ["X-Campaign"]

    async def test_reply_to_is_carried(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(monkeypatch, FakeHTTP(GraphResponse(202)))
        await outlook.send(
            creds(provider="outlook"), outbound(reply_to="recruiter@acme.test")
        )
        reply = http.calls[0]["json"]["message"]["replyTo"]
        assert reply[0]["emailAddress"]["address"] == "recruiter@acme.test"

    async def test_an_empty_202_still_yields_a_message_id(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # sendMail returns no body, but callers need something to correlate on.
        install(monkeypatch, FakeHTTP(GraphResponse(202)))
        result = await outlook.send(creds(provider="outlook"), outbound())
        assert result.provider_message_id

    async def test_the_request_id_header_is_preferred_as_the_id(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch, FakeHTTP(GraphResponse(202, headers={"request-id": "req-9"}))
        )
        result = await outlook.send(creds(provider="outlook"), outbound())
        assert result.provider_message_id == "req-9"

    async def test_a_403_is_permanent(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(GraphResponse(403, {"error": "denied"})))
        result = await outlook.send(creds(provider="outlook"), outbound())
        assert result.retryable is False

    async def test_a_transport_fault_is_reported_not_raised(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeHTTP(httpx.ConnectTimeout("slow")))
        result = await outlook.send(creds(provider="outlook"), outbound())
        assert not result.ok
        assert result.retryable is True

    async def test_refresh_requests_the_graph_scope(
        self, outlook: OutlookTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = install(
            monkeypatch,
            FakeHTTP(
                FakeResponse(200, {"access_token": "at-new", "expires_in": 3600}),
                GraphResponse(202),
            ),
        )
        await outlook.send(
            creds(
                provider="outlook", expires_at=datetime.now(UTC) - timedelta(hours=1)
            ),
            outbound(),
        )
        assert "scope" in http.calls[0]["data"]
        assert "login.microsoftonline.com" in http.calls[0]["url"]

    def test_a_missing_tenant_defaults_to_common(self) -> None:
        assert "/common/" in OutlookTransport(tenant="").token_url


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_ses_shares_the_smtp_transport(self) -> None:
        # SES exposes an SMTP endpoint, so it needs no separate client.
        email_api.set_transports(None)
        assert type(email_api.get_transport("ses")) is type(
            email_api.get_transport("smtp")
        )

    def test_an_unknown_vendor_is_unavailable_rather_than_an_error(self) -> None:
        transport = email_api.get_transport("carrier-pigeon")
        assert isinstance(transport, UnavailableTransport)

    async def test_an_unavailable_transport_fails_permanently(self) -> None:
        result = await UnavailableTransport("gmail", "not configured").send(
            creds(), outbound()
        )
        assert not result.ok
        assert result.retryable is False

    def test_oauth_transports_are_unavailable_when_unconfigured(self) -> None:
        email_api.set_transports(None)
        assert isinstance(email_api.get_transport("gmail"), UnavailableTransport)

    def test_sending_can_be_switched_off_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.core.config import settings

        monkeypatch.setattr(settings, "outreach_sending_enabled", False)
        email_api.set_transports(None)
        assert isinstance(email_api.get_transport("smtp"), UnavailableTransport)

    def test_set_transports_overrides_the_registry(self) -> None:
        stub = UnavailableTransport("smtp", "stubbed")
        email_api.set_transports({"smtp": stub})
        assert email_api.get_transport("smtp") is stub
