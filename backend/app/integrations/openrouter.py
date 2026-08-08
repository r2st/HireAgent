"""OpenRouter LLM gateway (design §2.1, §5).

All model calls route through here. Two properties matter to callers:

* **Never fatal.** If the key is missing, the model is rate-limited, or the
  response is unparseable, this returns ``None`` rather than raising. Every
  caller has a deterministic fallback, so a degraded LLM degrades quality
  rather than breaking the pipeline.
* **Structured.** ``complete_json`` extracts a JSON object even when a free
  model wraps it in prose or a markdown fence, which the free tiers routinely
  do.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Matches ```json ... ``` or ``` ... ``` fenced blocks.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Retried: transient rate limiting and upstream faults.
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResult:
    """A model response plus the metadata callers persist for auditability."""

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.timeout = timeout or settings.llm_timeout_seconds
        self.max_retries = (
            max_retries if max_retries is not None else settings.llm_max_retries
        )
        self._enabled = enabled if enabled is not None else settings.llm_enabled

    @property
    def is_available(self) -> bool:
        return bool(self._enabled and self.api_key)

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format_json: bool = False,
    ) -> LLMResult | None:
        """Single-turn completion. Returns ``None`` when unavailable or failing."""
        if not self.is_available:
            logger.debug("OpenRouter unavailable (disabled or no API key)")
            return None

        chosen_model = model or settings.llm_model_parsing
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter attributes traffic using these.
            "HTTP-Referer": "https://hireagent.app",
            "X-Title": "HireAgent",
        }

        last_error: str | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    started = _now_ms()
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=body,
                        headers=headers,
                    )
                    latency = _now_ms() - started

                if response.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "OpenRouter transient error (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        last_error,
                    )
                    continue
                if response.status_code >= 400:
                    logger.error(
                        "OpenRouter rejected the request: %s %s",
                        response.status_code,
                        response.text[:400],
                    )
                    return None

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    last_error = "no choices in response"
                    continue

                content = (choices[0].get("message") or {}).get("content") or ""
                if not content.strip():
                    last_error = "empty completion"
                    continue

                usage = data.get("usage") or {}
                return LLMResult(
                    content=content,
                    model=data.get("model", chosen_model),
                    latency_ms=latency,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    raw=data,
                )
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "OpenRouter call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

        logger.error("OpenRouter exhausted retries: %s", last_error)
        return None

    async def complete_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> tuple[dict | list | None, LLMResult | None]:
        """Completion parsed into JSON.

        Returns ``(parsed, result)``; ``parsed`` is ``None`` if the model gave
        nothing usable, while ``result`` still carries the raw text for logging.
        """
        result = await self.complete(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        if result is None:
            return None, None
        return extract_json(result.content), result


def extract_json(text: str) -> dict | list | None:
    """Pull the first JSON object/array out of a model response.

    Free-tier models frequently answer with a fenced block, a preamble
    sentence, or trailing commentary, so a bare ``json.loads`` is not enough.
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict | list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fall back to slicing between the outermost brackets.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(candidate[start : end + 1])
                    if isinstance(parsed, dict | list):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return None


def _now_ms() -> int:
    import time

    return int(time.perf_counter() * 1000)


_default_client: OpenRouterClient | None = None


def get_llm_client() -> OpenRouterClient:
    global _default_client
    if _default_client is None:
        _default_client = OpenRouterClient()
    return _default_client


def set_llm_client(client: OpenRouterClient | None) -> None:
    """Swap the process-wide client (used by tests)."""
    global _default_client
    _default_client = client
