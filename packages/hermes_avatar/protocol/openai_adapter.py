"""Generic OpenAI-compatible LLM adapter for the LiveEmote avatar.

The user asked for "a real working AI avatar" — not canned local responses. This
adapter calls any chat-completions-compatible endpoint (OpenAI, SambaNova,
Together, llama.cpp server, vLLM, LM Studio, OpenRouter) and asks the model to
return **structured JSON** containing the spoken text plus a small set of voice
and affect tags. The runtime then uses those tags to drive the avatar's voice
backend (LuxTTS / ElevenLabs) and the affect runtime's post-speech policy.

Design notes:

* The adapter talks HTTP via ``httpx`` only — no SDK install required. The user
  just provides ``OPENAI_COMPATIBLE_API_KEY``, ``OPENAI_COMPATIBLE_BASE_URL``,
  and ``OPENAI_COMPATIBLE_MODEL`` through the Freebuff API Keys UI.
* The JSON contract is enforced two ways at once: a strict prompt that asks for
  ``{"text": ..., "tags": {"voice": {...}, "affect": "..."}}`` and ``response_format``
  set to ``{"type": "json_object"}`` for endpoints that support it (OpenAI,
  SambaNova, Together). Endpoints that don't support ``response_format`` still
  work — the adapter falls back to parsing the first ``{...}`` block in the
  response.
* If no API key is configured the adapter returns an empty ``AgentResponse``,
  letting the orchestrator fall through to its offline reflect-only mode (the
  avatar still mirrors/reflects the user's affect, it just won't speak back
  with LLM-generated text). This keeps the demo runnable end-to-end during
  setup, before the user pastes their key.

This file does not touch any secrets. It only reads env vars by name; the
user is expected to set those in the Freebuff API Keys panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .agent_bridge import AgentResponse
from hermes_avatar.affect.state import UserAffectState
from hermes_avatar.util import (
    CircuitBreaker,
    compute_backoff_delay,
    is_retryable_error,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com")
DEFAULT_TIMEOUT_S = float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_S", "20"))


@dataclass
class AdapterConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Retry configuration (item 2.2). ``max_retries=2`` means up to 3 total
    # attempts per user-request (1 initial + 2 retries). Backoff uses the
    # shared ``compute_backoff_delay`` so far-future tuning happens in one
    # place. A 5xx / 429 / network-blip retry avoids round-tripping through
    # the breaker for short transient issues; the breaker still catches the
    # sustained outage through ``record_failure()`` *once* per user-request
    # when ALL retries are exhausted (or a non-retryable error short-circuits).
    max_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 4.0
    retry_jitter_factor: float = 0.1

    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)


SYSTEM_PROMPT = (
    "You are the voice of an autonomous live avatar driven by the user's webcam "
    "affect (focus + energy). The avatar has its OWN identity and its OWN "
    "voice — you are not narrating the user's face or moving it. Your job is "
    "to speak briefly, naturally, and in response to what the user just said. "
    "Sometimes the right answer is a short silence (return text=\"\" if the "
    "user's signal is already self-contained). Keep spoken text under 80 words.\n\n"
    "You MUST return a single JSON object with this exact shape and no other "
    "keys:\n\n"
    "    {\n"
    '      "text": "<the line the avatar speaks back, or empty string for a comfortable pause>",\n'
    '      "tags": {\n'
    '        "affect": "<grounded | focused | warm | curious | grounded_steady | spacious | amused | reassuring>",\n'
    '        "voice": {"pace": 0.30...0.60, "warmth": 0.30...0.85, "intensity": 0.20...0.70}\n'
    "      }\n"
    "    }\n\n"
    "Interpret the user's affect summary carefully:\n"
    "  - high attention + low valence + high arousal → frustrated/tense → be validating_grounded\n"
    "  - low attention + low arousal → tired/distracted → be spacious and warm, do not probe\n"
    "  - high arousal + positive valence → happy/engaged → be warm and lightly amused\n"
    "  - high tension regardless of valence → reduce intensity, validate first, never match anger\n\n"
    "Do not pretend to know the user. Do not narrate their face. Do not mirror "
    "their motion. Speak to them like a relaxed, observant peer."
)


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Find the first {...} JSON object in a model output."""
    if not text:
        return None
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Find first balanced {...} block by scanning.
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _coerce_voice(voice: Any) -> dict[str, float]:
    if not isinstance(voice, dict):
        return {"pace": 0.44, "warmth": 0.62, "intensity": 0.35}
    out = {"pace": 0.44, "warmth": 0.62, "intensity": 0.35}
    for key in ("pace", "warmth", "intensity"):
        try:
            v = float(voice.get(key, out[key]))
            out[key] = max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            continue
    return out


class OpenAICompatibleAdapter:
    """Calls any OpenAI chat-completions-compatible endpoint in JSON mode."""

    name: str = "openai_compatible"
    mode: str = "openai_compatible"

    def __init__(self, config: AdapterConfig | None = None) -> None:
        if config is None:
            config = AdapterConfig(
                api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
                base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
                model=os.getenv("OPENAI_COMPATIBLE_MODEL", DEFAULT_MODEL),
                timeout_s=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))),
            )
        self.config = config
        self.last_error: str | None = None
        self.last_latency_ms: int | None = None
        self.last_model: str = self.config.model
        # Retry counters -- surfaced via capability_status() so the demo
        # server's /api/health can report how aggressively this adapter is
        # retrying under load. ``last_retry_count`` resets each user-request,
        # ``total_retries`` is cumulative since adapter init.
        self.last_retry_count: int = 0
        self.total_retries: int = 0
        # Circuit breaker protects the avatar's LLM call stream from
        # cascading vendor outages. 3 consecutive failures / 30s open
        # window is more tolerant than LuxTTS=2 / 60s because HTTP 429/5xx
        # storms are common in real chat-completions endpoints and external
        # HTTP vendors typically self-recover in seconds rather than
        # minutes. Trip catches ALL exception paths (HTTPStatusError,
        # network errors, JSON parse failures) -- preventing model-spam
        # loops on bad keys / broken integration pairs (e.g. 401/404),
        # matching LuxTTS's except-Exception simplicity.
        self.cb = CircuitBreaker(failure_threshold=3, open_timeout=30.0, name="openai")

    def is_configured(self) -> bool:
        return self.config.is_configured()

    def capability_status(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "configured": self.is_configured(),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
            "circuit_breaker": self.cb.snapshot(),
            # Retry behaviour surface for /api/health monitoring. Operators
            # can spot a vendor in flap mode (``last_retry_count`` near
            # ``max_retries``) or chronically noisy (``total_retries`` large).
            "retry": {
                "max_retries": self.config.max_retries,
                "base_delay": self.config.retry_base_delay,
                "max_delay": self.config.retry_max_delay,
                "jitter_factor": self.config.retry_jitter_factor,
                "last_retry_count": self.last_retry_count,
                "total_retries": self.total_retries,
                "retry_via": "compute_backoff_delay + is_retryable_error",
            },
        }

    async def generate_response(
        self, user_text: str, affect_state: UserAffectState
    ) -> AgentResponse:
        if not self.is_configured():
            self.last_error = "OPENAI_COMPATIBLE_API_KEY not set"
            return AgentResponse(text="", tags={}, source="offline")

        # Circuit-breaker gate (after is_configured, before any HTTP).
        # The breaker measures BACKEND HEALTH, not user config status --
        # unconfigured callers fall through to offline mode WITHOUT
        # touching the breaker at all. On OPEN we emit the same offline
        # AgentResponse shape the existing except arms return -- the demo
        # contract that offline-mode mirroring/reflecting continues to
        # work during a vendor outage stays intact.
        if not self.cb.allow():
            self.last_error = "openai circuit breaker open"
            log.warning(
                "openai circuit breaker open; emit offline agent response",
                extra={"audit": {"event": "openai.cb_open"}},
            )
            return AgentResponse(text="", tags={}, source="offline")

        payload = self._build_payload(user_text, affect_state)
        # Reset per-user-request counter so ``last_retry_count`` reflects
        # the LAST completed request, not a moving sum.
        self.last_retry_count = 0
        # Retry loop (item 2.2). The breaker gate above guarantees we only
        # enter this loop while CLOSED or HALF_OPEN-probing. Each iteration:
        #   - on success: record_success() + break out
        #   - on retryable transient error + attempts left: sleep with
        #     exponential backoff, retry
        #   - on retryable transient error + exhausted: break out
        #   - on non-retryable error (401/403/404/JSON parse fails on 200):
        #     break out immediately (no point burning retries on a doomed call)
        # After the loop we record the breaker outcome ONCE per user-request so
        # a 3-attempt failure counts as a single breaker event, not three --
        # this is the explicit user spec for retry+breaker composition.
        last_exc: Exception | None = None
        data: dict[str, Any] | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                    response = await client.post(
                        f"{self.config.base_url}/v1/chat/completions",
                        headers={
                            "authorization": f"Bearer {self.config.api_key}",
                            "content-type": "application/json",
                            **self.config.extra_headers,
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                self.cb.record_success()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                # Last attempt: stop the loop here so we fall through to
                # the post-loop error path.
                if attempt == self.config.max_retries:
                    break
                # Non-retryable error: stop immediately. ``is_retryable_error``
                # returns False on 401/403/404 and on JSON parse failures
                # (since their message lacks the retryable-keyword substring
                # set), so 4xx and bad-shape responses don't burn retries.
                if not is_retryable_error(exc):
                    break
                # Retry-able and not last attempt: exponential backoff with
                # jitter, then loop. Counters tick ONCE per retry decision
                # so a 2-retry sequence lands at last_retry_count=2.
                self.last_retry_count += 1
                self.total_retries += 1
                delay = compute_backoff_delay(
                    attempt,
                    self.config.retry_base_delay,
                    self.config.retry_max_delay,
                    self.config.retry_jitter_factor,
                )
                await asyncio.sleep(delay)

        if last_exc is not None:
            self.cb.record_failure()
            if isinstance(last_exc, httpx.HTTPStatusError):
                self.last_error = (
                    f"HTTP {last_exc.response.status_code}: "
                    f"{last_exc.response.text[:200]}"
                )
                log.warning(
                    "openai_compatible HTTP error after retries: %s",
                    self.last_error,
                    extra={"audit": {"event": "openai.http_error", "status_code": last_exc.response.status_code}},
                )
            else:
                self.last_error = str(last_exc)
                log.warning(
                    "openai_compatible request failed after retries: %s",
                    self.last_error,
                    extra={"audit": {"event": "openai.request_failed", "error": str(last_exc)}},
                )
            return AgentResponse(text="", tags={}, source="offline")

        self.last_error = None
        text = self._extract_text(data)
        parsed = _extract_json_block(text) or {}
        spoken = str(parsed.get("text") or "").strip()
        tags = parsed.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        tags.setdefault("source_model", self.config.model)
        tags.setdefault("source_backend", self.name)
        tags["voice"] = _coerce_voice(tags.get("voice"))
        if "affect" in tags and not isinstance(tags["affect"], str):
            tags.pop("affect")
        return AgentResponse(text=spoken, tags=tags, source=self.name)

    # --- Build payload --------------------------------------------------

    def _build_payload(
        self, user_text: str, affect_state: UserAffectState
    ) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "temperature": 0.6,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"user_text: {user_text or '<silent>'}\n"
                        f"user_affect: {json.dumps(affect_state.to_dict())}"
                    ),
                },
            ],
        }
        # JSON mode is best-effort; some endpoints ignore it but still produce JSON.
        try:
            body["response_format"] = {"type": "json_object"}
        except Exception:
            pass
        return body

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        # Standard OpenAI shape.
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Some endpoints return list[{type,text}].
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join([part for part in parts if part])
        # Fallback shapes (a few providers).
        for key in ("text", "content", "response", "output"):
            if isinstance(data.get(key), str):
                return data[key]
        return ""
