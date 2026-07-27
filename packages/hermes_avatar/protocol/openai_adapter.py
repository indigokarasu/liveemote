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

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .agent_bridge import AgentResponse
from hermes_avatar.affect.state import UserAffectState

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
        }

    async def generate_response(
        self, user_text: str, affect_state: UserAffectState
    ) -> AgentResponse:
        if not self.is_configured():
            self.last_error = "OPENAI_COMPATIBLE_API_KEY not set"
            return AgentResponse(text="", tags={}, source="offline")

        payload = self._build_payload(user_text, affect_state)
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
        except httpx.HTTPStatusError as exc:
            self.last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            log.warning("openai_compatible HTTP error: %s", self.last_error)
            return AgentResponse(text="", tags={}, source="offline")
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("openai_compatible request failed: %s", self.last_error)
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
