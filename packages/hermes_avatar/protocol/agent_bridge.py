from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from hermes_avatar.affect.state import UserAffectState

log = logging.getLogger(__name__)

OFFLINE_MODES = {"none", "off", "offline", "disabled", "no_llm", "no-llm"}
FAKE_MODES = {"fake", "mock", "local"}
EXTERNAL_MODES = {"external", "agent", "harness", "openclaw", "hermes", "deerflow"}
OPENAI_COMPATIBLE_MODES = {"openai-compatible", "openai"}


@dataclass
class AgentResponse:
    text: str
    tags: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


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
    import re
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
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


class AgentBridge:
    """Harness-agnostic bridge for optional cognition/LLM runtimes.

    The avatar runtime does not require this bridge to be connected. In offline
    modes it returns an empty response so perception-driven mirroring and manual
    controls continue to work without any LLM, speech model, or agent harness.
    External modes use a compact JSON contract that common harnesses can adapt
    to: OpenClaw, Hermes, Deerflow, or any HTTP/WebSocket service that accepts a
    user transcript plus affect summary and returns text/tags.

    The ``openai-compatible`` mode calls any OpenAI-chat-completions-compatible
    endpoint directly via httpx.  Base URL and model are read from the config
    system (``agent.base_url`` / ``agent.model`` in defaults.yaml, overridable
    via ``AGENT__BASE_URL`` / ``AGENT__MODEL``).  The API key is read from the
    ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        mode: str = "fake",
        url: str | None = None,
        harness: str = "generic",
        base_url: str = "https://api.openai.com",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.mode = normalize_agent_mode(mode)
        self.url = url
        self.harness = harness or "generic"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return self.mode not in OFFLINE_MODES

    def capability_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "backend": "agent_bridge",
            "mode": self.mode,
            "harness": self.harness,
            "url_configured": bool(self.url),
            "available": self.available,
            "last_error": self.last_error,
        }
        if self.mode in OPENAI_COMPATIBLE_MODES:
            status["base_url"] = self.base_url
            status["model"] = self.model
        return status

    # -- dispatch ----------------------------------------------------------

    async def generate_response(
        self, user_text: str, affect_state: UserAffectState
    ) -> AgentResponse:
        if self.mode in OFFLINE_MODES:
            return AgentResponse(text="", tags={}, source="offline")
        if self.mode in OPENAI_COMPATIBLE_MODES:
            return await self._openai_compatible(user_text, affect_state)
        if self.mode in EXTERNAL_MODES and self.url:
            return await self._external(user_text, affect_state)
        if self.mode in EXTERNAL_MODES and not self.url:
            self.last_error = "external agent mode selected without an agent URL"
            return AgentResponse(text="", tags={}, source="offline")

        from hermes_avatar.demo.fake_hermes import generate_response

        response = generate_response(user_text, affect_state)
        return AgentResponse(text=response.text, tags=response.tags, source="fake")

    # -- OpenAI-compatible chat completions --------------------------------

    async def _openai_compatible(
        self, user_text: str, affect_state: UserAffectState
    ) -> AgentResponse:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.last_error = "OPENAI_API_KEY not set"
            return AgentResponse(text="", tags={}, source="offline")

        payload = {
            "model": self.model,
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
        try:
            payload["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            self.last_error = (
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            )
            log.warning("openai-compatible HTTP error: %s", self.last_error)
            return AgentResponse(text="", tags={}, source="offline")
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("openai-compatible request failed: %s", self.last_error)
            return AgentResponse(text="", tags={}, source="offline")

        self.last_error = None
        text = self._extract_text(data)
        parsed = _extract_json_block(text) or {}
        spoken = str(parsed.get("text") or "").strip()
        tags = parsed.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        tags.setdefault("source_model", self.model)
        tags.setdefault("source_backend", "openai-compatible")
        tags["voice"] = _coerce_voice(tags.get("voice"))
        if "affect" in tags and not isinstance(tags["affect"], str):
            tags.pop("affect")
        return AgentResponse(text=spoken, tags=tags, source="openai-compatible")

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join([part for part in parts if part])
        for key in ("text", "content", "response", "output"):
            if isinstance(data.get(key), str):
                return data[key]
        return ""

    # -- external agent harness (unchanged) --------------------------------

    async def _external(
        self, user_text: str, affect_state: UserAffectState
    ) -> AgentResponse:
        import websockets

        payload = {
            "type": "user.transcript",
            "schema": "liveemote.agent.v1",
            "harness": self.harness,
            "text": user_text,
            "affect": affect_state.to_dict(),
        }
        try:
            if self.url and self.url.startswith("ws"):
                async with websockets.connect(self.url) as ws:
                    await ws.send(json.dumps(payload))
                    data = json.loads(await ws.recv())
            else:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.post(self.url or "", json=payload)
                    response.raise_for_status()
                    data = response.json()
        except Exception as exc:
            self.last_error = str(exc)
            return AgentResponse(text="", tags={}, source="offline")
        self.last_error = None
        return normalize_agent_response(data, source=self.harness)


def normalize_agent_mode(mode: str | None) -> str:
    normalized = (mode or "fake").strip().lower().replace("_", "-")
    if normalized in {"no-llm", "no llm"}:
        return "offline"
    if normalized in {"openai", "openai-compatible"}:
        return "openai-compatible"
    return normalized


def normalize_agent_response(
    data: dict[str, Any], source: str = "external"
) -> AgentResponse:
    """Accept common agent response shapes without tying to one harness."""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    text = (
        data.get("text")
        or data.get("content")
        or data.get("response")
        or message.get("content")
        or message.get("text")
        or output.get("text")
        or ""
    )
    tags = (
        data.get("tags")
        or data.get("affect_tags")
        or data.get("emotion")
        or output.get("tags")
        or {}
    )
    if not isinstance(tags, dict):
        tags = {"value": tags}
    return AgentResponse(
        text=str(text), tags=tags, source=str(data.get("source") or source)
    )
