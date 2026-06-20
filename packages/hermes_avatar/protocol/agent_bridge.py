from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json

import httpx
import websockets

from hermes_avatar.affect.state import UserAffectState

OFFLINE_MODES = {"none", "off", "offline", "disabled", "no_llm", "no-llm"}
FAKE_MODES = {"fake", "mock", "local"}
EXTERNAL_MODES = {"external", "agent", "harness", "openclaw", "hermes", "deerflow"}


@dataclass
class AgentResponse:
    text: str
    tags: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


class AgentBridge:
    """Harness-agnostic bridge for optional cognition/LLM runtimes.

    The avatar runtime does not require this bridge to be connected. In offline
    modes it returns an empty response so perception-driven mirroring and manual
    controls continue to work without any LLM, speech model, or agent harness.
    External modes use a compact JSON contract that common harnesses can adapt
    to: OpenClaw, Hermes, Deerflow, or any HTTP/WebSocket service that accepts a
    user transcript plus affect summary and returns text/tags.
    """

    def __init__(self, mode: str = "fake", url: str | None = None, harness: str = "generic") -> None:
        self.mode = normalize_agent_mode(mode)
        self.url = url
        self.harness = harness or "generic"
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return self.mode not in OFFLINE_MODES

    def capability_status(self) -> dict[str, Any]:
        return {
            "backend": "agent_bridge",
            "mode": self.mode,
            "harness": self.harness,
            "url_configured": bool(self.url),
            "available": self.available,
            "last_error": self.last_error,
        }

    async def generate_response(self, user_text: str, affect_state: UserAffectState) -> AgentResponse:
        if self.mode in OFFLINE_MODES:
            return AgentResponse(text="", tags={}, source="offline")
        if self.mode in EXTERNAL_MODES and self.url:
            return await self._external(user_text, affect_state)
        if self.mode in EXTERNAL_MODES and not self.url:
            self.last_error = "external agent mode selected without an agent URL"
            return AgentResponse(text="", tags={}, source="offline")

        from hermes_avatar.demo.fake_hermes import generate_response

        response = generate_response(user_text, affect_state)
        return AgentResponse(text=response.text, tags=response.tags, source="fake")

    async def _external(self, user_text: str, affect_state: UserAffectState) -> AgentResponse:
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
    return normalized


def normalize_agent_response(data: dict[str, Any], source: str = "external") -> AgentResponse:
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
    tags = data.get("tags") or data.get("affect_tags") or data.get("emotion") or output.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {"value": tags}
    return AgentResponse(text=str(text), tags=tags, source=str(data.get("source") or source))
