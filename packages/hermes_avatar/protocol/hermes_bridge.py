from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import httpx
import websockets
from hermes_avatar.affect.state import UserAffectState, ConversationState

@dataclass
class HermesResponse:
    text: str
    tags: dict[str, Any] = field(default_factory=dict)

class HermesBridge:
    def __init__(self, mode: str = "fake", url: str | None = None) -> None:
        self.mode = mode
        self.url = url

    async def generate_response(self, user_text: str, affect_state: UserAffectState) -> HermesResponse:
        if self.mode == "external" and self.url:
            return await self._external(user_text, affect_state)
        from hermes_avatar.demo.fake_hermes import generate_response
        return generate_response(user_text, affect_state)

    async def _external(self, user_text: str, affect_state: UserAffectState) -> HermesResponse:
        payload = {"type": "user.transcript", "text": user_text, "affect": affect_state.to_dict()}
        if self.url.startswith("ws"):
            async with websockets.connect(self.url) as ws:
                await ws.send(__import__("json").dumps(payload))
                data = __import__("json").loads(await ws.recv())
        else:
            async with httpx.AsyncClient(timeout=20) as client:
                data = (await client.post(self.url, json=payload)).json()
        return HermesResponse(text=data.get("text", ""), tags=data.get("tags", {}))

def affect_summary(user: UserAffectState, conversation: ConversationState, window_ms: int = 3000) -> dict[str, Any]:
    return {
        "type": "affect.summary", "window_ms": window_ms,
        "user": {"speaking": user.speaking, "attention": user.attention, "valence": user.valence,
                 "arousal": user.arousal, "dominant_expression": user.dominant_expression},
        "conversation": {"turn_state": conversation.turn_state, "interruption_risk": conversation.interruption_risk,
                         "tension": conversation.tension},
    }
