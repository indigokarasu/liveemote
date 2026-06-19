from __future__ import annotations
from hermes_avatar.affect.state import UserAffectState
from hermes_avatar.protocol.agent_bridge import AgentResponse as HermesResponse

def generate_response(user_text: str, affect_state: UserAffectState) -> HermesResponse:
    affect = "grounded" if affect_state.tension > 0.45 else "focused"
    return HermesResponse(
        text="I am tracking the interaction state locally.",
        tags={"affect": affect, "delivery": "calm_precise", "gesture_hints": ["small_nod", "steady_gaze"], "voice": {"pace": 0.44, "warmth": 0.62, "intensity": 0.35}},
    )
