from __future__ import annotations
from hermes_avatar.affect.state import UserAffectState
from hermes_avatar.protocol.hermes_bridge import HermesResponse

def generate_response(user_text: str, affect_state: UserAffectState) -> HermesResponse:
    affect = "grounded" if affect_state.tension > 0.45 else "focused"
    return HermesResponse(
        text="I’m tracking the interaction state locally, then using speech only when it is my turn.",
        tags={"affect": affect, "delivery": "calm_precise", "gesture_hints": ["small_nod", "steady_gaze"], "voice": {"pace": 0.44, "warmth": 0.62, "intensity": 0.35}},
    )
