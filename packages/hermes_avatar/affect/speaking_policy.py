from __future__ import annotations
from .state import UserAffectState, AvatarBehaviorState

def speaking_behavior(user: UserAffectState, hermes_tags: dict | None, emote_id: str | None) -> AvatarBehaviorState:
    tags = hermes_tags or {}
    voice = tags.get("voice", {}) if isinstance(tags.get("voice", {}), dict) else {}
    affect = tags.get("affect") or ("grounded" if user.tension > 0.5 else "focused")
    return AvatarBehaviorState(mode="speaking", affect=str(affect), gaze_target="toward_user", emote_id=emote_id, intensity=float(voice.get("intensity", 0.4)), lip_sync_enabled=True, mirror_strength=0.0)
