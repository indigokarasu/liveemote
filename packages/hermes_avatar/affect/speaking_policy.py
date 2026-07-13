from __future__ import annotations
from .state import UserAffectState, AvatarBehaviorState


def speaking_behavior(
    user: UserAffectState,
    hermes_tags: dict | None,
    emote_id: str | None,
    out: AvatarBehaviorState | None = None,
) -> AvatarBehaviorState:
    tags = hermes_tags or {}
    voice = tags.get("voice", {}) if isinstance(tags.get("voice", {}), dict) else {}
    affect = tags.get("affect") or ("grounded" if user.tension > 0.5 else "focused")
    intensity = float(voice.get("intensity", 0.4))
    if out is None:
        return AvatarBehaviorState(
            mode="speaking",
            affect=str(affect),
            gaze_target="toward_user",
            emote_id=emote_id,
            intensity=intensity,
            lip_sync_enabled=True,
            mirror_strength=0.0,
        )
    out.mode = "speaking"
    out.affect = str(affect)
    out.gaze_target = "toward_user"
    out.emote_id = emote_id
    out.intensity = intensity
    out.lip_sync_enabled = True
    out.mirror_strength = 0.0
    out.delay_ms = 0
    out.full_body_pose = "standing_idle"
    return out
