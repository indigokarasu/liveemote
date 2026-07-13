from __future__ import annotations
from .state import UserAffectState, AvatarBehaviorState


def speaking_behavior(
    user: UserAffectState,
    hermes_tags: dict | None,
    emote_id: str | None,
    out: AvatarBehaviorState | None = None,
) -> AvatarBehaviorState:
    """Build the avatar behavior while it is the one talking (assistant turn).

    Affect is taken from the agent's ``hermes_tags`` when present (so the chosen
    voice/emotion styling drives the face), falling back to a grounded/focused read
    derived from the user's tension. Lip-sync is enabled and gaze stays toward the
    user. If ``out`` is provided it is filled in place (pooled hot path); otherwise a
    fresh ``AvatarBehaviorState`` is allocated.
    """
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
