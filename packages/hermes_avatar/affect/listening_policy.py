from __future__ import annotations
from .state import UserAffectState, ConversationState, AvatarBehaviorState


def listening_behavior(
    user: UserAffectState,
    conversation: ConversationState,
    emote_id: str | None,
    out: AvatarBehaviorState | None = None,
) -> AvatarBehaviorState:
    """Build the avatar behavior while the user is talking (avatar listening turn).

    The avatar stays attentive and gaze-locked on the user. A subtle "nod" affect is
    layered in once the user has paused (``silence_ms > 350``), a lightweight
    back-channeling cue that the avatar is still tracking. Intensity scales with the
    user's speech energy, clamped to a calm [0.25, 0.65] band so listening never reads
    as agitated. If ``out`` is provided it is filled in place (pooled hot path).
    """
    nod = conversation.silence_ms > 350
    affect = "attentive_soft" if not nod else "attentive_nod"
    intensity = max(0.25, min(0.65, user.speech_energy))
    if out is None:
        return AvatarBehaviorState(
            mode="listening",
            affect=affect,
            gaze_target="toward_user",
            emote_id=emote_id,
            intensity=intensity,
            lip_sync_enabled=False,
        )
    out.mode = "listening"
    out.affect = affect
    out.gaze_target = "toward_user"
    out.emote_id = emote_id
    out.intensity = intensity
    out.lip_sync_enabled = False
    out.mirror_strength = 0.0
    out.delay_ms = 0
    out.full_body_pose = "standing_idle"
    return out
