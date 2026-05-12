from __future__ import annotations
from .state import UserAffectState, ConversationState, AvatarBehaviorState

def listening_behavior(user: UserAffectState, conversation: ConversationState, emote_id: str | None) -> AvatarBehaviorState:
    nod = conversation.silence_ms > 350
    return AvatarBehaviorState(mode="listening", affect="attentive_soft" if not nod else "attentive_nod", gaze_target="toward_user", emote_id=emote_id, intensity=max(0.25, min(0.65, user.speech_energy)), lip_sync_enabled=False)
