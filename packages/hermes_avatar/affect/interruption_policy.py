from __future__ import annotations
from .state import UserAffectState, ConversationState

def interruption_risk(user: UserAffectState, conversation: ConversationState) -> float:
    if conversation.turn_state != "assistant_speaking":
        return 0.0
    return min(1.0, user.speech_energy * 0.7 + (0.3 if user.speaking else 0.0))
