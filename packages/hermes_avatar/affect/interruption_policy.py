from __future__ import annotations
from .state import UserAffectState, ConversationState

# Weights for the interruption heuristic. Speech energy is the dominant signal
# (a sudden loud burst while the avatar talks), with a smaller flat contribution
# when the user is actively speaking at all (over-talk / barge-in).
_ENERGY_WEIGHT = 0.7
_SPEAKING_BONUS = 0.3

def interruption_risk(user: UserAffectState, conversation: ConversationState) -> float:
    """Estimate how likely the user is about to interrupt the avatar mid-utterance.

    Only meaningful while the avatar is speaking (``assistant_speaking``); otherwise
    the risk is definitionally zero. The score combines the user's smoothed speech
    energy with a smaller bonus when the user is already producing speech, capped at
    1.0. Downstream code can threshold this to decide whether to yield the floor.
    """
    if conversation.turn_state != "assistant_speaking":
        return 0.0
    return min(1.0, user.speech_energy * _ENERGY_WEIGHT + (_SPEAKING_BONUS if user.speaking else 0.0))
