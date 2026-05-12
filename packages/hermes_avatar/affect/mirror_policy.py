from __future__ import annotations
from .state import UserAffectState

def mirrored_affect(user: UserAffectState) -> tuple[str, float]:
    if user.dominant_expression == "happy":
        return "small_delayed_smile", 0.28
    if user.dominant_expression in {"angry", "frustrated"}:
        return "grounded_not_angry", 0.12
    if user.dominant_expression == "sad":
        return "soft_concern", 0.18
    return "neutral_attentive", 0.12
