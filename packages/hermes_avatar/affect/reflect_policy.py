from __future__ import annotations
from .state import UserAffectState

def reflected_affect(user: UserAffectState) -> tuple[str, float]:
    if user.dominant_expression == "happy":
        return "warm_acknowledging", 0.36
    if user.dominant_expression in {"angry", "frustrated"} or user.tension > 0.55:
        return "apologetic_grounded", 0.26
    if user.dominant_expression == "sad":
        return "warm_steady_consoling", 0.32
    if user.gaze_direction == "away":
        return "thinking_glance_down", 0.24
    return "attentive_soft", 0.3
