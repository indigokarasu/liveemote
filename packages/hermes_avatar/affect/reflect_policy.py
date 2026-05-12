from __future__ import annotations
from .smoothing import clamp
from .state import UserAffectState


def reflected_affect(user: UserAffectState) -> tuple[str, float]:
    """Return a validating, regulated complement to the user's affect.

    Psychological model: reflection communicates accurate understanding without
    parroting or absorbing the user's emotion. It names the relational stance the
    avatar should embody: warm for positive affect, grounded for high tension,
    consoling for sadness, and spacious when attention/gaze moves away.
    """
    tension = clamp(user.tension, 0.0, 1.0)
    arousal = clamp(user.arousal, 0.0, 1.0)
    attention = clamp(user.attention, 0.0, 1.0)

    if user.dominant_expression == "happy" and user.valence >= 0:
        return "warm_acknowledging", clamp(
            0.28 + (0.10 * attention) + (0.04 * arousal), 0.28, 0.44
        )
    if user.dominant_expression in {"angry", "frustrated"} or tension > 0.55:
        return "validating_grounded", clamp(
            0.20 + (0.08 * tension) - (0.03 * arousal), 0.18, 0.30
        )
    if user.dominant_expression == "sad" or user.valence < -0.35:
        return "warm_steady_consoling", clamp(0.24 + (0.06 * attention), 0.24, 0.34)
    if user.dominant_expression == "tired" or arousal < 0.18:
        return "patient_low_energy_attentive", 0.22
    if user.gaze_direction == "away" or attention < 0.25:
        return "spacious_attentive", 0.22
    return "attentive_soft", 0.28
